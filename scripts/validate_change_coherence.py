"""Classify a structured change-coherence record without repository-specific cases."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, cast


class Outcome(str, Enum):
    """Next action derived from the current change record."""

    DIRECT = "DIRECT"
    SIBLING_SEARCH_REQUIRED = "SIBLING_SEARCH_REQUIRED"
    FIX_REQUIRED = "FIX_REQUIRED"
    CONTRACT_REPLAN_REQUIRED = "CONTRACT_REPLAN_REQUIRED"
    REVIEW_READY = "REVIEW_READY"


class RecordError(ValueError):
    """Raised when a change record does not satisfy the structural contract."""


STABILITY_DIMENSIONS = (
    "acceptance",
    "contract",
    "architecture",
    "ownership",
    "compatibility",
    "scope",
)
REPLAN_SIGNALS = {
    "repeated_invariant_family",
    "cascading_local_fixes",
    "guard_withdrawn",
    "material_behavior_change",
    "material_architecture_change",
    "material_ownership_change",
    "material_compatibility_change",
    "material_scope_change",
    "unexpected_production_domains",
    "unstable_acceptance",
}
MANDATORY_REPLAN_SIGNALS = REPLAN_SIGNALS - {
    "repeated_invariant_family",
    "cascading_local_fixes",
}


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecordError(f"{path} must be an object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, path: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise RecordError(f"{path} must be an array")
    return value


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecordError(f"{path} must be a non-empty string")
    return value.strip()


def _text_list(value: object, path: str, *, allow_empty: bool = False) -> list[str]:
    values = [_text(item, f"{path}[{index}]") for index, item in enumerate(_sequence(value, path))]
    if not allow_empty and not values:
        raise RecordError(f"{path} must not be empty")
    return values


def _choice(value: object, path: str, allowed: set[str]) -> str:
    selected = _text(value, path)
    if selected not in allowed:
        choices = ", ".join(sorted(allowed))
        raise RecordError(f"{path} must be one of: {choices}")
    return selected


def _validate_contract(record: Mapping[str, Any]) -> None:
    contract = _mapping(record.get("contract"), "contract")
    _text(contract.get("summary"), "contract.summary")
    _text_list(contract.get("acceptance"), "contract.acceptance")

    behavior = _mapping(contract.get("behavior"), "contract.behavior")
    for category in ("supported", "restricted", "invalid"):
        if category not in behavior:
            raise RecordError(f"contract.behavior.{category} is required")
        _text_list(behavior[category], f"contract.behavior.{category}", allow_empty=True)

    owners = _sequence(contract.get("owners"), "contract.owners")
    if not owners:
        raise RecordError("contract.owners must not be empty")
    for index, value in enumerate(owners):
        owner = _mapping(value, f"contract.owners[{index}]")
        _text(owner.get("state"), f"contract.owners[{index}].state")
        _text(owner.get("owner"), f"contract.owners[{index}].owner")

    scope = _mapping(record.get("scope"), "scope")
    _text_list(scope.get("initial"), "scope.initial")
    _text_list(scope.get("current"), "scope.current")


def _validate_stability(record: Mapping[str, Any]) -> bool:
    stability = _mapping(record.get("stability"), "stability")
    unstable = False
    for dimension in STABILITY_DIMENSIONS:
        state = _choice(stability.get(dimension), f"stability.{dimension}", {"stable", "changed", "uncertain"})
        unstable = unstable or state != "stable"
    return unstable


def _validate_replan(record: Mapping[str, Any]) -> bool:
    replan = _mapping(record.get("replan"), "replan")
    signals = set(_text_list(replan.get("signals"), "replan.signals", allow_empty=True))
    unknown = signals - REPLAN_SIGNALS
    if unknown:
        raise RecordError(f"replan.signals contains unknown values: {', '.join(sorted(unknown))}")

    decision = _choice(
        replan.get("decision"),
        "replan.decision",
        {"not_required", "continue", "required", "completed"},
    )
    if signals and decision == "not_required":
        raise RecordError("replan signals require an explicit continue, required, or completed decision")
    if not signals and decision not in {"not_required", "completed"}:
        raise RecordError("replan.decision requires at least one signal")
    if signals or decision == "completed":
        _text(replan.get("rationale"), "replan.rationale")

    scope = _mapping(record.get("scope"), "scope")
    initial_scope = set(_text_list(scope.get("initial"), "scope.initial"))
    current_scope = set(_text_list(scope.get("current"), "scope.current"))
    unassessed_scope_drift = initial_scope != current_scope and "material_scope_change" not in signals

    mandatory_signal_is_open = bool(signals & MANDATORY_REPLAN_SIGNALS) and decision != "completed"
    return decision == "required" or mandatory_signal_is_open or unassessed_scope_drift


def _validate_findings(record: Mapping[str, Any]) -> tuple[bool, bool]:
    findings = _sequence(record.get("findings"), "findings")
    sibling_search_missing = False
    fix_pending = False
    for index, value in enumerate(findings):
        finding = _mapping(value, f"findings[{index}]")
        _text(finding.get("id"), f"findings[{index}].id")
        _text(finding.get("invariant_family"), f"findings[{index}].invariant_family")
        status = _choice(
            finding.get("status"),
            f"findings[{index}].status",
            {"open", "resolved", "deferred"},
        )
        fix_pending = fix_pending or status == "open"
        if status == "deferred":
            _text(finding.get("tracking"), f"findings[{index}].tracking")

        search_value = finding.get("sibling_search")
        if search_value is None:
            sibling_search_missing = True
            continue
        search = _mapping(search_value, f"findings[{index}].sibling_search")
        _text_list(search.get("boundaries"), f"findings[{index}].sibling_search.boundaries")
        _text_list(search.get("checked"), f"findings[{index}].sibling_search.checked")
        _text(search.get("result"), f"findings[{index}].sibling_search.result")
    return sibling_search_missing, fix_pending


def _validate_review_evidence(record: Mapping[str, Any]) -> tuple[bool, bool]:
    open_decisions = _text_list(record.get("open_decisions"), "open_decisions", allow_empty=True)

    deferred_work = _sequence(record.get("deferred_work"), "deferred_work")
    for index, value in enumerate(deferred_work):
        deferred = _mapping(value, f"deferred_work[{index}]")
        _text(deferred.get("description"), f"deferred_work[{index}].description")
        _text(deferred.get("tracking"), f"deferred_work[{index}].tracking")

    alignment = _mapping(record.get("alignment"), "alignment")
    alignment_pending = False
    for concern in ("implementation", "tests", "documentation", "tracking"):
        state = _choice(
            alignment.get(concern),
            f"alignment.{concern}",
            {"aligned", "not_applicable", "drifted"},
        )
        if concern in {"implementation", "tests"} and state == "not_applicable":
            raise RecordError(f"alignment.{concern} cannot be not_applicable")
        alignment_pending = alignment_pending or state == "drifted"

    validation = _sequence(record.get("validation"), "validation")
    if not validation:
        raise RecordError("validation must not be empty")
    validation_pending = False
    for index, value in enumerate(validation):
        evidence = _mapping(value, f"validation[{index}]")
        _text(evidence.get("command"), f"validation[{index}].command")
        result = _choice(
            evidence.get("result"),
            f"validation[{index}].result",
            {"passed", "failed", "skipped"},
        )
        validation_pending = validation_pending or result == "failed"
        if result == "skipped":
            _text(evidence.get("notes"), f"validation[{index}].notes")

    _text_list(record.get("residual_risk"), "residual_risk", allow_empty=True)

    review = _mapping(record.get("review"), "review")
    current_head = _text(review.get("current_head"), "review.current_head")
    evidence_head = _text(review.get("evidence_head"), "review.evidence_head")
    current_revision = _text(review.get("contract_revision"), "review.contract_revision")
    evidence_revision = _text(review.get("evidence_contract_revision"), "review.evidence_contract_revision")
    revision_changed = current_revision != evidence_revision or bool(open_decisions)
    evidence_pending = current_head != evidence_head
    return revision_changed, alignment_pending or validation_pending or evidence_pending


def evaluate_record(record: Mapping[str, Any]) -> Outcome:
    """Validate ``record`` and derive the next coherent action."""

    if record.get("schema_version") != 1:
        raise RecordError("schema_version must be 1")
    risk = _choice(record.get("risk"), "risk", {"low", "medium", "high"})
    action = _choice(record.get("requested_action"), "requested_action", {"direct", "fix", "review"})
    _text(record.get("summary"), "summary")

    if action == "direct":
        if risk != "low":
            raise RecordError("requested_action direct is limited to low-risk changes")
        if record.get("findings") not in (None, []):
            raise RecordError("direct changes cannot carry review findings")
        return Outcome.DIRECT

    _validate_contract(record)
    unstable = _validate_stability(record)
    replan_required = _validate_replan(record)
    sibling_search_missing, fix_pending = _validate_findings(record)

    if unstable or replan_required:
        return Outcome.CONTRACT_REPLAN_REQUIRED
    if sibling_search_missing:
        return Outcome.SIBLING_SEARCH_REQUIRED
    if action == "fix":
        if not fix_pending:
            raise RecordError("requested_action fix requires at least one open finding")
        return Outcome.FIX_REQUIRED

    revision_changed, review_fix_pending = _validate_review_evidence(record)
    if revision_changed:
        return Outcome.CONTRACT_REPLAN_REQUIRED
    if fix_pending or review_fix_pending:
        return Outcome.FIX_REQUIRED
    return Outcome.REVIEW_READY


def _load_record(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RecordError(f"cannot read {path}: {error}") from error
    return _mapping(value, "record")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line validator."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path, help="JSON change-coherence record")
    parser.add_argument("--expect", choices=[outcome.value for outcome in Outcome])
    args = parser.parse_args(argv)
    try:
        outcome = evaluate_record(_load_record(args.record))
    except RecordError as error:
        print(f"INVALID: {error}", file=sys.stderr)
        return 2

    print(outcome.value)
    if args.expect is not None and args.expect != outcome.value:
        print(f"expected {args.expect}, got {outcome.value}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
