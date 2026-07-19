from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.validate_change_coherence import Outcome, RecordError, evaluate_record, main

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_RECORD = (
    REPO_ROOT / ".agents" / "skills" / "wandas-change-coherence" / "references" / "change-record.example.json"
)


def _risk_record(*, action: str = "review") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "risk": "high",
        "requested_action": action,
        "summary": "Keep one state owner across every public boundary.",
        "contract": {
            "summary": "The owner validates state before it reaches consumers.",
            "acceptance": ["Every supported entry path preserves the same state invariant."],
            "behavior": {
                "supported": ["State produced through the owner"],
                "restricted": ["State adapted at an explicit boundary"],
                "invalid": ["State that bypasses owner validation"],
            },
            "owners": [{"state": "shared state", "owner": "the construction boundary"}],
        },
        "scope": {
            "initial": ["construction and public access"],
            "current": ["construction and public access"],
        },
        "stability": {
            "acceptance": "stable",
            "contract": "stable",
            "architecture": "stable",
            "ownership": "stable",
            "compatibility": "stable",
            "scope": "stable",
        },
        "replan": {"signals": [], "decision": "not_required"},
        "findings": [],
        "open_decisions": [],
        "deferred_work": [],
        "alignment": {
            "implementation": "aligned",
            "tests": "aligned",
            "documentation": "aligned",
            "tracking": "not_applicable",
        },
        "validation": [{"command": "focused validation", "result": "passed"}],
        "residual_risk": [],
        "review": {
            "current_head": "head-2",
            "evidence_head": "head-2",
            "contract_revision": "contract-2",
            "evidence_contract_revision": "contract-2",
        },
    }


def _open_finding() -> dict[str, Any]:
    return {
        "id": "finding-a",
        "invariant_family": "state ownership is bypassed",
        "status": "open",
    }


def test_small_clear_change_can_run_directly_without_full_record() -> None:
    record = {
        "schema_version": 1,
        "risk": "low",
        "requested_action": "direct",
        "summary": "Correct one unambiguous local label.",
    }

    assert evaluate_record(record) is Outcome.DIRECT


def test_repository_example_is_a_valid_review_ready_record() -> None:
    record = json.loads(EXAMPLE_RECORD.read_text(encoding="utf-8"))

    assert evaluate_record(record) is Outcome.REVIEW_READY


def test_finding_requires_bounded_sibling_search_before_fix() -> None:
    record = _risk_record(action="fix")
    record["findings"] = [_open_finding()]

    assert evaluate_record(record) is Outcome.SIBLING_SEARCH_REQUIRED

    finding = record["findings"][0]
    finding["sibling_search"] = {
        "boundaries": ["construction", "transformation", "public access"],
        "checked": ["all paths that consume the same state owner"],
        "result": "One sibling shares the cause; serialization is outside the owner boundary.",
    }

    assert evaluate_record(record) is Outcome.FIX_REQUIRED


@pytest.mark.parametrize(
    ("dimension", "state"),
    [
        ("acceptance", "uncertain"),
        ("contract", "uncertain"),
        ("architecture", "changed"),
        ("ownership", "changed"),
        ("compatibility", "uncertain"),
        ("scope", "changed"),
    ],
)
def test_unstable_material_boundary_cannot_be_review_ready(dimension: str, state: str) -> None:
    record = _risk_record()
    record["stability"][dimension] = state

    assert evaluate_record(record) is Outcome.CONTRACT_REPLAN_REQUIRED


def test_replan_signal_may_continue_only_after_explicit_assessment() -> None:
    record = _risk_record(action="fix")
    finding = _open_finding()
    finding["sibling_search"] = {
        "boundaries": ["construction", "mutation"],
        "checked": ["both paths"],
        "result": "The shared owner contains both cases.",
    }
    record["findings"] = [finding]
    record["replan"] = {
        "signals": ["repeated_invariant_family"],
        "decision": "continue",
        "rationale": "The contract is unchanged and the bounded family has one shared fix.",
    }

    assert evaluate_record(record) is Outcome.FIX_REQUIRED

    record["replan"]["decision"] = "required"
    assert evaluate_record(record) is Outcome.CONTRACT_REPLAN_REQUIRED


def test_material_drift_signal_requires_completed_replan() -> None:
    record = _risk_record()
    record["replan"] = {
        "signals": ["material_architecture_change"],
        "decision": "continue",
        "rationale": "A local patch was proposed without revising the contract.",
    }

    assert evaluate_record(record) is Outcome.CONTRACT_REPLAN_REQUIRED

    record["replan"]["decision"] = "completed"
    record["replan"]["rationale"] = "The new architecture is represented by contract-2 and current evidence."
    assert evaluate_record(record) is Outcome.REVIEW_READY


def test_responsibility_scope_drift_requires_replan_assessment() -> None:
    record = _risk_record()
    record["scope"]["current"] = ["construction", "public access", "persistence"]

    assert evaluate_record(record) is Outcome.CONTRACT_REPLAN_REQUIRED

    record["replan"] = {
        "signals": ["material_scope_change"],
        "decision": "required",
        "rationale": "Production work reached an unplanned responsibility domain.",
    }
    assert evaluate_record(record) is Outcome.CONTRACT_REPLAN_REQUIRED

    record["replan"]["decision"] = "completed"
    record["replan"]["rationale"] = "The revised contract and evidence include the expanded scope."
    assert evaluate_record(record) is Outcome.REVIEW_READY


def test_review_ready_requires_current_contract_and_head_evidence() -> None:
    record = _risk_record()
    assert evaluate_record(record) is Outcome.REVIEW_READY

    stale_head = copy.deepcopy(record)
    stale_head["review"]["evidence_head"] = "head-1"
    assert evaluate_record(stale_head) is Outcome.FIX_REQUIRED

    stale_contract = copy.deepcopy(record)
    stale_contract["review"]["evidence_contract_revision"] = "contract-1"
    assert evaluate_record(stale_contract) is Outcome.CONTRACT_REPLAN_REQUIRED


@pytest.mark.parametrize("concern", ["implementation", "tests", "documentation", "tracking"])
def test_review_ready_rejects_alignment_drift(concern: str) -> None:
    record = _risk_record()
    record["alignment"][concern] = "drifted"

    assert evaluate_record(record) is Outcome.FIX_REQUIRED


def test_review_ready_rejects_material_open_decision_and_failed_validation() -> None:
    open_decision = _risk_record()
    open_decision["open_decisions"] = ["Which boundary owns compatibility behavior?"]
    assert evaluate_record(open_decision) is Outcome.CONTRACT_REPLAN_REQUIRED

    failed_validation = _risk_record()
    failed_validation["validation"][0]["result"] = "failed"
    assert evaluate_record(failed_validation) is Outcome.FIX_REQUIRED


def test_deferred_work_requires_tracking() -> None:
    record = _risk_record()
    record["deferred_work"] = [{"description": "Evaluate an independent extension."}]

    with pytest.raises(RecordError, match=r"deferred_work\[0\]\.tracking"):
        evaluate_record(record)


def test_cli_expectation_is_a_gate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "change.json"
    path.write_text(json.dumps(_risk_record()), encoding="utf-8")

    assert main([str(path), "--expect", "REVIEW_READY"]) == 0
    assert capsys.readouterr().out.strip() == "REVIEW_READY"

    assert main([str(path), "--expect", "CONTRACT_REPLAN_REQUIRED"]) == 1
    captured = capsys.readouterr()
    assert captured.out.strip() == "REVIEW_READY"
    assert "expected CONTRACT_REPLAN_REQUIRED" in captured.err
