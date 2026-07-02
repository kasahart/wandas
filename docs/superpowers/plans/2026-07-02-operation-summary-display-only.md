# Operation Summary Display-Only Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `operation_summaries` / `to_summary()` as display-only, strict-JSON-safe operation history summaries with no persistence or portability contract.

**Architecture:** Add JSON-safe display conversion helpers in `wandas/processing/base.py`, keep `AudioOperation.to_summary()` and `BinaryOperation.to_summary()` thin, and expose `BaseFrame.operation_summaries` as a read-only lineage projection. Close PR #250 and open a new PR from `develop` so the old portability-review history does not define the new contract.

**Tech Stack:** Python 3.10, Dask, NumPy, pytest, ruff, ty, GitHub CLI.

---

## File Structure

- Modify `wandas/processing/base.py`
  - Add `OperationSummary`.
  - Add `_summary_value()` and `_summary_sort_key()`.
  - Add `AudioOperation.to_summary()`.
  - Add `BinaryOperation.to_summary()`.
  - Keep `_operand_descriptor()` as the binary operand display descriptor helper.
- Modify `wandas/processing/custom.py`
  - Add `_callable_reference()`.
  - Add `CustomOperation.to_summary()` with an `implementation` display field.
- Modify `wandas/core/base_frame.py`
  - Add `_operation_summary()`.
  - Add `_lineage_to_summaries()`.
  - Add `operation_summaries`.
  - Update display-only docstrings.
- Modify `tests/processing/test_base_operations.py`
  - Add direct summary tests for `AudioOperation` and `BinaryOperation`.
- Modify `tests/processing/test_custom_operation.py`
  - Add direct summary test for `CustomOperation`.
- Modify `tests/core/test_base_frame_lineage.py`
  - Add `operation_summaries` lineage, strict JSON, and no-compute tests.
- GitHub operations
  - Edit PR #250 body to remove `Closes #244` and `Part of #232`.
  - Comment on PR #250 that the persistence-oriented contract is withdrawn.
  - Close PR #250.
  - Push this branch and open a replacement PR.

---

## Task 1: Add Display Summary Conversion and Operation Methods

**Files:**
- Modify: `wandas/processing/base.py`
- Test: `tests/processing/test_base_operations.py`

- [ ] **Step 1: Write failing tests for display-only operation summaries**

Add imports near the top of `tests/processing/test_base_operations.py`:

```python
import json
from fractions import Fraction
```

Add these tests after `test_binary_operation_params_delegate_to_params()`:

```python
def test_binary_operation_to_summary_returns_display_only_summary() -> None:
    operation = BinaryOperation(symbol="+", operand_kind="scalar", operand=2.0)

    assert operation.to_summary() == {
        "operation": "+",
        "params": {
            "symbol": "+",
            "operand_kind": "scalar",
            "operand": {"type": "float", "value": 2.0},
        },
    }


def test_binary_operation_to_summary_describes_array_operand_without_values() -> None:
    operation = BinaryOperation(symbol="+", operand_kind="scalar", operand=np.array([0.5, 1.5]))

    summary = operation.to_summary()

    assert summary == {
        "operation": "+",
        "params": {
            "symbol": "+",
            "operand_kind": "scalar",
            "operand": {"type": "ndarray", "shape": [2], "dtype": "float64"},
        },
    }
    json.dumps(summary, allow_nan=False)
```

Add these methods inside `TestAudioOperation` after `test_process_array_removed_from_audio_operation()`:

```python
    def test_audio_operation_to_summary_returns_display_only_summary(self) -> None:
        test_op_cls = self._make_test_op_class()
        op = test_op_cls(16000, gain=2.0, enabled=True)

        assert op.to_summary() == {
            "operation": "test_op",
            "params": {"gain": 2.0, "enabled": True},
        }

    def test_audio_operation_summary_sanitizes_display_values_for_strict_json(self) -> None:
        test_op_cls = self._make_test_op_class()

        def transform(x: NDArrayReal) -> NDArrayReal:
            return x

        op = test_op_cls(
            16000,
            norm=np.inf,
            ratio=Fraction(1, 3),
            weights=np.array([0.1, 0.9]),
            config={1: "linear", "callable": transform},
            channels={2, 1},
        )

        summary = op.to_summary()

        assert summary == {
            "operation": "test_op",
            "params": {
                "norm": {"type": "float", "value": "inf"},
                "ratio": {"type": "Fraction"},
                "weights": {"type": "ndarray", "shape": [2], "dtype": "float64"},
                "config": {
                    "1": "linear",
                    "callable": {
                        "type": "callable",
                        "name": (
                            "TestAudioOperation."
                            "test_audio_operation_summary_sanitizes_display_values_for_strict_json."
                            "<locals>.transform"
                        ),
                    },
                },
                "channels": [1, 2],
            },
        }
        json.dumps(summary, allow_nan=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest \
  tests/processing/test_base_operations.py::test_binary_operation_to_summary_returns_display_only_summary \
  tests/processing/test_base_operations.py::test_binary_operation_to_summary_describes_array_operand_without_values \
  tests/processing/test_base_operations.py::TestAudioOperation::test_audio_operation_to_summary_returns_display_only_summary \
  tests/processing/test_base_operations.py::TestAudioOperation::test_audio_operation_summary_sanitizes_display_values_for_strict_json \
  -q
```

Expected: FAIL because `BinaryOperation` and `AudioOperation` do not have `to_summary()`.

- [ ] **Step 3: Implement display summary helpers and methods**

In `wandas/processing/base.py`, add `json` and `numbers` imports:

```python
import json
import logging
import numbers
```

Add this alias after the existing type aliases:

```python
OperationSummary = dict[str, Any]
```

Add these helpers after `_operand_descriptor()`:

```python
def _summary_sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _summary_value(value: Any) -> Any:
    """Return a lightweight, JSON-safe value for display summaries."""
    if callable(value):
        return {"type": "callable", "name": getattr(value, "__qualname__", type(value).__name__)}
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, np.timedelta64 | np.datetime64):
        return {"type": type(value).__name__}
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Rational):
        return {"type": type(value).__name__}
    if isinstance(value, numbers.Real):
        numeric = float(value)
        if np.isfinite(numeric):
            return numeric
        if np.isnan(numeric):
            return {"type": "float", "value": "nan"}
        if numeric > 0:
            return {"type": "float", "value": "inf"}
        return {"type": "float", "value": "-inf"}
    if isinstance(value, numbers.Complex):
        return {
            "type": "complex",
            "real": _summary_value(value.real),
            "imag": _summary_value(value.imag),
        }
    if isinstance(value, Mapping):
        return {str(key): _summary_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_summary_value(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted((_summary_value(item) for item in value), key=_summary_sort_key)
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": [_summary_value(item) for item in value.shape], "dtype": str(value.dtype)}
    if isinstance(value, DaArray):
        return {
            "type": "dask.array",
            "shape": [_summary_value(item) for item in value.shape],
            "dtype": str(value.dtype),
            "chunks": [[_summary_value(item) for item in chunk] for chunk in value.chunks],
        }
    return {"type": type(value).__name__}
```

Add this method to `BinaryOperation` after `to_params()`:

```python
    def to_summary(self) -> OperationSummary:
        """Return a lightweight display summary for this operation."""
        params = self.to_params()
        return {
            "operation": self.symbol,
            "params": {key: _summary_value(value) for key, value in params.items()},
        }
```

Add this method to `AudioOperation` after `to_params()`:

```python
    def to_summary(self) -> OperationSummary:
        """Return a lightweight display summary for this operation."""
        params = self.to_params()
        return {
            "operation": self.name,
            "params": {str(key): _summary_value(value) for key, value in params.items()},
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run pytest \
  tests/processing/test_base_operations.py::test_binary_operation_to_summary_returns_display_only_summary \
  tests/processing/test_base_operations.py::test_binary_operation_to_summary_describes_array_operand_without_values \
  tests/processing/test_base_operations.py::TestAudioOperation::test_audio_operation_to_summary_returns_display_only_summary \
  tests/processing/test_base_operations.py::TestAudioOperation::test_audio_operation_summary_sanitizes_display_values_for_strict_json \
  -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add wandas/processing/base.py tests/processing/test_base_operations.py
git commit -m "Add display-only operation summaries"
```

---

## Task 2: Add Frame `operation_summaries`

**Files:**
- Modify: `wandas/core/base_frame.py`
- Test: `tests/core/test_base_frame_lineage.py`

- [ ] **Step 1: Write failing frame summary tests**

Add `json` import to `tests/core/test_base_frame_lineage.py`:

```python
import json
```

Add these tests after `test_operation_history_public_behavior_is_read_only_lineage_view()`:

```python
def test_operation_summaries_returns_display_lineage_summaries() -> None:
    result = _frame().high_pass_filter(100).normalize()

    assert result.operation_summaries == [
        {"operation": "highpass_filter", "params": {"cutoff": 100.0, "order": 4}},
        {"operation": "normalize", "params": {"norm": {"type": "float", "value": "inf"}, "axis": 1}},
    ]


def test_operation_summaries_do_not_compute_data() -> None:
    result = _frame().normalize()

    with mock.patch("dask.array.core.Array.compute") as compute:
        summaries = result.operation_summaries

    compute.assert_not_called()
    assert summaries == [{"operation": "normalize", "params": {"norm": {"type": "float", "value": "inf"}, "axis": 1}}]


def test_operation_summaries_are_strict_json_serializable() -> None:
    summaries = _frame().normalize().operation_summaries

    json.dumps(summaries, allow_nan=False)


def test_operation_summaries_include_multi_input_lineage() -> None:
    left = _frame().normalize()
    right = _frame().remove_dc()

    result = left + right

    assert [summary["operation"] for summary in result.operation_summaries] == ["normalize", "remove_dc", "+"]
    assert result.operation_summaries[-1]["params"] == {
        "symbol": "+",
        "operand_kind": "frame",
        "operand": {"type": "frame", "label": "lineage"},
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest \
  tests/core/test_base_frame_lineage.py::test_operation_summaries_returns_display_lineage_summaries \
  tests/core/test_base_frame_lineage.py::test_operation_summaries_do_not_compute_data \
  tests/core/test_base_frame_lineage.py::test_operation_summaries_are_strict_json_serializable \
  tests/core/test_base_frame_lineage.py::test_operation_summaries_include_multi_input_lineage \
  -q
```

Expected: FAIL because `BaseFrame` does not expose `operation_summaries`.

- [ ] **Step 3: Implement frame summary projection**

In `wandas/core/base_frame.py`, add this helper after `_lineage_to_graph()`:

```python
    @classmethod
    def _operation_summary(cls, operation: Any) -> dict[str, Any]:
        if hasattr(operation, "to_summary"):
            return cast(dict[str, Any], operation.to_summary())
        from wandas.processing.base import _summary_value

        return {
            "operation": cls._operation_name(operation),
            "params": _summary_value(cls._operation_params(operation)),
        }
```

Add this helper after `_operation_summary()`:

```python
    @classmethod
    def _lineage_to_summaries(cls, lineage: "LineageNode | None") -> list[dict[str, Any]]:
        if lineage is None:
            return []
        records: list[dict[str, Any]] = []
        for input_lineage in lineage.inputs:
            records.extend(cls._lineage_to_summaries(input_lineage))
        records.append(cls._operation_summary(lineage.operation))
        return records
```

Add this property after `operation_graph`:

```python
    @property
    def operation_summaries(self) -> list[dict[str, Any]]:
        """Return lightweight display summaries derived from ``lineage``."""
        return self._lineage_to_summaries(self.lineage)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run pytest \
  tests/core/test_base_frame_lineage.py::test_operation_summaries_returns_display_lineage_summaries \
  tests/core/test_base_frame_lineage.py::test_operation_summaries_do_not_compute_data \
  tests/core/test_base_frame_lineage.py::test_operation_summaries_are_strict_json_serializable \
  tests/core/test_base_frame_lineage.py::test_operation_summaries_include_multi_input_lineage \
  -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add wandas/core/base_frame.py tests/core/test_base_frame_lineage.py
git commit -m "Expose frame operation summaries"
```

---

## Task 3: Add Custom Operation Display Summary

**Files:**
- Modify: `wandas/processing/custom.py`
- Test: `tests/processing/test_custom_operation.py`

- [ ] **Step 1: Write failing custom summary test**

Add this test after `test_custom_operation_dask_pure_controls_dask_purity_not_params()`:

```python
    def test_custom_operation_summary_includes_callable_reference_for_display(self) -> None:
        def scale(x: np.ndarray, gain: float) -> np.ndarray:
            return x * gain

        operation = CustomOperation(16000, func=scale, gain=2.0)

        summary = operation.to_summary()

        assert summary["operation"] == "custom"
        assert summary["params"] == {"gain": 2.0}
        assert summary["implementation"] is not scale
        assert summary["implementation"].endswith(".scale")
        assert "portable" not in summary
        assert "schema_version" not in summary
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/processing/test_custom_operation.py::TestCustomOperation::test_custom_operation_summary_includes_callable_reference_for_display -q
```

Expected: FAIL because `CustomOperation.to_summary()` does not include `implementation`.

- [ ] **Step 3: Implement custom summary**

In `wandas/processing/custom.py`, add this helper after imports:

```python
def _callable_reference(func: Callable[..., Any]) -> str:
    module = getattr(func, "__module__", type(func).__module__)
    qualname = getattr(func, "__qualname__", type(func).__qualname__)
    return f"{module}.{qualname}"
```

Add this method to `CustomOperation` after `get_display_name()`:

```python
    def to_summary(self) -> dict[str, Any]:
        """Return a lightweight display summary for this custom callable."""
        summary = super().to_summary()
        summary["implementation"] = _callable_reference(self.func)
        return summary
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run pytest tests/processing/test_custom_operation.py::TestCustomOperation::test_custom_operation_summary_includes_callable_reference_for_display -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add wandas/processing/custom.py tests/processing/test_custom_operation.py
git commit -m "Summarize custom operation callables for display"
```

---

## Task 4: Remove Persistence Language from Tests and Docs

**Files:**
- Modify: `tests/core/test_base_frame_lineage.py`
- Modify: `tests/processing/test_base_operations.py`
- Modify: `tests/processing/test_custom_operation.py`
- Confirm: no old portability docs exist on this branch except this display-only spec and plan.

- [ ] **Step 1: Search for forbidden summary contract terms**

Run:

```bash
rg -n "schema_version|portable|persistence|restore|restoration|_summary_is_portable|_operand_descriptor_is_portable|SUMMARY_SCHEMA_VERSION" wandas tests docs/superpowers
```

Expected: matches only this display-only spec/plan where these terms explain removed scope. There should be no matches in `wandas/` or summary tests.

- [ ] **Step 2: Remove accidental test assertions if any remain**

If the search finds `schema_version` or `portable` in new summary tests, remove those assertions and replace them with exact display-only shape assertions:

```python
assert summary == {
    "operation": "test_op",
    "params": {"gain": 2.0},
}
```

If the search finds `_summary_is_portable`, `_operand_descriptor_is_portable`, or `SUMMARY_SCHEMA_VERSION` in `wandas/`, delete those names because the display-only design does not use them.

- [ ] **Step 3: Run summary-focused tests**

Run:

```bash
uv run pytest tests/core/test_base_frame_lineage.py tests/processing/test_base_operations.py tests/processing/test_custom_operation.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit if Step 2 changed files**

Run only if files changed:

```bash
git add wandas tests
git commit -m "Remove persistence contract from operation summaries"
```

---

## Task 5: Full Validation

**Files:**
- No planned source edits.

- [ ] **Step 1: Run relevant pytest suite**

Run:

```bash
uv run pytest tests/core/test_base_frame_lineage.py tests/processing/test_base_operations.py tests/processing/test_custom_operation.py tests/frames/test_channel_processing.py
```

Expected: all tests pass.

- [ ] **Step 2: Run frame binary and lineage-adjacent tests**

Run:

```bash
uv run pytest tests/core/test_base_frame_additional.py tests/core/test_base_frame.py tests/frames/test_channel_frame.py -q
```

Expected: all tests pass. Existing warnings are acceptable if unrelated.

- [ ] **Step 3: Run lint**

Run:

```bash
uv run ruff check wandas tests --config=pyproject.toml
```

Expected: `All checks passed!`

- [ ] **Step 4: Run type check**

Run:

```bash
uv run ty check wandas tests
```

Expected: `All checks passed!`

- [ ] **Step 5: Confirm validation did not leave tracked changes**

Run:

```bash
git status --short
```

Expected: no tracked source, test, or doc files changed. Ignore generated local
coverage artifacts if they are untracked.

---

## Task 6: Close PR #250 and Open Replacement PR

**Files:**
- No local source edits expected.

- [ ] **Step 1: Push the new branch**

Run:

```bash
git push -u origin codex/operation-summaries-display-only
```

Expected: branch pushes to GitHub.

- [ ] **Step 2: Edit PR #250 body to remove issue linkage**

Run:

```bash
gh pr view 250 --repo kasahart/wandas --json body --jq .body
```

Then replace it with this body:

```bash
gh pr edit 250 --repo kasahart/wandas --body '## Status
Closed in favor of a replacement PR.

## Reason
This branch treated `operation_summaries` / `to_summary()` as if they were a portable persistence contract. Review feedback showed that this makes the feature larger than the immediate need and creates ongoing edge-case churn.

The replacement work restarts from `develop` and limits summaries to display-only, strict-JSON-safe operation history inspection. Save/restore is intentionally out of scope and should be handled by a separate `to_spec()` / `from_spec()` design if needed later.'
```

The replacement body must not contain issue-closing keywords or issue references.

- [ ] **Step 3: Comment on PR #250**

Run:

```bash
gh pr comment 250 --repo kasahart/wandas --body 'Closing this PR because the design direction changed. This branch treated `operation_summaries` / `to_summary()` as if they were a portable persistence contract, which led to a growing set of edge-case fixes and review churn. The replacement work restarts from `develop` and limits summaries to display-only, strict-JSON-safe operation history inspection. Save/restore will be handled by a separate `to_spec()` / `from_spec()` design if needed later.'
```

Expected: comment is added.

- [ ] **Step 4: Close PR #250**

Run:

```bash
gh pr close 250 --repo kasahart/wandas
```

Expected: PR #250 is closed and no issue is closed by it.

- [ ] **Step 5: Open replacement PR**

Run:

```bash
gh pr create \
  --repo kasahart/wandas \
  --base develop \
  --head codex/operation-summaries-display-only \
  --title 'Add display-only operation summaries' \
  --body '## Summary
- Add `operation_summaries` as a display-only lineage projection.
- Add `AudioOperation.to_summary()` and `BinaryOperation.to_summary()` with `operation` and JSON-safe display `params`.
- Add `CustomOperation.to_summary()` with a callable implementation reference for human inspection.
- Keep summaries strict-JSON serializable and avoid Dask compute.
- Explicitly avoid `schema_version`, `portable`, and save/restore semantics.

## Design
- Display-only contract documented in `docs/superpowers/specs/2026-07-02-operation-summary-display-only-design.md`.
- Save/restore is intentionally out of scope; future persistence should use a separate `to_spec()` / `from_spec()` contract.

## Linked Issues
- Related to #232
- Related to #244

## Validation
- `uv run pytest tests/core/test_base_frame_lineage.py tests/processing/test_base_operations.py tests/processing/test_custom_operation.py tests/frames/test_channel_processing.py`
- `uv run pytest tests/core/test_base_frame_additional.py tests/core/test_base_frame.py tests/frames/test_channel_frame.py -q`
- `uv run ruff check wandas tests --config=pyproject.toml`
- `uv run ty check wandas tests`'
```

Expected: a new PR URL is printed.

- [ ] **Step 6: Verify PR state**

Run:

```bash
gh pr view --repo kasahart/wandas --head codex/operation-summaries-display-only --json number,url,state,headRefOid
gh pr view 250 --repo kasahart/wandas --json state,body
```

Expected:

- replacement PR is open
- PR #250 is closed
- PR #250 body does not contain `Closes #244` or `Part of #232`

- [ ] **Step 7: Monitor automatic review after push**

For at least 10 minutes after opening the replacement PR, check for new review
threads every few minutes:

```bash
REPLACEMENT_PR=$(gh pr view --repo kasahart/wandas --head codex/operation-summaries-display-only --json number --jq .number)
gh pr view "$REPLACEMENT_PR" --repo kasahart/wandas --json reviews,comments,headRefOid
uv run python /home/vscode/.codex/plugins/cache/openai-curated/github/3fdeeb49/skills/gh-address-comments/scripts/fetch_comments.py
```

Expected after 10 minutes: either no review appears, or any new review comments
are visible and can be triaged. If no review has appeared after 30 minutes, treat
additional automated review as unlikely for this push.
