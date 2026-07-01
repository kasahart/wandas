# Operation Summary Portability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `operation_summaries` strict-JSON safe and mark summaries portable only when parameter values are faithfully represented.

**Architecture:** Keep `AudioOperation.to_summary()` thin. Extend the private summary helpers in `wandas/processing/base.py`: `_summary_value()` produces JSON-safe display descriptors, `_summary_is_portable()` decides whether the original parameter values are faithfully represented.

**Tech Stack:** Python 3.10, NumPy, Dask array, pytest, ruff, ty, uv.

---

## File Structure

- Modify `wandas/processing/base.py`
  - Add a local stable JSON sort key helper for deterministic set/frozenset summaries.
  - Update Dask array descriptor generation to normalize `shape` and `chunks` recursively.
  - Add set/frozenset summary conversion.
  - Tighten `_summary_is_portable()` so opaque values are non-portable.
- Modify `tests/processing/test_base_operations.py`
  - Add `Path` and `dask.array as da` imports.
  - Add focused tests for Dask unknown shape/chunks, set/frozenset preservation, and opaque object non-portability.
- No frame method changes.
- No WDF persistence changes.

---

### Task 1: Add Dask Unknown Shape Regression Test

**Files:**
- Modify: `tests/processing/test_base_operations.py`

- [ ] **Step 1: Add imports for this test**

At the top of `tests/processing/test_base_operations.py`, add:

```python
from pathlib import Path
```

Add the Dask array module import near the existing third-party imports:

```python
import dask.array as da
```

Add the delayed helper import near the other Dask imports:

```python
from dask import delayed
```

- [ ] **Step 2: Write the failing Dask descriptor test**

Add this method after `test_audio_operation_summary_sanitizes_array_like_params`:

```python
    def test_audio_operation_summary_sanitizes_dask_shape_and_chunks_for_strict_json(self) -> None:
        test_op_cls = self._make_test_op_class()
        weights = da.from_delayed(delayed(lambda: np.array([1.0, 2.0]))(), shape=(np.nan,), dtype=float)
        op = test_op_cls(16000, weights=weights)

        summary = op.to_summary()

        assert summary["params"]["weights"] == {
            "type": "dask.array",
            "shape": [{"type": "float", "value": "nan"}],
            "dtype": "float64",
            "chunks": [[{"type": "float", "value": "nan"}]],
        }
        assert summary["portable"] is True
        json.dumps(summary, allow_nan=False)
```

- [ ] **Step 3: Run the failing test**

Run:

```bash
uv run pytest tests/processing/test_base_operations.py::TestAudioOperation::test_audio_operation_summary_sanitizes_dask_shape_and_chunks_for_strict_json -q
```

Expected: FAIL because `shape` or `chunks` still contain raw `nan`.

---

### Task 2: Add Set/Frozenset Regression Test

**Files:**
- Modify: `tests/processing/test_base_operations.py`

- [ ] **Step 1: Write the failing set/frozenset test**

Add this method after the Dask descriptor test from Task 1:

```python
    def test_audio_operation_summary_preserves_set_values_deterministically(self) -> None:
        test_op_cls = self._make_test_op_class()
        op = test_op_cls(
            16000,
            channels={2, 1},
            bands=frozenset({np.float32(0.5), np.float32(0.25)}),
        )

        summary = op.to_summary()

        assert summary["params"]["channels"] == [1, 2]
        assert summary["params"]["bands"] == [0.25, 0.5]
        assert summary["portable"] is True
        json.dumps(summary, allow_nan=False)
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
uv run pytest tests/processing/test_base_operations.py::TestAudioOperation::test_audio_operation_summary_preserves_set_values_deterministically -q
```

Expected: FAIL because sets currently fall back to `{"type": "set"}` or `{"type": "frozenset"}`.

---

### Task 3: Add Opaque Object Portability Test

**Files:**
- Modify: `tests/processing/test_base_operations.py`

- [ ] **Step 1: Write the failing opaque object test**

Add this method after the set/frozenset test from Task 2:

```python
    def test_audio_operation_summary_marks_opaque_params_non_portable(self) -> None:
        test_op_cls = self._make_test_op_class()
        op = test_op_cls(16000, impulse_response=Path("ir.wav"), resource=object())

        summary = op.to_summary()

        assert summary["params"] == {
            "impulse_response": {"type": "PosixPath"},
            "resource": {"type": "object"},
        }
        assert summary["portable"] is False
        json.dumps(summary, allow_nan=False)
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
uv run pytest tests/processing/test_base_operations.py::TestAudioOperation::test_audio_operation_summary_marks_opaque_params_non_portable -q
```

Expected: FAIL because `_summary_is_portable()` currently returns `True` for opaque non-callable values.

---

### Task 4: Implement JSON-Safe Summary Values

**Files:**
- Modify: `wandas/processing/base.py`

- [ ] **Step 1: Add JSON import**

Add this import near the existing standard library imports:

```python
import json
```

- [ ] **Step 2: Add stable sort helper**

Add this helper immediately before `_summary_value()`:

```python
def _summary_sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
```

- [ ] **Step 3: Update `_summary_value()` container and Dask branches**

Replace the mapping/list/array/Dask tail of `_summary_value()` with:

```python
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

- [ ] **Step 4: Run Task 1 and Task 2 tests**

Run:

```bash
uv run pytest tests/processing/test_base_operations.py::TestAudioOperation::test_audio_operation_summary_sanitizes_dask_shape_and_chunks_for_strict_json tests/processing/test_base_operations.py::TestAudioOperation::test_audio_operation_summary_preserves_set_values_deterministically -q
```

Expected: both tests PASS.

---

### Task 5: Implement Strict Portability Boundary

**Files:**
- Modify: `wandas/processing/base.py`

- [ ] **Step 1: Replace `_summary_is_portable()`**

Replace `_summary_is_portable()` with:

```python
def _summary_is_portable(value: Any) -> bool:
    """Return whether a value can be represented as a portable summary."""
    if callable(value):
        return False
    if value is None or isinstance(value, str | bool | np.bool_):
        return True
    if isinstance(value, numbers.Number):
        return True
    if isinstance(value, np.ndarray | DaArray):
        return True
    if isinstance(value, Mapping):
        return all(_summary_is_portable(item) for item in value.values())
    if isinstance(value, tuple | list | set | frozenset):
        return all(_summary_is_portable(item) for item in value)
    return False
```

- [ ] **Step 2: Run Task 3 test**

Run:

```bash
uv run pytest tests/processing/test_base_operations.py::TestAudioOperation::test_audio_operation_summary_marks_opaque_params_non_portable -q
```

Expected: PASS.

- [ ] **Step 3: Run existing callable and scalar summary tests**

Run:

```bash
uv run pytest tests/processing/test_base_operations.py::TestAudioOperation::test_audio_operation_summary_preserves_nested_callable_params_as_non_portable tests/processing/test_base_operations.py::TestAudioOperation::test_audio_operation_summary_preserves_numpy_and_complex_scalars_for_strict_json -q
```

Expected: both tests PASS.

---

### Task 6: Run Focused Summary Test Set

**Files:**
- Test only.

- [ ] **Step 1: Run all AudioOperation summary tests**

Run:

```bash
uv run pytest tests/processing/test_base_operations.py -q
```

Expected: all tests in `test_base_operations.py` PASS.

- [ ] **Step 2: Commit implementation**

Run:

```bash
git add wandas/processing/base.py tests/processing/test_base_operations.py
git commit -m "Tighten operation summary portability"
```

Expected: commit hooks PASS.

---

### Task 7: Validate PR Scope and Update PR

**Files:**
- Modify only PR metadata through `gh`.

- [ ] **Step 1: Run relevant validation**

Run:

```bash
uv run pytest tests/core/test_base_frame_lineage.py tests/processing/test_base_operations.py tests/processing/test_custom_operation.py tests/frames/test_channel_processing.py
```

Expected: all tests PASS. The count should increase by 3 from the previous 195 because three tests were added.

Run:

```bash
uv run ruff check wandas tests --config=pyproject.toml -v
```

Expected: `All checks passed!`

Run:

```bash
uv run ty check wandas tests
```

Expected: `All checks passed!`

- [ ] **Step 2: Push branch**

Run:

```bash
git push
```

Expected: `codex/operation-summaries` updates on GitHub.

- [ ] **Step 3: Update PR body validation and summary**

Run:

```bash
gh pr edit 250 --repo kasahart/wandas --body $'## Summary\n- Add `AudioOperation.to_summary()` and `BinaryOperation.to_summary()` for lightweight portable operation summaries.\n- Mark `CustomOperation` summaries as non-portable and include a callable implementation reference.\n- Add `BaseFrame.operation_summaries` as a read-only projection from existing `LineageNode` provenance without changing `previous` or `operation_history`.\n- Sanitize non-finite numeric summary params so summaries are strict-JSON serializable.\n- Preserve nested callable params as explicit descriptors and mark affected summaries as non-portable.\n- Normalize Dask descriptor shape/chunks, preserve set/frozenset contents deterministically, and mark opaque type-only params non-portable.\n\n## Linked Issues\n- Closes #244\n- Part of #232\n\n## Validation\n- `uv run pytest tests/core/test_base_frame_lineage.py tests/processing/test_base_operations.py tests/processing/test_custom_operation.py tests/frames/test_channel_processing.py` -> 198 passed\n- `uv run ruff check wandas tests --config=pyproject.toml -v` -> All checks passed\n- `uv run ty check wandas tests` -> All checks passed\n- commit hooks: ruff check, ruff format, ty -> Passed\n\n## Notes\n- WDF/persist summary snapshots are intentionally left for a later phase.\n- `previous` retention is unchanged because frames are lazy and this phase focuses on summary contracts.'
```

Expected: command prints the PR URL.

- [ ] **Step 4: Reply to and resolve the three new review threads**

Use the same thread-aware GraphQL workflow used earlier. Reply with:

```text
Addressed in `<commit>`: Dask descriptor shape/chunks are now recursively normalized through `_summary_value()`, so unknown dimensions/chunks remain strict-JSON safe while the descriptor stays portable. Added regression coverage with `json.dumps(..., allow_nan=False)`.
```

```text
Addressed in `<commit>`: set and frozenset params now preserve their contents as deterministic summary lists. Summaries remain portable when all contained values are portable, and the regression test covers stable values and strict JSON encoding.
```

```text
Addressed in `<commit>`: opaque non-callable params still get display-safe type descriptors, but `_summary_is_portable()` now marks them non-portable because the original value is not faithfully represented. Added coverage for `Path` and `object()`.
```

Then call `resolveReviewThread` for each thread and confirm all new threads have `isResolved: true`.

---

## Self-Review

- Spec coverage: Dask unknown shape/chunks is covered by Task 1 and Task 4. Set/frozenset preservation is covered by Task 2 and Task 4. Opaque non-portability is covered by Task 3 and Task 5. Validation and PR update are covered by Task 7.
- Incomplete-marker scan: no unfinished markers remain. All code changes and commands are explicit.
- Type consistency: helper names `_summary_value`, `_summary_is_portable`, and `_summary_sort_key` are consistent across tasks. Test class and file paths match the current codebase.
