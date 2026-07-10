# Legacy Lineage History Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pin the first #246 cleanup slice: `previous` stays a stable debug/compat accessor, while `operation_history` and `operation_summaries` are derived only from runtime lineage or inspection snapshots.

**Architecture:** Add focused contract tests around `BaseFrame` lineage/history boundaries, then update developer-facing docs and docstrings to match. Avoid behavior changes unless a test shows legacy attrs or `previous` are still acting as provenance sources.

**Tech Stack:** Python, pytest, Dask arrays, Wandas `ChannelFrame`, Markdown docs.

---

## File Map

- Modify `tests/core/test_base_frame_lineage.py`
  - Add focused tests for `previous`, runtime `lineage`, `operation_history`, and `operation_summaries`.
- Modify `wandas/core/base_frame.py`
  - Update `previous`, `operation_history`, and `operation_summaries` docstrings only unless tests fail.
- Modify `docs/src/explanation/pipeline-recipe-developer-guide.md`
  - Add beginner-friendly core-term entries for `previous` and `operation_summaries`.
  - Clarify that Recipe extraction reads `operation_graph`, while `previous` is only a debug/compat pointer.
- Keep `docs/superpowers/specs/2026-07-06-legacy-lineage-history-cleanup-design.md`
  - This is the approved design source for the PR.

## Task 1: Pin Previous And Lineage Boundaries

**Files:**
- Modify: `tests/core/test_base_frame_lineage.py`

- [ ] **Step 1: Add tests proving `previous` is not a history source**

Add these tests after `test_operation_history_public_behavior_is_read_only_lineage_view`:

```python
def test_previous_is_stable_debug_accessor_not_history_source() -> None:
    previous = _frame().normalize()
    frame = ChannelFrame(
        da_from_array(np.array([[1.0, 2.0, 3.0]]), chunks=(1, -1)),
        sampling_rate=16000,
        previous=previous,
    )

    assert frame.previous is previous
    assert frame.lineage is None
    assert frame.operation_history == []
    assert frame.operation_summaries == []
    assert frame.operation_graph is None


def test_operation_history_comes_from_lineage_without_previous() -> None:
    frame = ChannelFrame(
        da_from_array(np.array([[1.0, 2.0, 3.0]]), chunks=(1, -1)),
        sampling_rate=16000,
        lineage=LineageNode(Normalize(16000)),
        previous=None,
    )

    assert frame.previous is None
    assert [record["operation"] for record in frame.operation_history] == ["normalize"]
    assert [summary["operation"] for summary in frame.operation_summaries] == ["normalize"]
    assert frame.operation_graph is not None
    assert frame.operation_graph["operation"] == "normalize"
```

- [ ] **Step 2: Run the new tests**

Run:

```bash
uv run pytest \
  tests/core/test_base_frame_lineage.py::test_previous_is_stable_debug_accessor_not_history_source \
  tests/core/test_base_frame_lineage.py::test_operation_history_comes_from_lineage_without_previous \
  -q
```

Expected: PASS. These tests should pin current behavior. If they fail, fix only the failing provenance boundary in `wandas/core/base_frame.py`.

- [ ] **Step 3: Commit the boundary tests**

Run:

```bash
git add tests/core/test_base_frame_lineage.py
git commit -m "test: pin previous and lineage history boundary"
```

## Task 2: Pin Legacy Operation History Attr Boundaries

**Files:**
- Modify: `tests/core/test_base_frame_lineage.py`

- [ ] **Step 1: Add tests proving legacy attrs are not summary sources**

Add these tests near the operation summaries snapshot tests:

```python
def test_operation_summaries_ignore_legacy_operation_history_attrs() -> None:
    frame = _frame().normalize()
    frame._xr.attrs["operation_history"] = [{"operation": "legacy", "params": {"gain": 2.0}}]

    assert [summary["operation"] for summary in frame.operation_summaries] == ["normalize"]
    assert [record["operation"] for record in frame.operation_history] == ["normalize"]


def test_snapshot_operation_summaries_ignore_legacy_operation_history_attrs() -> None:
    frame = ChannelFrame(
        da_from_array(np.array([[1.0, 2.0, 3.0]]), chunks=(1, -1)),
        sampling_rate=16000,
        operation_summaries_snapshot=[{"operation": "loaded", "params": {"gain": 2.0}}],
    )
    frame._xr.attrs["operation_history"] = [{"operation": "legacy", "params": {"gain": 3.0}}]

    assert frame.operation_summaries == [{"operation": "loaded", "params": {"gain": 2.0}}]
    assert frame.operation_history == []
```

- [ ] **Step 2: Run the new tests**

Run:

```bash
uv run pytest \
  tests/core/test_base_frame_lineage.py::test_operation_summaries_ignore_legacy_operation_history_attrs \
  tests/core/test_base_frame_lineage.py::test_snapshot_operation_summaries_ignore_legacy_operation_history_attrs \
  -q
```

Expected: PASS. If either fails, remove the legacy attr read path from the smallest affected code path.

- [ ] **Step 3: Run the focused lineage test file**

Run:

```bash
uv run pytest tests/core/test_base_frame_lineage.py -q
```

Expected: all tests in the file pass.

- [ ] **Step 4: Commit the legacy attr tests**

Run:

```bash
git add tests/core/test_base_frame_lineage.py
git commit -m "test: pin operation summary provenance sources"
```

## Task 3: Update Public Documentation And Docstrings

**Files:**
- Modify: `wandas/core/base_frame.py`
- Modify: `docs/src/explanation/pipeline-recipe-developer-guide.md`

- [ ] **Step 1: Update `BaseFrame.previous` and view docstrings**

In `wandas/core/base_frame.py`, replace the current docstrings for `previous`, `operation_history`, and `operation_summaries` with:

```python
    @property
    def previous(self) -> "BaseFrame[Any] | None":
        """Return the immediate prior frame for compatibility/debug inspection.

        This strong reference is not the source of truth for processing
        history. Runtime lineage drives ``operation_history``,
        ``operation_summaries``, ``operation_graph``, and Recipe extraction.
        """
        return self._previous
```

```python
    @property
    def operation_history(self) -> list[dict[str, Any]]:
        """Return a flat read-only compatibility view derived from ``lineage``."""
        return self._lineage_to_history(self.lineage)
```

```python
    @property
    def operation_summaries(self) -> list[dict[str, Any]]:
        """Return display summaries derived from lineage or a boundary snapshot."""
        if self._operation_summaries_snapshot is not None:
            return copy.deepcopy(self._operation_summaries_snapshot)
        return self._lineage_to_summaries(self.lineage)
```

- [ ] **Step 2: Update the developer guide term table**

In `docs/src/explanation/pipeline-recipe-developer-guide.md`, replace the first three term rows under `## Core Terms` with:

```markdown
| `previous` | Stable compatibility/debug pointer to the immediate prior frame when available. It is useful for inspection, but it is not the source of truth for history or Recipe extraction. |
| `lineage` | Runtime tree stored on a frame. It records the operation object and parent lineage nodes used to create the frame. |
| `operation_history` | Backward-compatible linear list derived from `lineage`. Good for user inspection, but not enough for graph extraction because it loses parent structure. It is not backed by `previous` or `_xr.attrs["operation_history"]`. |
| `operation_summaries` | Display-oriented operation list derived from runtime `lineage`, or from an inspection-only snapshot after WDF load or `persist()`. |
| `operation_graph` | JSON-like tree derived from `lineage`. Recipe extraction reads this structure because it preserves operation params, parent edges, source leaves, and selected custom metadata. |
```

- [ ] **Step 3: Update the data flow explanation**

In the same file, after the text block showing `lineage -> operation_history -> operation_graph`, add:

```markdown
`previous` is separate from this flow. It may point to the prior frame for debugging or compatibility, but changing or missing `previous` must not change `operation_history`, `operation_summaries`, `operation_graph`, or Recipe extraction.

At WDF and `persist()` boundaries, `operation_summaries` may come from an inspection-only snapshot instead of live runtime lineage. That snapshot is for display continuity only; it does not rebuild `operation_history` or executable Recipe lineage.
```

- [ ] **Step 4: Run docs/text-focused checks**

Run:

```bash
uv run ruff check wandas/core/base_frame.py tests/core/test_base_frame_lineage.py
uv run ty check wandas/core tests/core/test_base_frame_lineage.py
```

Expected: both commands pass.

- [ ] **Step 5: Commit docs and docstrings**

Run:

```bash
git add wandas/core/base_frame.py docs/src/explanation/pipeline-recipe-developer-guide.md
git commit -m "docs: clarify previous and lineage history roles"
```

## Task 4: Final Verification And PR Prep

**Files:**
- Verify all modified files.
- Update PR body when the PR is created.

- [ ] **Step 1: Run focused tests**

Run:

```bash
uv run pytest tests/core/test_base_frame_lineage.py tests/core/test_xarray_storage_only.py tests/io/test_wdf_io.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run lint and scoped type checks**

Run:

```bash
uv run ruff check
uv run ty check wandas/core tests/core/test_base_frame_lineage.py tests/core/test_xarray_storage_only.py tests/io/test_wdf_io.py
```

Expected: both commands pass. If full `uv run ty check` is also run and reports unrelated existing diagnostics, record the affected files separately.

- [ ] **Step 3: Check generated artifacts**

Run:

```bash
git status --short --ignored
```

Expected: only intentional tracked changes. Remove unintended `.coverage`, `coverage.xml`, `.pytest_cache/`, `.ruff_cache/`, and `__pycache__/` artifacts before reporting readiness.

- [ ] **Step 4: Open PR**

Run:

```bash
git push -u origin codex/issue-246-lineage-history-cleanup
gh pr create \
  --base develop \
  --head codex/issue-246-lineage-history-cleanup \
  --title "[codex] Clarify legacy lineage history boundaries" \
  --body "## Summary

- Documents the #246 first-slice contract: previous stays a stable debug/compat accessor, not a provenance source.
- Adds tests pinning operation_history as lineage-derived and operation_summaries as lineage/snapshot-derived.
- Clarifies contributor docs around previous, lineage, operation_history, operation_summaries, and operation_graph.

## Validation

- \`uv run pytest tests/core/test_base_frame_lineage.py tests/core/test_xarray_storage_only.py tests/io/test_wdf_io.py -q\`
- \`uv run ruff check\`
- \`uv run ty check wandas/core tests/core/test_base_frame_lineage.py tests/core/test_xarray_storage_only.py tests/io/test_wdf_io.py\`

Related #246"
```

Expected: PR opens against `develop`. Use `Related #246`, not `Closes #246`, because this is the first cleanup slice and does not finish all remaining #246 cleanup.
