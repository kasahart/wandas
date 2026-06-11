# xarray Migration Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the xarray migration documentation and public wording so persistent design decisions live in `docs/design/`, completed implementation plans are removed, and stale concepts are no longer presented as active API.

**Architecture:** This is a documentation and public API cleanup phase. The implementation deletes completed `docs/superpowers/plans` logs, adds one concise ADR under `docs/design/`, adds a short public documentation pointer, and uses targeted `rg` checks to clean stale wording. It does not change runtime behavior, xarray storage, operation execution, or channel metadata storage.

**Tech Stack:** Markdown docs, existing `docs/design` ADR style, `rg`, `ruff`, `pytest`.

---

## File Structure

- Delete: `docs/superpowers/plans/2026-06-09-xarray-storage-only.md`
- Delete: `docs/superpowers/plans/2026-06-10-xarray-backed-frame-state.md`
- Delete: `docs/superpowers/plans/2026-06-11-channel-metadata-dataclass.md`
- Delete: `docs/superpowers/plans/2026-06-11-xarray-migration-consolidation.md`
- Create: `docs/design/2026-06-11-xarray-migration-consolidation.md`
- Modify: `docs/src/explanation/index.md`
- Modify if searches show active stale wording: `README.md`, `docs/src/**/*.md`, `wandas/**/*.py`, `tests/**/*.py`

The ADR is the durable design record. The `docs/superpowers/plans` files are task execution artifacts and should not remain after this phase lands.

---

### Task 1: Add Durable xarray Migration ADR

**Files:**
- Create: `docs/design/2026-06-11-xarray-migration-consolidation.md`

- [ ] **Step 1: Read nearby ADR style**

Run:

```bash
sed -n '1,220p' docs/design/2026-06-10-xarray-dimension-semantics.md
sed -n '1,180p' docs/design/2025-11-19-channel-wise-chunking.md
```

Expected: both files are short Markdown design records with headings and decision-focused prose.

- [ ] **Step 2: Create the consolidation ADR**

Create `docs/design/2026-06-11-xarray-migration-consolidation.md` with exactly this content:

```markdown
# xarray Migration Consolidation

Date: 2026-06-11
Status: Accepted

## Context

Wandas is moving toward an xarray-backed internal data model, but the migration is intentionally staged. Earlier phases introduced xarray as storage and state backing without changing the operation execution model.

The repository also accumulated detailed implementation plans under `docs/superpowers/plans/`. Those plans were useful while executing the phases, but they are too verbose and procedural to serve as long-term architecture documentation.

## Decision

Keep durable xarray migration decisions in `docs/design/` and remove completed implementation plans from `docs/superpowers/plans/`.

The current responsibility boundary is:

```text
xarray
  - data container
  - named dimensions
  - selected coordinates
  - frame-level attrs storage

Wandas
  - validation
  - waveform/audio domain semantics
  - operation history semantics
  - channel metadata objects
  - operation execution
```

## Current State

- `BaseFrame` owns `_xr: xarray.DataArray` as the internal storage object.
- `_data` is a read-only compatibility alias to `_xr.data`.
- Supported frame schemas use semantic xarray dims and a centralized channel coord.
- Frame-level state such as `label`, `sampling_rate`, `metadata`, and `operation_history` is backed by `_xr.attrs`.
- `FrameMetadata` has been removed. Frame metadata is a plain `dict[str, Any]`.
- Source file metadata is stored as `metadata["_source_file"]`.
- `ChannelMetadata` is a standard-library dataclass, not a Pydantic model.
- Explicit `ref` values are preserved, including `ref=1.0`.
- Operation execution remains on the existing Wandas/Dask path.

## Migration Notes

Code that previously used `FrameMetadata` should use plain dictionaries:

```python
frame.metadata = {"operator": "lab-a"}
```

Code that previously used `metadata.source_file` should use the reserved metadata key:

```python
frame.metadata["_source_file"]
```

Code that previously used `metadata.merged(...)` should use normal dict merging:

```python
metadata = {**frame.metadata, "window": "hann"}
```

Code that previously used Pydantic-specific `ChannelMetadata` APIs such as `model_copy`, `model_fields`, or Pydantic validation errors must use standard-library equivalents such as `copy.deepcopy()`, `ChannelMetadata._MODEL_FIELDS`, `to_json()`, and `from_json()`.

## Non-Goals

This consolidation does not introduce:

- `from_xarray`
- NetCDF or Zarr support
- xarray-native operation dispatch
- `xr.apply_ufunc` or `map_blocks` operation rewrites
- channel metadata storage migration to xarray coords or attrs
- dense time coordinates
- new frame schemas

## Consequences

The documentation surface is smaller and clearer. Contributors should read `docs/design/` for lasting architecture decisions instead of completed implementation plans.

The migration remains conservative: xarray stores labelled data and frame-level state, while Wandas keeps the signal-processing semantics and public domain API.
```

- [ ] **Step 3: Verify the ADR renders as expected**

Run:

```bash
sed -n '1,260p' docs/design/2026-06-11-xarray-migration-consolidation.md
```

Expected: the file exists, has one `#` heading, and includes Context, Decision, Current State, Migration Notes, Non-Goals, and Consequences sections.

- [ ] **Step 4: Commit the ADR**

Run:

```bash
git add docs/design/2026-06-11-xarray-migration-consolidation.md
git commit -m "docs: add xarray migration consolidation adr"
```

Expected: commit succeeds with one created ADR file.

---

### Task 2: Add a Short Public Documentation Pointer

**Files:**
- Modify: `docs/src/explanation/index.md`

- [ ] **Step 1: Inspect the explanation index**

Run:

```bash
sed -n '1,220p' docs/src/explanation/index.md
```

Expected: identify a suitable location for a short architecture pointer without turning the page into a full design doc.

- [ ] **Step 2: Add the xarray-backed architecture pointer**

In `docs/src/explanation/index.md`, add this section near the existing architecture/explanation content. If the file is very short, append it after the introductory text.

```markdown
## xarray-backed architecture

Wandas uses xarray internally as the labelled storage and frame-state layer while keeping Wandas as the waveform analysis API. In the current migration stage, xarray owns data, named dimensions, selected coordinates, and frame-level attrs. Wandas still owns validation, channel metadata objects, operation history semantics, and operation execution.

For the durable design record, see [xarray Migration Consolidation](../../design/2026-06-11-xarray-migration-consolidation.md).
```

- [ ] **Step 3: Verify the relative link target resolves within the docs tree**

Run:

```bash
find docs -maxdepth 3 -type f | sort | rg "docs/(design/2026-06-11-xarray-migration-consolidation.md|src/explanation/index.md)"
```

Expected: both files appear. The relative path from `docs/src/explanation/index.md` to `docs/design/2026-06-11-xarray-migration-consolidation.md` is `../../design/2026-06-11-xarray-migration-consolidation.md`.

- [ ] **Step 4: Run Markdown search for duplicated long architecture text**

Run:

```bash
rg -n "xarray-backed architecture|xarray Migration Consolidation|labelled storage" docs/src docs/design
```

Expected: the new explanation section and ADR are the only relevant public references. If duplicate long copies appear, reduce the explanation page to the short pointer above.

- [ ] **Step 5: Commit the public docs pointer**

Run:

```bash
git add docs/src/explanation/index.md
git commit -m "docs: point to xarray migration adr"
```

Expected: commit succeeds with one docs page changed.

---

### Task 3: Remove Completed Superpowers Implementation Plans

**Files:**
- Delete: `docs/superpowers/plans/2026-06-09-xarray-storage-only.md`
- Delete: `docs/superpowers/plans/2026-06-10-xarray-backed-frame-state.md`
- Delete: `docs/superpowers/plans/2026-06-11-channel-metadata-dataclass.md`
- Delete: `docs/superpowers/plans/2026-06-11-xarray-migration-consolidation.md`

- [ ] **Step 1: Confirm current plan files**

Run:

```bash
find docs/superpowers/plans -maxdepth 1 -type f | sort
```

Expected: the four completed implementation plan files listed above are present.

- [ ] **Step 2: Delete completed plans**

Run:

```bash
git rm docs/superpowers/plans/2026-06-09-xarray-storage-only.md docs/superpowers/plans/2026-06-10-xarray-backed-frame-state.md docs/superpowers/plans/2026-06-11-channel-metadata-dataclass.md docs/superpowers/plans/2026-06-11-xarray-migration-consolidation.md
```

Expected: all four files are staged for deletion.

- [ ] **Step 3: Verify `docs/superpowers/plans` is empty or absent**

Run:

```bash
find docs/superpowers/plans -maxdepth 1 -type f | sort
```

Expected: no output. If the directory itself remains empty, that is acceptable; git will not track the empty directory.

- [ ] **Step 4: Commit plan removal**

Run:

```bash
git commit -m "docs: remove completed implementation plans"
```

Expected: commit succeeds and removes the completed plans.

---

### Task 4: Clean Active Stale Wording

**Files:**
- Modify if needed: `README.md`
- Modify if needed: `docs/src/**/*.md`
- Modify if needed: `wandas/**/*.py`
- Modify if needed: `tests/**/*.py`

- [ ] **Step 1: Search for Pydantic references outside historical specs**

Run:

```bash
rg -n "Pydantic" README.md docs/src docs/design wandas tests pyproject.toml
```

Expected: no output outside the new ADR's migration notes if the ADR mentions Pydantic-specific APIs. If output appears in active API docs, production code, tests, or README, replace it with current dataclass wording.

Use this replacement when a public API sentence says ChannelMetadata is Pydantic-backed:

```markdown
`ChannelMetadata` is a standard-library dataclass.
```

- [ ] **Step 2: Search for FrameMetadata active references**

Run:

```bash
rg -n "FrameMetadata" README.md docs/src docs/design wandas tests pyproject.toml
```

Expected: references are only in migration notes that describe removed behavior. If an active API doc says to import or instantiate `FrameMetadata`, replace it with plain dict metadata:

```markdown
Frame metadata is represented as a plain dictionary, for example `frame.metadata = {"operator": "lab-a"}`.
```

- [ ] **Step 3: Search for old source_file metadata access**

Run:

```bash
rg -n "metadata\.source_file" README.md docs/src docs/design wandas tests
```

Expected: references are only in migration notes that describe removed behavior. If active docs or tests use it as current API, replace with:

```python
frame.metadata["_source_file"]
```

- [ ] **Step 4: Search for old metadata merging API**

Run:

```bash
rg -n "metadata\.merged" README.md docs/src docs/design wandas tests
```

Expected: references are only in migration notes that describe removed behavior. If active docs or tests use it as current API, replace with:

```python
metadata = {**frame.metadata, "key": value}
```

- [ ] **Step 5: Search for old xarray bridge framing**

Run:

```bash
rg -n "xarray bridge|xarray-bridge" README.md docs/src docs/design wandas tests
```

Expected: no active public reference describes the current implementation as an xarray bridge. If active docs use that term, replace with:

```markdown
xarray-backed storage/state
```

- [ ] **Step 6: Commit stale wording cleanup if any files changed**

Run:

```bash
git status --short
```

If only expected docs/code wording changed, commit them:

```bash
git add README.md docs/src docs/design wandas tests
git commit -m "docs: clean stale xarray migration wording"
```

Expected: commit succeeds if there were changes. If `git status --short` shows no changes, skip this commit.

---

### Task 5: Verification and PR Update

**Files:**
- No production files expected
- PR body update through GitHub UI/API after verification

- [ ] **Step 1: Run consolidated stale concept search**

Run:

```bash
rg -n "Pydantic|FrameMetadata|metadata\.source_file|metadata\.merged|xarray bridge|xarray-bridge" README.md docs wandas tests pyproject.toml
```

Expected: matches may remain in `docs/superpowers/specs/*.md` and the new `docs/design/2026-06-11-xarray-migration-consolidation.md` only when they describe removed behavior or migration notes. No match should present the old concepts as current public API.

- [ ] **Step 2: Run formatting checks**

Run:

```bash
uv run ruff check .
uv run ruff format --check .
```

Expected: both commands pass.

- [ ] **Step 3: Run focused regression tests**

Run:

```bash
uv run pytest tests/core/test_metadata.py tests/core/test_xarray_storage_only.py -q
```

Expected: tests pass. The exact count may vary as the branch evolves, but failures indicate this docs cleanup accidentally changed behavior or imports.

- [ ] **Step 4: Inspect final diff**

Run:

```bash
git status --short
git diff --stat origin/develop...HEAD
```

Expected: final diff includes the new ADR, public docs pointer, removal of completed plans, the Phase 5 spec, and any stale wording cleanup. It should not include operation execution changes or new xarray APIs.

- [ ] **Step 5: Push branch**

Run:

```bash
git push
```

Expected: branch pushes to the existing PR branch. If the command prints `fatal: could not read Username for 'https://github.com': terminal prompts disabled` but also prints a successful `To https://github.com/... old..new branch -> branch` line, treat the push as successful.

- [ ] **Step 6: Update PR body verification section**

Update PR #217 body to include Phase 5 docs cleanup and the verification commands from this task. The verification section should include:

```markdown
- `rg -n "Pydantic|FrameMetadata|metadata\.source_file|metadata\.merged|xarray bridge|xarray-bridge" README.md docs wandas tests pyproject.toml` -> no stale active references
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run pytest tests/core/test_metadata.py tests/core/test_xarray_storage_only.py -q`
```

Expected: PR body describes Phase 4 plus Phase 5 cleanup accurately.