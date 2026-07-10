# Legacy Lineage And History Cleanup Design

## Context

Issue #246 is the Phase 5 cleanup for legacy lineage and history behavior. Recent work already moved the main provenance model away from stored `operation_history` state:

- Runtime `lineage` is held on frames and drives executable replay.
- `operation_graph` is derived from runtime `lineage` and preserves parent structure for Recipe extraction.
- `operation_history` is a read-only compatibility view derived from runtime `lineage`.
- `operation_summaries` is a display view derived from runtime `lineage`, unless a WDF or `persist()` boundary supplies an inspection-only summaries snapshot.
- WDF stores inspection-only operation summaries snapshots, not executable Recipe lineage.

The first #246 PR should not remove or weaken `previous`. It should clarify the remaining contract and add tests where the current behavior is easy to misunderstand.

## Decision

Keep `_previous` and the public `previous` accessor as strong references for now.

`previous` is a stable compatibility/debug accessor. It can help users inspect the frame they just transformed, and some user-facing examples rely on that stable behavior. It is not the source of truth for processing history, Recipe extraction, or WDF summaries.

Do not weak-reference `previous` in this issue. Wandas frame data is Dask-backed, so computation dependencies already live in the Dask graph. Turning `previous` into a weak reference would not make computation meaningfully safer, but it would make a public/debug accessor depend on garbage collection timing. That is poor beginner UX: a frame that was visible a moment ago could unexpectedly become `None`.

If future memory work needs to deprecate, remove, or weak-reference `previous`, that should be a separate issue with an explicit deprecation plan.

## Source Of Truth Contract

Wandas should expose four distinct concepts:

1. `previous`
   - Strong reference to the immediate prior frame when a frame operation creates a new frame.
   - Compatibility/debug convenience only.
   - Not used to build `operation_history`, `operation_summaries`, `operation_graph`, or Recipe specs.

2. `lineage`
   - Runtime source of truth for operation provenance inside the current process.
   - Used to derive `operation_history`, `operation_summaries`, and `operation_graph` when no boundary snapshot is present.
   - Not exported as public xarray attrs or WDF executable state.

3. `operation_history`
   - Backward-compatible flat list for user inspection.
   - Read-only derived view from runtime `lineage`.
   - Not backed by `_xr.attrs["operation_history"]`.
   - Mutating a returned list must not mutate frame state.

4. `operation_summaries`
   - Display-oriented summary view.
   - Derived from runtime `lineage` during normal processing.
   - Derived from an inspection-only snapshot after WDF load or `persist()`.
   - Not backed by `_xr.attrs["operation_history"]`, WDF `operation_history` groups, or manual append paths.
   - WDF `operation_history` groups are not a supported load contract; WDF history display uses the
     `operation_summaries_json` snapshot instead.

## First PR Scope

The first #246 PR should be small and contract-focused:

- Keep `_previous` and `previous` as strong references.
- Update docstrings or docs so `previous` is described as a compatibility/debug accessor, not a history source of truth.
- Rename or adjust misleading tests such as `previous_property_tracks_lineage` so they no longer imply `previous` powers lineage.
- Add focused tests proving that `operation_history` does not read from `previous`.
- Add focused tests proving that unsupported `operation_history` attrs do not affect `operation_summaries`.
- Add or update docs for the beginner-facing distinction between `previous`, `operation_history`, `operation_summaries`, `operation_graph`, and WDF snapshots.

Production code changes should be minimal. If the new tests already pass, the PR can be mostly docs and test contract pinning. If a new test exposes legacy state being treated as authoritative, fix only that path.

## Explicit Non-Goals

The first #246 PR should not:

- weak-reference `_previous`;
- remove `previous`;
- deprecate public `operation_history`;
- implement executable Recipe WDF persistence;
- change Recipe graph identity or DAG behavior;
- remove constructor compatibility in a broad sweep;
- change Dask graph construction or numerical processing behavior.

## Later #246 Cleanup Slices

After the first PR, remaining #246 work can be split into smaller issues or PRs:

1. Constructor compatibility cleanup
   - Audit positional constructor compatibility around `previous`, `source_time_offset`, `lineage`, and `operation_summaries_snapshot`.
   - Prefer explicit keyword-only future APIs if a breaking change is approved.

2. Legacy `operation_history` storage cleanup
   - Remove or tighten any remaining compatibility paths that accept stored `operation_history` as input.
   - Do not add compatibility for WDF `operation_history` groups unless a migration requirement is explicitly added.

3. Metadata/history duplication cleanup
   - Ensure metadata remains durable frame state while history remains runtime lineage or inspection snapshot state.
   - Avoid storing the same processing fact in both metadata and lineage.

4. Public deprecation decisions
   - If `previous` or `operation_history` should change public behavior, open a separate issue with migration text and deprecation timing.

## Test Strategy

Add tests before production changes:

- `previous` remains stable across a normal operation and is not weakref/GC-dependent.
- A frame with `previous` but no `lineage` has empty `operation_history` and `operation_graph`.
- A frame with runtime `lineage` but no `previous` still exposes `operation_history`.
- Injecting unsupported `_xr.attrs["operation_history"]` does not affect `operation_history` or `operation_summaries`.
- A snapshot-backed frame returns snapshot `operation_summaries` even if unsupported `operation_history` attrs are present.

Run focused checks first:

- `uv run pytest tests/core/test_base_frame_lineage.py tests/core/test_xarray_storage_only.py -q`
- `uv run pytest tests/io/test_wdf_io.py -q`

Before PR readiness, run:

- `uv run pytest tests/core/test_base_frame_lineage.py tests/core/test_xarray_storage_only.py tests/io/test_wdf_io.py -q`
- `uv run ruff check`
- `uv run ty check wandas/core tests/core/test_base_frame_lineage.py tests/core/test_xarray_storage_only.py tests/io/test_wdf_io.py`

Full `uv run ty check` may still report pre-existing diagnostics outside this PR. If so, report them separately with the affected files.

## Acceptance

The first #246 PR is complete when:

- `previous` is explicitly documented and tested as a stable debug/compat accessor, not a history source of truth.
- `operation_history` is explicitly tested as a lineage-derived read-only compatibility view.
- `operation_summaries` is explicitly tested as lineage-derived or snapshot-derived, never unsupported history attr-derived.
- No new public behavior depends on unsupported `operation_history` attrs or manual append paths.
- Relevant tests, lint, and scoped type checks pass.
