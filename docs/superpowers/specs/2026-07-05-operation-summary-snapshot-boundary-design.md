# Operation Summary Snapshot Boundary Design

## Context

PR #252 added `operation_summaries` as a display-only projection derived from runtime frame lineage. That projection is useful while a frame still has live lineage, but `persist()` and WDF save/load are boundary operations where live lineage or Dask operation tasks may no longer be available.

Issue #245 should therefore focus on snapshotting display summaries at these boundaries. It should not become Recipe persistence, live operation serialization, or legacy history cleanup.

## Goal

Preserve display-only operation summaries across `persist()` and WDF round trips without retaining old Dask graphs, live `AudioOperation` objects, or executable Recipe specs.

## Contract

`operation_summaries` remains the public display API. For ordinary live frames, it is derived from runtime lineage. For persisted or loaded frames that no longer have live lineage, it may return a stored summary snapshot.

The snapshot is a JSON-safe list of dictionaries produced from the existing `operation_summaries` projection. It is not an executable representation and cannot reconstruct `frame.operations`, Dask tasks, or `AudioOperation` instances.

`operation_history` remains a compatibility view. This issue should not deprecate it or perform broader cleanup; that belongs to #246.

## Persist Boundary

Before `BaseFrame.persist()` materializes the Dask collection, it snapshots `self.operation_summaries`. The new persisted frame receives the persisted data and the summary snapshot.

The persisted frame must not retain a strong reference to the old graph or live operation objects solely for summary display. If live lineage remains naturally available through existing frame construction, the snapshot must still be the stable source for summaries after persistence.

Expected behavior:

- `persisted.operation_summaries == original.operation_summaries`
- retrieving summaries does not call `.compute()`
- summary records are plain JSON-safe values
- `persist()` itself is not recorded as a semantic operation

## WDF Boundary

WDF save writes only the display summary snapshot, not live operations and not executable Recipe specs.

Use a schema-versioned JSON field at the file root:

```text
operation_summaries_schema = 1
operation_summaries_json = "[...]"
```

WDF load restores that JSON into the frame as a summary snapshot. Loaded frames expose `operation_summaries` from the snapshot, but do not regain executable lineage or operation instances.

The legacy `operation_history` HDF5 group remains a compatibility concern. For this issue, keep its behavior explicit and covered by tests. The recommended default is to ignore legacy groups rather than silently translating unstable old history into the new summary schema.

## Recipe Boundary

Recipe persistence is separate. `RecipeSpec`, `GraphRecipeSpec`, and `NodeGraphRecipeSpec` may later define `recipe_json` or another executable schema in #257.

This issue must not save executable Recipe specs as operation summaries, and it must not infer Recipe specs from loaded summaries.

## Implementation Shape

Add a small internal snapshot helper on frames or a focused utility near frame lineage handling. It should:

- take a defensive deep copy of summary records
- validate JSON safety with strict JSON encoding
- avoid references to arrays, frames, callables, Dask collections, and operation objects
- provide one path used by both `persist()` and WDF save

Frame constructors or `_create_new_instance()` need a private way to carry a summary snapshot. Keep it internal and avoid adding public mutable state that must be synchronized with lineage.

## Tests

Add focused tests for:

- `persist()` preserves `operation_summaries`
- persisted summaries do not require compute when read
- persisted summaries do not retain live operation objects
- WDF save writes schema-versioned `operation_summaries_json`
- WDF load restores `operation_summaries`
- WDF load does not restore executable `operations` or live lineage
- legacy `operation_history` HDF5 groups keep the clarified compatibility behavior
- built-in, custom non-portable, and multi-input operation summaries round trip as display summaries

Run relevant `uv run pytest` targets for core frame lineage and WDF I/O, then `uv run ruff check` and `uv run ty check`.

## Out Of Scope

- saving live `AudioOperation` objects
- rebuilding Dask graphs from persisted or loaded summaries
- Recipe persistence or replay schema
- broad `operation_history` cleanup
- public deprecation of `operation_history`
