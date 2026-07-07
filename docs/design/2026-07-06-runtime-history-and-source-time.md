# ADR: Runtime History, Operation Summaries, and Source Time

- **Status**: Accepted
- **Date**: 2026-07-06
- **Context**: Consolidates the durable decisions from the temporary superpowers specs after checking the current implementation.

## Context

Wandas keeps two different records of processing work:

1. runtime lineage, which is the executable parent graph for recipe extraction; and
2. operation summaries, which are display-safe history records for users and WDF inspection.

Temporary specs explored several overlapping designs for xarray attrs, operation-history display, WDF save/load behavior, and source-time offsets. The implemented contract is smaller: runtime lineage remains in memory, while WDF persists display summaries only.

Source-time offsets also moved from one frame-wide scalar toward per-channel state. The implementation stores scalar offsets as frame attrs when possible and per-channel offsets as xarray coordinates when the channel dimension is present.

## Decision

Runtime `lineage` is the source of truth for executable provenance. `operation_history` is a backward-compatible, read-only linear view derived from lineage. Recipe extraction reads `operation_graph` derived from lineage, not display summaries.

`operation_summaries` are an inspection/display boundary. They intentionally convert runtime values into JSON-safe summaries. WDF stores these summaries as an inspection-only snapshot and does not rebuild lineage, `operation_graph`, or executable recipes from them on load.

`source_time_offset` is a public frame property with one finite numeric offset per channel. Scalar inputs are normalized to all channels. Channel selection and continuous time slicing propagate offsets through existing frame APIs, and raw-data `add_channel(...)` records `source_time_offset` as a public replay option for graph recipes.

## Implementation Contract

### Runtime lineage

- Frame operations create lineage nodes that preserve operation names, params, and parent edges.
- `operation_history` remains a user-inspection view and may look linear even when the runtime graph has multiple parents.
- `operation_graph` preserves source leaves and parent structure for Recipe extraction.
- xarray exports do not use `operation_history` or `operation_graph` attrs as authoritative state.

### Operation summaries

- `operation_summaries` returns display-safe records, not executable replay payloads.
- A loaded WDF frame may carry an `operation_summaries_snapshot` without runtime lineage for pre-load operations.
- New operations after load are recorded in runtime lineage. Display summaries are composed as the load-time snapshot plus the post-load lineage delta.
- Saving a frame stores the composed summaries as the next inspection snapshot.
- Repeated `save -> load -> process -> save -> load` cycles must not duplicate, drop, or reorder display summaries.
- `RecipeSpec.from_frame(...)`, `GraphRecipeSpec.from_frame(...)`, and `NodeGraphRecipeSpec.from_frame(...)` must not recover pre-load snapshot entries as executable steps.

### Source time

- `source_time_offset` accepts a finite scalar or a one-dimensional sequence whose length matches the number of channels.
- Per-channel offsets are stored as the `source_time_offset` coordinate when a semantic channel dimension is available; otherwise offsets are stored in attrs.
- Channel slicing, channel selection, `add_channel(...)`, and continuous time slicing carry the appropriate per-channel offsets forward.
- Continuous time slicing adds `start / sampling_rate` to the selected channels' offsets.
- Unsupported or non-contiguous time selection remains outside the replay contract unless the frame API itself defines stable source-time behavior.

## Non-Goals

This decision does not add:

- executable Recipe persistence in WDF;
- broad `from_xarray(...)` restoration;
- operation provenance backed by xarray attrs;
- source-time alignment logic inside Recipe replay; or
- Recipe-side reconstruction of metadata, arrays, Dask graphs, or source-time updates.

## Validation Evidence

The current implementation aligns with this contract:

- `BaseFrame` accepts `source_time_offset` and `operation_summaries_snapshot` during construction and stores the snapshot separately from lineage.
- `operation_summaries` returns the snapshot when present, while storage uses a display-safe summary snapshot.
- `_operation_summaries_with_lineage_delta(...)` composes snapshot history with the runtime lineage delta after load.
- WDF save/load writes and reads operation summaries JSON separately from frame data and channel metadata.
- `source_time_offset` is normalized, validated, and stored as either a channel coordinate or attrs depending on frame shape.
- `GraphRecipeSpec` and `NodeGraphRecipeSpec` extract from runtime `operation_graph` and use explicit or mechanical input names rather than display summaries.
