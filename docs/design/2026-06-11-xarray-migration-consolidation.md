# ADR: Xarray Migration Consolidation

- **Status**: Accepted
- **Date**: 2026-06-11

The mutable state-update portion of this record is superseded by
[Immutable Frame state updates](2026-07-21-immutable-frame-state-updates.md).

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
- Supported frame schemas use semantic xarray dims and centralized channel coords.
- Channel `label`, `unit`, and `ref` are stored as xarray coords when a semantic `channel` dimension is present; channel `extra` is stored in `_xr.attrs` by stable channel id.
- Frame-level state such as `label`, `sampling_rate`, and `metadata` is backed by `_xr.attrs`.
- Operation provenance is runtime-only: `lineage` is the source of truth, `operation_history` is a read-only derived compatibility view, and neither `operation_history` nor `operation_graph` is exported through xarray attrs.
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

Explicit `ChannelMetadata.ref` values are preserved during this migration, including `ref=1.0`.

## Non-Goals

This consolidation does not introduce:

- `from_xarray`
- NetCDF or Zarr support
- xarray-native operation dispatch
- `xr.apply_ufunc` or `map_blocks` operation rewrites
- xarray-native channel metadata ownership or public APIs; Wandas still exposes and validates channel metadata through `ChannelMetadata` views
- dense time coordinates
- new frame schemas

## Consequences

The documentation surface is smaller and clearer. Contributors should read `docs/design/` for lasting architecture decisions instead of completed implementation plans.

The migration remains conservative: xarray stores labelled data and frame-level state, while Wandas keeps the signal-processing semantics and public domain API.
