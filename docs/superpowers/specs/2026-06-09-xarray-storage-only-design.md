# Phase 1: xarray-backed storage only

## Purpose

This phase introduces xarray only as an axis-aware data container for Wandas frames.
It is not an xarray-native execution, metadata, or storage migration.

Current provenance note: this phase predates the runtime-lineage cleanup. In the current model, operation provenance is runtime-only: `lineage` is the source of truth, and `operation_history` is a read-only derived compatibility view that is not backed by `_xr.attrs`.

The goal is to evaluate whether replacing `BaseFrame`'s direct Dask-array ownership with
`xr.DataArray` reduces long-term complexity without changing the existing Wandas API or
operation semantics.

## Background

The previous `feat/xarray-bridge` branch grew beyond a bridge. It included authoritative
xarray storage, attrs-backed metadata, broad `from_xarray()` restoration, NetCDF helpers,
and xarray-aware operation dispatch. That made the migration hard to review and made it
unclear whether xarray itself was reducing complexity or adding synchronization work.

This replacement branch starts from `develop` and deliberately limits Phase 1 to:

```text
xarray as storage, not execution engine.
```

## Responsibility split

In Phase 1, responsibilities are split as follows:

```text
xarray:
  data
  dims
  coords

Wandas:
  sampling_rate
  label
  metadata
  lineage
  channel_metadata
  previous
  domain API
```

`xr.DataArray.attrs` may be populated for public export, but attrs are not the internal
source of truth for Wandas state.

## Architecture

`BaseFrame` owns one internal xarray object:

```python
class BaseFrame:
    _xr: xr.DataArray

    @property
    def _data(self):
        return self._xr.data
```

`_data` remains as a read-only compatibility alias for existing operation code.
There is no `_data` setter.

Production code that currently assigns to `_data` must stop doing so. If an in-place
method needs to replace frame data, it should call an explicit private method:

```python
def _replace_data(self, data: DaArray) -> None:
    self._xr = self._build_xarray(data)
```

`_replace_data()` updates only the xarray data container. It must not mutate or synchronize
`sampling_rate`, `label`, `metadata`, runtime provenance, or `_channel_metadata`.

## Data model

`BaseFrame.__init__` keeps the existing Dask normalization and channel-wise rechunk policy:

```text
1D input:
  reshape to (1, -1)

2D frame:
  chunks=(1, -1)

3D+ frame:
  chunks=(1, -1, -1, ...)
```

After normalization, the Dask array is wrapped in `xr.DataArray`.

The initial schema is intentionally small. `BaseFrame` may define hooks such as:

```python
def _xarray_dims(self, data: DaArray) -> tuple[str, ...]: ...
def _xarray_coords(self, data: DaArray) -> dict[str, Any]: ...
```

Subclasses can override these hooks only where they already have the required axis
information. In Phase 1 the concrete schema remains intentionally conservative:

```text
BaseFrame default:
  dims=("dim_0", "dim_1", ...)
  coords: none

ChannelFrame:
  dims=("channel", "time")
  coords: channel labels only when metadata length matches the channel count
  no generated time coordinate

SpectralFrame:
  BaseFrame default dims/coords in this PR

SpectrogramFrame:
  BaseFrame default dims/coords in this PR
```

If a frame does not have enough domain information for a rich coordinate, the implementation
should use neutral dimension names instead of adding validation-heavy schema logic.
Time and frequency coordinates are deferred to a later schema phase.

## Public API

Phase 1 adds:

```python
frame.to_xarray()
frame.xr
```

`to_xarray()` returns a public shallow copy of the internal `DataArray`. It may attach export
attrs such as:

```text
wandas_frame_type
sampling_rate
label
metadata
```

These attrs are export metadata only. They do not back the live Wandas frame state.
Current exports omit runtime provenance attrs (`operation_history` and `operation_graph`).

`.xr` is a convenience alias for `to_xarray()`.

## Operation execution

Existing operation execution remains unchanged:

```python
processed_data = operation.process(self._data)
```

Because `_data` is an alias to `_xr.data`, existing Dask-backed operations should keep the
same behavior and laziness.

Phase 1 must not add:

```text
process_xarray()
process_dataarray()
xr.apply_ufunc()
xarray operation dispatch
blockwise or map_overlap execution modes
```

## Explicitly out of scope

Phase 1 does not include:

```text
metadata backed by _xr.attrs
runtime provenance backed by _xr.attrs
channel_metadata backed by _xr.attrs
sampling_rate or label backed by _xr.attrs
wd.from_xarray() broad restoration
NetCDF support
Zarr support
xarray accessor API
operation chunk-policy integration
```

These are later-phase topics and should not enter this PR.
The later lineage cleanup chose runtime-only provenance instead of `operation_history` attrs backing.

## Lazy execution requirements

Lazy execution is a core requirement for this phase. The implementation must verify that
xarray wrapping does not materialize data early.

Required tests:

```text
BaseFrame construction does not compute.
Accessing frame._data does not compute.
Accessing frame.to_xarray() / frame.xr does not compute.
get_channel() and channel slicing do not compute.
add_channel() and remove_channel() do not compute.
apply_operation() does not compute.
Existing transform methods such as fft() and stft() do not compute.
compute() and data access are the points that materialize data.
Dask graph structure remains lazy after wrapping in xarray.
Default chunks remain channel-wise: channel=1 and signal axes=-1.
```

Tests should use Dask delayed or mocked compute paths where needed, so they fail if
construction/export accidentally calls `compute()`.

## Error handling and compatibility

`_data` assignment should fail because `_data` is a read-only property. Existing tests that
assign fake objects to `_data` should be rewritten to exercise public behavior or
`_replace_data()` where appropriate.

`_replace_data()` should be private and narrow. It exists for existing in-place frame methods,
not as a general extension point.

If xarray coord generation fails for a frame, the implementation should prefer a small,
predictable fallback schema over complex recovery logic.

## Complexity evaluation

This phase succeeds only if the implementation stays simpler than the previous broad branch.

Positive signals:

```text
BaseFrame has a single data owner: _xr.
Production code no longer assigns to _data.
Operation paths do not gain xarray-vs-legacy branching.
Metadata synchronization code does not appear.
Added code is mostly small construction/export helpers.
Existing tests continue to pass with minimal behavioral changes.
```

Negative signals:

```text
_data setter returns.
Metadata attrs become authoritative.
Subclass schema branching grows large.
to_xarray() needs complex validation or restoration logic.
Operation dispatch changes are needed to keep tests passing.
Production code grows substantially without removing ownership ambiguity.
```

After implementation, the PR description should include a short complexity review:

```text
What became simpler?
What became more complex?
Which added pieces are permanent?
Which added pieces are candidates for deletion or Phase 2 replacement?
```

## Verification

Required verification before opening the PR:

```text
uv run pytest -q
uv run ruff check wandas tests
uv run ty check wandas tests
```

The PR should also include focused tests for the xarray-backed storage behavior and lazy
execution requirements above.
