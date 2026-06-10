# Phase 3 Design: Xarray-Backed Frame State

## Status

Proposed. This spec assumes Phase 2 has introduced xarray-backed storage and exact-rank semantic dimensions for target frames.

## Goal

Move frame-level state into the internal `xarray.DataArray.attrs` so Wandas has one storage location for frame state while preserving the public property API.

This phase is intentionally about state ownership, not operation execution or channel metadata migration.

## Motivation

After Phase 1 and Phase 2, `BaseFrame` stores data, dims, and conservative coords in `_xr`, but still stores frame-level state as separate Python attributes:

```text
BaseFrame
  _xr
    data
    dims
    coords

  sampling_rate
  label
  metadata
  operation_history
  _channel_metadata
```

That split keeps Phase 1/2 safe, but it leaves duplicated state ownership. Phase 3 makes `_xr.attrs` the backing store for frame-level state:

```text
BaseFrame
  _xr
    data
    dims
    coords
    attrs
      sampling_rate
      label
      metadata
      operation_history

  _channel_metadata
  _previous
```

Wandas properties keep the domain semantics. xarray is only the storage location.

## Scope

Move these frame-level fields to `_xr.attrs`:

- `sampling_rate`
- `label`
- `metadata`
- `operation_history`

Remove these `FrameMetadata` concepts from the public API and implementation:

- `FrameMetadata`
- `FrameMetadata.copy()`
- `FrameMetadata.merged()`
- `FrameMetadata.source_file`
- WDF special handling for `FrameMetadata.source_file`

Use plain dictionaries for frame metadata. `source_file` moves to a reserved metadata key:

```python
frame.metadata["_source_file"] = "input.wav"
```

## Non-Goals

Phase 3 does not change:

- `ChannelMetadata` storage or behavior
- channel `unit`, `ref`, or `extra` handling
- generated numeric coordinates such as dense time, frequency, band, or bark coordinates
- operation execution
- xarray-native operation dispatch
- `from_xarray`
- NetCDF or Zarr I/O
- accepted input shapes

`ChannelMetadata` remains in `wandas/core/metadata.py` for now. Removing Pydantic or moving channel metadata to xarray coords/attrs is a later phase.

## Public API

These existing public attributes remain available as properties:

```python
frame.sampling_rate
frame.label
frame.metadata
frame.operation_history
```

Their backing store changes to `_xr.attrs`.

### `sampling_rate`

`sampling_rate` is validated through `validate_sampling_rate()` in the setter and stored as `float`:

```text
frame.sampling_rate = 0      -> ValueError
frame.sampling_rate = -1     -> ValueError
frame.sampling_rate = 48000  -> _xr.attrs["sampling_rate"] = 48000.0
```

### `label`

`label` is stored in `_xr.attrs["label"]` and mirrored to `_xr.name`. `None` or an empty string becomes `"unnamed_frame"`. Non-string non-`None` values raise `TypeError`.

### `metadata`

`metadata` is a plain `dict[str, Any]`:

```text
metadata=None        -> {}
metadata=dict        -> shallow dict copy on set
metadata=other type  -> TypeError
```

The getter returns the internal metadata dict so existing mutation patterns continue to work:

```python
frame.metadata["calibration"] = 1.0
```

### `operation_history`

`operation_history` is stored as a list of dictionaries. The setter deep-copies incoming history to avoid sharing mutable state between frames.

## Initialization Flow

`BaseFrame.__init__` should build `_xr` before setting frame-level properties because those properties write to `_xr.attrs`.

Order:

1. Normalize data.
2. Build `_channel_metadata`, because channel coords depend on labels.
3. Build `_xr` with data, dims, coords, and name.
4. Set `label`, `sampling_rate`, `metadata`, and `operation_history` through property setters.
5. Set `_previous`.

`_build_xarray()` must not read `self.label` before `_xr` exists. It should accept a `name` argument or otherwise receive the name explicitly.

## Data Replacement

`_replace_data()` replaces data/dims/coords while preserving frame-level attrs:

```python
attrs = copy.deepcopy(self._xr.attrs)
self._xr = self._build_xarray(normalized_data, name=self.label)
self._xr.attrs = attrs
```

This keeps metadata and operation history stable across data replacement.

## New Frame Creation

`_create_new_instance()` must keep the current no-shared-mutable-state behavior:

- default `metadata` is `copy.deepcopy(self.metadata)`
- default `operation_history` is `copy.deepcopy(self.operation_history)`
- default channel metadata remains `copy.deepcopy(self._channel_metadata)`

Any use of `metadata.merged(...)` must be replaced with plain dict merging, for example:

```python
metadata={**self.metadata, **params}
```

or an equivalent local expression. Do not add a broad abstraction unless repeated code becomes hard to read.

## Xarray Export

`to_xarray()` continues to return a public shallow copy of `_xr`, not the internal object. Data remains shallow; attrs are deep-copied:

```python
exported = self._xr.copy(deep=False)
exported.attrs = copy.deepcopy(self._xr.attrs)
return exported
```

External mutation of `frame.xr.attrs["metadata"]` or `frame.xr.attrs["operation_history"]` must not mutate the frame.

## WDF and File Readers

WDF should store metadata as plain dictionary entries. `_source_file`, when present, is just another metadata key.

WAV and URL readers should populate:

```python
metadata={"_source_file": source}
```

Round trips should verify `metadata["_source_file"]`, not `metadata.source_file`.

## Error Handling

- Invalid `metadata` types raise `TypeError`.
- Invalid `operation_history` types raise `TypeError`.
- Invalid `sampling_rate` values raise `ValueError` from `validate_sampling_rate()`.
- Invalid `label` types raise `TypeError`.
- Missing attrs are repaired lazily by getters where safe: `metadata` becomes `{}` and `operation_history` becomes `[]`. Missing `sampling_rate` should remain an error because a frame without sampling rate is invalid.

## Tests

Add or update tests for:

- `sampling_rate`, `label`, `metadata`, and `operation_history` using `_xr.attrs` as source of truth
- property setters updating `_xr.attrs` and `_xr.name`
- invalid `sampling_rate`, `label`, `metadata`, and `operation_history` values
- `to_xarray()` deep-copying attrs
- `_replace_data()` preserving attrs
- `_create_new_instance()` deep-copying metadata and operation history
- WDF source file round trip through `metadata["_source_file"]`
- WAV/URL source file metadata through `metadata["_source_file"]`
- removal of `FrameMetadata` tests while preserving `ChannelMetadata` tests

Run the full verification suite before completion:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check wandas tests
uv run pytest
```

## Success Criteria

Phase 3 is complete when:

1. `FrameMetadata` is removed from production code.
2. `frame.metadata` is a plain dict.
3. `frame.label`, `frame.sampling_rate`, `frame.metadata`, and `frame.operation_history` are backed by `_xr.attrs`.
4. Public property access remains available.
5. `to_xarray()` returns a public copy with deep-copied attrs.
6. `ChannelMetadata` remains unchanged.
7. Generated numeric coords are not added.
8. Operation execution remains unchanged.
9. Production code removes the FrameMetadata-specific paths instead of adding a compatibility layer.
