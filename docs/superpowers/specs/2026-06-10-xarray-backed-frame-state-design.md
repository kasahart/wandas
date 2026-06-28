# Phase 3 Design: Xarray-Backed Frame State

## Status

Superseded by the current xarray-backed runtime model. This note is retained as design history, but operation provenance has since moved out of `_xr.attrs`: `lineage` is the runtime source of truth, and `operation_history` is a read-only derived compatibility view.

## Goal

Move durable frame-level state into the internal `xarray.DataArray.attrs` so Wandas has one storage location for frame state while preserving the public property API. Runtime operation provenance is explicitly excluded from durable attrs.

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
  lineage
  _channel_metadata
```

That split keeps Phase 1/2 safe, but it leaves duplicated state ownership. Phase 3 makes `_xr.attrs` the backing store for durable frame-level state while keeping operation provenance runtime-only:

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

  _lineage
  _channel_metadata
  _previous
```

Wandas properties keep the domain semantics. xarray is only the storage location.

## Scope

Move these frame-level fields to `_xr.attrs`:

- `sampling_rate`
- `label`
- `metadata`

Do not move operation provenance to `_xr.attrs`. `lineage` is held on the frame at runtime, and `operation_history` is derived from that lineage.

Remove these `FrameMetadata` concepts from the public API and implementation:

- `FrameMetadata`
- `FrameMetadata.copy()`
- `FrameMetadata.merged()`
- `FrameMetadata.source_file`
- WDF special handling for `FrameMetadata.source_file`

Use plain dictionaries for frame metadata. This is an intentional breaking change for code importing `FrameMetadata` or using `frame.metadata.source_file`. `source_file` moves to a reserved metadata key:

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
frame.lineage
```

The durable state backing store changes to `_xr.attrs`. `lineage` remains runtime-only, and `operation_history` is a read-only compatibility view derived from lineage.

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

### `lineage` and `operation_history`

`lineage` is the runtime source of truth for operation provenance. It is set during construction and propagated through `_create_new_instance(..., lineage=...)`. `operation_history` has no setter and is generated from lineage on access.

## Initialization Flow

`BaseFrame.__init__` should build `_xr` before setting frame-level properties because those properties write to `_xr.attrs`.

Order:

1. Normalize data.
2. Build `_channel_metadata`, because channel coords depend on labels.
3. Build `_xr` with data, dims, coords, and name.
4. Set `label`, `sampling_rate`, and `metadata` through property setters, and assign internal runtime `_lineage` from the constructor argument.
5. Set `_previous`.

`_build_xarray()` must not read `self.label` before `_xr` exists. It should accept a `name` argument or otherwise receive the name explicitly.

## Data Replacement

`_replace_data()` replaces data/dims/coords while preserving frame-level attrs:

```python
attrs = copy.deepcopy(self._xr.attrs)
self._xr = self._build_xarray(normalized_data, name=self.label)
self._xr.attrs = attrs
```

This keeps metadata stable across data replacement. Runtime lineage is preserved separately on the frame.

## New Frame Creation

`_create_new_instance()` must keep the current no-shared-mutable-state behavior:

- default `metadata` is `copy.deepcopy(self.metadata)`
- default `lineage` is `self.lineage`
- default channel metadata remains `copy.deepcopy(self._channel_metadata)`

Any use of `metadata.merged(...)` must be replaced with plain dict merging, for example:

```python
metadata={**self.metadata, **domain_metadata}
```

or an equivalent local expression. Do not duplicate operation parameters into `metadata[operation_name]`; provenance parameters belong to lineage. Do not add a broad abstraction unless repeated code becomes hard to read.

## Xarray Export

`to_xarray()` continues to return a public shallow copy of `_xr`, not the internal object. Data remains shallow; attrs are deep-copied:

```python
exported = self._xr.copy(deep=False)
exported.attrs = copy.deepcopy(self._xr.attrs)
return exported
```

External mutation of `frame.xr.attrs["metadata"]` must not mutate the frame. Legacy provenance attrs, if present on external xarray objects, are ignored by Wandas frames.
`to_xarray()` must omit `operation_history` and `operation_graph`; runtime provenance is not exported as attrs.

## WDF and File Readers

WDF should store metadata as plain dictionary entries. `_source_file`, when present, is just another metadata key.

WAV and URL readers should populate:

```python
metadata={"_source_file": source}
```

Round trips should verify `metadata["_source_file"]`, not `metadata.source_file`.

## Error Handling

- Invalid `metadata` types raise `TypeError`.
- Assigning `operation_history` or `lineage` directly raises `AttributeError`; use construction or `_create_new_instance(..., lineage=...)`.
- Invalid `sampling_rate` values raise `ValueError` from `validate_sampling_rate()`.
- Invalid `label` types raise `TypeError`.
- Missing attrs are repaired lazily by getters where safe: `metadata` becomes `{}`. `operation_history` ignores attrs and returns the lineage-derived view, or `[]` when lineage is absent. Missing `sampling_rate` should remain an error because a frame without sampling rate is invalid.

## Tests

Add or update tests for:

- `sampling_rate`, `label`, and `metadata` using `_xr.attrs` as source of truth
- `lineage` as the source of truth for derived `operation_history` and `operation_graph`
- property setters updating `_xr.attrs` and `_xr.name`
- invalid `sampling_rate`, `label`, and `metadata` values
- direct assignment rejection for `lineage` and `operation_history`
- `to_xarray()` deep-copying attrs
- `to_xarray()` omitting `operation_history` and `operation_graph`
- `_replace_data()` preserving attrs
- `_create_new_instance()` deep-copying metadata and preserving or replacing lineage through its keyword argument
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
3. `frame.label`, `frame.sampling_rate`, and `frame.metadata` are backed by `_xr.attrs`.
4. `frame.lineage` is runtime-only, and `frame.operation_history` / `frame.operation_graph` are derived views that are not stored or exported.
5. Public property access remains available.
6. `to_xarray()` returns a public copy with deep-copied attrs.
7. `ChannelMetadata` remains unchanged.
8. Generated numeric coords are not added.
9. Operation execution remains unchanged.
10. Production code removes the FrameMetadata-specific paths instead of adding a compatibility layer.
