# PR 2: xarray Dimension Semantics and Centralized Channel Coord

## Purpose

PR 2 builds on Phase 1 by moving channel dimension semantics into xarray named dimensions.
The goal is not to make execution xarray-native and not to broaden accepted input shapes.
The goal is to let xarray own the names and sizes of frame dimensions, then remove Wandas
code that separately tracks channel axis and channel coordinate state.

In one sentence:

```text
Use xarray named dimensions for channel semantics, but keep operations and metadata ownership unchanged.
```

## Scope

This PR covers these frame families:

```text
ChannelFrame:     (..., channel, time)
SpectralFrame:    (..., channel, frequency)
SpectrogramFrame: (..., channel, frequency, time)
NOctFrame:        (..., channel, band)
```

The `...` prefix is a schema design direction only. PR 2 must keep the current input validation:

```text
ChannelFrame:     1D or 2D input only
SpectralFrame:    1D or 2D input only
SpectrogramFrame: 2D or 3D input only
NOctFrame:        1D or 2D input only
```

`RoughnessFrame` is out of scope for PR 2 because it has a special mono 2D layout and a
multi-channel 3D layout. It should keep its existing channel count behavior until a later,
Roughness-specific schema pass.

## Non-Goals

PR 2 must not add:

- time coordinates
- frequency coordinates
- band coordinates
- leading-dimension semantics
- broader input dimensionality
- `from_xarray()`
- NetCDF or Zarr I/O
- xarray accessor APIs
- xarray-native operation dispatch
- `xr.apply_ufunc()` or `map_blocks()` execution
- attrs-backed internal metadata ownership

## Data Model

Each target frame declares a suffix of semantic dimension names. BaseFrame uses that suffix
to generate xarray dims while preserving neutral names for any leading dimensions.

```python
class BaseFrame:
    _xarray_dim_suffix: ClassVar[tuple[str, ...]] = ()

    def _xarray_dims(self, data: DaArray) -> tuple[str, ...]:
        suffix = self._xarray_dim_suffix
        prefix_count = data.ndim - len(suffix)
        if prefix_count < 0:
            return tuple(f"dim_{i}" for i in range(data.ndim))
        return tuple(f"dim_{i}" for i in range(prefix_count)) + suffix
```

Target frames set only the suffix:

```python
class ChannelFrame(BaseFrame):
    _xarray_dim_suffix = ("channel", "time")

class SpectralFrame(BaseFrame):
    _xarray_dim_suffix = ("channel", "frequency")

class SpectrogramFrame(BaseFrame):
    _xarray_dim_suffix = ("channel", "frequency", "time")

class NOctFrame(BaseFrame):
    _xarray_dim_suffix = ("channel", "band")
```

The suffix helper is future-compatible with leading dimensions, but PR 2 does not change
current constructors to accept those leading dimensions.

## Channel Count

`BaseFrame.n_channels` should prefer the xarray `"channel"` dimension size.

```python
_CHANNEL_DIM: ClassVar[str] = "channel"

@property
def _n_channels(self) -> int:
    if self._CHANNEL_DIM in self._xr.sizes:
        return int(self._xr.sizes[self._CHANNEL_DIM])
    return self._channel_count_from_data(self._data)
```

This makes xarray dims the authoritative channel count for frames that declare a channel
dimension. `_channel_count_from_data()` remains as a compatibility fallback and for out-of-scope
frames such as `RoughnessFrame`.

After PR 2, target frames should not need `_channel_axis` declarations. If `_channel_axis`
remains in `BaseFrame`, it should be treated as fallback-only and not as the primary channel
semantics for the target frames.

## Channel Coordinates

Channel labels should be attached as an xarray coordinate on the `"channel"` dimension when
and only when the number of labels matches the xarray channel size.

```python
def _xarray_coords(self, data: DaArray) -> dict[str, Any]:
    coords: dict[str, Any] = {}
    dims = self._xarray_dims(data)
    if self._CHANNEL_DIM not in dims:
        return coords

    labels = [ch.label for ch in self._channel_metadata]
    channel_size = int(data.shape[dims.index(self._CHANNEL_DIM)])
    if len(labels) == channel_size:
        coords[self._CHANNEL_DIM] = labels
    return coords
```

The coordinate should remain conservative:

- add `channel` coord only when lengths match
- omit `channel` coord when user-provided metadata length does not match the data
- do not synthesize time/frequency/band coords
- do not move channel metadata ownership into xarray attrs or coords

## Channel Coord Refresh

Phase 1 has ChannelFrame-specific channel coord refresh logic. PR 2 should centralize that
behavior in `BaseFrame` so any frame with a `"channel"` dimension can use it.

A private helper such as `_refresh_xarray_channel_coord()` can live on `BaseFrame`:

```python
def _refresh_xarray_channel_coord(self) -> None:
    if self._CHANNEL_DIM not in self._xr.dims:
        return

    labels = [ch.label for ch in self._channel_metadata]
    channel_size = int(self._xr.sizes[self._CHANNEL_DIM])
    if len(labels) != channel_size:
        self._xr = self._xr.drop_vars(self._CHANNEL_DIM, errors="ignore")
        return

    self._xr = self._xr.assign_coords({self._CHANNEL_DIM: labels})
```

Existing label update paths should call the centralized helper. PR 2 should not introduce new
public APIs for coord refresh.

## Expected Internal Schemas

The expected xarray dims after construction are:

```text
ChannelFrame from 1D input:
  data normalized to shape (1, n_time)
  dims=("channel", "time")

ChannelFrame from 2D input:
  dims=("channel", "time")

SpectralFrame from 1D input:
  data normalized to shape (1, n_frequency)
  dims=("channel", "frequency")

SpectralFrame from 2D input:
  dims=("channel", "frequency")

SpectrogramFrame from 2D input:
  data expanded to shape (1, n_frequency, n_time)
  dims=("channel", "frequency", "time")

SpectrogramFrame from 3D input:
  dims=("channel", "frequency", "time")

NOctFrame from 1D input:
  data normalized to shape (1, n_band)
  dims=("channel", "band")

NOctFrame from 2D input:
  dims=("channel", "band")
```

Only the `channel` coordinate is in scope. `time`, `frequency`, and `band` coordinates must not
exist after this PR unless they already existed for unrelated reasons before the PR.

## Error Handling and Compatibility

PR 2 should preserve current validation errors for unsupported dimensionality. The suffix schema
must not silently accept higher-dimensional inputs.

If a frame has no `"channel"` dim, `n_channels` should fall back to existing behavior. This keeps
`RoughnessFrame` and any neutral BaseFrame test doubles stable.

If metadata length does not match channel size, xarray should not receive a stale or invalid
channel coord. The Wandas `_channel_metadata` list remains untouched for compatibility.

## Testing Strategy

Add or update tests that prove:

- target frames expose expected suffix dims
- `n_channels` reads from xarray `"channel"` dim when present
- default channel metadata length matches the xarray channel size
- channel coord exists only when metadata length matches the channel size
- channel coord refresh is centralized and works for label update paths
- time/frequency/band coords are still absent
- existing constructor dimensionality restrictions remain unchanged
- operations remain lazy and Dask-backed
- no existing tests are removed

Suggested focused tests:

```text
test_target_frames_use_semantic_suffix_dims
test_n_channels_prefers_xarray_channel_size
test_spectral_frame_adds_channel_coord_without_frequency_coord
test_spectrogram_frame_adds_channel_coord_without_time_or_frequency_coords
test_noct_frame_adds_channel_coord_without_band_coord
test_metadata_length_mismatch_omits_channel_coord
test_constructor_dimension_constraints_are_unchanged
```

Run the existing full verification after implementation:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check wandas tests
uv run pytest
```

## Success Criteria

PR 2 is successful when:

- target frame dims are semantic suffix dims
- channel count for target frames is read from xarray `"channel"` sizes
- channel coord logic is shared instead of ChannelFrame-only
- `_channel_axis` is no longer the primary channel semantics for target frames
- no time/frequency/band coords are introduced
- no accepted input shapes change
- operation execution remains unchanged
- full test suite passes
