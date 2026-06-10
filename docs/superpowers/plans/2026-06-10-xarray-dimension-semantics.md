# xarray Dimension Semantics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move channel dimension semantics for ChannelFrame, SpectralFrame, SpectrogramFrame, and NOctFrame into xarray named dims and centralized channel coords without changing execution, I/O, metadata ownership, or accepted input shapes.

**Architecture:** BaseFrame gains a suffix-based xarray dim helper and a centralized channel coord helper. Target frames declare only their semantic dim suffix. `n_channels` prefers the xarray `"channel"` dim size and falls back to legacy channel counting for neutral frames and out-of-scope RoughnessFrame.

**Tech Stack:** Python, Dask Array, xarray, pytest, ty, ruff.

---

## File Structure

- `wandas/core/base_frame.py`
  - Add `_CHANNEL_DIM` and `_xarray_dim_suffix` class variables.
  - Implement suffix-based `_xarray_dims()` and centralized `_xarray_coords()`.
  - Make `_n_channels` prefer `self._xr.sizes["channel"]`.
  - Move `_refresh_xarray_channel_coord()` from `ChannelFrame` to `BaseFrame`.

- `wandas/frames/channel.py`
  - Replace ChannelFrame-specific `_xarray_dims()`, `_xarray_coords()`, and `_refresh_xarray_channel_coord()` with `_xarray_dim_suffix = ("channel", "time")`.
  - Keep existing validation and label update calls.

- `wandas/frames/spectral.py`
  - Add `_xarray_dim_suffix = ("channel", "frequency")`.
  - Do not add frequency coords.

- `wandas/frames/spectrogram.py`
  - Add `_xarray_dim_suffix = ("channel", "frequency", "time")`.
  - Remove or demote `_channel_axis = -3` if BaseFrame fallback still covers legacy use.
  - Do not add frequency/time coords.

- `wandas/frames/noct.py`
  - Add `_xarray_dim_suffix = ("channel", "band")`.
  - Do not add band coords.

- `tests/core/test_xarray_storage_only.py`
  - Add focused tests for target dims, channel-size authority, channel coord behavior, and preserved non-goals.
  - Update the Phase 1 AxisOnlyFrame test so it validates fallback behavior separately from suffix semantics.

---

### Task 1: Add RED tests for semantic suffix dims

**Files:**
- Modify: `tests/core/test_xarray_storage_only.py`

- [ ] **Step 1: Import target frame classes**

Add imports near the existing frame imports:

```python
from wandas.frames.noct import NOctFrame
from wandas.frames.spectral import SpectralFrame
```

- [ ] **Step 2: Add failing semantic dims test**

Add this test after `test_base_frame_owns_xarray_dataarray`:

```python
def test_target_frames_use_semantic_suffix_dims() -> None:
    channel = ChannelFrame.from_numpy(np.ones((2, 8)), sampling_rate=8.0)
    spectral = SpectralFrame(
        data=da.ones((2, 5), chunks=(1, 5)) + 0j,
        sampling_rate=8.0,
        n_fft=8,
    )
    spectrogram = SpectrogramFrame(
        data=da.ones((2, 5, 3), chunks=(1, 5, 3)) + 0j,
        sampling_rate=8.0,
        n_fft=8,
        hop_length=2,
    )
    noct = NOctFrame(
        data=da.ones((2, 4), chunks=(1, 4)),
        sampling_rate=8.0,
        fmin=20.0,
        fmax=2000.0,
    )

    assert channel._xr.dims == ("channel", "time")
    assert spectral._xr.dims == ("channel", "frequency")
    assert spectrogram._xr.dims == ("channel", "frequency", "time")
    assert noct._xr.dims == ("channel", "band")
```

- [ ] **Step 3: Verify RED**

Run:

```bash
uv run pytest tests/core/test_xarray_storage_only.py::test_target_frames_use_semantic_suffix_dims -q
```

Expected: FAIL because SpectralFrame, SpectrogramFrame, and NOctFrame still expose neutral `dim_*` dims.

- [ ] **Step 4: Commit RED test**

```bash
git add tests/core/test_xarray_storage_only.py
git commit -m "test: define xarray semantic frame dims"
```

---

### Task 2: Implement suffix-based dims for target frames

**Files:**
- Modify: `wandas/core/base_frame.py`
- Modify: `wandas/frames/channel.py`
- Modify: `wandas/frames/spectral.py`
- Modify: `wandas/frames/spectrogram.py`
- Modify: `wandas/frames/noct.py`
- Test: `tests/core/test_xarray_storage_only.py`

- [ ] **Step 1: Add class variables and suffix dim helper to BaseFrame**

In `wandas/core/base_frame.py`, replace the class variable block:

```python
    _channel_axis: ClassVar[int | None] = -2
    _xr: xr.DataArray
```

with:

```python
    _CHANNEL_DIM: ClassVar[str] = "channel"
    _channel_axis: ClassVar[int | None] = -2
    _xarray_dim_suffix: ClassVar[tuple[str, ...]] = ()
    _xr: xr.DataArray
```

Replace `_xarray_dims()` with:

```python
    def _xarray_dims(self, data: DaArray) -> tuple[str, ...]:
        """Return xarray dimension names using any declared semantic suffix."""
        suffix = self._xarray_dim_suffix
        prefix_count = data.ndim - len(suffix)
        if not suffix or prefix_count < 0:
            return tuple(f"dim_{i}" for i in range(data.ndim))
        return tuple(f"dim_{i}" for i in range(prefix_count)) + suffix
```

- [ ] **Step 2: Replace ChannelFrame xarray hooks with suffix declaration**

In `wandas/frames/channel.py`, add this class variable immediately below the class declaration/docstring attributes area, before `__init__`:

```python
    _xarray_dim_suffix = ("channel", "time")
```

Remove `ChannelFrame._xarray_dims()` only. Keep `_xarray_coords()` and `_refresh_xarray_channel_coord()` for now; they will be centralized in a later task.

- [ ] **Step 3: Add suffix declarations to target frames**

In `wandas/frames/spectral.py`, add inside `class SpectralFrame` before `n_fft: int`:

```python
    _xarray_dim_suffix = ("channel", "frequency")
```

In `wandas/frames/spectrogram.py`, replace:

```python
    _channel_axis = -3
```

with:

```python
    _channel_axis = -3
    _xarray_dim_suffix = ("channel", "frequency", "time")
```

Keep `_channel_axis = -3` for now. It remains a fallback until channel count is centralized in Task 3.

In `wandas/frames/noct.py`, add inside `class NOctFrame` before `fmin: float`:

```python
    _xarray_dim_suffix = ("channel", "band")
```

- [ ] **Step 4: Verify semantic dims test passes**

Run:

```bash
uv run pytest tests/core/test_xarray_storage_only.py::test_target_frames_use_semantic_suffix_dims -q
```

Expected: PASS.

- [ ] **Step 5: Run focused frame tests**

Run:

```bash
uv run pytest tests/core/test_xarray_storage_only.py tests/frames/test_spectral_frame.py tests/frames/test_spectrogram_frame.py tests/frames/test_noct_frame.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit suffix dims implementation**

```bash
git add wandas/core/base_frame.py wandas/frames/channel.py wandas/frames/spectral.py wandas/frames/spectrogram.py wandas/frames/noct.py
git commit -m "feat: declare semantic xarray dimension suffixes"
```

---

### Task 3: Add RED tests for xarray channel-size authority and channel coords

**Files:**
- Modify: `tests/core/test_xarray_storage_only.py`

- [ ] **Step 1: Add test proving n_channels prefers xarray channel size**

Add this helper class near `AxisOnlyFrame`:

```python
class SuffixOnlyFrame(BaseFrame[np.ndarray]):
    _xarray_dim_suffix = ("channel", "frequency", "time")

    def plot(self, plot_type: str = "default", ax=None, **kwargs):
        raise NotImplementedError

    def _get_dataframe_index(self):
        return None
```

Add this test near the existing `test_base_frame_channel_axis_drives_default_metadata_and_n_channels`:

```python
def test_n_channels_prefers_xarray_channel_dim_size() -> None:
    frame = SuffixOnlyFrame(data=da.ones((2, 3, 4), chunks=(1, 3, 4)), sampling_rate=1.0)

    assert frame._xr.dims == ("channel", "frequency", "time")
    assert frame.n_channels == 2
    assert len(frame.channels) == 2
```

This fails before Task 4 because the legacy `_channel_axis=-2` fallback reads size 3, while the xarray `"channel"` dim size is 2.

- [ ] **Step 2: Add tests for target frame channel coords and absent non-channel coords**

Add:

```python
def test_spectral_frame_adds_channel_coord_without_frequency_coord() -> None:
    frame = SpectralFrame(
        data=da.ones((2, 5), chunks=(1, 5)) + 0j,
        sampling_rate=8.0,
        n_fft=8,
        channel_metadata=[
            ChannelMetadata(label="left"),
            ChannelMetadata(label="right"),
        ],
    )

    assert list(frame._xr.coords["channel"].values) == ["left", "right"]
    assert "frequency" not in frame._xr.coords


def test_spectrogram_frame_adds_channel_coord_without_frequency_or_time_coords() -> None:
    frame = SpectrogramFrame(
        data=da.ones((2, 5, 3), chunks=(1, 5, 3)) + 0j,
        sampling_rate=8.0,
        n_fft=8,
        hop_length=2,
        channel_metadata=[
            ChannelMetadata(label="left"),
            ChannelMetadata(label="right"),
        ],
    )

    assert list(frame._xr.coords["channel"].values) == ["left", "right"]
    assert "frequency" not in frame._xr.coords
    assert "time" not in frame._xr.coords


def test_noct_frame_adds_channel_coord_without_band_coord() -> None:
    frame = NOctFrame(
        data=da.ones((2, 4), chunks=(1, 4)),
        sampling_rate=8.0,
        fmin=20.0,
        fmax=2000.0,
        channel_metadata=[
            ChannelMetadata(label="left"),
            ChannelMetadata(label="right"),
        ],
    )

    assert list(frame._xr.coords["channel"].values) == ["left", "right"]
    assert "band" not in frame._xr.coords
```

- [ ] **Step 3: Add mismatch test**

Add:

```python
def test_channel_coord_omitted_when_metadata_length_differs_for_target_frames() -> None:
    spectral = SpectralFrame(
        data=da.ones((2, 5), chunks=(1, 5)) + 0j,
        sampling_rate=8.0,
        n_fft=8,
        channel_metadata=[ChannelMetadata(label="only-one")],
    )

    assert spectral.labels == ["only-one"]
    assert "channel" not in spectral._xr.coords
```

- [ ] **Step 4: Verify RED**

Run:

```bash
uv run pytest tests/core/test_xarray_storage_only.py::test_spectral_frame_adds_channel_coord_without_frequency_coord tests/core/test_xarray_storage_only.py::test_spectrogram_frame_adds_channel_coord_without_frequency_or_time_coords tests/core/test_xarray_storage_only.py::test_noct_frame_adds_channel_coord_without_band_coord -q
```

Expected: FAIL because channel coord is still ChannelFrame-only.

- [ ] **Step 5: Commit RED tests**

```bash
git add tests/core/test_xarray_storage_only.py
git commit -m "test: define centralized channel coord semantics"
```

---

### Task 4: Centralize channel count and channel coord in BaseFrame

**Files:**
- Modify: `wandas/core/base_frame.py`
- Modify: `wandas/frames/channel.py`
- Modify: `wandas/frames/spectrogram.py`
- Test: `tests/core/test_xarray_storage_only.py`

- [ ] **Step 1: Add channel-size helper to BaseFrame**

In `wandas/core/base_frame.py`, add this method after `_xarray_coords()`:

```python
    def _channel_size_from_xarray_dims(self, data: DaArray) -> int | None:
        """Return the channel size implied by xarray dims, if present."""
        dims = self._xarray_dims(data)
        if self._CHANNEL_DIM not in dims:
            return None
        return int(data.shape[dims.index(self._CHANNEL_DIM)])
```

- [ ] **Step 2: Update `_xarray_coords()` in BaseFrame**

Replace BaseFrame `_xarray_coords()` with:

```python
    def _xarray_coords(self, data: DaArray) -> dict[str, Any]:
        """Return conservative coordinates for declared xarray dimensions."""
        channel_size = self._channel_size_from_xarray_dims(data)
        if channel_size is None:
            return {}

        labels = [ch.label for ch in self._channel_metadata]
        if len(labels) != channel_size:
            return {}
        return {self._CHANNEL_DIM: labels}
```

- [ ] **Step 3: Make channel metadata defaults use xarray dims first**

Replace the default channel metadata block in `BaseFrame.__init__`:

```python
            self._channel_metadata = [
                ChannelMetadata(label=f"ch{i}", unit="", extra={})
                for i in range(self._channel_count_from_data(normalized_data))
            ]
```

with:

```python
            channel_count = self._channel_size_from_xarray_dims(normalized_data)
            if channel_count is None:
                channel_count = self._channel_count_from_data(normalized_data)
            self._channel_metadata = [
                ChannelMetadata(label=f"ch{i}", unit="", extra={})
                for i in range(channel_count)
            ]
```

- [ ] **Step 4: Make `_n_channels` prefer xarray sizes**

Replace `_n_channels` with:

```python
    @property
    def _n_channels(self) -> int:
        """Returns the number of channels from the xarray channel dimension when available."""
        if self._CHANNEL_DIM in self._xr.sizes:
            return int(self._xr.sizes[self._CHANNEL_DIM])
        return self._channel_count_from_data(self._data)
```

- [ ] **Step 5: Move channel coord refresh to BaseFrame**

Add this method to `BaseFrame` after `_n_channels`:

```python
    def _refresh_xarray_channel_coord(self) -> None:
        """Refresh the internal xarray channel coordinate after label changes."""
        if self._CHANNEL_DIM not in self._xr.dims:
            return

        labels = [ch.label for ch in self._channel_metadata]
        channel_size = int(self._xr.sizes[self._CHANNEL_DIM])
        if len(labels) != channel_size:
            self._xr = self._xr.drop_vars(self._CHANNEL_DIM, errors="ignore")
            return

        self._xr = self._xr.assign_coords({self._CHANNEL_DIM: labels})
```

- [ ] **Step 6: Remove ChannelFrame-specific xarray coord methods**

In `wandas/frames/channel.py`, remove `ChannelFrame._xarray_coords()` and `ChannelFrame._refresh_xarray_channel_coord()`.

Keep `_set_channel_labels()` unchanged; it should continue calling `self._refresh_xarray_channel_coord()`, now inherited from `BaseFrame`.

- [ ] **Step 7: Remove target-frame `_channel_axis` declarations where xarray dims now cover channel count**

In `wandas/frames/spectrogram.py`, remove:

```python
    _channel_axis = -3
```

Do not remove `_channel_axis` from `BaseFrame`; it is still the fallback for neutral frames and RoughnessFrame behavior.

- [ ] **Step 8: Verify RED tests are GREEN**

Run:

```bash
uv run pytest tests/core/test_xarray_storage_only.py::test_n_channels_prefers_xarray_channel_dim_size tests/core/test_xarray_storage_only.py::test_spectral_frame_adds_channel_coord_without_frequency_coord tests/core/test_xarray_storage_only.py::test_spectrogram_frame_adds_channel_coord_without_frequency_or_time_coords tests/core/test_xarray_storage_only.py::test_noct_frame_adds_channel_coord_without_band_coord tests/core/test_xarray_storage_only.py::test_channel_coord_omitted_when_metadata_length_differs_for_target_frames -q
```

Expected: PASS.

- [ ] **Step 9: Run focused tests**

Run:

```bash
uv run pytest tests/core/test_xarray_storage_only.py tests/frames/test_channel_label_updates.py tests/frames/test_spectral_frame.py tests/frames/test_spectrogram_frame.py tests/frames/test_noct_frame.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit centralized channel semantics**

```bash
git add wandas/core/base_frame.py wandas/frames/channel.py wandas/frames/spectrogram.py tests/core/test_xarray_storage_only.py
git commit -m "refactor: centralize xarray channel semantics"
```

---

### Task 5: Preserve constructor dimensionality constraints

**Files:**
- Modify: `tests/core/test_xarray_storage_only.py`

- [ ] **Step 1: Add tests that high-dimensional inputs are still rejected**

Add:

```python
def test_constructor_dimension_constraints_remain_unchanged() -> None:
    with pytest.raises(ValueError, match="Invalid data shape for ChannelFrame"):
        ChannelFrame(data=da.ones((1, 2, 3), chunks=(1, 2, 3)), sampling_rate=8.0)

    with pytest.raises(ValueError, match="Data must be 1-dimensional or 2-dimensional"):
        SpectralFrame(
            data=da.ones((1, 2, 3), chunks=(1, 2, 3)) + 0j,
            sampling_rate=8.0,
            n_fft=8,
        )

    with pytest.raises(ValueError, match="Invalid data dimensions"):
        SpectrogramFrame(
            data=da.ones((1, 2, 5, 3), chunks=(1, 2, 5, 3)) + 0j,
            sampling_rate=8.0,
            n_fft=8,
            hop_length=2,
        )

    with pytest.raises(ValueError, match="1D or 2D"):
        NOctFrame(
            data=da.ones((1, 2, 3), chunks=(1, 2, 3)),
            sampling_rate=8.0,
            fmin=20.0,
            fmax=2000.0,
        )
```

If the NOctFrame error message differs, inspect `wandas/frames/noct.py` and adjust only the regex to match the existing public error. Do not change production validation text in this task.

- [ ] **Step 2: Verify constraints test passes**

Run:

```bash
uv run pytest tests/core/test_xarray_storage_only.py::test_constructor_dimension_constraints_remain_unchanged -q
```

Expected: PASS.

- [ ] **Step 3: Commit constructor constraint tests**

```bash
git add tests/core/test_xarray_storage_only.py
git commit -m "test: preserve xarray dimension input constraints"
```

---

### Task 6: Full verification and PR prep

**Files:**
- No production files expected unless verification reveals a real issue.

- [ ] **Step 1: Verify no tests were deleted**

Run:

```bash
git diff --name-status origin/develop...HEAD -- tests
git diff origin/develop...HEAD -- tests | rg "^-def test_" || true
```

Expected: no deleted test files and no removed test functions. Modified tests are acceptable when they are updated for the new contract.

- [ ] **Step 2: Run lint and type checks**

Run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check wandas tests
```

Expected: all pass.

- [ ] **Step 3: Run full test suite**

Run:

```bash
uv run pytest
```

Expected: all tests pass. Record the final pass count for the PR body.

- [ ] **Step 4: Inspect final diff for scope creep**

Run:

```bash
git diff --stat origin/develop...HEAD
git diff origin/develop...HEAD -- wandas/core/base_frame.py wandas/frames/channel.py wandas/frames/spectral.py wandas/frames/spectrogram.py wandas/frames/noct.py tests/core/test_xarray_storage_only.py
```

Confirm the diff does not include:

- time/frequency/band coordinate generation
- operation dispatch changes
- `from_xarray()`
- NetCDF/Zarr
- RoughnessFrame production changes

- [ ] **Step 5: Push branch**

Run:

```bash
git push -u origin feat/xarray-dimension-semantics
```

- [ ] **Step 6: Create PR against `develop`**

Use title:

```text
PR 2: centralize xarray channel dimension semantics
```

PR body should include:

```markdown
## Summary

PR 2 of the xarray migration. This keeps xarray as a storage container but gives target frames semantic suffix dims and centralizes channel count/coord handling through the xarray `channel` dimension.

- add suffix-based xarray dims for ChannelFrame, SpectralFrame, SpectrogramFrame, and NOctFrame
- make `n_channels` prefer `self._xr.sizes["channel"]`
- centralize channel coord creation and refresh in BaseFrame
- keep time/frequency/band coords out of scope
- keep operation execution, metadata ownership, accepted input shapes, from_xarray, and I/O unchanged
- leave RoughnessFrame out of scope

## Verification

- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run ty check wandas tests`
- `uv run pytest` -> record the final pass count from Step 3

## Non-Goals

- no xarray-native operation dispatch
- no time/frequency/band coords
- no from_xarray
- no NetCDF/Zarr
- no accepted input shape expansion
```
