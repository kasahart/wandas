# xarray Storage-Only Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `BaseFrame`'s direct Dask-array storage with `xr.DataArray` storage while keeping Wandas operations, metadata ownership, and lazy execution unchanged.

**Architecture:** `BaseFrame` owns `_xr: xr.DataArray` as the only data container. `_data` becomes a read-only compatibility property returning `_xr.data`, and explicit `_replace_data()` handles the few internal in-place update paths. xarray is used only for data, dims, and coords; Wandas continues to own sampling rate, labels, metadata, operation history, channel metadata, and operation execution.

**Tech Stack:** Python, Dask array, xarray, pytest, ruff, ty.

---

## File Structure

- Modify `pyproject.toml`
  - Add `xarray` runtime dependency.
- Modify `uv.lock`
  - Updated by `uv lock` or dependency sync after adding xarray.
- Modify `wandas/core/base_frame.py`
  - Import xarray.
  - Replace direct `_data` ownership with `_xr`.
  - Add read-only `_data` property.
  - Add `_replace_data()`, `_build_xarray()`, `_xarray_dims()`, `_xarray_coords()`, `to_xarray()`, and `xr`.
  - Keep metadata, operation history, channel metadata, and operation execution unchanged.
- Modify `wandas/frames/channel.py`
  - Replace the production `_data = new_data` in `_finalize_channel_update()` with `_replace_data(new_data)`.
- Create `tests/core/test_xarray_storage_only.py`
  - Focused tests for `_xr`, read-only `_data`, attrs non-ownership, coords, chunking, and lazy execution.
- Modify existing tests that directly assign fake objects to `_data`
  - Replace fake `_data` assignment tests with public behavior or `_replace_data()` tests where the same behavior remains meaningful.

---

### Task 1: Add xarray dependency

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [ ] **Step 1: Add dependency**

In `pyproject.toml`, add xarray to the runtime dependencies:

```toml
"xarray>=2025.6.1",
```

Place it near the existing scientific stack dependencies.

- [ ] **Step 2: Update lockfile**

Run:

```bash
uv lock
```

Expected: `uv.lock` updates successfully and includes xarray if it was not already locked.

- [ ] **Step 3: Commit dependency change**

Run:

```bash
git add pyproject.toml uv.lock
git commit -m "build: add xarray dependency"
```

Expected: commit succeeds.

---

### Task 2: Write RED tests for xarray-backed storage

**Files:**
- Create: `tests/core/test_xarray_storage_only.py`

- [ ] **Step 1: Create failing storage tests**

Create `tests/core/test_xarray_storage_only.py` with:

```python
import pytest
import xarray as xr
from dask import array as da, delayed

from wandas import ChannelFrame
from wandas.core.metadata import ChannelMetadata, FrameMetadata


def test_base_frame_owns_xarray_dataarray() -> None:
    data = da.ones((2, 8), chunks=(2, 4))
    frame = ChannelFrame(
        data=data,
        sampling_rate=4.0,
        label="signal",
        channel_metadata=[
            ChannelMetadata(label="left", unit="Pa"),
            ChannelMetadata(label="right", unit="Pa"),
        ],
    )

    assert isinstance(frame._xr, xr.DataArray)
    assert frame._xr.dims == ("channel", "time")
    assert frame._data is frame._xr.data
    assert frame._data.chunks == ((1, 1), (8,))
    assert list(frame._xr.coords["channel"].values) == ["left", "right"]
    assert "time" not in frame._xr.coords


def test_data_alias_is_read_only() -> None:
    frame = ChannelFrame.from_numpy([1.0, 2.0, 3.0], sampling_rate=3.0)

    with pytest.raises(AttributeError):
        frame._data = da.zeros((1, 3), chunks=(1, -1))  # type: ignore[misc]


def test_replace_data_updates_xarray_container_only() -> None:
    metadata = FrameMetadata({"source": "test"}, source_file="input.wav")
    frame = ChannelFrame.from_numpy(
        [[1.0, 2.0, 3.0]],
        sampling_rate=3.0,
        label="original",
        metadata=metadata,
    )
    original_metadata = frame.metadata
    original_history = frame.operation_history
    replacement = da.full((1, 4), 2.0, chunks=(1, -1))

    frame._replace_data(replacement)

    assert frame._data is frame._xr.data
    assert frame._data.shape == (1, 4)
    assert frame._data.chunks == ((1,), (4,))
    assert frame.metadata is original_metadata
    assert frame.operation_history is original_history
    assert frame.label == "original"
    assert frame.sampling_rate == 3.0


def test_internal_xarray_attrs_do_not_own_wandas_metadata() -> None:
    frame = ChannelFrame.from_numpy(
        [[1.0, 2.0]],
        sampling_rate=2.0,
        label="owned-by-wandas",
        metadata={"gain": 1.5},
    )

    frame._xr.attrs["label"] = "not-authoritative"
    frame._xr.attrs["metadata"] = {"gain": 999}

    assert frame.label == "owned-by-wandas"
    assert frame.metadata["gain"] == 1.5
```

- [ ] **Step 2: Run RED tests**

Run:

```bash
uv run pytest tests/core/test_xarray_storage_only.py -q
```

Expected: FAIL because `BaseFrame` does not yet define `_xr`, `_data` is still assignable, and `_replace_data()` does not exist.

- [ ] **Step 3: Commit RED tests**

Run:

```bash
git add tests/core/test_xarray_storage_only.py
git commit -m "test: define xarray storage-only expectations"
```

Expected: commit succeeds with failing tests documented by the previous command output.

---

### Task 3: Implement BaseFrame xarray storage

**Files:**
- Modify: `wandas/core/base_frame.py`

- [ ] **Step 1: Import xarray and update class annotations**

In `wandas/core/base_frame.py`, add:

```python
import xarray as xr
```

Replace the class annotation:

```python
_data: DaArray
```

with:

```python
_xr: xr.DataArray
```

- [ ] **Step 2: Add helper methods and read-only alias**

Add these methods inside `BaseFrame` after `__init__` and before `_n_channels`:

```python
    @property
    def _data(self) -> DaArray:
        """Compatibility alias for the Dask array stored in ``_xr``."""
        data = self._xr.data
        if not isinstance(data, DaArray):
            raise TypeError(f"Internal xarray data is not a Dask array: {type(data).__name__}")
        return data

    def _replace_data(self, data: DaArray) -> None:
        """Replace the internal xarray data container without touching Wandas metadata."""
        normalized = self._normalize_data(data)
        self._xr = self._build_xarray(normalized)

    def _normalize_data(self, data: DaArray) -> DaArray:
        """Normalize Dask data shape and chunks using Wandas channel-wise policy."""
        try:
            normalized = data.reshape((1, -1)) if data.ndim == 1 else data
            if normalized.ndim >= 2:
                chunks = tuple([1] + [-1] * (normalized.ndim - 1))
            else:
                chunks = tuple([-1] * normalized.ndim)
            return normalized.rechunk(chunks)
        except Exception as e:
            logger.warning(f"Rechunk failed: {e!r}. Falling back to chunks=-1.")
            return data.rechunk(chunks=-1)

    def _build_xarray(self, data: DaArray) -> xr.DataArray:
        """Build the internal xarray container for frame data, dims, and coords."""
        return xr.DataArray(
            data,
            dims=self._xarray_dims(data),
            coords=self._xarray_coords(data),
            name=self.label,
        )

    def _xarray_dims(self, data: DaArray) -> tuple[str, ...]:
        """Return neutral dimension names for the internal xarray container."""
        return tuple(f"dim_{i}" for i in range(data.ndim))

    def _xarray_coords(self, data: DaArray) -> dict[str, Any]:
        """Return conservative base coordinates for the internal xarray container."""
        return {}
```

`ChannelFrame` may then provide the only Phase 1 semantic override:

```python
    def _xarray_dims(self, data: DaArray) -> tuple[str, ...]:
        """Return ChannelFrame dimension names for the internal xarray container."""
        return ("channel", "time")

    def _xarray_coords(self, data: DaArray) -> dict[str, Any]:
        """Return cheap ChannelFrame coordinates owned by the frame."""
        labels = [ch.label for ch in self._channel_metadata]
        if len(labels) != self._channel_count_from_data(data):
            return {}
        return {"channel": labels}
```

- [ ] **Step 3: Refactor `__init__` to create `_xr`**

In `BaseFrame.__init__`, remove the existing try/except block that assigns `self._data`.
After `self._channel_metadata` is initialized, add:

```python
        normalized_data = self._normalize_data(data)
        self._xr = self._build_xarray(normalized_data)
```

The order matters: channel metadata must exist before `_build_xarray()` creates channel coords.

Move the Dask graph logging after `_xr` is assigned and keep it using `self._data`.

- [ ] **Step 4: Run storage tests**

Run:

```bash
uv run pytest tests/core/test_xarray_storage_only.py -q
```

Expected: PASS for the storage tests, except failures that point to subclass-specific coord hooks. Fix only minimal hook behavior needed for `ChannelFrame`.

- [ ] **Step 5: Commit BaseFrame storage implementation**

Run:

```bash
git add wandas/core/base_frame.py tests/core/test_xarray_storage_only.py
git commit -m "feat: store frame data in xarray containers"
```

Expected: commit succeeds.

---

### Task 4: Replace production `_data` assignment

**Files:**
- Modify: `wandas/frames/channel.py`
- Test: `tests/core/test_xarray_storage_only.py`

- [ ] **Step 1: Update in-place channel data replacement**

In `wandas/frames/channel.py`, change `_finalize_channel_update()` from:

```python
        if inplace:
            self._data = new_data
            self._channel_metadata = new_chmeta
            return self
```

to:

```python
        if inplace:
            self._channel_metadata = new_chmeta
            self._replace_data(new_data)
            return self
```

Channel metadata must be updated before `_replace_data()` so the rebuilt xarray channel coord uses the new labels.

- [ ] **Step 2: Add lazy in-place tests**

Append to `tests/core/test_xarray_storage_only.py`:

```python
def test_add_channel_inplace_updates_xarray_without_compute() -> None:
    calls: list[str] = []

    def build() -> list[list[float]]:
        calls.append("computed")
        return [[1.0, 2.0, 3.0]]

    lazy_data = da.from_delayed(
        delayed(build)(),
        shape=(1, 3),
        dtype=float,
    )
    frame = ChannelFrame(data=lazy_data, sampling_rate=3.0)

    result = frame.add_channel([4.0, 5.0, 6.0], label="extra", inplace=True)

    assert result is frame
    assert calls == []
    assert frame._data is frame._xr.data
    assert frame._data.shape == (2, 3)
    assert list(frame._xr.coords["channel"].values) == ["ch0", "extra"]


def test_remove_channel_inplace_updates_xarray_without_compute() -> None:
    calls: list[str] = []

    def build() -> list[list[float]]:
        calls.append("computed")
        return [[1.0, 2.0], [3.0, 4.0]]

    lazy_data = da.from_delayed(
        delayed(build)(),
        shape=(2, 2),
        dtype=float,
    )
    frame = ChannelFrame(
        data=lazy_data,
        sampling_rate=2.0,
        channel_metadata=[
            ChannelMetadata(label="left"),
            ChannelMetadata(label="right"),
        ],
    )

    result = frame.remove_channel("left", inplace=True)

    assert result is frame
    assert calls == []
    assert frame._data is frame._xr.data
    assert frame._data.shape == (1, 2)
    assert list(frame._xr.coords["channel"].values) == ["right"]
```

- [ ] **Step 3: Run targeted tests**

Run:

```bash
uv run pytest tests/core/test_xarray_storage_only.py -q
```

Expected: PASS.

- [ ] **Step 4: Search for production `_data` assignments**

Run:

```bash
rg -n "self\\._data\\s*=" wandas
```

Expected: no production matches. If a match remains, replace it with `_replace_data()` only when it is an in-place data replacement. Do not add a `_data` setter.

- [ ] **Step 5: Commit production assignment removal**

Run:

```bash
git add wandas/frames/channel.py tests/core/test_xarray_storage_only.py
git commit -m "refactor: replace internal data assignment"
```

Expected: commit succeeds.

---

### Task 5: Add public xarray export API

**Files:**
- Modify: `wandas/core/base_frame.py`
- Test: `tests/core/test_xarray_storage_only.py`

- [ ] **Step 1: Write RED export tests**

Append to `tests/core/test_xarray_storage_only.py`:

```python
def test_to_xarray_returns_public_shallow_copy_with_export_attrs() -> None:
    frame = ChannelFrame.from_numpy(
        [[1.0, 2.0]],
        sampling_rate=2.0,
        label="exported",
        metadata={"source": "unit-test"},
    )
    frame.operation_history.append({"operation": "normalize", "params": {}})

    exported = frame.to_xarray()

    assert isinstance(exported, xr.DataArray)
    assert exported is not frame._xr
    assert exported.data is frame._data
    assert exported.attrs["wandas_frame_type"] == "ChannelFrame"
    assert exported.attrs["sampling_rate"] == 2.0
    assert exported.attrs["label"] == "exported"
    assert exported.attrs["metadata"] == {"source": "unit-test"}
    assert exported.attrs["operation_history"] == [{"operation": "normalize", "params": {}}]

    exported.attrs["label"] = "changed-export"
    assert frame.label == "exported"


def test_xr_property_matches_to_xarray_contract() -> None:
    frame = ChannelFrame.from_numpy([1.0, 2.0], sampling_rate=2.0)

    exported = frame.xr

    assert isinstance(exported, xr.DataArray)
    assert exported is not frame._xr
    assert exported.data is frame._data
```

- [ ] **Step 2: Run RED export tests**

Run:

```bash
uv run pytest tests/core/test_xarray_storage_only.py::test_to_xarray_returns_public_shallow_copy_with_export_attrs tests/core/test_xarray_storage_only.py::test_xr_property_matches_to_xarray_contract -q
```

Expected: FAIL because `to_xarray()` and `xr` are not implemented.

- [ ] **Step 3: Implement export API**

Add to `BaseFrame` near `compute()`:

```python
    def to_xarray(self) -> xr.DataArray:
        """Return a public xarray view of this frame without changing Wandas ownership."""
        exported = self._xr.copy(deep=False)
        exported.attrs = {
            "wandas_frame_type": type(self).__name__,
            "sampling_rate": self.sampling_rate,
            "label": self.label,
            "metadata": dict(self.metadata),
            "operation_history": copy.deepcopy(self.operation_history),
        }
        return exported

    @property
    def xr(self) -> xr.DataArray:
        """Return a public xarray view of this frame."""
        return self.to_xarray()
```

- [ ] **Step 4: Run export tests**

Run:

```bash
uv run pytest tests/core/test_xarray_storage_only.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit export API**

Run:

```bash
git add wandas/core/base_frame.py tests/core/test_xarray_storage_only.py
git commit -m "feat: expose xarray frame views"
```

Expected: commit succeeds.

---

### Task 6: Preserve lazy execution through common workflows

**Files:**
- Test: `tests/core/test_xarray_storage_only.py`

- [ ] **Step 1: Add lazy execution regression tests**

Append to `tests/core/test_xarray_storage_only.py`:

```python

def _lazy_frame_with_counter(calls: list[str]) -> ChannelFrame:
    def build() -> list[list[float]]:
        calls.append("computed")
        return [[1.0, 2.0, 3.0, 4.0]]

    lazy_data = da.from_delayed(delayed(build)(), shape=(1, 4), dtype=float)
    return ChannelFrame(data=lazy_data, sampling_rate=4.0)


def test_construction_and_xarray_export_do_not_compute() -> None:
    calls: list[str] = []
    frame = _lazy_frame_with_counter(calls)

    _ = frame._data
    _ = frame.to_xarray()
    _ = frame.xr

    assert calls == []


def test_selection_and_operation_do_not_compute_until_compute() -> None:
    calls: list[str] = []
    frame = _lazy_frame_with_counter(calls)

    selected = frame.get_channel(0)
    normalized = selected.normalize()

    assert calls == []

    result = normalized.compute()

    assert calls == ["computed"]
    assert result.shape == (1, 4)


def test_transform_methods_remain_lazy() -> None:
    calls: list[str] = []
    frame = _lazy_frame_with_counter(calls)

    spectrum = frame.fft()
    spectrogram = frame.stft(nperseg=4, noverlap=2)

    assert calls == []
    assert spectrum._data is spectrum._xr.data
    assert spectrogram._data is spectrogram._xr.data
```

- [ ] **Step 2: Run lazy tests**

Run:

```bash
uv run pytest tests/core/test_xarray_storage_only.py -q
```

Expected: PASS. If a test computes early, fix the xarray construction/export path. Do not modify operation dispatch.

- [ ] **Step 3: Commit lazy tests**

Run:

```bash
git add tests/core/test_xarray_storage_only.py
git commit -m "test: verify xarray storage stays lazy"
```

Expected: commit succeeds.

---

### Task 7: Update tests that assign `_data`

**Files:**
- Modify: `tests/core/test_base_frame.py`
- Modify: `tests/core/test_base_frame_additional.py`
- Modify other test files only if `rg` finds direct `_data` assignments.

- [ ] **Step 1: Find direct `_data` assignments in tests**

Run:

```bash
rg -n "\\._data\\s*=" tests
```

Expected: a short list of tests that mutate internals.

- [ ] **Step 2: Replace assignment-based tests**

For tests that assign fake objects only to force graph visualization errors, replace direct `_data` mutation with monkeypatching the property target or with a real Dask object.

Example replacement for a visualization failure test:

```python
def test_visualize_graph_handles_visualize_failure(monkeypatch):
    frame = make_frame()

    def fail_visualize(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(frame._data, "visualize", fail_visualize)

    assert frame.visualize_graph() is None
```

For tests that need to replace frame data, use:

```python
frame._replace_data(new_dask_data)
```

Do not add a `_data` setter for tests.

- [ ] **Step 3: Run updated test files**

Run:

```bash
uv run pytest tests/core/test_base_frame.py tests/core/test_base_frame_additional.py tests/core/test_xarray_storage_only.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit test updates**

Run:

```bash
git add tests/core/test_base_frame.py tests/core/test_base_frame_additional.py tests/core/test_xarray_storage_only.py
git commit -m "test: stop assigning frame data internals"
```

Expected: commit succeeds.

---

### Task 8: Full verification and complexity review

**Files:**
- Modify: `docs/superpowers/specs/2026-06-09-xarray-storage-only-design.md` only if implementation reveals a needed clarification.

- [ ] **Step 1: Run focused xarray storage tests**

Run:

```bash
uv run pytest tests/core/test_xarray_storage_only.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run full test suite**

Run:

```bash
uv run pytest -q
```

Expected: all existing tests pass.

- [ ] **Step 3: Run lint and type checks**

Run:

```bash
uv run ruff check wandas tests
uv run ty check wandas tests
```

Expected: both commands pass.

- [ ] **Step 4: Check implementation guardrails**

Run:

```bash
rg -n "process_xarray|process_dataarray|apply_ufunc|from_xarray|to_netcdf|open_netcdf|to_zarr|open_zarr" wandas tests
rg -n "self\\._data\\s*=" wandas
```

Expected:

```text
No operation-dispatch or storage-I/O symbols were added.
No production self._data assignment remains.
```

- [ ] **Step 5: Inspect code size and ownership**

Run:

```bash
git diff --stat develop...HEAD
git diff -- wandas/core/base_frame.py wandas/frames/channel.py tests/core/test_xarray_storage_only.py
```

Expected: changes are concentrated in `BaseFrame`, the single ChannelFrame in-place update path, and focused tests.

- [ ] **Step 6: Write PR complexity summary**

Use this text as the PR body summary and update it with final numbers from verification:

```markdown
## Summary

This PR restarts the xarray migration as Phase 1 only: xarray is used as an axis-aware data container, not as metadata storage, operation execution, or persistence.

Implemented:
- Add xarray dependency.
- Store frame data in `BaseFrame._xr: xr.DataArray`.
- Keep `_data` as a read-only compatibility alias to `_xr.data`.
- Replace production `_data` assignment with explicit `_replace_data()`.
- Add `frame.to_xarray()` and `frame.xr` as public export views.
- Preserve existing operation execution through `operation.process(self._data)`.
- Add lazy execution tests to ensure xarray wrapping does not call `compute()`.

Explicitly not included:
- No attrs-backed metadata.
- No xarray operation dispatch.
- No `from_xarray()`.
- No NetCDF or Zarr.
- No chunk-policy integration changes.

## Complexity Review

What became simpler:
- `BaseFrame` has one data owner: `_xr`.
- Production `_data` assignment is removed.
- Operation paths remain unchanged.

What became more complex:
- `BaseFrame` now has small xarray construction/export helpers.
- Coordinate generation introduces a new hook surface.

Permanent pieces:
- `_xr` as internal data container.
- Read-only `_data` compatibility property.
- `to_xarray()` / `.xr`.

Candidates for later cleanup:
- Existing operation call sites can continue using `_data` until Phase 3.
- Coordinate hooks may move to frame-specific schema helpers if they grow.

## Verification

- `uv run pytest -q`: passes
- `uv run ruff check wandas tests`: passes
- `uv run ty check wandas tests`: passes
```

- [ ] **Step 7: Commit any final documentation clarification**

If the design doc needed updates, run:

```bash
git add docs/superpowers/specs/2026-06-09-xarray-storage-only-design.md
git commit -m "docs: clarify xarray storage scope"
```

Expected: no commit is needed unless the implementation forced a real design clarification.
