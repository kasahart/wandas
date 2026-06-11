# Xarray-Backed Frame State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move frame-level state from standalone `BaseFrame` attributes and `FrameMetadata` into `_xr.attrs`, while keeping public property access and leaving `ChannelMetadata` unchanged.

**Architecture:** `BaseFrame._xr` remains the internal data container and becomes the backing store for `sampling_rate`, `label`, `metadata`, and `operation_history`. Wandas properties validate and interpret those attrs; xarray only stores them. `FrameMetadata` is removed instead of bridged for compatibility.

**Tech Stack:** Python 3.10, xarray `DataArray.attrs`, Dask arrays, pytest, ruff, ty, h5py/WDF tests.

---

## File Structure

- Modify: `wandas/core/base_frame.py`
  - Add property-backed frame state over `_xr.attrs`.
  - Reorder initialization so `_xr` exists before frame-level setters run.
  - Preserve attrs in `_replace_data()`.
  - Deep-copy attrs in `to_xarray()`.
  - Remove `FrameMetadata` imports and type paths.
- Modify: `wandas/core/metadata.py`
  - Remove `FrameMetadata`; keep `ChannelMetadata` unchanged.
- Modify: `wandas/core/__init__.py`
  - Stop exporting `FrameMetadata`.
- Modify: `wandas/frames/channel.py`, `wandas/frames/spectral.py`, `wandas/frames/spectrogram.py`, `wandas/frames/mixins/channel_transform_mixin.py`, `wandas/frames/mixins/channel_processing_mixin.py`, `wandas/frames/mixins/protocols.py`
  - Replace `FrameMetadata` annotations and `.merged()` calls with `dict[str, Any]` and plain dict merging.
  - Change WAV source metadata to `{"_source_file": source}`.
- Modify: `wandas/io/wdf_io.py`
  - Treat metadata as plain dict.
  - Store and load `_source_file` as a normal metadata key; read legacy `meta.attrs["source_file"]` into `_source_file` for old WDF files.
- Modify tests:
  - `tests/core/test_xarray_storage_only.py`
  - `tests/core/test_metadata.py`
  - `tests/core/test_base_frame.py`
  - `tests/core/test_base_frame_additional.py`
  - `tests/io/test_wdf_io.py`
  - `tests/io/test_wav_io.py`
  - `tests/frames/test_channel_additional.py`
  - `tests/frames/test_channel_processing.py`
  - any other tests that still import `FrameMetadata` or assert `.source_file`.

---

### Task 1: Add Failing Tests for Attr-Backed Frame State

**Files:**
- Modify: `tests/core/test_xarray_storage_only.py`

- [ ] **Step 1: Replace FrameMetadata import in the xarray storage test file**

Change the import near the top of `tests/core/test_xarray_storage_only.py` from:

```python
from wandas.core.metadata import ChannelMetadata, FrameMetadata
```

to:

```python
from wandas.core.metadata import ChannelMetadata
```

- [ ] **Step 2: Replace `test_replace_data_updates_xarray_container_only` with an attrs preservation test**

Replace the existing function with:

```python
def test_replace_data_preserves_xarray_attrs_backed_frame_state() -> None:
    frame = ChannelFrame.from_numpy(
        np.array([[1.0, 2.0, 3.0]]),
        sampling_rate=3.0,
        label="original",
        metadata={"source": "test", "nested": {"x": 1}},
    )
    frame.operation_history = [{"operation": "load", "params": {"path": "input.wav"}}]
    replacement = da.full((1, 4), 2.0, chunks=(1, -1))

    frame._replace_data(replacement)

    assert frame._data is frame._xr.data
    assert frame._data.shape == (1, 4)
    assert frame.sampling_rate == 3.0
    assert frame.label == "original"
    assert frame._xr.attrs["sampling_rate"] == 3.0
    assert frame._xr.attrs["label"] == "original"
    assert frame.metadata == {"source": "test", "nested": {"x": 1}}
    assert frame.operation_history == [{"operation": "load", "params": {"path": "input.wav"}}]
```

- [ ] **Step 3: Replace `test_to_xarray_deep_copies_exported_frame_metadata`**

Replace the function with:

```python
def test_to_xarray_deep_copies_exported_metadata_dict() -> None:
    frame = ChannelFrame.from_numpy(
        np.array([[1.0, 2.0, 3.0]]),
        sampling_rate=3.0,
        metadata={"nested": {"x": 1}, "_source_file": "input.wav"},
    )

    exported = frame.to_xarray()
    exported.attrs["metadata"]["nested"]["x"] = 99
    exported.attrs["metadata"]["_source_file"] = "changed.wav"

    assert type(frame.metadata) is dict
    assert frame.metadata["nested"]["x"] == 1
    assert frame.metadata["_source_file"] == "input.wav"
```

- [ ] **Step 4: Add direct attrs-backed property tests**

Add these tests after the existing `to_xarray` tests:

```python
def test_frame_state_properties_are_backed_by_xarray_attrs() -> None:
    frame = ChannelFrame.from_numpy(
        np.array([[1.0, 2.0, 3.0]]),
        sampling_rate=3.0,
        label="stateful",
        metadata={"owner": "attrs"},
        operation_history=[{"operation": "load", "params": {}}],
    )

    assert frame._xr.attrs["sampling_rate"] == 3.0
    assert frame._xr.attrs["label"] == "stateful"
    assert frame._xr.attrs["metadata"] == {"owner": "attrs"}
    assert frame._xr.attrs["operation_history"] == [{"operation": "load", "params": {}}]

    frame._xr.attrs["sampling_rate"] = 8.0
    frame._xr.attrs["label"] = "from-attrs"
    frame._xr.attrs["metadata"] = {"owner": "mutated"}
    frame._xr.attrs["operation_history"] = [{"operation": "mutated", "params": {"x": 1}}]

    assert frame.sampling_rate == 8.0
    assert frame.label == "from-attrs"
    assert frame.metadata == {"owner": "mutated"}
    assert frame.operation_history == [{"operation": "mutated", "params": {"x": 1}}]


def test_frame_state_property_setters_update_xarray_attrs() -> None:
    frame = ChannelFrame.from_numpy(np.array([[1.0, 2.0, 3.0]]), sampling_rate=3.0)

    frame.sampling_rate = 6
    frame.label = "updated"
    frame.metadata = {"nested": {"x": 1}}
    frame.operation_history = [{"operation": "gain", "params": {"factor": 2.0}}]

    assert frame._xr.attrs["sampling_rate"] == 6.0
    assert frame._xr.attrs["label"] == "updated"
    assert frame._xr.name == "updated"
    assert frame._xr.attrs["metadata"] == {"nested": {"x": 1}}
    assert frame._xr.attrs["operation_history"] == [{"operation": "gain", "params": {"factor": 2.0}}]


def test_frame_state_property_setters_validate_inputs() -> None:
    frame = ChannelFrame.from_numpy(np.array([[1.0, 2.0, 3.0]]), sampling_rate=3.0)

    with pytest.raises(ValueError, match="Invalid sampling_rate"):
        frame.sampling_rate = 0

    with pytest.raises(TypeError, match="Label must be a string or None"):
        frame.label = 123  # type: ignore[assignment]

    with pytest.raises(TypeError, match="Metadata must be a dictionary"):
        frame.metadata = "invalid"  # type: ignore[assignment]

    with pytest.raises(TypeError, match="Operation history must be a list"):
        frame.operation_history = {"operation": "bad"}  # type: ignore[assignment]
```

- [ ] **Step 5: Run tests and verify they fail for the right reason**

Run:

```bash
uv run pytest tests/core/test_xarray_storage_only.py -q
```

Expected: fails because `FrameMetadata` is still imported/used in production or because state is still stored on plain attributes rather than `_xr.attrs`.

- [ ] **Step 6: Commit failing tests**

```bash
git add tests/core/test_xarray_storage_only.py
git commit -m "test: define attrs-backed frame state"
```

---

### Task 2: Implement Attr-Backed State in BaseFrame

**Files:**
- Modify: `wandas/core/base_frame.py`
- Test: `tests/core/test_xarray_storage_only.py`

- [ ] **Step 1: Update imports and annotations**

In `wandas/core/base_frame.py`, replace:

```python
from wandas.utils.types import NDArrayComplex, NDArrayReal

from .metadata import ChannelMetadata, FrameMetadata
```

with:

```python
from wandas.utils import validate_sampling_rate
from wandas.utils.types import NDArrayComplex, NDArrayReal

from .metadata import ChannelMetadata
```

Update class annotations from:

```python
sampling_rate: float
label: str
metadata: FrameMetadata
operation_history: list[dict[str, Any]]
```

to a comment-free removal of those storage annotations. The public properties added below provide the typed interface. Keep `_xr`, `_channel_metadata`, and `_previous` annotations.

- [ ] **Step 2: Reorder `__init__` to build `_xr` before property setters**

Replace the beginning of `BaseFrame.__init__` through `_previous` setup with this structure:

```python
        normalized_data = self._normalize_data(data)
        frame_label = label or "unnamed_frame"

        if channel_metadata:
            def _to_channel_metadata(ch: ChannelMetadata | dict[str, Any], index: int) -> ChannelMetadata:
                if isinstance(ch, ChannelMetadata):
                    return copy.deepcopy(ch)
                if isinstance(ch, dict):
                    try:
                        return ChannelMetadata(**ch)
                    except ValidationError as e:
                        raise ValueError(
                            f"Invalid channel_metadata at index {index}\n"
                            f"  Got: {ch}\n"
                            f"  Validation error: {e}\n"
                            f"Ensure all dict keys match ChannelMetadata fields "
                            f"(label, unit, ref, extra) and have correct types."
                        ) from e
                raise TypeError(
                    f"Invalid type in channel_metadata at index {index}\n"
                    f"  Got: {type(ch).__name__} ({ch!r})\n"
                    f"  Expected: ChannelMetadata or dict\n"
                    f"Use ChannelMetadata objects or dicts with valid fields."
                )

            self._channel_metadata = [
                _to_channel_metadata(cast(ChannelMetadata | dict[str, Any], ch), i)
                for i, ch in enumerate(channel_metadata)
            ]
        else:
            channel_count = self._channel_size_from_xarray_dims(normalized_data)
            if channel_count is None:
                channel_count = self._channel_count_from_data(normalized_data)
            self._channel_metadata = [ChannelMetadata(label=f"ch{i}", unit="", extra={}) for i in range(channel_count)]

        self._xr = self._build_xarray(normalized_data, name=frame_label)
        self.label = label
        self.sampling_rate = sampling_rate
        self.metadata = metadata
        self.operation_history = operation_history
        self._previous = previous
```

- [ ] **Step 3: Change `_build_xarray()` and `_replace_data()`**

Replace `_replace_data()` and `_build_xarray()` with:

```python
    def _replace_data(self, data: DaArray) -> None:
        """Replace the internal xarray data container without touching frame state."""
        normalized = self._normalize_data(data)
        attrs = copy.deepcopy(self._xr.attrs)
        self._xr = self._build_xarray(normalized, name=self.label)
        self._xr.attrs = attrs

    def _build_xarray(self, data: DaArray, *, name: str) -> xr.DataArray:
        """Build the internal xarray container for frame data, dims, and coords."""
        return xr.DataArray(
            data,
            dims=self._xarray_dims(data),
            coords=self._xarray_coords(data),
            name=name,
        )
```

- [ ] **Step 4: Add frame state properties after `previous` property**

Add:

```python
    @property
    def sampling_rate(self) -> float:
        """Return the frame sampling rate from xarray attrs."""
        return float(self._xr.attrs["sampling_rate"])

    @sampling_rate.setter
    def sampling_rate(self, value: float) -> None:
        validate_sampling_rate(value)
        self._xr.attrs["sampling_rate"] = float(value)

    @property
    def label(self) -> str:
        """Return the frame label from xarray attrs."""
        value = self._xr.attrs.get("label", self._xr.name)
        if value is None or value == "":
            return "unnamed_frame"
        return str(value)

    @label.setter
    def label(self, value: str | None) -> None:
        if value is not None and not isinstance(value, str):
            raise TypeError("Label must be a string or None")
        label = value or "unnamed_frame"
        self._xr.attrs["label"] = label
        self._xr.name = label

    @property
    def metadata(self) -> dict[str, Any]:
        """Return mutable frame metadata stored in xarray attrs."""
        value = self._xr.attrs.get("metadata")
        if value is None:
            value = {}
            self._xr.attrs["metadata"] = value
        if not isinstance(value, dict):
            raise TypeError(f"Internal metadata attrs must be a dictionary, got {type(value).__name__}")
        return value

    @metadata.setter
    def metadata(self, value: dict[str, Any] | None) -> None:
        if value is None:
            self._xr.attrs["metadata"] = {}
            return
        if not isinstance(value, dict):
            raise TypeError("Metadata must be a dictionary")
        self._xr.attrs["metadata"] = dict(value)

    @property
    def operation_history(self) -> list[dict[str, Any]]:
        """Return mutable operation history stored in xarray attrs."""
        value = self._xr.attrs.get("operation_history")
        if value is None:
            value = []
            self._xr.attrs["operation_history"] = value
        if not isinstance(value, list):
            raise TypeError(f"Internal operation_history attrs must be a list, got {type(value).__name__}")
        return value

    @operation_history.setter
    def operation_history(self, value: list[dict[str, Any]] | None) -> None:
        if value is None:
            self._xr.attrs["operation_history"] = []
            return
        if not isinstance(value, list):
            raise TypeError("Operation history must be a list")
        self._xr.attrs["operation_history"] = copy.deepcopy(value)
```

- [ ] **Step 5: Update `to_xarray()`**

Replace the method body with:

```python
        exported = self._xr.copy(deep=False)
        exported.attrs = copy.deepcopy(self._xr.attrs)
        exported.attrs["wandas_frame_type"] = type(self).__name__
        return exported
```

- [ ] **Step 6: Update `_create_new_instance()` metadata validation**

Replace:

```python
        metadata = kwargs.pop("metadata", copy.deepcopy(self.metadata))
        if not isinstance(metadata, (dict, FrameMetadata)):
            raise TypeError("Metadata must be a dictionary or FrameMetadata")
```

with:

```python
        metadata = kwargs.pop("metadata", copy.deepcopy(self.metadata))
        if not isinstance(metadata, dict):
            raise TypeError("Metadata must be a dictionary")

        operation_history = kwargs.pop("operation_history", copy.deepcopy(self.operation_history))
        if not isinstance(operation_history, list):
            raise TypeError("Operation history must be a list")
```

And pass `operation_history=operation_history` into `type(self)(...)`.

- [ ] **Step 7: Update `_binary_op()` and `_updated_metadata_and_history()` types**

In `_binary_op()`, replace:

```python
        metadata: FrameMetadata = self.metadata.copy() if self.metadata is not None else FrameMetadata()
        operation_history: list[dict[str, Any]] = self.operation_history.copy() if self.operation_history else []
```

with:

```python
        metadata = copy.deepcopy(self.metadata)
        operation_history = copy.deepcopy(self.operation_history)
```

Change `_updated_metadata_and_history()` return annotation and docstring from `FrameMetadata` to `dict[str, Any]`, and use:

```python
        new_metadata = copy.deepcopy(self.metadata)
```

- [ ] **Step 8: Run targeted tests**

Run:

```bash
uv run pytest tests/core/test_xarray_storage_only.py -q
```

Expected: the new attr-backed tests pass, but other repository tests may still fail because `FrameMetadata` remains in other modules.

- [ ] **Step 9: Commit BaseFrame implementation**

```bash
git add wandas/core/base_frame.py tests/core/test_xarray_storage_only.py
git commit -m "feat: back frame state with xarray attrs"
```

---

### Task 3: Remove FrameMetadata From Production Imports and Merges

**Files:**
- Modify: `wandas/core/metadata.py`
- Modify: `wandas/core/__init__.py`
- Modify: `wandas/frames/channel.py`
- Modify: `wandas/frames/spectral.py`
- Modify: `wandas/frames/spectrogram.py`
- Modify: `wandas/frames/mixins/channel_transform_mixin.py`
- Modify: `wandas/frames/mixins/channel_processing_mixin.py`
- Modify: `wandas/frames/mixins/protocols.py`

- [ ] **Step 1: Remove FrameMetadata class**

In `wandas/core/metadata.py`, delete the `FrameMetadata` class and the unused import alias:

```python
import copy as _copy
```

The file should start with:

```python
from typing import Any

from pydantic import BaseModel, Field

from wandas.utils.util import unit_to_ref
```

Keep `ChannelMetadata` unchanged.

- [ ] **Step 2: Stop exporting FrameMetadata**

Replace `wandas/core/__init__.py` with:

```python
from .base_frame import BaseFrame

__all__ = [
    "BaseFrame",
]
```

- [ ] **Step 3: Update frame constructor annotations/imports**

In `wandas/frames/channel.py`, replace:

```python
from ..core.metadata import ChannelMetadata, FrameMetadata
```

with:

```python
from ..core.metadata import ChannelMetadata
```

Replace each constructor or factory annotation of:

```python
metadata: "FrameMetadata | dict[str, Any] | None" = None
```

with:

```python
metadata: dict[str, Any] | None = None
```

Do the same replacement in `wandas/frames/spectral.py` and `wandas/frames/spectrogram.py`.

- [ ] **Step 4: Update WAV source metadata**

In `ChannelFrame.from_file()`, replace:

```python
            metadata=FrameMetadata(source_file=source_file),
```

with:

```python
            metadata={"_source_file": source_file} if source_file is not None else None,
```

- [ ] **Step 5: Replace `.merged()` in transform mixin**

In `wandas/frames/mixins/channel_transform_mixin.py`, replace each occurrence:

```python
metadata=self.metadata.merged(**params),
```

with:

```python
metadata={**self.metadata, **params},
```

Replace:

```python
metadata=self.metadata.merged(window=window, n_fft=_n_fft),
```

with:

```python
metadata={**self.metadata, "window": window, "n_fft": _n_fft},
```

- [ ] **Step 6: Replace `.merged()` in processing mixin**

In `wandas/frames/mixins/channel_processing_mixin.py`, replace:

```python
        new_metadata = self.metadata.merged(**params)
```

with:

```python
        new_metadata = {**self.metadata, **params}
```

- [ ] **Step 7: Update protocols**

In `wandas/frames/mixins/protocols.py`, replace:

```python
from wandas.core.metadata import ChannelMetadata, FrameMetadata
```

with:

```python
from wandas.core.metadata import ChannelMetadata
```

Replace protocol metadata annotation:

```python
    metadata: FrameMetadata
```

with:

```python
    metadata: dict[str, Any]
```

Replace `_updated_metadata_and_history()` return annotation:

```python
    ) -> tuple[FrameMetadata, list[dict[str, Any]]]: ...
```

with:

```python
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]: ...
```

- [ ] **Step 8: Search for production FrameMetadata references**

Run:

```bash
rg -n "FrameMetadata|metadata\.merged|metadata\.source_file" wandas -g '*.py'
```

Expected: no output.

- [ ] **Step 9: Run focused tests**

Run:

```bash
uv run pytest tests/core/test_xarray_storage_only.py tests/frames/test_channel_transform.py tests/frames/test_channel_processing.py -q
```

Expected: passes or only fails in tests that still import `FrameMetadata`, which Task 5 handles. Production import errors must be fixed in this task before committing.

- [ ] **Step 10: Commit production cleanup**

```bash
git add wandas/core/metadata.py wandas/core/__init__.py wandas/frames/channel.py wandas/frames/spectral.py wandas/frames/spectrogram.py wandas/frames/mixins/channel_transform_mixin.py wandas/frames/mixins/channel_processing_mixin.py wandas/frames/mixins/protocols.py
git commit -m "refactor: remove FrameMetadata from production code"
```

---

### Task 4: Update WDF and Reader Source Metadata

**Files:**
- Modify: `wandas/io/wdf_io.py`
- Modify: `tests/io/test_wdf_io.py`
- Modify: `tests/io/test_wav_io.py`
- Modify: `tests/frames/test_channel_additional.py`

- [ ] **Step 1: Update WDF source-file tests first**

In `tests/io/test_wdf_io.py`, replace `test_source_file_roundtrip()` with:

```python
def test_source_file_roundtrip(tmp_path: Path) -> None:
    """Source file metadata survives a save/load round-trip as a plain dict key."""
    source = "audio/input.wav"
    data = np.array([[1, 2, 3]], dtype=np.float32)
    sr = 16000
    cf = ChannelFrame.from_numpy(
        data,
        sr,
        metadata={"key": "val", "_source_file": source},
    )
    assert type(cf.metadata) is dict
    assert cf.metadata["_source_file"] == source

    path = tmp_path / "test_source_file.wdf"
    cf.to_wdf(path)
    cf2 = ChannelFrame.from_wdf(path)

    assert type(cf2.metadata) is dict
    assert cf2.metadata["_source_file"] == source
    assert cf2.metadata["key"] == "val"
```

Replace `test_source_file_none_roundtrip()` with:

```python
def test_source_file_absent_roundtrip(tmp_path: Path) -> None:
    """Round-trip without source file keeps metadata as a plain dict."""
    data = np.array([[1, 2, 3]], dtype=np.float32)
    sr = 16000
    cf = ChannelFrame.from_numpy(data, sr, metadata={"only": "dict"})
    assert type(cf.metadata) is dict
    assert "_source_file" not in cf.metadata

    path = tmp_path / "test_no_source_file.wdf"
    cf.to_wdf(path)
    cf2 = ChannelFrame.from_wdf(path)

    assert type(cf2.metadata) is dict
    assert cf2.metadata == {"only": "dict"}
```

- [ ] **Step 2: Update WAV/URL source tests**

In `tests/io/test_wav_io.py`, replace:

```python
    assert cf.metadata.source_file == url, "source_file metadata not preserved"
```

with:

```python
    assert cf.metadata["_source_file"] == url, "source file metadata not preserved"
```

In `tests/frames/test_channel_additional.py`, replace:

```python
    assert cf.metadata.source_file == "my_file.wav"
```

with:

```python
    assert cf.metadata["_source_file"] == "my_file.wav"
```

- [ ] **Step 3: Run source metadata tests and verify failure**

Run:

```bash
uv run pytest tests/io/test_wdf_io.py::test_source_file_roundtrip tests/io/test_wdf_io.py::test_source_file_absent_roundtrip tests/io/test_wav_io.py::test_read_wav_from_url_sets_source_file tests/frames/test_channel_additional.py::test_read_wav_source_file_metadata -q
```

Expected: fails until WDF and reader implementation are updated. If exact test names differ, use `rg -n "source_file|_source_file" tests/io tests/frames/test_channel_additional.py` and run the matching node ids.

- [ ] **Step 4: Update WDF writer metadata block**

In `wandas/io/wdf_io.py`, remove `FrameMetadata` import. Replace the frame metadata block with:

```python
        if frame.metadata:
            meta_grp = f.create_group("meta")
            meta_grp.attrs["json"] = json.dumps(dict(frame.metadata))

            for k, v in frame.metadata.items():
                if isinstance(v, (str, int, float, bool, np.number)):
                    meta_grp.attrs[k] = v
```

- [ ] **Step 5: Update WDF reader metadata block**

Replace:

```python
        frame_metadata = FrameMetadata()
        if "meta" in f:
            meta_json = f["meta"].attrs.get("json", "{}")
            if isinstance(meta_json, (bytes, np.bytes_)):
                meta_json = _decode_hdf5_str(meta_json)
            frame_metadata.update(json.loads(meta_json))
            source_file = f["meta"].attrs.get("source_file", None)
            if source_file is not None:
                frame_metadata.source_file = _decode_hdf5_str(source_file)
```

with:

```python
        frame_metadata: dict[str, Any] = {}
        if "meta" in f:
            meta_json = f["meta"].attrs.get("json", "{}")
            if isinstance(meta_json, (bytes, np.bytes_)):
                meta_json = _decode_hdf5_str(meta_json)
            frame_metadata.update(json.loads(meta_json))
            legacy_source_file = f["meta"].attrs.get("source_file", None)
            if legacy_source_file is not None and "_source_file" not in frame_metadata:
                frame_metadata["_source_file"] = _decode_hdf5_str(legacy_source_file)
```

Ensure `Any` is imported in `wdf_io.py`; if not present, add:

```python
from typing import Any
```

- [ ] **Step 6: Run WDF/WAV source metadata tests**

Run:

```bash
uv run pytest tests/io/test_wdf_io.py tests/io/test_wav_io.py tests/frames/test_channel_additional.py -q
```

Expected: passes for source metadata behavior.

- [ ] **Step 7: Commit I/O changes**

```bash
git add wandas/io/wdf_io.py tests/io/test_wdf_io.py tests/io/test_wav_io.py tests/frames/test_channel_additional.py
git commit -m "refactor: store source file in metadata dict"
```

---

### Task 5: Update Metadata Tests and Remaining FrameMetadata Test Usage

**Files:**
- Modify: `tests/core/test_metadata.py`
- Modify: `tests/core/test_base_frame.py`
- Modify: `tests/core/test_base_frame_additional.py`
- Modify: `tests/frames/test_channel_processing.py`
- Modify: `tests/core/test_xarray_storage_only.py`
- Any test found by `rg -n "FrameMetadata|metadata\.source_file|metadata\.merged" tests -g '*.py'`

- [ ] **Step 1: Remove FrameMetadata tests from metadata test file**

In `tests/core/test_metadata.py`, remove `FrameMetadata` from the import:

```python
from wandas.core.metadata import ChannelMetadata
```

Delete the entire `TestFrameMetadata` class. Keep all `ChannelMetadata` tests unchanged.

- [ ] **Step 2: Update channel processing metadata preservation test**

In `tests/frames/test_channel_processing.py`, remove `FrameMetadata` import. Replace the test setup around the metadata preservation test with:

```python
        metadata = {"source": "test", "_source_file": "input.wav"}
        operation_history = [{"operation": "normalize", "params": {"method": "peak"}}]
```

Replace assertions:

```python
        assert result.metadata.source_file == "input.wav"
        assert frame.metadata.source_file == "input.wav"
```

with:

```python
        assert result.metadata["_source_file"] == "input.wav"
        assert frame.metadata["_source_file"] == "input.wav"
```

Replace any test assignment:

```python
        self.channel_frame.metadata = FrameMetadata({"foo": "bar"})
```

with:

```python
        self.channel_frame.metadata = {"foo": "bar"}
```

- [ ] **Step 3: Update invalid metadata error messages**

In tests that assert the old message:

```python
"Metadata must be a dictionary or FrameMetadata"
```

replace it with:

```python
"Metadata must be a dictionary"
```

- [ ] **Step 4: Search tests for stale FrameMetadata usage**

Run:

```bash
rg -n "FrameMetadata|metadata\.source_file|metadata\.merged" tests -g '*.py'
```

Expected: no output. If output remains, replace `FrameMetadata(...)` with plain dict construction, replace `metadata.source_file` assertions with `metadata["_source_file"]`, and replace `metadata.merged(**params)` with `{**metadata, **params}`.

- [ ] **Step 5: Run metadata-focused tests**

Run:

```bash
uv run pytest tests/core/test_metadata.py tests/core/test_base_frame.py tests/core/test_base_frame_additional.py tests/frames/test_channel_processing.py tests/core/test_xarray_storage_only.py -q
```

Expected: passes.

- [ ] **Step 6: Commit test cleanup**

```bash
git add tests/core/test_metadata.py tests/core/test_base_frame.py tests/core/test_base_frame_additional.py tests/frames/test_channel_processing.py tests/core/test_xarray_storage_only.py
git commit -m "test: remove FrameMetadata expectations"
```

---

### Task 6: Repository-Wide Cleanup and Verification

**Files:**
- Modify any remaining files found by search.
- No new production abstraction unless a search shows repeated unreadable code.

- [ ] **Step 1: Search production and tests for removed API**

Run:

```bash
rg -n "FrameMetadata|metadata\.source_file|metadata\.merged" wandas tests -g '*.py'
```

Expected: no output.

- [ ] **Step 2: Search docs for stale public API examples**

Run:

```bash
rg -n "FrameMetadata|metadata\.source_file|metadata\.merged" docs README.md -g '*.md'
```

Expected: no output in user-facing docs, or only historical planning/design docs under `docs/superpowers/`. If user-facing docs mention old API, update them to plain dict metadata and `_source_file`.

- [ ] **Step 3: Run type and lint checks**

Run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check wandas tests
```

Expected: all pass.

- [ ] **Step 4: Run full test suite**

Run:

```bash
uv run pytest
```

Expected: full suite passes. Record the measured pass/skip count for the PR body.

- [ ] **Step 5: Update or add short ADR if implementation changed the spec materially**

If implementation exactly follows the spec, no new ADR is required. If implementation chooses a materially different behavior, add a short ADR under `docs/design/` explaining the final behavior.

- [ ] **Step 6: Commit final cleanup if any**

If Step 1-5 changed files, commit them:

```bash
git status --short
git add docs README.md wandas tests
git commit -m "chore: finalize xarray-backed frame state"
```

If no files changed, do not create an empty commit.

---

### Task 7: PR Preparation

**Files:**
- No code files unless PR body template is stored in repo.

- [ ] **Step 1: Inspect final diff**

Run:

```bash
git status --short
git diff --stat origin/develop...HEAD
git log --oneline origin/develop..HEAD
```

Expected: worktree clean; diff shows Phase 3 only.

- [ ] **Step 2: Push branch**

Run:

```bash
git push -u origin feat/xarray-frame-state
```

Expected: branch pushed. If Git prints `fatal: could not read Username` but also prints a `To https://github.com/...` update line, treat the push as successful.

- [ ] **Step 3: Create PR body**

Use this body, replacing the test count with the measured result from Task 6:

```markdown
## Summary

Phase 3 of the xarray migration. This moves frame-level state onto `_xr.attrs` and removes `FrameMetadata` instead of keeping a compatibility adapter.

- store `sampling_rate`, `label`, `metadata`, and `operation_history` in `_xr.attrs`
- keep public property access for frame state
- make `frame.metadata` a plain `dict[str, Any]`
- move source file tracking to `metadata["_source_file"]`
- remove `FrameMetadata`, `.merged()`, and `.source_file` production paths
- keep `ChannelMetadata` unchanged
- keep operation execution, generated coords, from_xarray, NetCDF, and Zarr out of scope

## Compatibility Notes

- This intentionally removes the `FrameMetadata` public API.
- Code using `frame.metadata.source_file` must use `frame.metadata["_source_file"]`.
- WDF loading still maps legacy `meta.attrs["source_file"]` to `metadata["_source_file"]` when present.

## Verification

- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run ty check wandas tests`
- `uv run pytest` -> use the measured pass/skip count printed by Task 6 Step 4, for example `1463 passed, 3 skipped`

## Non-Goals

- no ChannelMetadata migration
- no unit/ref coords
- no generated time/frequency/band/bark coords
- no xarray-native operation dispatch
- no from_xarray
- no NetCDF/Zarr
```

- [ ] **Step 4: Create PR**

Create a PR against `develop` titled:

```text
Phase 3: back frame state with xarray attrs
```

- [ ] **Step 5: Report final state**

Report:

- PR URL
- latest commit SHA
- verification commands and measured results
- any intentional breaking changes
