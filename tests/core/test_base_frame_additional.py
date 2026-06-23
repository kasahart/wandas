from typing import Any, cast

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from wandas.core.base_frame import BaseFrame, _constructor_accepts_kwarg, _mutable_config_value
from wandas.core.metadata import ChannelMetadata
from wandas.frames.channel import ChannelFrame
from wandas.processing.base import AudioOperation
from wandas.utils.dask_helpers import da_from_array


class DummyFrame(BaseFrame[np.ndarray]):
    def __init__(self, data, sampling_rate: float = 1.0, **kwargs):
        super().__init__(data, sampling_rate, **kwargs)

    @property
    def _n_channels(self) -> int:
        return int(self._data.shape[0])

    def plot(self, plot_type: str = "default", ax=None, **kwargs):
        raise NotImplementedError

    def _get_additional_init_kwargs(self) -> dict:
        return {}

    def _binary_op(self, other, op, symbol):
        # return a shallow copy for testing
        return self._create_new_instance(data=self._data)

    def _apply_operation_impl(self, operation_name: str, **params):
        new_history = [*self.operation_history, {"operation": operation_name, **params}]
        return self._create_new_instance(data=self._data, operation_history=new_history)

    def _get_dataframe_index(self) -> pd.Index:
        # index should be length of samples
        length = self._data.shape[-1]
        return pd.RangeIndex(stop=length)

    def _debug_info_impl(self) -> None:
        return None


class LegacyFrame(BaseFrame[np.ndarray]):
    def __init__(
        self,
        data,
        sampling_rate: float = 1.0,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
        operation_history: list[dict[str, Any]] | None = None,
        channel_metadata=None,
        channel_ids: list[str] | None = None,
        source_time_offset=0.0,
        previous=None,
    ):
        super().__init__(
            data=data,
            sampling_rate=sampling_rate,
            label=label,
            metadata=metadata,
            operation_history=operation_history,
            channel_metadata=channel_metadata,
            channel_ids=channel_ids,
            source_time_offset=source_time_offset,
            previous=previous,
        )

    @property
    def _n_channels(self) -> int:
        return int(self._data.shape[0])

    def plot(self, plot_type: str = "default", ax=None, **kwargs):
        raise NotImplementedError

    def _get_additional_init_kwargs(self) -> dict:
        return {}

    def _binary_op(self, other, op, symbol):
        return self._create_new_instance(data=self._data)

    def _apply_operation_impl(self, operation_name: str, **params):
        new_history = [*self.operation_history, {"operation": operation_name, **params}]
        return self._create_new_instance(data=self._data, operation_history=new_history)

    def _get_dataframe_index(self) -> pd.Index:
        return pd.RangeIndex(stop=self._data.shape[-1])

    def _debug_info_impl(self) -> None:
        return None


def make_frame(arr: np.ndarray | da.Array, **kwargs) -> DummyFrame:
    if isinstance(arr, np.ndarray):
        darr = da_from_array(arr, chunks=arr.shape)
    else:
        darr = arr
    return DummyFrame(darr, sampling_rate=100.0, **kwargs)


def test_rechunk_failure_fallback_logs_warning(caplog):
    """Test rechunk failure during init logs warning and falls back."""
    arr = da_from_array(np.arange(6).reshape(2, 3), chunks=(2, 3))
    original = arr.rechunk
    state = {"called": False}

    def bad_rechunk(chunks=None, **kwargs):
        if not state["called"]:
            state["called"] = True
            raise RuntimeError("boom")
        return original(chunks, **kwargs)

    arr.rechunk = bad_rechunk  # ty: ignore[invalid-assignment]

    with caplog.at_level("WARNING"):
        f = make_frame(arr)
    assert "Rechunk failed" in caplog.text
    assert hasattr(f, "_data")


def test_get_channel_query_no_match_raises_key_error():
    """Test get_channel with unmatched query raises KeyError."""
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    with pytest.raises(KeyError, match=r"No channels match query"):
        f.get_channel(query="nope")


def test_get_channel_query_unknown_key_raises_key_error():
    """Test get_channel with unknown dict key raises KeyError."""
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    with pytest.raises(KeyError, match=r"Unknown channel metadata key\(s\): unknown"):
        f.get_channel(query={"unknown": "x"})


def test_get_channel_query_regex_and_callable_returns_matches():
    """Test get_channel with regex and callable queries returns matches."""
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    import re as _re

    # regex query returns matching channel with Dask preserved
    res = f.get_channel(query=_re.compile(r"ch0"))
    assert len(res) == 1
    assert res.labels == ["ch0"]
    assert isinstance(res._data, da.Array)

    # callable query returns matching channel with Dask preserved
    res2 = f.get_channel(query=lambda ch: ch.label == "ch1")
    assert len(res2) == 1
    assert res2.labels == ["ch1"]
    assert isinstance(res2._data, da.Array)


def test_get_channel_dict_query_regex_value_returns_matches():
    """Test get_channel with regex dict query returns matching channels."""
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    import re as _re

    res = f.get_channel(query={"label": _re.compile(r"ch")})
    assert len(res) == 2
    assert isinstance(res._data, da.Array)
    assert res is not f


def test_getitem_boolean_mask_wrong_length_raises_value_error():
    """Test boolean mask with wrong length raises ValueError."""
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    mask = np.array([True])
    with pytest.raises(ValueError, match=r"Boolean mask length 1 does not match number of channels 2"):
        _ = f[mask]


def test_getitem_numpy_float_array_raises_type_error():
    """Test numpy float array indexing raises TypeError."""
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    a = np.array([0.1, 1.0])
    with pytest.raises(TypeError, match=r"NumPy array must be of integer or boolean type"):
        _ = f[a]


def test_getitem_empty_list_raises_value_error():
    """Test empty list indexing raises ValueError."""
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    with pytest.raises(ValueError, match=r"Cannot index with an empty list"):
        _ = f[[]]  # empty list index triggers ValueError


def test_getitem_mixed_list_types_raises_type_error():
    """Test mixed int/str list indexing raises TypeError."""
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    with pytest.raises(TypeError, match=r"List must contain all str or all int"):
        _ = f[[0, "ch0"]]  # ty: ignore[invalid-argument-type]


def test_multidim_indexing_invalid_key_length_raises_value_error():
    """Test tuple indexing with wrong length raises ValueError."""
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    with pytest.raises(ValueError, match=r"Invalid key length"):
        _ = f[(0, slice(None), slice(None))]


def test_label2index_nonexistent_label_raises_key_error():
    """Test label2index with nonexistent label raises KeyError."""
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    with pytest.raises(KeyError, match=r"Channel label 'nope' not found"):
        f.label2index("nope")


def test_shape_single_channel_returns_1d():
    """Test shape property for single channel returns 1D shape."""
    arr = np.arange(3)
    f = make_frame(arr)
    assert f.shape == (3,)


def test_compute_non_ndarray_result_raises_value_error():
    """Test compute raising ValueError when result is not ndarray."""
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(type(f._data), "compute", lambda self: [1, 2, 3])
        with pytest.raises(ValueError, match=r"Computed result is not a np.ndarray"):
            f.compute()


def test_visualize_graph_failure_logs_warning_returns_none(caplog):
    """Test visualize_graph logs warning and returns None on failure."""
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            type(f._data),
            "visualize",
            lambda self, filename=None: (_ for _ in ()).throw(RuntimeError("no graphviz")),
        )
        with caplog.at_level("WARNING"):
            res = f.visualize_graph()
    assert res is None
    assert "Failed to visualize the graph" in caplog.text


def test_to_tensor_unsupported_framework_raises_value_error():
    """Test to_tensor raises ValueError for unsupported framework."""
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    with pytest.raises(ValueError, match=r"Unsupported framework"):
        f.to_tensor(framework="mxnet")


def test_to_tensor_torch_not_installed_raises_import_error(monkeypatch):
    """Test to_tensor raises ImportError when torch is not installed."""
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)

    def raise_missing_torch(key, *, feature):
        assert key == "torch"
        raise ImportError(f'{feature} requires optional dependency {key!r}.\nInstall it with: pip install "wandas[ml]"')

    monkeypatch.setattr("wandas.core.base_frame.require_dependency", raise_missing_torch)
    with pytest.raises(ImportError, match=r'pip install "wandas\[ml\]"'):
        f.to_tensor(framework="torch")


def test_to_tensor_tensorflow_not_installed_raises_import_error(monkeypatch):
    """Test to_tensor raises ImportError when tensorflow is not installed."""
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)

    def raise_missing_tensorflow(key, *, feature):
        assert key == "tensorflow"
        raise ImportError(f'{feature} requires optional dependency {key!r}.\nInstall it with: pip install "wandas[ml]"')

    monkeypatch.setattr("wandas.core.base_frame.require_dependency", raise_missing_tensorflow)
    with pytest.raises(ImportError, match=r'pip install "wandas\[ml\]"'):
        f.to_tensor(framework="tensorflow")


def test_to_dataframe_single_channel_correct_shape():
    """Test to_dataframe for single channel returns correct shape."""
    arr = np.arange(3)
    f = make_frame(arr)
    df = f.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 1)


def test_to_dataframe_multi_channel_correct_shape():
    """Test to_dataframe for multi channel returns correct shape."""
    arr = np.arange(12).reshape(3, 4)
    f = make_frame(arr)
    df = f.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (4, 3)


def test_array_protocol_dtype_conversion_preserves_type():
    """Test __array__ with dtype converts to requested type."""
    arr = np.arange(6).reshape(2, 3).astype(np.float64)
    f = make_frame(arr)
    a = f.__array__(dtype=np.float32)
    assert a.dtype == np.float32


def test_print_operation_history_empty_shows_empty_label(capsys):
    """Test print_operation_history shows '<empty>' for empty history."""
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    f.operation_history = []
    f.print_operation_history()
    out = capsys.readouterr().out
    assert "Operation history: <empty>" in out


def test_print_operation_history_populated_shows_indexed_entries(capsys):
    """Test print_operation_history shows indexed entries for populated history."""
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    f.operation_history = [{"operation": "normalize"}, {"name": "filter", "cutoff": 1000}]
    f.print_operation_history()
    out = capsys.readouterr().out
    assert "Operation history (2):" in out
    assert "1: normalize {}" in out
    assert "2: filter {'cutoff': 1000}" in out


def test_relabel_channels_adds_prefix_to_labels():
    """Test _relabel_channels adds operation prefix to all channel labels."""
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    new_meta = f._relabel_channels("norm")
    assert all(m.label.startswith("norm(") for m in new_meta)


def test_create_new_instance_invalid_label_raises_type_error():
    """Test _create_new_instance raises TypeError for non-string label."""
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    with pytest.raises(TypeError, match=r"Label must be a string"):
        f._create_new_instance(data=f._data, label=123)


def test_create_new_instance_invalid_metadata_raises_type_error():
    """Test _create_new_instance raises TypeError for non-dict metadata."""
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    with pytest.raises(TypeError, match=r"Metadata must be a dictionary"):
        f._create_new_instance(data=f._data, metadata=123)


def test_create_new_instance_invalid_channel_metadata_raises_type_error():
    """Test _create_new_instance raises TypeError for non-list channel_metadata."""
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    with pytest.raises(TypeError, match=r"Channel metadata must be a list"):
        f._create_new_instance(data=f._data, channel_metadata=123)


def test_persist_returns_persisted_data():
    """Test persist() calls persist on internal data."""
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    persisted = da_from_array(np.zeros((2, 3)), chunks=(2, 3))
    calls = {"count": 0}

    def fake_persist(self):
        calls["count"] += 1
        return persisted

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(type(f._data), "persist", fake_persist)
        newf = f.persist()

    assert calls["count"] == 1
    assert newf._data is newf._xr.data
    np.testing.assert_array_equal(newf.compute(), np.zeros((2, 3)))


def test_channel_metadata_invalid_dict_raises_value_error():
    """Test invalid dict value in channel_metadata raises ValueError."""
    arr = np.arange(6).reshape(2, 3)
    with pytest.raises(ValueError, match=r"Invalid channel_metadata at index 0"):
        make_frame(arr, channel_metadata=[{"label": "x", "ref": "bad"}])


def test_channel_metadata_invalid_type_raises_type_error():
    """Test invalid type in channel_metadata list raises TypeError."""
    arr = np.arange(6).reshape(2, 3)
    with pytest.raises(TypeError, match=r"Invalid type in channel_metadata at index 0"):
        make_frame(arr, channel_metadata=[123])


def test_get_channel_query_no_validate_unknown_key_raises_no_match():
    """Test get_channel with validate_query_keys=False and unknown key raises no match."""
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    with pytest.raises(KeyError, match=r"No channels match query"):
        f.get_channel(query={"unknown": "x"}, validate_query_keys=False)


def test_get_channel_query_extra_key_match_returns_channel():
    """Test get_channel with extra key query returns matching channel."""
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(
        arr,
        channel_metadata=[
            {"label": "a", "extra": {"foo": "bar"}},
            {"label": "b", "extra": {"foo": "baz"}},
        ],
    )
    original_data = f._data.compute().copy()
    res = f.get_channel(query={"foo": "bar"})
    assert len(res) == 1
    assert res.labels == ["a"]
    assert isinstance(res._data, da.Array)
    # Pillar 1: Immutability
    assert res is not f
    np.testing.assert_array_equal(f._data.compute(), original_data)


def test_len_returns_channel_count():
    """Test len() returns number of channels."""
    arr = np.arange(12).reshape(3, 4)
    f = make_frame(arr)
    assert len(f) == 3


def test_iter_yields_single_channel_dask_frames():
    """Test iterating yields single-channel Dask-backed frames."""
    arr = np.arange(12).reshape(3, 4)
    f = make_frame(arr)
    original_data = f._data.compute().copy()

    items = list(iter(f))
    assert len(items) == 3
    for i, chf in enumerate(items):
        assert chf is not f
        assert chf.n_channels == 1
        assert chf.labels == [f"ch{i}"]
        assert isinstance(chf._data, da.Array)

    # Pillar 1: Iteration does not mutate original
    np.testing.assert_array_equal(f._data.compute(), original_data)


def test_debug_info_logs_debug_output(caplog):
    """Test debug_info logs expected debug output."""
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    with caplog.at_level("DEBUG"):
        f.debug_info()
    assert "Debug Info" in caplog.text or f"=== {f.__class__.__name__} Debug Info ===" in caplog.text


def test_print_operation_history_empty_shows_none(capsys):
    """Test _print_operation_history shows 'None' for empty history."""
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    f.operation_history = []
    f._print_operation_history()
    out = capsys.readouterr().out
    assert "Operations Applied: None" in out


def test_print_operation_history_populated_shows_count(capsys):
    """Test _print_operation_history shows correct count for populated history."""
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    f.operation_history = [{"operation": "a"}, {"operation": "b"}]
    f._print_operation_history()
    out = capsys.readouterr().out
    assert "Operations Applied: 2" in out


def test_to_tensor_torch_and_tensorflow_fake_modules_succeed(monkeypatch):
    """Test to_tensor succeeds with fake torch and tensorflow modules."""
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)

    import sys

    # Fake torch (module-like)
    class FakeDevice:
        def __init__(self) -> None:
            self.type: str = "cpu"
            self.index: int | None = None

    class FakeTorchTensor:
        def __init__(self, arr):
            self._arr = arr
            self.device = FakeDevice()

        def to(self, device):
            # device may be 'cpu', 'cuda', 'cuda:0' etc.
            if isinstance(device, str) and device.startswith("cuda"):
                self.device.type = "cuda"
                if ":" in device:
                    self.device.index = int(device.split(":", 1)[1])
                else:
                    self.device.index = None
            else:
                self.device.type = "cpu"
                self.device.index = None
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

        @property
        def shape(self):
            return self._arr.shape

    class FakeTorchModule:
        def __init__(self):
            class Cuda:
                @staticmethod
                def is_available():
                    return False

            self.cuda = Cuda()

        @staticmethod
        def from_numpy(x):
            return FakeTorchTensor(x)

    monkeypatch.setitem(sys.modules, "torch", FakeTorchModule())

    t = f.to_tensor(framework="torch", device="cpu")
    assert hasattr(t, "numpy")
    assert t.device.type == "cpu"

    # Fake tensorflow (module-like)
    class FakeTFModule:
        def __init__(self):
            self.__spec__ = object()

            class Config:
                @staticmethod
                def list_physical_devices(kind):
                    return []

            self.config = Config()

        class _DeviceCtx:
            def __init__(self, device):
                self.device = device

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        def device(self, device):
            return FakeTFModule._DeviceCtx(device)

        @staticmethod
        def convert_to_tensor(x):
            class T:
                def __init__(self, arr):
                    self._arr = arr
                    self.shape = arr.shape

                def numpy(self):
                    return self._arr

            return T(x)

    monkeypatch.setitem(sys.modules, "tensorflow", FakeTFModule())
    t2 = f.to_tensor(framework="tensorflow", device="/CPU:0")
    assert hasattr(t2, "numpy")
    assert t2.shape == f.to_numpy().shape


class SingleChannelAxislessFrame(BaseFrame[np.ndarray]):
    _channel_axis = None

    def plot(self, plot_type: str = "default", ax=None, **kwargs):
        raise NotImplementedError

    def _get_dataframe_index(self) -> pd.Index:
        return pd.RangeIndex(stop=self._data.shape[-1])

    def _debug_info_impl(self) -> None:
        return None


def test_axisless_frame_stores_source_time_offset_in_attrs() -> None:
    data = da_from_array(np.arange(4, dtype=float), chunks=(4,))
    frame = SingleChannelAxislessFrame(data, sampling_rate=100.0, source_time_offset=2.5)

    xr_data = frame.to_xarray()

    assert "source_time_offset" not in xr_data.coords
    np.testing.assert_array_equal(xr_data.attrs["source_time_offset"], np.array([2.5]))
    np.testing.assert_array_equal(frame.source_time_offset, np.array([2.5]))


def test_data_property_rejects_non_dask_xarray_storage():
    f = make_frame(np.arange(6).reshape(2, 3))
    f._xr = xr.DataArray(np.arange(6).reshape(2, 3), dims=("dim_0", "dim_1"))

    with pytest.raises(TypeError, match="Internal xarray data is not a Dask array"):
        _ = f._data


def test_replace_data_preserves_existing_attrs():
    f = make_frame(np.arange(6).reshape(2, 3), metadata={"source": "old"}, label="original")
    replacement = da_from_array(np.arange(8).reshape(2, 4), chunks=(1, -1))

    f._replace_data(replacement)

    assert f.label == "original"
    assert f.metadata == {"source": "old"}
    assert f._data.shape == (2, 4)


def test_axisless_frame_counts_as_one_channel():
    data = da_from_array(np.arange(4), chunks=(4,))

    f = SingleChannelAxislessFrame(data, sampling_rate=100.0)

    assert f.n_channels == 1
    assert f.labels == ["ch0"]


def test_xarray_coords_without_pending_metadata_are_conservative():
    f = make_frame(np.arange(6).reshape(2, 3))

    assert f._xarray_coords(f._data) == {}


def test_channel_id_validation_rejects_wrong_length_and_duplicates():
    data = da_from_array(np.arange(6).reshape(2, 3), chunks=(1, -1))

    with pytest.raises(ValueError, match="length must match"):
        DummyFrame(data, sampling_rate=100.0, channel_ids=["a"])
    with pytest.raises(ValueError, match="must be unique"):
        DummyFrame(data, sampling_rate=100.0, channel_ids=["a", "a"])


def test_channel_metadata_short_lists_are_padded_and_long_lists_rejected():
    data = da_from_array(np.arange(6).reshape(2, 3), chunks=(1, -1))

    f = DummyFrame(data, sampling_rate=100.0, channel_metadata=[{"label": "only"}])
    assert f.labels == ["only", "ch1"]

    with pytest.raises(ValueError, match="must not exceed number of channels"):
        DummyFrame(
            data,
            sampling_rate=100.0,
            channel_metadata=[{"label": "a"}, {"label": "b"}, {"label": "c"}],
        )


def test_channel_metadata_accepts_view_like_objects():
    source = make_frame(np.arange(6).reshape(2, 3))

    f = make_frame(np.arange(6).reshape(2, 3), channel_metadata=[source.channels[0], source.channels[1]])

    assert f.channels.to_list() == source.channels.to_list()


def test_attrs_backed_channel_metadata_mutation_and_refresh():
    f = make_frame(np.arange(6).reshape(2, 3))

    f.channels[0].label = "left"
    f.channels[0].unit = "V"
    f._channel_metadata = [ChannelMetadata(label="front"), ChannelMetadata(label="rear")]
    f._refresh_xarray_channel_coord()

    assert f._xr.attrs["channel_label"] == ["front", "rear"]
    assert f._xr.attrs["channel_unit"] == ["", ""]


def test_empty_channel_ids_are_defaulted_when_setting_metadata():
    f = make_frame(np.arange(6).reshape(2, 3))
    f._xr.attrs["channel_ids"] = []

    f._set_channel_metadata([ChannelMetadata(label="a"), ChannelMetadata(label="b")])

    assert f._channel_ids == ["c0", "c1"]


def test_label_metadata_and_operation_history_attrs_validation():
    f = make_frame(np.arange(6).reshape(2, 3))

    f.label = None
    assert f.label == "unnamed_frame"
    f._xr.attrs["label"] = ""
    assert f.label == "unnamed_frame"
    with pytest.raises(TypeError, match="Label must be a string or None"):
        f.label = cast(Any, 123)

    f.metadata = None
    assert f.metadata == {}
    f._xr.attrs["metadata"] = "bad"
    with pytest.raises(TypeError, match="Internal metadata attrs must be a dictionary"):
        _ = f.metadata
    with pytest.raises(TypeError, match="Metadata must be a dictionary"):
        f.metadata = cast(Any, "bad")

    f._xr.attrs["operation_history"] = None
    assert f.operation_history == []
    f._xr.attrs["operation_history"] = "bad"
    with pytest.raises(TypeError, match="Internal operation_history attrs must be a list"):
        _ = f.operation_history
    with pytest.raises(TypeError, match="Operation history must be a list"):
        f.operation_history = cast(Any, "bad")


def test_create_new_instance_validates_operation_history_and_channel_ids():
    f = make_frame(np.arange(6).reshape(2, 3))

    with pytest.raises(TypeError, match="Operation history must be a list"):
        f._create_new_instance(data=f._data, operation_history="bad")
    with pytest.raises(TypeError, match="Operations must be a tuple"):
        f._create_new_instance(data=f._data, operations=["bad"])
    with pytest.raises(TypeError, match="Operations must be a tuple"):
        make_frame(np.arange(6).reshape(2, 3), operations=["bad"])
    with pytest.raises(TypeError, match="Channel ids must be a list"):
        f._create_new_instance(data=f._data, channel_ids=("a", "b"))


def test_create_new_instance_omits_operations_for_legacy_constructor():
    data = da_from_array(np.arange(6).reshape(2, 3).astype(float), chunks=(2, 3))
    frame = LegacyFrame(
        data,
        sampling_rate=100.0,
        operation_history=[{"operation": "fake"}],
    )

    result = frame._create_new_instance(
        data=frame._data,
        operation_history=[*frame.operation_history],
        operations=(AudioOperation(100.0),),
    )

    assert isinstance(result, LegacyFrame)
    assert result.previous is frame
    assert result.operation_history == [{"operation": "fake"}]
    assert result.operations == ()


def test_constructor_accepts_kwarg_falls_back_when_signature_unavailable(monkeypatch):
    def raise_value_error(_):
        raise ValueError("signature unavailable")

    monkeypatch.setattr("wandas.core.base_frame.inspect.signature", raise_value_error)

    assert _constructor_accepts_kwarg(LegacyFrame, "operations")


def test_binary_operations_cover_scalar_helpers_and_frame_mismatch_errors():
    f = ChannelFrame(da_from_array(np.arange(6).reshape(2, 3).astype(float), chunks=(1, -1)), sampling_rate=100.0)

    assert (f - 1).operation_history[-1] == {"operation": "-", "with": "1"}
    assert (f * 2).operation_history[-1] == {"operation": "*", "with": "2"}
    assert (f / 2).operation_history[-1] == {"operation": "/", "with": "2"}
    assert BaseFrame._format_operand_str(1 + 2j) == "complex(1.0, 2.0)"
    assert BaseFrame._format_operand_str(object()) == "object"

    one_channel = ChannelFrame(
        da_from_array(np.arange(3).reshape(1, 3).astype(float), chunks=(1, -1)),
        sampling_rate=100.0,
    )
    with pytest.raises(ValueError, match="Channel count mismatch"):
        _ = f + one_channel

    different_shape = ChannelFrame(
        da_from_array(np.arange(8).reshape(2, 4).astype(float), chunks=(1, -1)),
        sampling_rate=100.0,
    )
    with pytest.raises(ValueError, match="Frame shape mismatch"):
        _ = f + different_shape


def test_binary_frame_operation_merges_left_and_right_operation_lineage():
    left = ChannelFrame(da_from_array(np.array([[1.0, 2.0, 4.0]]), chunks=(1, -1)), sampling_rate=100.0)
    right = ChannelFrame(da_from_array(np.array([[1.0, 1.0, 1.0]]), chunks=(1, -1)), sampling_rate=100.0)

    result = left.normalize() + right.remove_dc()

    assert [operation.name for operation in result.operations] == ["normalize", "remove_dc"]


def test_apply_operation_helpers_update_metadata_and_history(monkeypatch):
    f = make_frame(np.arange(6).reshape(2, 3).astype(float))
    dependency_calls = 0

    class FakeOperation:
        name = "fake"
        params = {"amount": 2}

        def ensure_dependencies(self):
            nonlocal dependency_calls
            dependency_calls += 1

        def process(self, data):
            return data + 1

        def get_metadata_updates(self):
            return {"sampling_rate": 200.0}

        def get_display_name(self):
            return "Fake"

    result = f._apply_operation_instance(FakeOperation())

    assert result.sampling_rate == 200.0
    assert result.metadata["fake"] == {"amount": 2}
    assert result.operation_history[-1] == {"operation": "fake", "params": {"amount": 2}}
    assert result.labels == ["Fake(ch0)", "Fake(ch1)"]
    assert dependency_calls == 1

    class CreatedOperation(FakeOperation):
        name = "created"
        params = {"gain": 3}

    monkeypatch.setattr("wandas.processing.create_operation", lambda name, sampling_rate, **params: CreatedOperation())
    applied = BaseFrame._apply_operation_impl(f, "created", gain=3)

    assert applied.metadata["created"] == {"gain": 3}
    assert dependency_calls == 2

    class TrimOperation(FakeOperation):
        name = "trim"
        params = {"start": 0.01}

        def get_metadata_updates(self):
            return {}

    trimmed = f._apply_operation_instance(TrimOperation())
    assert trimmed.source_time_offset[0] == pytest.approx(0.01)

    class CreatedAudioOperation(AudioOperation[np.ndarray, np.ndarray]):
        name = "created_audio"

        def ensure_dependencies(self):
            nonlocal dependency_calls
            dependency_calls += 1

        def _process_array(self, x: np.ndarray) -> np.ndarray:
            return x

    monkeypatch.setattr(
        "wandas.processing.create_operation",
        lambda name, sampling_rate, **params: CreatedAudioOperation(sampling_rate),
    )
    applied_with_operation = BaseFrame._apply_operation_impl(f, "created_audio")

    assert len(applied_with_operation.operations) == 1
    assert applied_with_operation.operations[0].name == "created_audio"
    assert dependency_calls == 4


def test_mutable_config_value_converts_containers_for_history():
    source_array = np.array([1.0, 2.0])
    converted = _mutable_config_value(
        {
            "array": source_array,
            "tuple": (source_array,),
            "frozenset": frozenset({1, 2}),
        }
    )

    assert converted["array"] == [1.0, 2.0]
    assert converted["tuple"] == [[1.0, 2.0]]
    assert sorted(converted["frozenset"]) == [1, 2]


def test_frame_operations_returns_live_operation_with_defensive_params():
    data = np.array([[1.0, 2.0, 4.0]])
    frame = ChannelFrame(da_from_array(data, chunks=(1, -1)), sampling_rate=100.0)

    result = frame.normalize()

    assert isinstance(result.operations, tuple)
    assert len(result.operations) == 1
    assert isinstance(result.operations[0], AudioOperation)
    op = result.operations[0]
    assert op.name == "normalize"

    params = op.params
    params["norm"] = 2
    assert op.params["norm"] == 2
    assert op.norm == np.inf

    op.norm = None
    assert op.norm == np.inf

    np.testing.assert_allclose(result.compute(), data / 4.0)


def test_channel_metadata_update_paths_preserve_operations_lineage():
    data = np.array([[1.0, 2.0, 4.0]])
    frame = ChannelFrame(da_from_array(data, chunks=(1, -1)), sampling_rate=100.0)

    result = frame.normalize().rename_channels({0: "renamed"})

    assert isinstance(result.operations, tuple)
    assert [operation.name for operation in result.operations] == ["normalize"]
    np.testing.assert_allclose(result.compute(), data / 4.0)


def test_apply_operation_instance_output_frame_validation_and_constructor_errors():
    f = make_frame(np.arange(6).reshape(2, 3).astype(float))

    class FakeOperation:
        name = "fake"
        params = {}

        def process(self, data):
            return data

        def get_metadata_updates(self):
            return {}

        def get_display_name(self):
            return "fake"

    with pytest.raises(TypeError, match="Invalid output_frame_class"):
        f._apply_operation_instance(FakeOperation(), output_frame_class=cast(Any, object))

    class NeedsExtra(DummyFrame):
        def __init__(self, data, sampling_rate: float = 1.0, *, required, **kwargs):
            super().__init__(data, sampling_rate, **kwargs)
            self.required = required

    with pytest.raises(TypeError, match="Invalid output_frame_class constructor"):
        f._apply_operation_instance(FakeOperation(), output_frame_class=NeedsExtra)

    transitioned = f._apply_operation_instance(
        FakeOperation(),
        operation_name="renamed",
        output_frame_class=NeedsExtra,
        output_frame_kwargs={"required": "ok"},
    )

    assert isinstance(transitioned, NeedsExtra)
    assert transitioned.required == "ok"
    assert transitioned.operation_history[-1]["operation"] == "renamed"

    class AudioFakeOperation(AudioOperation[np.ndarray, np.ndarray]):
        name = "audio_fake"

        def _process_array(self, x: np.ndarray) -> np.ndarray:
            return x

    audio_transitioned = f._apply_operation_instance(
        AudioFakeOperation(f.sampling_rate),
        output_frame_class=NeedsExtra,
        output_frame_kwargs={"required": "ok"},
    )

    assert len(audio_transitioned.operations) == 1
    assert audio_transitioned.operations[0].name == "audio_fake"

    legacy_transitioned = f._apply_operation_instance(
        AudioFakeOperation(f.sampling_rate),
        output_frame_class=LegacyFrame,
    )

    assert isinstance(legacy_transitioned, LegacyFrame)
    assert legacy_transitioned.previous is f
    assert legacy_transitioned.operations == ()
    assert legacy_transitioned.operation_history[-1]["operation"] == "audio_fake"


def test_indexing_variants_cover_base_frame_selection_paths():
    f = make_frame(
        np.arange(12).reshape(3, 4),
        channel_metadata=[{"label": "left"}, {"label": "right"}, {"label": "rear"}],
    )

    assert f.get_channel(query={"label": "right"}).labels == ["right"]
    with pytest.raises(TypeError, match="Unsupported query type"):
        f.get_channel(query=object())  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="Either 'channel_idx' or 'query'"):
        f.get_channel()

    assert f["right"].labels == ["right"]
    assert f[np.array([True, False, True])].labels == ["left", "rear"]
    assert f[np.array([2, 0])].labels == ["rear", "left"]
    assert f[["rear", "left"]].labels == ["rear", "left"]
    assert f[cast(Any, [1, np.int64(2)])].labels == ["right", "rear"]
    assert f[1:].labels == ["right", "rear"]
    assert f[("left",)].labels == ["left"]
    assert f[([0, 2], slice(1, 3))]._data.shape == (2, 2)

    with pytest.raises(TypeError, match="Invalid channel key type in tuple"):
        _ = f[cast(Any, (1.5, slice(None)))]


def test_public_xarray_and_array_protocol_paths():
    f = make_frame(np.arange(6).reshape(2, 3), label="raw")

    exported = f.xr
    assert exported.name == "raw"
    assert exported.attrs["wandas_frame_type"] == "DummyFrame"
    np.testing.assert_array_equal(f.__array__(), np.arange(6).reshape(2, 3))


def test_channel_frame_binary_op_frame_success_and_remaining_operand_formats():
    left = ChannelFrame(
        da_from_array(np.arange(6).reshape(2, 3).astype(float), chunks=(1, -1)),
        sampling_rate=100.0,
        label="left_frame",
        channel_metadata=[{"label": "l0"}, {"label": "l1"}],
    )
    right = ChannelFrame(
        da_from_array(np.ones((2, 3)), chunks=(1, -1)),
        sampling_rate=100.0,
        label="right_frame",
        channel_metadata=[{"label": "r0"}, {"label": "r1"}],
    )

    added = left + right
    powered = left**2

    assert added.label == "(left_frame + right_frame)"
    assert added.labels == ["(l0 + r0)", "(l1 + r1)"]
    assert added.operation_history[-1] == {"operation": "+", "with": "right_frame"}
    np.testing.assert_array_equal(added.compute(), left.compute() + right.compute())
    assert powered.operation_history[-1] == {"operation": "**", "with": "2"}
    assert BaseFrame._format_operand_str(np.zeros((2, 3))) == "ndarray(2, 3)"
    assert BaseFrame._format_operand_str(da_from_array(np.zeros((2, 3)), chunks=(1, -1))) == "dask.array(2, 3)"

    mismatched_rate = ChannelFrame(da_from_array(np.ones((2, 3)), chunks=(1, -1)), sampling_rate=200.0)
    with pytest.raises(ValueError, match="Sampling rate mismatch"):
        _ = left + mismatched_rate


def test_apply_operation_dispatches_to_subclass_impl():
    f = make_frame(np.arange(6).reshape(2, 3))

    result = f.apply_operation("noop", amount=1)

    assert result.operation_history[-1] == {"operation": "noop", "amount": 1}


def test_lazy_metadata_and_previous_accessors():
    f = make_frame(np.arange(6).reshape(2, 3))
    f._xr.attrs.pop("metadata")

    assert f.metadata == {}
    assert f.previous is None


def test_tensorflow_tensor_conversion_without_device(monkeypatch):
    f = make_frame(np.arange(6).reshape(2, 3))

    class FakeTf:
        @staticmethod
        def convert_to_tensor(value):
            return {"tensor": value}

    import sys

    monkeypatch.setitem(sys.modules, "tensorflow", FakeTf)

    tensor = f.to_tensor(framework="tensorflow")

    np.testing.assert_array_equal(tensor["tensor"], f.to_numpy())


def test_base_frame_remaining_coordinate_and_indexing_edges():
    f = make_frame(np.arange(6).reshape(2, 3))

    assert f._next_channel_id(["c0", "c1"]) == "c2"
    assert f._channel_ids_for_selection([0, 0]) == ["c0", "c2"]
    with pytest.raises(TypeError, match="Invalid key type"):
        _ = f[cast(Any, object())]

    cf = ChannelFrame(da_from_array(np.arange(6).reshape(2, 3), chunks=(1, -1)), sampling_rate=100.0)
    cf._set_channel_coord_value("channel_label", 0, "front")
    exported = cf.to_xarray()
    exported.coords["channel_label"].values[0] = "mutated"

    assert cf.channels[0].label == "front"
    assert exported.attrs["wandas_frame_type"] == "ChannelFrame"


def test_xarray_coords_are_omitted_when_pending_metadata_length_mismatches():
    data = da_from_array(np.arange(6).reshape(2, 3), chunks=(1, -1))
    f = ChannelFrame(data, sampling_rate=100.0)
    f._pending_channel_metadata = [ChannelMetadata(label="only")]
    f._pending_channel_ids = ["c0", "c1"]

    assert f._xarray_coords(data) == {}

    del f._pending_channel_metadata
    del f._pending_channel_ids


def test_zero_dimensional_axisless_frame_uses_scalar_chunk_policy():
    data = da_from_array(np.array(1.0), chunks=())

    f = SingleChannelAxislessFrame(data, sampling_rate=100.0)

    assert f._data.ndim == 0
    assert f.n_channels == 1


def test_init_debug_logging_fallback_when_dask_details_unavailable(caplog):
    class DebugFallbackFrame(BaseFrame[np.ndarray]):
        _xarray_dim_suffix = ("channel", "time")

        @property
        def _data(self):
            raise RuntimeError("no dask details")

        def plot(self, plot_type: str = "default", ax=None, **kwargs):
            raise NotImplementedError

        def _get_dataframe_index(self) -> pd.Index:
            return pd.RangeIndex(stop=0)

        def _debug_info_impl(self) -> None:
            return None

    with caplog.at_level("DEBUG"):
        DebugFallbackFrame(da_from_array(np.arange(6).reshape(2, 3), chunks=(1, -1)), sampling_rate=100.0)

    assert "Dask graph visualization details unavailable" in caplog.text
