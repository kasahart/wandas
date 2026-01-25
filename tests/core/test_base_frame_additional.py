import dask.array as da
import numpy as np
import pandas as pd
import pytest

from wandas.core.base_frame import BaseFrame


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
        new_history = self.operation_history.copy() if self.operation_history else []
        new_history.append({"operation": operation_name, **params})
        return self._create_new_instance(data=self._data, operation_history=new_history)

    def _get_dataframe_columns(self) -> list[str]:
        return [ch.label for ch in self._channel_metadata]

    def _get_dataframe_index(self) -> pd.Index:
        # index should be length of samples
        length = self._data.shape[-1]
        return pd.RangeIndex(stop=length)

    def _debug_info_impl(self) -> None:
        return None


def make_frame(arr: np.ndarray | da.Array, **kwargs) -> DummyFrame:
    if isinstance(arr, np.ndarray):
        darr = da.from_array(arr, chunks=arr.shape)
    else:
        darr = arr
    return DummyFrame(darr, sampling_rate=100.0, **kwargs)


def test_rechunk_fallback_logs_warning(caplog):
    arr = da.from_array(np.arange(6).reshape(2, 3), chunks=(2, 3))
    original = arr.rechunk
    state = {"called": False}

    def bad_rechunk(chunks=None, **kwargs):
        if not state["called"]:
            state["called"] = True
            raise RuntimeError("boom")
        return original(chunks, **kwargs)

    arr.rechunk = bad_rechunk

    with caplog.at_level("WARNING"):
        f = make_frame(arr)
    assert "Rechunk failed" in caplog.text
    assert hasattr(f, "_data")


def test_get_channel_query_no_match_raises():
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    with pytest.raises(KeyError, match=r"No channels match query"):
        f.get_channel(query="nope")


def test_get_channel_query_unknown_key_raises():
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    with pytest.raises(KeyError, match=r"Unknown channel metadata key\(s\): unknown"):
        f.get_channel(query={"unknown": "x"})


def test_get_channel_query_regex_and_callable():
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    import re as _re

    # regex
    res = f.get_channel(query=_re.compile(r"ch0"))
    assert len(res) == 1
    assert res.labels == ["ch0"]

    # callable
    res2 = f.get_channel(query=lambda ch: ch.label == "ch1")
    assert len(res2) == 1
    assert res2.labels == ["ch1"]


def test_get_channel_query_dict_regex_value():
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    import re as _re

    res = f.get_channel(query={"label": _re.compile(r"ch")})
    assert len(res) == 2


def test_getitem_boolean_mask_length_mismatch():
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    mask = np.array([True])
    with pytest.raises(ValueError, match=r"Boolean mask length 1 does not match number of channels 2"):
        _ = f[mask]


def test_getitem_numpy_array_wrong_dtype():
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    a = np.array([0.1, 1.0])
    with pytest.raises(TypeError, match=r"NumPy array must be of integer or boolean type"):
        _ = f[a]


def test_getitem_empty_list_and_mixed_list_types():
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    with pytest.raises(ValueError, match=r"Cannot index with an empty list"):
        _ = f[[]]  # empty list index triggers ValueError


def test_getitem_mixed_list_types_explicit():
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    with pytest.raises(TypeError, match=r"List must contain all str or all int"):
        _ = f[[0, "ch0"]]


def test_handle_multidim_indexing_invalid_key_length():
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    with pytest.raises(ValueError, match=r"Invalid key length"):
        _ = f[(0, slice(None), slice(None))]


def test_label2index_keyerror():
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    with pytest.raises(KeyError, match=r"Channel label 'nope' not found"):
        f.label2index("nope")


def test_shape_single_channel():
    arr = np.arange(3)
    f = make_frame(arr)
    assert f.shape == (3,)


def test_compute_non_numpy_raises():
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)

    class Bad:
        def compute(self):
            return [1, 2, 3]

    f._data = Bad()
    with pytest.raises(ValueError, match=r"Computed result is not a np.ndarray"):
        f.compute()


def test_visualize_graph_failure_logs_and_returns_none(caplog):
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)

    class BadVis:
        def visualize(self, filename=None):
            raise RuntimeError("no graphviz")

    f._data = BadVis()

    with caplog.at_level("WARNING"):
        res = f.visualize_graph()
    assert res is None
    assert "Failed to visualize the graph" in caplog.text


def test_to_tensor_unsupported_framework():
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    with pytest.raises(ValueError, match=r"Unsupported framework"):
        f.to_tensor(framework="mxnet")


def test_to_tensor_missing_torch(monkeypatch):
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    import importlib.util as iu

    monkeypatch.setattr(iu, "find_spec", lambda name: None)
    with pytest.raises(ImportError, match=r"PyTorch is not installed"):
        f.to_tensor(framework="torch")


def test_to_tensor_missing_tf(monkeypatch):
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    import importlib.util as iu

    monkeypatch.setattr(iu, "find_spec", lambda name: None)
    with pytest.raises(ImportError, match=r"TensorFlow is not installed"):
        f.to_tensor(framework="tensorflow")


def test_to_dataframe_single_and_multi():
    # single channel
    arr1 = np.arange(3)
    f1 = make_frame(arr1)
    df1 = f1.to_dataframe()
    assert isinstance(df1, pd.DataFrame)
    assert df1.shape == (3, 1)

    # multi channel
    arr2 = np.arange(12).reshape(3, 4)
    f2 = make_frame(arr2)
    df2 = f2.to_dataframe()
    assert isinstance(df2, pd.DataFrame)
    assert df2.shape == (4, 3)


def test_array_protocol_dtype():
    arr = np.arange(6).reshape(2, 3).astype(np.float64)
    f = make_frame(arr)
    a = f.__array__(dtype=np.float32)
    assert a.dtype == np.float32


def test_print_operation_history_empty_and_nonempty(capsys):
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    f.operation_history = []
    f.print_operation_history()
    out = capsys.readouterr().out
    assert "Operation history: <empty>" in out

    f.operation_history = [{"operation": "normalize"}, {"name": "filter", "cutoff": 1000}]
    f.print_operation_history()
    out2 = capsys.readouterr().out
    assert "Operation history (2):" in out2
    assert "1: normalize {}" in out2
    assert "2: filter {'cutoff': 1000}" in out2


def test_relabel_and_create_new_instance_and_persist_and_type_checks():
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)

    # relabel
    new_meta = f._relabel_channels("norm")
    assert all(m.label.startswith("norm(") for m in new_meta)

    # create new instance type checks
    with pytest.raises(TypeError, match=r"Label must be a string"):
        f._create_new_instance(data=f._data, label=123)

    with pytest.raises(TypeError, match=r"Metadata must be a dictionary"):
        f._create_new_instance(data=f._data, metadata=123)

    with pytest.raises(TypeError, match=r"Channel metadata must be a list"):
        f._create_new_instance(data=f._data, channel_metadata=123)

    # persist

    class Persister:
        def __init__(self):
            self.ndim = 2
            self.shape = (2, 3)

        def persist(self):
            return self

        def rechunk(self, *args, **kwargs):
            return self

        def compute(self):
            import numpy as _np

            return _np.zeros(self.shape)

    p = Persister()
    f._data = p
    newf = f.persist()
    assert newf._data is p


def test_channel_metadata_invalid_types_and_validation():
    arr = np.arange(6).reshape(2, 3)
    # invalid dict value that fails pydantic validation (ref must be float)
    with pytest.raises(ValueError, match=r"Invalid channel_metadata at index 0"):
        make_frame(arr, channel_metadata=[{"label": "x", "ref": "bad"}])

    # invalid type in channel_metadata list
    with pytest.raises(TypeError, match=r"Invalid type in channel_metadata at index 0"):
        make_frame(arr, channel_metadata=[123])


def test_get_channel_query_unknown_key_no_validate_raises_no_match():
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    with pytest.raises(KeyError, match=r"No channels match query"):
        f.get_channel(query={"unknown": "x"}, validate_query_keys=False)


def test_get_channel_query_matches_extra_key():
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(
        arr,
        channel_metadata=[
            {"label": "a", "extra": {"foo": "bar"}},
            {"label": "b", "extra": {"foo": "baz"}},
        ],
    )
    res = f.get_channel(query={"foo": "bar"})
    assert len(res) == 1
    assert res.labels == ["a"]


def test_len_and_iter_and_getitem_single_channel():
    arr = np.arange(12).reshape(3, 4)
    f = make_frame(arr)
    assert len(f) == 3
    items = list(iter(f))
    assert len(items) == 3
    # each iterated item should be a single-channel frame
    for i, chf in enumerate(items):
        assert chf.n_channels == 1
        assert chf.labels == [f"ch{i}"]


def test_debug_info_logs_and__print_operation_history(capsys, caplog):
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    with caplog.at_level("DEBUG"):
        f.debug_info()
    assert "Debug Info" in caplog.text or f"=== {f.__class__.__name__} Debug Info ===" in caplog.text

    # _print_operation_history prints None and count
    f.operation_history = []
    f._print_operation_history()
    out = capsys.readouterr().out
    assert "Operations Applied: None" in out

    f.operation_history = [{"operation": "a"}, {"operation": "b"}]
    f._print_operation_history()
    out2 = capsys.readouterr().out
    assert "Operations Applied: 2" in out2


def test_to_tensor_torch_and_tensorflow_success(monkeypatch):
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)

    import importlib.util as iu
    import sys

    # Fake torch (module-like)
    class FakeTorchTensor:
        def __init__(self, arr):
            self._arr = arr
            self.device = type("D", (), {"type": "cpu", "index": None})()

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

    monkeypatch.setattr(iu, "find_spec", lambda name: object())
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
