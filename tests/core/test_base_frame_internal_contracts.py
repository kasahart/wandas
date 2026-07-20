"""Internal BaseFrame invariants not covered by its public contract suite."""

from typing import Any, cast

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tests.frame_helpers import channel_first_values
from wandas.core.base_frame import BaseFrame
from wandas.core.metadata import ChannelMetadata
from wandas.frames.channel import ChannelFrame
from wandas.processing.semantic import SemanticOperation, freeze_params, source_lineage, thaw_params
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
        lineage = self._required_semantic_lineage()
        return self._create_new_instance(data=self._data, lineage=lineage)

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
        lineage=None,
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
            channel_metadata=channel_metadata,
            channel_ids=channel_ids,
            source_time_offset=source_time_offset,
            previous=previous,
            lineage=lineage,
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
        lineage = self._required_semantic_lineage()
        return self._create_new_instance(data=self._data, lineage=lineage)

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


def test_print_operation_history_empty_shows_none(capsys):
    """Test _print_operation_history shows 'None' for empty history."""
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(arr)
    f._print_operation_history()
    out = capsys.readouterr().out
    assert "Operations Applied: None" in out


def test_print_operation_history_populated_shows_count(capsys):
    """Test _print_operation_history shows correct count for populated history."""
    arr = np.arange(6).reshape(2, 3)
    f = make_frame(
        arr,
        operation_history_prefix=[
            {"operation": "a", "version": 1, "params": {}},
            {"operation": "b", "version": 1, "params": {}},
        ],
    )
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

    assert "source_time_offset" not in frame._xr.coords
    np.testing.assert_array_equal(frame._xr.attrs["source_time_offset"], np.array([2.5]))
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


def test_label_metadata_attrs_and_lineage_assignment_validation():
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

    with pytest.raises(AttributeError):
        setattr(f, "operation_history", cast(Any, "bad"))
    with pytest.raises(AttributeError):
        setattr(f, "lineage", source_lineage())


def test_create_new_instance_rejects_legacy_history_and_validates_channel_ids():
    f = make_frame(np.arange(6).reshape(2, 3))

    with pytest.raises(TypeError, match="unexpected keyword argument 'operation_history'"):
        f._create_new_instance(data=f._data, operation_history="bad")
    with pytest.raises(TypeError, match="unexpected keyword argument 'operation_graph'"):
        f._create_new_instance(data=f._data, operation_graph={"operation": "bad"})
    with pytest.raises(TypeError, match="unexpected keyword argument 'operations'"):
        make_frame(np.arange(6).reshape(2, 3), operations=["bad"])
    with pytest.raises(TypeError, match="Channel ids must be a list"):
        f._create_new_instance(data=f._data, channel_ids=("a", "b"))


def test_create_new_instance_preserves_lineage_for_constructor():
    data = da_from_array(np.arange(6).reshape(2, 3).astype(float), chunks=(2, 3))
    lineage = source_lineage([{"operation": "fake", "version": 1, "params": {}}])
    frame = LegacyFrame(
        data,
        sampling_rate=100.0,
        lineage=lineage,
    )

    result = frame._create_new_instance(
        data=frame._data,
    )

    assert isinstance(result, LegacyFrame)
    assert result.previous is frame
    assert result.operation_history == [{"operation": "fake", "version": 1, "params": {}}]
    assert result.lineage is lineage


def test_binary_operations_cover_scalar_helpers_and_frame_mismatch_errors():
    f = ChannelFrame(da_from_array(np.arange(6).reshape(2, 3).astype(float), chunks=(1, -1)), sampling_rate=100.0)

    assert (f - 1).operation_history[-1] == {
        "operation": "wandas.operator.subtract",
        "version": 1,
        "params": {"operand": 1},
    }
    assert (f * 2).operation_history[-1]["operation"] == "wandas.operator.multiply"
    assert (f / 2).operation_history[-1]["operation"] == "wandas.operator.divide"
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

    left_branch = left.normalize()
    right_branch = right.remove_dc()
    result = left_branch + right_branch

    assert [record["operation"] for record in result.operation_history] == [
        "wandas.audio.normalize",
        "wandas.audio.remove_dc",
        "wandas.operator.add",
    ]
    assert result.lineage.inputs == (left_branch.lineage, right_branch.lineage)


def test_lineage_keeps_semantic_operation_without_runtime_operation_state():
    data = np.array([[1.0, 2.0, 4.0]])
    frame = ChannelFrame(da_from_array(data, chunks=(1, -1)), sampling_rate=100.0)

    result = frame.normalize(norm=np.inf)

    op = result.lineage.operation
    assert isinstance(op, SemanticOperation)
    assert op.operation_id == "wandas.audio.normalize"
    assert thaw_params(op.params)["norm"] == np.inf

    with pytest.raises(AttributeError):
        setattr(op, "params", freeze_params({"norm": 2.0}))

    np.testing.assert_allclose(channel_first_values(result), data / 4.0)


def test_none_valued_config_reassignment_is_blocked_and_delayed_result_is_stable():
    data = np.array([[1.0, 2.0, 4.0]])
    frame = ChannelFrame(da_from_array(data, chunks=(1, -1)), sampling_rate=100.0)

    result = frame.normalize(norm=None)
    op = result.lineage.operation
    assert isinstance(op, SemanticOperation)

    with pytest.raises(AttributeError):
        setattr(op, "params", freeze_params({"norm": np.inf}))

    assert thaw_params(op.params)["norm"] is None
    np.testing.assert_allclose(channel_first_values(result), data)


def test_power_operation_exp_alias_reassignment_does_not_change_delayed_result():
    data = np.array([[2.0, 3.0, 4.0]])
    frame = ChannelFrame(da_from_array(data, chunks=(1, -1)), sampling_rate=100.0)

    result = frame.power(2.0)
    op = result.lineage.operation
    assert isinstance(op, SemanticOperation)

    with pytest.raises(AttributeError):
        setattr(op, "params", freeze_params({"exponent": 3.0}))

    assert thaw_params(op.params)["exponent"] == 2.0
    np.testing.assert_allclose(channel_first_values(result), data**2)


def test_channel_metadata_update_paths_preserve_operations_lineage():
    data = np.array([[1.0, 2.0, 4.0]])
    frame = ChannelFrame(da_from_array(data, chunks=(1, -1)), sampling_rate=100.0)

    result = frame.normalize().rename_channels({0: "renamed"})

    assert result.lineage is not None
    assert [record["operation"] for record in result.operation_history] == [
        "wandas.audio.normalize",
        "wandas.channel.rename_channels",
    ]
    np.testing.assert_allclose(channel_first_values(result), data / 4.0)


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

    with pytest.raises(TypeError, match="Invalid channel selector type"):
        _ = f[cast(Any, (1.5, slice(None)))]


def test_array_protocol_matches_data():
    f = make_frame(np.arange(6).reshape(2, 3), label="raw")

    np.testing.assert_array_equal(np.asarray(f), f.data)


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
    assert added.operation_history[-1] == {
        "operation": "wandas.operator.add",
        "version": 1,
        "params": {},
    }
    np.testing.assert_array_equal(channel_first_values(added), channel_first_values(left) + channel_first_values(right))
    assert powered.operation_history[-1]["params"]["operand"] == 2
    assert BaseFrame._format_operand_str(np.zeros((2, 3))) == "ndarray(2, 3)"
    assert BaseFrame._format_operand_str(da_from_array(np.zeros((2, 3)), chunks=(1, -1))) == "dask.array(2, 3)"

    mismatched_rate = ChannelFrame(da_from_array(np.ones((2, 3)), chunks=(1, -1)), sampling_rate=200.0)
    with pytest.raises(ValueError, match="Sampling rate mismatch"):
        _ = left + mismatched_rate


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
    with pytest.raises(TypeError, match="Invalid channel selector type"):
        _ = f[cast(Any, object())]

    cf = ChannelFrame(da_from_array(np.arange(6).reshape(2, 3), chunks=(1, -1)), sampling_rate=100.0)
    cf._set_channel_coord_value("channel_label", 0, "front")
    assert cf.channels[0].label == "front"
    assert cf._xr.coords["channel_label"].values[0] == "front"


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


def test_semantic_lineage_guard_rejects_internal_bypass() -> None:
    frame = make_frame(np.arange(6).reshape(2, 3))

    with pytest.raises(RuntimeError, match="semantic lineage capture is not active"):
        frame._required_semantic_lineage()


def test_selector_decoder_rejects_unknown_canonical_kind() -> None:
    with pytest.raises(TypeError, match="Unsupported canonical selector"):
        BaseFrame._selector_from_intent({"indexing": "unknown"})


def test_channel_indexing_rejects_out_of_range_scalar_and_list() -> None:
    frame = ChannelFrame.from_numpy(
        np.arange(6).reshape(2, 3),
        sampling_rate=100,
        ch_labels=["left", "right"],
    )

    with pytest.raises(IndexError, match="Channel index out of range"):
        _ = frame[2]
    with pytest.raises(IndexError, match="Channel index out of range"):
        _ = frame[[0, 2]]


def test_bool_scalar_operand_display_is_stable() -> None:
    assert BaseFrame._format_operand_str(np.bool_(True)) == "True"
