from typing import Any
from unittest import mock

import dask.array as da
import numpy as np
import pytest
from dask.delayed import delayed

from wandas.frames.channel import ChannelFrame
from wandas.lineage import extract_operations
from wandas.processing.base import AudioOperation, _execute_wandas_operation
from wandas.processing.custom import CustomOperation
from wandas.processing.effects import Normalize
from wandas.processing.filters import HighPassFilter
from wandas.processing.spectral import STFT
from wandas.utils.dask_helpers import da_from_array
from wandas.utils.types import NDArrayReal


def _frame() -> ChannelFrame:
    data = np.linspace(-1.0, 1.0, 4096, dtype=np.float64).reshape(1, -1)
    return ChannelFrame(da_from_array(data, chunks=(1, -1)), sampling_rate=16000, label="lineage")


def _stereo_frame() -> ChannelFrame:
    data = np.vstack(
        [
            np.linspace(-1.0, 1.0, 4096, dtype=np.float64),
            np.linspace(1.0, -1.0, 4096, dtype=np.float64),
        ]
    )
    return ChannelFrame(da_from_array(data, chunks=(1, -1)), sampling_rate=16000, label="lineage")


def _int_stereo_frame() -> ChannelFrame:
    data = np.vstack(
        [
            np.arange(4096, dtype=np.int16),
            np.arange(4096, dtype=np.int16) * -1,
        ]
    )
    return ChannelFrame(da_from_array(data, chunks=(1, -1)), sampling_rate=16000, label="lineage")


class _GraphCollection:
    def __init__(self, graph: dict[object, object]):
        self._graph = graph

    def __dask_graph__(self) -> dict[object, object]:
        return self._graph


class _LineageTestOperation(AudioOperation[NDArrayReal, NDArrayReal]):
    name = "lineage_test"

    def _process(self, x: NDArrayReal) -> NDArrayReal:
        return x


def test_operations_extracts_serial_chain_in_dependency_order() -> None:
    result = _frame().high_pass_filter(100).normalize().stft(n_fft=64, hop_length=16)

    operations = result.operations

    assert [operation.name for operation in operations] == ["highpass_filter", "normalize", "stft"]
    assert isinstance(operations[0], HighPassFilter)
    assert isinstance(operations[1], Normalize)
    assert isinstance(operations[2], STFT)


def test_operations_keeps_repeated_operations_as_distinct_instances() -> None:
    result = _frame().normalize().normalize()

    operations = result.operations

    assert [operation.name for operation in operations] == ["normalize", "normalize"]
    assert operations[0] is not operations[1]


def test_operations_ignores_dask_internal_tasks_rechunk_and_from_delayed() -> None:
    delayed_data = delayed(lambda: np.ones((1, 32), dtype=np.float64))()
    dask_data = da.from_delayed(delayed_data, shape=(1, 32), dtype=np.float64).rechunk((1, -1))
    frame = ChannelFrame(dask_data, sampling_rate=16000)

    assert frame.operations == ()


def test_extract_operations_rejects_non_dask_collection() -> None:
    with pytest.raises(TypeError, match="Expected a Dask collection with __dask_graph__"):
        extract_operations(object())


def test_operations_preserves_tuple_key_dependencies_in_legacy_graph() -> None:
    first = _LineageTestOperation(16000)
    second = _LineageTestOperation(16000)
    first_key = ("first", 0, 0)
    graph: dict[Any, Any] = {
        "second": (_execute_wandas_operation, second, first_key),
        first_key: (_execute_wandas_operation, first, "source"),
    }

    operations = extract_operations(_GraphCollection(graph))

    assert operations == (first, second)


def test_operations_stable_after_compute_on_same_frame_graph() -> None:
    result = _frame().high_pass_filter(100).normalize()
    before = result.operations

    _ = result.compute()

    assert result.operations == before


def test_operations_does_not_compute_data() -> None:
    result = _frame().normalize()

    with mock.patch("dask.array.core.Array.compute") as compute:
        operations = result.operations

    compute.assert_not_called()
    assert [operation.name for operation in operations] == ["normalize"]


def test_operations_extracts_custom_operation_instance() -> None:
    def scale(x: NDArrayReal, gain: float) -> NDArrayReal:
        return x * gain

    result = _frame().apply(scale, gain=2.0)

    operations = result.operations

    assert len(operations) == 1
    assert isinstance(operations[0], CustomOperation)
    assert operations[0].func is scale
    assert operations[0].params == {"gain": 2.0}


def test_operations_includes_stats_operation_before_normalize() -> None:
    operations = _frame().abs().normalize().operations

    assert [operation.name for operation in operations] == ["abs", "normalize"]


def test_operations_deduplicates_chunked_stats_markers() -> None:
    operations = _stereo_frame().abs().operations

    assert [operation.name for operation in operations] == ["abs"]


def test_operations_includes_power_params() -> None:
    operations = _frame().power(exponent=2.0).operations

    assert [operation.name for operation in operations] == ["power"]
    assert operations[0].params == {"exponent": 2.0}


def test_stats_operations_keep_dask_native_dtype_and_chunks() -> None:
    frame = _int_stereo_frame()
    meaned = frame.mean()
    absolute = frame.abs()

    assert meaned._data.dtype == np.float64
    assert meaned.compute().dtype == np.float64
    assert absolute._data.chunks == frame._data.chunks


def test_operations_includes_channel_reductions() -> None:
    summed = _stereo_frame().sum()
    averaged = _stereo_frame().mean()

    assert [operation.name for operation in summed.operations] == ["sum"]
    assert [operation.name for operation in averaged.operations] == ["mean"]
    assert summed.n_channels == 1
    assert averaged.n_channels == 1


def test_operations_includes_channel_difference() -> None:
    operations = _stereo_frame().channel_difference(other_channel=0).operations

    assert [operation.name for operation in operations] == ["channel_difference"]
    assert operations[0].params == {"other_channel": 0}


def test_operations_preserves_negative_channel_difference_indices() -> None:
    frame = _stereo_frame()
    result = frame.channel_difference(other_channel=-1)

    assert [operation.name for operation in result.operations] == ["channel_difference"]
    np.testing.assert_allclose(result.compute(), frame.compute() - frame.compute()[-1])


def test_operations_prefers_nested_marker_over_alias() -> None:
    operations = _stereo_frame().normalize().sum().operations

    assert [operation.name for operation in operations] == ["normalize", "sum"]


def test_operations_preserves_fused_native_markers_before_normalize() -> None:
    operations = _frame().abs().power(exponent=2.0).normalize().operations

    assert [operation.name for operation in operations] == ["abs", "power", "normalize"]


def test_operations_preserves_fused_native_markers_before_stft() -> None:
    operations = _frame().abs().power(exponent=2.0).stft(n_fft=64, hop_length=16).operations

    assert [operation.name for operation in operations] == ["abs", "power", "stft"]


def test_operation_history_public_behavior_is_read_only_lineage_view() -> None:
    frame = _frame()

    with pytest.raises(AttributeError):
        setattr(frame, "operation_history", [{"operation": "load", "params": {"path": "input.wav"}}])

    result = frame.normalize()
    history = result.operation_history
    history.append({"operation": "mutated", "params": {}})

    assert [record["operation"] for record in result.operation_history] == ["normalize"]
    assert all(record["operation"] != "mutated" for record in result.operation_history)
    assert "operation_history" not in result._xr.attrs


def test_operations_property_is_read_only_sequence() -> None:
    result = _frame().normalize()

    assert isinstance(result.operations, tuple)
    with pytest.raises(AttributeError):
        result.operations = ()  # ty: ignore[invalid-assignment]
