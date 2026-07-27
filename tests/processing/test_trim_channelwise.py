"""Channel-wise execution contracts for the Trim operation."""

from typing import Any
from unittest import mock

import dask.array as da
import numpy as np
import pytest
from dask import delayed
from dask.array.core import Array as DaArray

from wandas.processing.temporal import Trim
from wandas.utils.dask_helpers import da_from_array

_SAMPLING_RATE = 1_000
_N_SAMPLES = 16
_START = 0.003
_END = 0.011
_EXPECTED_SLICE = slice(3, 11)


def _values(channels: int, dtype: np.dtype[Any]) -> np.ndarray:
    return np.arange(channels * _N_SAMPLES, dtype=dtype).reshape(channels, _N_SAMPLES)


def _wandas_operation_task_keys(result: DaArray) -> tuple[str, ...]:
    return tuple(
        sorted(key for key in result.dask if isinstance(key, str) and key.startswith("_execute_wandas_operation-"))
    )


@pytest.mark.parametrize("channels", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", [np.dtype(np.int16), np.dtype(np.float32), np.dtype(np.float64)])
def test_trim_channel_wise_kernel_matches_numpy_and_whole_frame_exactly(
    monkeypatch: pytest.MonkeyPatch,
    channels: int,
    dtype: np.dtype[Any],
) -> None:
    values = _values(channels, dtype)
    values_before = values.copy()
    source = da_from_array(values, chunks=(1, -1))
    operation = Trim(_SAMPLING_RATE, start=_START, end=_END)
    original_process = operation._process
    kernel_shapes: list[tuple[int, ...]] = []

    def observed_process(channel_values: np.ndarray) -> np.ndarray:
        kernel_shapes.append(channel_values.shape)
        return original_process(channel_values)

    monkeypatch.setattr(operation, "_process", observed_process)
    with mock.patch.object(DaArray, "compute") as compute:
        channel_wise = operation.process(source)
        compute.assert_not_called()

    whole_frame = operation._build_whole_frame_graph(
        source,
        (),
        output_shape=(channels, 8),
        output_dtype=dtype,
    )

    assert isinstance(channel_wise, DaArray)
    assert channel_wise.shape == (channels, 8)
    assert channel_wise.dtype == dtype
    assert channel_wise.chunks == (channels * (1,), (8,))

    channel_values = channel_wise.compute(scheduler="synchronous")
    assert kernel_shapes == channels * [(1, _N_SAMPLES)]

    kernel_shapes.clear()
    whole_values = whole_frame.compute(scheduler="synchronous")
    assert kernel_shapes == [(channels, _N_SAMPLES)]

    np.testing.assert_array_equal(channel_values, values[:, _EXPECTED_SLICE])
    np.testing.assert_array_equal(channel_values, whole_values)
    np.testing.assert_array_equal(values, values_before)


def test_trim_repeated_graph_builds_use_stable_wandas_task_keys() -> None:
    source = da_from_array(_values(4, np.dtype(np.float64)), chunks=(1, -1))
    operation = Trim(_SAMPLING_RATE, start=_START, end=_END)

    first = operation.process(source)
    second = operation.process(source)

    first_keys = _wandas_operation_task_keys(first)
    assert len(first_keys) == 4
    assert first_keys == _wandas_operation_task_keys(second)


def test_trim_zero_channel_input_uses_whole_frame_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = da.from_array(np.empty((0, _N_SAMPLES), dtype=np.float64), chunks=(0, _N_SAMPLES))
    operation = Trim(_SAMPLING_RATE, start=_START, end=_END)
    original_process = operation._process
    kernel_shapes: list[tuple[int, ...]] = []

    def observed_process(values: np.ndarray) -> np.ndarray:
        kernel_shapes.append(values.shape)
        return original_process(values)

    monkeypatch.setattr(operation, "_process", observed_process)
    result = operation.process(source)

    np.testing.assert_array_equal(
        result.compute(scheduler="synchronous"),
        np.empty((0, 8), dtype=np.float64),
    )
    assert kernel_shapes == [(0, _N_SAMPLES)]
    assert result.chunks == ((0,), (8,))


def test_trim_unknown_channel_count_uses_whole_frame_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _values(3, np.dtype(np.float64))
    source = da.from_delayed(
        delayed(np.asarray)(values),
        shape=(np.nan, _N_SAMPLES),
        dtype=values.dtype,
    )
    operation = Trim(_SAMPLING_RATE, start=_START, end=_END)
    original_process = operation._process
    kernel_shapes: list[tuple[int, ...]] = []

    def observed_process(input_values: np.ndarray) -> np.ndarray:
        kernel_shapes.append(input_values.shape)
        return original_process(input_values)

    monkeypatch.setattr(operation, "_process", observed_process)
    result = operation.process(source)

    np.testing.assert_array_equal(
        result.compute(scheduler="synchronous"),
        values[:, _EXPECTED_SLICE],
    )
    assert kernel_shapes == [(3, _N_SAMPLES)]


def test_trim_rejects_extra_runtime_input_before_graph_construction() -> None:
    source = da_from_array(_values(2, np.dtype(np.float64)), chunks=(1, -1))

    with pytest.raises(ValueError, match=r"Expected exactly one input for Trim"):
        Trim(_SAMPLING_RATE, start=_START, end=_END).process(source, source)


@pytest.mark.parametrize(
    ("start_sample", "end_sample"),
    [
        pytest.param(-2, 8, id="negative-start"),
        pytest.param(10, 14, id="start-after-input"),
        pytest.param(6, 2, id="reverse-range-direct"),
        pytest.param(4, 4, id="empty-range"),
        pytest.param(2, 12, id="end-clipped-to-input"),
    ],
)
def test_trim_slice_boundaries_advertise_and_compute_numpy_shape_exactly(
    start_sample: int,
    end_sample: int,
) -> None:
    values = np.arange(16, dtype=np.float64).reshape(2, 8)
    source = da_from_array(values, chunks=(1, -1))
    operation = Trim(
        sampling_rate=8,
        start=start_sample / 8,
        end=end_sample / 8,
    )
    expected = values[..., start_sample:end_sample]
    advertised_shape = operation.calculate_output_shape(values.shape)

    assert advertised_shape == expected.shape

    channel_wise = operation.process(source)
    whole_frame = operation._build_whole_frame_graph(
        source,
        (),
        output_shape=advertised_shape,
        output_dtype=values.dtype,
    )

    assert channel_wise.shape == expected.shape
    assert whole_frame.shape == expected.shape
    np.testing.assert_array_equal(channel_wise.compute(scheduler="synchronous"), expected)
    np.testing.assert_array_equal(whole_frame.compute(scheduler="synchronous"), expected)
