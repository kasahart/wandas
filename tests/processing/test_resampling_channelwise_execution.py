"""Channel-wise execution contracts for ReSampling."""

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

import wandas.processing.temporal as temporal
from wandas.processing.temporal import ReSampling
from wandas.utils.dask_helpers import da_from_array


def _source(channels: int, samples: int = 1001) -> DaArray:
    time_axis = np.arange(samples) / 44_100
    values = np.stack([np.sin(2 * np.pi * (440 + 20 * index) * time_axis) for index in range(channels)])
    return da_from_array(values, chunks=(1, 200))


@pytest.mark.parametrize("channels", [1, 2, 4, 8])
def test_resampling_channel_wise_kernel_matches_whole_frame_exactly(
    monkeypatch: pytest.MonkeyPatch,
    channels: int,
) -> None:
    source = _source(channels)
    operation = ReSampling(44_100, target_sr=16_000)
    original_process = operation._process
    kernel_shapes: list[tuple[int, ...]] = []

    def observed_process(values: np.ndarray) -> np.ndarray:
        kernel_shapes.append(values.shape)
        return original_process(values)

    monkeypatch.setattr(operation, "_process", observed_process)
    channel_wise = operation.process(source)
    output_shape = operation.calculate_output_shape(source.shape)
    whole_frame = operation._build_whole_frame_graph(
        source,
        (),
        output_shape=output_shape,
        output_dtype=operation.calculate_output_dtype(source.dtype),
    )

    assert output_shape == (channels, 364)
    assert channel_wise.shape == output_shape
    assert channel_wise.dtype == np.dtype(np.float64)
    assert channel_wise.chunks == (channels * (1,), (364,))
    channel_values = channel_wise.compute(scheduler="synchronous")
    assert kernel_shapes == channels * [(1, 1001)]

    kernel_shapes.clear()
    whole_values = whole_frame.compute(scheduler="synchronous")
    assert kernel_shapes == [(channels, 1001)]
    np.testing.assert_array_equal(channel_values, whole_values)


def test_resampling_zero_channel_input_uses_whole_frame_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = ReSampling(44_100, target_sr=16_000)
    original_process = operation._process
    kernel_shapes: list[tuple[int, ...]] = []

    def observed_process(values: np.ndarray) -> np.ndarray:
        kernel_shapes.append(values.shape)
        return original_process(values)

    monkeypatch.setattr(operation, "_process", observed_process)
    source = da.from_array(np.empty((0, 1001)), chunks=(0, 1001))

    result = operation.process(source)

    np.testing.assert_array_equal(result.compute(scheduler="synchronous"), np.empty((0, 364)))
    assert kernel_shapes == [(0, 1001)]
    assert result.chunks == ((0,), (364,))


def test_resampling_exact_length_fft_fallback_matches_whole_frame_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(2)
    operation = ReSampling(1_000.0, target_sr=1_000.0001)
    original_process = operation._process
    original_resample = temporal.resample
    kernel_shapes: list[tuple[int, ...]] = []
    fallback_shapes: list[tuple[int, ...]] = []

    def observed_process(values: np.ndarray) -> np.ndarray:
        kernel_shapes.append(values.shape)
        return original_process(values)

    def observed_resample(values: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        fallback_shapes.append(values.shape)
        return original_resample(values, *args, **kwargs)

    monkeypatch.setattr(operation, "_process", observed_process)
    monkeypatch.setattr(temporal, "resample", observed_resample)
    channel_wise = operation.process(source)
    output_shape = operation.calculate_output_shape(source.shape)
    output_dtype = operation.calculate_output_dtype(source.dtype)
    whole_frame = operation._build_whole_frame_graph(
        source,
        (),
        output_shape=output_shape,
        output_dtype=output_dtype,
    )

    assert output_shape == (2, 1002)
    assert channel_wise.shape == output_shape
    assert channel_wise.dtype == np.dtype(np.float64)
    assert channel_wise.chunks == ((1, 1), (1002,))

    channel_values = channel_wise.compute(scheduler="synchronous")
    assert kernel_shapes == [(1, 1001), (1, 1001)]
    assert fallback_shapes == [(1, 1001), (1, 1001)]

    kernel_shapes.clear()
    fallback_shapes.clear()
    whole_values = whole_frame.compute(scheduler="synchronous")
    assert kernel_shapes == [(2, 1001)]
    assert fallback_shapes == [(2, 1001)]
    np.testing.assert_array_equal(channel_values, whole_values)
