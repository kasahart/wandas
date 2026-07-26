"""Channel-wise execution contracts for the Butterworth filter family."""

from collections.abc import Callable
from typing import Any

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.processing.base import AudioOperation
from wandas.processing.filters import BandPassFilter, HighPassFilter, LowPassFilter
from wandas.utils.dask_helpers import da_from_array

FilterFactory = Callable[[], AudioOperation[Any, Any]]


def _low_pass() -> LowPassFilter:
    return LowPassFilter(8_000, cutoff=1_000, order=2)


def _high_pass() -> HighPassFilter:
    return HighPassFilter(8_000, cutoff=1_000, order=2)


def _band_pass() -> BandPassFilter:
    return BandPassFilter(8_000, low_cutoff=500, high_cutoff=1_500, order=2)


def _deterministic_input(channels: int) -> DaArray:
    time_axis = np.arange(256) / 8_000
    values = np.stack(
        [
            np.sin(2 * np.pi * (200 + 100 * index) * time_axis) + 0.25 * np.sin(2 * np.pi * 2_000 * time_axis)
            for index in range(channels)
        ]
    )
    return da_from_array(values, chunks=(1, 64))


@pytest.mark.parametrize("factory", [_low_pass, _high_pass, _band_pass])
@pytest.mark.parametrize("channels", [1, 2, 4, 8])
def test_butterworth_channel_wise_kernel_matches_whole_frame_exactly(
    monkeypatch: pytest.MonkeyPatch,
    factory: FilterFactory,
    channels: int,
) -> None:
    source = _deterministic_input(channels)
    operation = factory()
    original_process = operation._process
    kernel_shapes: list[tuple[int, ...]] = []

    def observed_process(values: np.ndarray) -> np.ndarray:
        kernel_shapes.append(values.shape)
        return original_process(values)

    monkeypatch.setattr(operation, "_process", observed_process)
    channel_wise = operation.process(source)
    whole_frame = operation._build_whole_frame_graph(
        source,
        (),
        output_shape=source.shape,
        output_dtype=np.dtype(np.float64),
    )

    assert channel_wise.shape == source.shape
    assert channel_wise.dtype == np.dtype(np.float64)
    assert channel_wise.chunks == (channels * (1,), (256,))
    channel_values = channel_wise.compute(scheduler="synchronous")
    assert kernel_shapes == channels * [(1, 256)]

    kernel_shapes.clear()
    whole_values = whole_frame.compute(scheduler="synchronous")
    assert kernel_shapes == [(channels, 256)]
    # Both paths call the same scipy.signal.filtfilt kernel on independent rows.
    np.testing.assert_array_equal(channel_values, whole_values)


def test_butterworth_zero_channel_input_uses_whole_frame_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = _low_pass()
    original_process = operation._process
    kernel_shapes: list[tuple[int, ...]] = []

    def observed_process(values: np.ndarray) -> np.ndarray:
        kernel_shapes.append(values.shape)
        return original_process(values)

    monkeypatch.setattr(operation, "_process", observed_process)
    source = da.from_array(np.empty((0, 64)), chunks=(0, 64))

    result = operation.process(source)

    np.testing.assert_array_equal(result.compute(scheduler="synchronous"), np.empty((0, 64)))
    assert kernel_shapes == [(0, 64)]
    assert result.chunks == ((0,), (64,))
