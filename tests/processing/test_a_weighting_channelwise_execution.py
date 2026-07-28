"""Channel-wise execution contracts for A-weighting."""

from unittest import mock

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray
from dask.delayed import delayed

from wandas.processing.base import ChannelIndependentAudioOperation
from wandas.processing.filters import AWeighting
from wandas.utils.dask_helpers import da_from_array

_SAMPLING_RATE = 48_000
_SAMPLES = 1_024


def _values(channels: int, dtype: np.dtype[np.generic]) -> np.ndarray:
    time = np.arange(_SAMPLES) / _SAMPLING_RATE
    values = np.stack(
        [
            (index + 1)
            * (1_000 * np.sin(2 * np.pi * (200 + 100 * index) * time) + 300 * np.cos(2 * np.pi * 2_000 * time))
            for index in range(channels)
        ]
    )
    return values.astype(dtype)


def _source(channels: int, dtype: np.dtype[np.generic]) -> DaArray:
    return da_from_array(_values(channels, dtype), chunks=(1, 128))


def _kernel_keys(array: DaArray) -> tuple[str, ...]:
    return tuple(sorted(repr(key) for key in array.dask if "_execute_wandas_operation" in repr(key)))


def test_a_weighting_declares_channel_independent_contract() -> None:
    assert issubclass(AWeighting, ChannelIndependentAudioOperation)


@pytest.mark.parametrize("channels", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "dtype",
    [np.dtype(np.float32), np.dtype(np.float64), np.dtype(np.int16)],
    ids=["float32", "float64", "int16"],
)
def test_a_weighting_channel_wise_kernel_matches_whole_frame_exactly(
    monkeypatch: pytest.MonkeyPatch,
    channels: int,
    dtype: np.dtype[np.generic],
) -> None:
    source = _source(channels, dtype)
    source_before = source.compute(scheduler="synchronous").copy()
    operation = AWeighting(_SAMPLING_RATE)
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

    assert isinstance(channel_wise, DaArray)
    assert channel_wise.shape == whole_frame.shape == source.shape
    assert channel_wise.dtype == whole_frame.dtype == np.dtype(np.float64)
    assert channel_wise.chunks == (channels * (1,), (_SAMPLES,))
    assert kernel_shapes == []

    channel_values = channel_wise.compute(scheduler="synchronous")
    assert kernel_shapes == channels * [(1, _SAMPLES)]

    kernel_shapes.clear()
    whole_values = whole_frame.compute(scheduler="synchronous")
    assert kernel_shapes == [(channels, _SAMPLES)]
    np.testing.assert_array_equal(channel_values, whole_values)
    np.testing.assert_array_equal(source.compute(scheduler="synchronous"), source_before)


def test_a_weighting_repeated_graphs_have_stable_pure_output_and_kernel_keys() -> None:
    source = _source(4, np.dtype(np.float64))
    operation = AWeighting(_SAMPLING_RATE)

    with mock.patch("wandas.processing.base.delayed", wraps=delayed) as delayed_builder:
        first = operation.process(source)
        second = operation.process(source)

    assert operation.pure is True
    assert first.name == second.name
    assert first.__dask_keys__() == second.__dask_keys__()
    assert _kernel_keys(first) == _kernel_keys(second)
    assert len(_kernel_keys(first)) == 4
    assert len(delayed_builder.call_args_list) == 8
    assert all(call.kwargs["pure"] is True for call in delayed_builder.call_args_list)


def test_a_weighting_zero_channel_input_uses_whole_frame_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = AWeighting(_SAMPLING_RATE)
    original_process = operation._process
    kernel_shapes: list[tuple[int, ...]] = []

    def observed_process(values: np.ndarray) -> np.ndarray:
        kernel_shapes.append(values.shape)
        return original_process(values)

    monkeypatch.setattr(operation, "_process", observed_process)
    source = da.from_array(np.empty((0, _SAMPLES)), chunks=(0, _SAMPLES))

    result = operation.process(source)

    assert result.dtype == np.dtype(np.float64)
    assert result.chunks == ((0,), (_SAMPLES,))
    np.testing.assert_array_equal(
        result.compute(scheduler="synchronous"),
        np.empty((0, _SAMPLES)),
    )
    assert kernel_shapes == [(0, _SAMPLES)]


def test_a_weighting_unknown_channel_count_uses_whole_frame_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _values(2, np.dtype(np.float64))
    source = da.from_delayed(
        delayed(np.asarray)(values),
        shape=(np.nan, _SAMPLES),
        dtype=values.dtype,
    )
    operation = AWeighting(_SAMPLING_RATE)
    original_process = operation._process
    kernel_shapes: list[tuple[int, ...]] = []

    def observed_process(array: np.ndarray) -> np.ndarray:
        kernel_shapes.append(array.shape)
        return original_process(array)

    monkeypatch.setattr(operation, "_process", observed_process)
    result = operation.process(source)
    expected = original_process(values)

    assert np.isnan(result.shape[0])
    assert result.shape[1:] == (_SAMPLES,)
    assert result.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(
        result.compute(scheduler="synchronous"),
        expected,
    )
    assert kernel_shapes == [(2, _SAMPLES)]


def test_a_weighting_extra_runtime_input_keeps_existing_error() -> None:
    source = _source(2, np.dtype(np.float64))

    with pytest.raises(
        ValueError,
        match=r"Expected exactly one input for AWeighting; got 2",
    ):
        AWeighting(_SAMPLING_RATE).process(source, source)
