"""Parameter-dependent channel-wise execution contracts for Normalize."""

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray
from dask.delayed import delayed

from wandas.processing.base import AudioOperation, ChannelIndependentAudioOperation
from wandas.processing.effects import Normalize
from wandas.utils.dask_helpers import da_from_array


def _source(channels: int, samples: int = 256) -> DaArray:
    base = np.linspace(-1.0, 1.0, samples)
    return da_from_array(
        np.stack([(index + 1) * base for index in range(channels)]),
        chunks=(1, 64),
    )


def test_normalize_keeps_parameter_dependent_base_contract() -> None:
    assert issubclass(Normalize, AudioOperation)
    assert not issubclass(Normalize, ChannelIndependentAudioOperation)


@pytest.mark.parametrize("channels", [1, 2, 4, 8])
@pytest.mark.parametrize("axis", [-1, 1])
def test_normalize_last_axis_kernel_matches_whole_frame_exactly(
    monkeypatch: pytest.MonkeyPatch,
    channels: int,
    axis: int,
) -> None:
    source = _source(channels)
    operation = Normalize(8_000, norm=2, axis=axis)
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
        output_dtype=operation.calculate_output_dtype(source.dtype),
    )

    assert channel_wise.shape == source.shape
    assert channel_wise.dtype == np.dtype(np.float64)
    assert channel_wise.chunks == (channels * (1,), (256,))
    channel_values = channel_wise.compute(scheduler="synchronous")
    assert kernel_shapes == channels * [(1, 256)]

    kernel_shapes.clear()
    whole_values = whole_frame.compute(scheduler="synchronous")
    assert kernel_shapes == [(channels, 256)]
    np.testing.assert_array_equal(channel_values, whole_values)


@pytest.mark.parametrize(
    ("axis", "norm"),
    [
        (None, 2),
        (0, 2),
        (-2, 2),
        (-1, None),
    ],
)
def test_normalize_dependent_or_noop_configuration_uses_whole_frame(
    monkeypatch: pytest.MonkeyPatch,
    axis: int | None,
    norm: float | None,
) -> None:
    source = _source(4)
    operation = Normalize(8_000, norm=norm, axis=axis)
    original_process = operation._process
    kernel_shapes: list[tuple[int, ...]] = []

    def observed_process(values: np.ndarray) -> np.ndarray:
        kernel_shapes.append(values.shape)
        return original_process(values)

    monkeypatch.setattr(operation, "_process", observed_process)

    result = operation.process(source)
    result.compute(scheduler="synchronous")

    assert kernel_shapes == [(4, 256)]
    assert result.chunks == ((4,), (256,))


def test_normalize_unhashable_axis_uses_whole_frame_before_kernel_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(2)
    operation = Normalize(8_000, norm=2, axis=[-1])  # ty: ignore[invalid-argument-type]
    original_process = operation._process
    kernel_shapes: list[tuple[int, ...]] = []

    def observed_process(values: np.ndarray) -> np.ndarray:
        kernel_shapes.append(values.shape)
        return original_process(values)

    monkeypatch.setattr(operation, "_process", observed_process)

    result = operation.process(source)
    with pytest.raises(TypeError):
        result.compute(scheduler="synchronous")

    assert kernel_shapes == [(2, 256)]


def test_normalize_zero_channel_input_uses_whole_frame_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = Normalize(8_000, norm=np.inf, axis=-1)
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


def test_normalize_unknown_channel_count_uses_whole_frame_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.arange(8, dtype=float).reshape(2, 4)
    source = da.from_delayed(
        delayed(np.asarray)(values),
        shape=(np.nan, 4),
        dtype=float,
    )
    operation = Normalize(8_000, norm=np.inf, axis=-1)
    original_process = operation._process
    kernel_shapes: list[tuple[int, ...]] = []

    def observed_process(array: np.ndarray) -> np.ndarray:
        kernel_shapes.append(array.shape)
        return original_process(array)

    monkeypatch.setattr(operation, "_process", observed_process)

    result = operation.process(source)
    computed = result.compute(scheduler="synchronous")

    expected = values / np.max(np.abs(values), axis=-1, keepdims=True)
    np.testing.assert_array_equal(computed, expected)
    assert np.isnan(result.shape[0])
    assert result.shape[1:] == (4,)
    assert result.dtype == np.dtype(np.float64)
    assert kernel_shapes == [(2, 4)]


@pytest.mark.parametrize(
    ("dtype", "norm", "threshold", "fill"),
    [
        (np.dtype(np.int16), np.inf, None, None),
        (np.dtype(np.float32), -np.inf, None, None),
        (np.dtype(np.float64), 1, None, None),
        (np.dtype(np.float32), 2, None, None),
        (np.dtype(np.int16), 0, None, False),
        (np.dtype(np.float64), 2, 1e6, True),
    ],
)
def test_normalize_dtype_norm_threshold_fill_matches_whole_frame_exactly(
    dtype: np.dtype[np.generic],
    norm: float,
    threshold: float | None,
    fill: bool | None,
) -> None:
    source = da.from_array(
        np.array([[1, -2, 3, -4], [2, -3, 4, -5]], dtype=dtype),
        chunks=(1, 2),
    )
    operation = Normalize(
        8_000,
        norm=norm,
        axis=-1,
        threshold=threshold,
        fill=fill,
    )

    channel_wise = operation.process(source)
    whole_frame = operation._build_whole_frame_graph(
        source,
        (),
        output_shape=source.shape,
        output_dtype=operation.calculate_output_dtype(source.dtype),
    )

    assert channel_wise.shape == whole_frame.shape == source.shape
    expected_dtype = operation.calculate_output_dtype(source.dtype)
    assert channel_wise.dtype == whole_frame.dtype == expected_dtype
    assert channel_wise.chunks == ((1, 1), (4,))
    np.testing.assert_array_equal(
        channel_wise.compute(scheduler="synchronous"),
        whole_frame.compute(scheduler="synchronous"),
    )
