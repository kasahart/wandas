"""Channel-wise execution contracts for the librosa HPSS family."""

from typing import Any
from unittest import mock

import dask.array as da
import numpy as np
import pytest
from dask import delayed
from dask.array.core import Array as DaArray
from librosa.util.exceptions import ParameterError

import wandas.processing.effects as effects_module
from wandas.processing.base import ChannelIndependentAudioOperation
from wandas.processing.effects import HpssHarmonic, HpssPercussive
from wandas.utils.dask_helpers import da_from_array
from wandas.utils.optional_imports import require_librosa_effects

_SAMPLING_RATE = 8_000
_SAMPLES = 512
_PARAMS = {
    "kernel_size": (7, 9),
    "power": 2.0,
    "margin": (1.0, 2.0),
    "n_fft": 64,
    "hop_length": 16,
    "win_length": 64,
    "window": "hann",
    "center": True,
    "pad_mode": "constant",
}
_HPSS_CLASSES = (HpssHarmonic, HpssPercussive)


def _values(channels: int, dtype: np.dtype[Any]) -> np.ndarray:
    time = np.arange(_SAMPLES, dtype=np.float64) / _SAMPLING_RATE
    values = np.stack(
        [
            np.sin(2 * np.pi * (220 + 40 * index) * time)
            + 0.35 * np.sin(2 * np.pi * (1_200 + 30 * index) * time)
            + 0.5 * ((np.arange(_SAMPLES) + 7 * index) % 97 == 0)
            for index in range(channels)
        ]
    )
    if np.issubdtype(dtype, np.integer):
        values *= 1_000
    return values.astype(dtype)


def _source(channels: int, dtype: np.dtype[Any]) -> DaArray:
    return da_from_array(_values(channels, dtype), chunks=(1, 128))


def _operation_keys(result: DaArray) -> tuple[str, ...]:
    return tuple(sorted(repr(key) for key in result.dask if "_execute_wandas_operation" in repr(key)))


@pytest.mark.parametrize("operation_class", _HPSS_CLASSES)
def test_hpss_family_declares_channel_independent_contract(operation_class: type[Any]) -> None:
    assert issubclass(operation_class, ChannelIndependentAudioOperation)


@pytest.mark.parametrize("operation_class", _HPSS_CLASSES)
@pytest.mark.parametrize("channels", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "dtype",
    [np.dtype(np.float32), np.dtype(np.float64)],
    ids=["float32", "float64"],
)
def test_hpss_channel_wise_kernel_matches_whole_frame_and_librosa_exactly(
    monkeypatch: pytest.MonkeyPatch,
    operation_class: type[Any],
    channels: int,
    dtype: np.dtype[Any],
) -> None:
    values = _values(channels, dtype)
    values_before = values.copy()
    source = da_from_array(values, chunks=(1, 128))
    operation = operation_class(_SAMPLING_RATE, **_PARAMS)
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
        output_shape=source.shape,
        output_dtype=dtype,
    )

    assert isinstance(channel_wise, DaArray)
    assert channel_wise.shape == whole_frame.shape == values.shape
    assert channel_wise.dtype == whole_frame.dtype == dtype
    assert channel_wise.chunks == (channels * (1,), (_SAMPLES,))
    assert kernel_shapes == []

    channel_values = channel_wise.compute(scheduler="synchronous")
    assert kernel_shapes == channels * [(1, _SAMPLES)]

    kernel_shapes.clear()
    whole_values = whole_frame.compute(scheduler="synchronous")
    assert kernel_shapes == [(channels, _SAMPLES)]

    librosa_effects = require_librosa_effects(operation.name)
    authority = getattr(librosa_effects, operation._extract_func)(values, **_PARAMS)
    np.testing.assert_array_equal(channel_values, whole_values)
    np.testing.assert_array_equal(channel_values, authority)
    np.testing.assert_array_equal(values, values_before)


@pytest.mark.parametrize("operation_class", _HPSS_CLASSES)
def test_hpss_scalar_parameters_match_librosa_authority_exactly(operation_class: type[Any]) -> None:
    params = {
        "kernel_size": 7,
        "power": 1.0,
        "margin": 1.0,
        "n_fft": 64,
        "hop_length": 16,
        "win_length": 64,
        "window": "hann",
        "center": False,
        "pad_mode": "constant",
    }
    values = _values(2, np.dtype(np.float64))
    operation = operation_class(_SAMPLING_RATE, **params)

    actual = operation.process(da_from_array(values, chunks=(1, -1))).compute(scheduler="synchronous")
    librosa_effects = require_librosa_effects(operation.name)
    expected = getattr(librosa_effects, operation._extract_func)(values, **params)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("operation_class", _HPSS_CLASSES)
@pytest.mark.parametrize(
    ("invalid_params", "expected_message"),
    [
        pytest.param(
            {"margin": 0.5},
            "Margins must be >= 1.0. A typical range is between 1 and 10.",
            id="margin-below-one",
        ),
        pytest.param(
            {"power": 0},
            "power must be strictly positive",
            id="non-positive-power",
        ),
        pytest.param(
            {"hop_length": 0},
            "hop_length=0 must be a positive integer",
            id="non-positive-hop-length",
        ),
    ],
)
def test_hpss_invalid_parameters_match_librosa_and_whole_frame_exception_exactly(
    operation_class: type[Any],
    invalid_params: dict[str, Any],
    expected_message: str,
) -> None:
    params = {**_PARAMS, **invalid_params}
    values = _values(2, np.dtype(np.float64))
    source = da_from_array(values, chunks=(1, -1))
    operation = operation_class(_SAMPLING_RATE, **params)
    librosa_effects = require_librosa_effects(operation.name)

    with pytest.raises(ParameterError) as authority_error:
        getattr(librosa_effects, operation._extract_func)(values, **params)
    assert str(authority_error.value) == expected_message

    channel_wise = operation.process(source)
    whole_frame = operation._build_whole_frame_graph(
        source,
        (),
        output_shape=source.shape,
        output_dtype=source.dtype,
    )

    with pytest.raises(ParameterError) as channel_error:
        channel_wise.compute(scheduler="synchronous")
    with pytest.raises(ParameterError) as whole_error:
        whole_frame.compute(scheduler="synchronous")

    assert type(channel_error.value) is type(authority_error.value)
    assert type(whole_error.value) is type(authority_error.value)
    assert str(channel_error.value) == str(authority_error.value)
    assert str(whole_error.value) == str(authority_error.value)


@pytest.mark.parametrize("operation_class", _HPSS_CLASSES)
@pytest.mark.parametrize("channels", [1, 2, 4, 8])
def test_hpss_integer_input_keeps_whole_frame_exception(
    operation_class: type[Any],
    channels: int,
) -> None:
    source = _source(channels, np.dtype(np.int16))
    operation = operation_class(_SAMPLING_RATE, **_PARAMS)
    channel_wise = operation.process(source)
    whole_frame = operation._build_whole_frame_graph(
        source,
        (),
        output_shape=source.shape,
        output_dtype=source.dtype,
    )

    with pytest.raises(ParameterError) as channel_error:
        channel_wise.compute(scheduler="synchronous")
    with pytest.raises(ParameterError) as whole_error:
        whole_frame.compute(scheduler="synchronous")

    assert type(channel_error.value) is type(whole_error.value)
    assert type(channel_error.value).__name__ == "ParameterError"
    assert str(channel_error.value) == str(whole_error.value) == "Audio data must be floating-point"


@pytest.mark.parametrize("operation_class", _HPSS_CLASSES)
def test_hpss_repeated_graphs_have_stable_pure_task_keys(operation_class: type[Any]) -> None:
    source = _source(4, np.dtype(np.float64))
    operation = operation_class(_SAMPLING_RATE, **_PARAMS)

    with mock.patch("wandas.processing.base.delayed", wraps=delayed) as delayed_builder:
        first = operation.process(source)
        second = operation.process(source)

    assert operation.pure is True
    assert first.name == second.name
    assert first.__dask_keys__() == second.__dask_keys__()
    assert _operation_keys(first) == _operation_keys(second)
    assert len(_operation_keys(first)) == 4
    assert len(delayed_builder.call_args_list) == 8
    assert all(call.kwargs["pure"] is True for call in delayed_builder.call_args_list)


@pytest.mark.parametrize("operation_class", _HPSS_CLASSES)
def test_hpss_zero_channel_input_uses_whole_frame_fallback(
    monkeypatch: pytest.MonkeyPatch,
    operation_class: type[Any],
) -> None:
    operation = operation_class(_SAMPLING_RATE, **_PARAMS)
    kernel_shapes: list[tuple[int, ...]] = []

    def observed_process(values: np.ndarray) -> np.ndarray:
        kernel_shapes.append(values.shape)
        return values

    monkeypatch.setattr(operation, "_process", observed_process)
    source = da.from_array(np.empty((0, _SAMPLES), dtype=np.float64), chunks=(0, _SAMPLES))
    result = operation.process(source)

    assert result.chunks == ((0,), (_SAMPLES,))
    np.testing.assert_array_equal(
        result.compute(scheduler="synchronous"),
        np.empty((0, _SAMPLES), dtype=np.float64),
    )
    assert kernel_shapes == [(0, _SAMPLES)]


@pytest.mark.parametrize("operation_class", _HPSS_CLASSES)
def test_hpss_unknown_channel_count_uses_whole_frame_fallback(
    monkeypatch: pytest.MonkeyPatch,
    operation_class: type[Any],
) -> None:
    values = _values(3, np.dtype(np.float64))
    source = da.from_delayed(
        delayed(np.asarray)(values),
        shape=(np.nan, _SAMPLES),
        dtype=values.dtype,
    )
    operation = operation_class(_SAMPLING_RATE, **_PARAMS)
    kernel_shapes: list[tuple[int, ...]] = []

    def observed_process(input_values: np.ndarray) -> np.ndarray:
        kernel_shapes.append(input_values.shape)
        return input_values

    monkeypatch.setattr(operation, "_process", observed_process)
    result = operation.process(source)

    assert np.isnan(result.shape[0])
    np.testing.assert_array_equal(result.compute(scheduler="synchronous"), values)
    assert kernel_shapes == [(3, _SAMPLES)]


@pytest.mark.parametrize("operation_class", _HPSS_CLASSES)
def test_hpss_rejects_extra_runtime_input_before_graph_construction(operation_class: type[Any]) -> None:
    source = _source(2, np.dtype(np.float64))

    with pytest.raises(
        ValueError,
        match=rf"Expected exactly one input for {operation_class.__name__}; got 2",
    ):
        operation_class(_SAMPLING_RATE, **_PARAMS).process(source, source)


@pytest.mark.parametrize("operation_class", _HPSS_CLASSES)
def test_hpss_missing_optional_dependency_keeps_initialization_error(
    monkeypatch: pytest.MonkeyPatch,
    operation_class: type[Any],
) -> None:
    def raise_missing_librosa(feature: str) -> None:
        assert feature == operation_class.name
        raise ImportError(f"{feature} requires optional dependency 'librosa.effects'.")

    monkeypatch.setattr(effects_module, "require_librosa_effects", raise_missing_librosa)

    with pytest.raises(
        ImportError,
        match=rf"{operation_class.name} requires optional dependency 'librosa.effects'",
    ):
        operation_class(_SAMPLING_RATE, **_PARAMS)
