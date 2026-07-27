"""Channel-wise execution contracts for the N-octave spectrum operation."""

from typing import Any
from unittest import mock

import dask.array as da
import numpy as np
import pytest
from dask import delayed
from dask.array.core import Array as DaArray
from mosqito.sound_level_meter import noct_spectrum

import wandas.processing.spectral as spectral_module
from wandas.processing.base import AudioOperation, ChannelIndependentAudioOperation
from wandas.processing.spectral import NOctSpectrum, NOctSynthesis, _NOctBase
from wandas.utils.dask_helpers import da_from_array

_SAMPLING_RATE = 8_192
_SAMPLES = 2_048
_FMIN = 250.0
_FMAX = 2_000.0
_G = 10
_FR = 1_000


def _values(channels: int, dtype: np.dtype[Any]) -> np.ndarray:
    time = np.arange(_SAMPLES, dtype=np.float64) / _SAMPLING_RATE
    values = np.stack(
        [
            0.55 * np.sin(2 * np.pi * (375 + 125 * index) * time)
            + 0.25 * np.cos(2 * np.pi * (750 + 125 * index) * time)
            + 0.05 * ((np.arange(_SAMPLES) + 7 * index) % 31)
            for index in range(channels)
        ]
    )
    if np.issubdtype(dtype, np.integer):
        return np.rint(values * 2_048).astype(dtype)
    return values.astype(dtype)


def _direct_mosqito(values: np.ndarray, *, n: int) -> np.ndarray:
    spectrum, _ = noct_spectrum(
        sig=values.T,
        fs=_SAMPLING_RATE,
        fmin=_FMIN,
        fmax=_FMAX,
        n=n,
        G=_G,
        fr=_FR,
    )
    spectrum = np.asarray(spectrum)
    return np.expand_dims(spectrum, axis=0) if spectrum.ndim == 1 else spectrum.T


def _wandas_operation_task_keys(result: DaArray) -> tuple[str, ...]:
    return tuple(sorted(repr(key) for key in result.dask if "_execute_wandas_operation" in repr(key)))


def _patch_fixed_bands(monkeypatch: pytest.MonkeyPatch, bands: int = 4) -> None:
    center_frequencies = np.arange(1, bands + 1, dtype=np.float64)
    monkeypatch.setattr(
        spectral_module,
        "_center_freq",
        lambda **_kwargs: (np.arange(bands), center_frequencies),
    )


def test_noct_spectrum_alone_declares_the_channel_independent_mro() -> None:
    assert NOctSpectrum.__mro__[:4] == (
        NOctSpectrum,
        _NOctBase,
        ChannelIndependentAudioOperation,
        AudioOperation,
    )
    assert NOctSynthesis.__mro__[:3] == (
        NOctSynthesis,
        _NOctBase,
        AudioOperation,
    )
    assert not issubclass(NOctSynthesis, ChannelIndependentAudioOperation)


@pytest.mark.parametrize("channels", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", [np.dtype(np.int16), np.dtype(np.float32), np.dtype(np.float64)])
@pytest.mark.parametrize("n", [1, 3])
def test_noct_spectrum_channel_wise_matches_whole_frame_and_mosqito_exactly(
    monkeypatch: pytest.MonkeyPatch,
    channels: int,
    dtype: np.dtype[Any],
    n: int,
) -> None:
    values = _values(channels, dtype)
    values_before = values.copy()
    source = da_from_array(values, chunks=(1, 512))
    operation = NOctSpectrum(
        _SAMPLING_RATE,
        fmin=_FMIN,
        fmax=_FMAX,
        n=n,
        G=_G,
        fr=_FR,
    )
    original_process = operation._process
    kernel_shapes: list[tuple[int, ...]] = []

    def observed_process(channel_values: np.ndarray) -> np.ndarray:
        kernel_shapes.append(channel_values.shape)
        return original_process(channel_values)

    monkeypatch.setattr(operation, "_process", observed_process)
    output_shape = operation.calculate_output_shape(source.shape)
    output_dtype = operation.calculate_output_dtype(source.dtype)

    with mock.patch.object(DaArray, "compute") as compute:
        channel_wise = operation.process(source)
        compute.assert_not_called()

    whole_frame = operation._build_whole_frame_graph(
        source,
        (),
        output_shape=output_shape,
        output_dtype=output_dtype,
    )

    assert isinstance(channel_wise, DaArray)
    assert output_shape[0] == channels
    assert channel_wise.shape == output_shape
    assert channel_wise.dtype == np.dtype(np.float64)
    assert channel_wise.chunks == (channels * (1,), (output_shape[1],))

    channel_values = channel_wise.compute(scheduler="synchronous")
    assert kernel_shapes == channels * [(1, _SAMPLES)]

    kernel_shapes.clear()
    whole_values = whole_frame.compute(scheduler="synchronous")
    assert kernel_shapes == [(channels, _SAMPLES)]

    expected = _direct_mosqito(values, n=n)
    assert expected.dtype == np.dtype(np.float64)
    assert channel_values.dtype == np.dtype(np.float64)
    assert channel_values.shape == expected.shape
    np.testing.assert_array_equal(channel_values, whole_values)
    np.testing.assert_array_equal(channel_values, expected)
    np.testing.assert_array_equal(values, values_before)


@pytest.mark.parametrize("input_dtype", [np.dtype(np.int16), np.dtype(np.float32), np.dtype(np.float64)])
def test_noct_spectrum_advertises_its_actual_float64_dtype_without_changing_shared_base(
    input_dtype: np.dtype[Any],
) -> None:
    operation = NOctSpectrum(
        _SAMPLING_RATE,
        fmin=_FMIN,
        fmax=_FMAX,
        n=3,
        G=_G,
        fr=_FR,
    )

    assert "calculate_output_dtype" in NOctSpectrum.__dict__
    assert "calculate_output_dtype" not in _NOctBase.__dict__
    assert operation.calculate_output_dtype(input_dtype) == np.dtype(np.float64)


def test_noct_spectrum_repeated_graphs_use_pure_stable_per_channel_keys() -> None:
    source = da_from_array(_values(4, np.dtype(np.float64)), chunks=(1, 512))
    operation = NOctSpectrum(
        _SAMPLING_RATE,
        fmin=_FMIN,
        fmax=_FMAX,
        n=3,
        G=_G,
        fr=_FR,
    )

    first = operation.process(source)
    second = operation.process(source)

    first_keys = _wandas_operation_task_keys(first)
    assert operation.pure is True
    assert len(first_keys) == 4
    assert first_keys == _wandas_operation_task_keys(second)


def test_noct_spectrum_zero_channel_count_uses_whole_frame_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bands = 4
    source = da.from_array(
        np.empty((0, _SAMPLES), dtype=np.float64),
        chunks=(0, _SAMPLES),
    )
    operation = NOctSpectrum(
        _SAMPLING_RATE,
        fmin=_FMIN,
        fmax=_FMAX,
        n=3,
        G=_G,
        fr=_FR,
    )
    kernel_shapes: list[tuple[int, ...]] = []
    _patch_fixed_bands(monkeypatch, bands)
    monkeypatch.setattr(operation, "ensure_dependencies", lambda: None)

    def fake_process(values: np.ndarray) -> np.ndarray:
        kernel_shapes.append(values.shape)
        return np.empty((values.shape[0], bands), dtype=np.float64)

    monkeypatch.setattr(operation, "_process", fake_process)
    result = operation.process(source)

    np.testing.assert_array_equal(
        result.compute(scheduler="synchronous"),
        np.empty((0, bands), dtype=np.float64),
    )
    assert kernel_shapes == [(0, _SAMPLES)]
    assert result.chunks == ((0,), (bands,))


def test_noct_spectrum_unknown_channel_count_uses_whole_frame_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bands = 4
    values = _values(3, np.dtype(np.float64))
    source = da.from_delayed(
        delayed(np.asarray)(values),
        shape=(np.nan, _SAMPLES),
        dtype=values.dtype,
    )
    operation = NOctSpectrum(
        _SAMPLING_RATE,
        fmin=_FMIN,
        fmax=_FMAX,
        n=3,
        G=_G,
        fr=_FR,
    )
    kernel_shapes: list[tuple[int, ...]] = []
    _patch_fixed_bands(monkeypatch, bands)
    monkeypatch.setattr(operation, "ensure_dependencies", lambda: None)

    def fake_process(input_values: np.ndarray) -> np.ndarray:
        kernel_shapes.append(input_values.shape)
        return np.zeros((input_values.shape[0], bands), dtype=np.float64)

    monkeypatch.setattr(operation, "_process", fake_process)
    result = operation.process(source)

    np.testing.assert_array_equal(
        result.compute(scheduler="synchronous"),
        np.zeros((3, bands), dtype=np.float64),
    )
    assert kernel_shapes == [(3, _SAMPLES)]


def test_noct_spectrum_rejects_extra_runtime_input_before_graph_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = da_from_array(_values(2, np.dtype(np.float64)), chunks=(1, 512))
    operation = NOctSpectrum(
        _SAMPLING_RATE,
        fmin=_FMIN,
        fmax=_FMAX,
        n=3,
        G=_G,
        fr=_FR,
    )
    monkeypatch.setattr(operation, "ensure_dependencies", lambda: None)

    with pytest.raises(ValueError, match=r"Expected exactly one input for NOctSpectrum"):
        operation.process(source, source)


def test_noct_spectrum_channel_count_changing_output_uses_whole_frame_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bands = 4
    source = da_from_array(_values(2, np.dtype(np.float64)), chunks=(1, 512))
    operation = NOctSpectrum(
        _SAMPLING_RATE,
        fmin=_FMIN,
        fmax=_FMAX,
        n=3,
        G=_G,
        fr=_FR,
    )
    kernel_shapes: list[tuple[int, ...]] = []
    monkeypatch.setattr(operation, "ensure_dependencies", lambda: None)
    monkeypatch.setattr(operation, "calculate_output_shape", lambda _shape: (3, bands))

    def fake_process(values: np.ndarray) -> np.ndarray:
        kernel_shapes.append(values.shape)
        return np.zeros((3, bands), dtype=np.float64)

    monkeypatch.setattr(operation, "_process", fake_process)
    result = operation.process(source)

    np.testing.assert_array_equal(
        result.compute(scheduler="synchronous"),
        np.zeros((3, bands), dtype=np.float64),
    )
    assert kernel_shapes == [(2, _SAMPLES)]


def test_noct_spectrum_dependency_failure_precedes_graph_and_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_error = ImportError('Install it with: pip install "wandas[psychoacoustic]"')
    calls: list[str] = []

    def raise_import_error(feature: str) -> None:
        calls.append(feature)
        raise original_error

    operation = NOctSpectrum(
        _SAMPLING_RATE,
        fmin=_FMIN,
        fmax=_FMAX,
        n=3,
        G=_G,
        fr=_FR,
    )
    monkeypatch.setattr(spectral_module, "require_mosqito_center_freq", raise_import_error)
    monkeypatch.setattr(operation, "_process", lambda _data: pytest.fail("kernel ran without MoSQITo"))

    with pytest.raises(ImportError) as exc_info:
        operation.process(
            da_from_array(_values(2, np.dtype(np.float64)), chunks=(1, 512)),
        )

    assert exc_info.value is original_error
    assert calls == ["NOctFrame"]


def test_noct_synthesis_keeps_one_whole_frame_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bands = 4
    source = da_from_array(_values(2, np.dtype(np.float64)), chunks=(1, 512))
    operation = NOctSynthesis(
        _SAMPLING_RATE,
        fmin=_FMIN,
        fmax=_FMAX,
        n=3,
        G=_G,
        fr=_FR,
    )
    kernel_shapes: list[tuple[int, ...]] = []
    _patch_fixed_bands(monkeypatch, bands)
    monkeypatch.setattr(operation, "ensure_dependencies", lambda: None)

    def fake_process(values: np.ndarray) -> np.ndarray:
        kernel_shapes.append(values.shape)
        return np.zeros((values.shape[0], bands), dtype=values.dtype)

    monkeypatch.setattr(operation, "_process", fake_process)
    result = operation.process(source)

    np.testing.assert_array_equal(
        result.compute(scheduler="synchronous"),
        np.zeros((2, bands), dtype=np.float64),
    )
    assert kernel_shapes == [(2, _SAMPLES)]
