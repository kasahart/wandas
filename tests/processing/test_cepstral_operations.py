"""Numerical contracts for real-cepstrum processing operations."""

from unittest import mock

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray
from scipy.signal import get_window

from wandas.processing import create_operation, get_operation
from wandas.processing.cepstral import Cepstrum, Lifter, SpectralEnvelope
from wandas.utils.dask_helpers import da_from_array

_SAMPLING_RATE = 8_000
_LOG_FLOOR = 1e-12


def _normalized_rfft_magnitude(signal: np.ndarray, n_fft: int, window: str) -> np.ndarray:
    """Return the independently calculated Wandas one-sided FFT magnitude."""
    analysis = signal[..., :n_fft]
    window_values = get_window(window, analysis.shape[-1])
    spectrum = np.fft.rfft(analysis * window_values, n=n_fft, axis=-1)
    spectrum[..., 1:-1] *= 2.0
    spectrum /= np.sum(window_values)
    return np.abs(spectrum)


def test_cepstral_operations_registry_uses_stable_public_keys() -> None:
    assert get_operation("cepstrum") is Cepstrum
    assert get_operation("lifter") is Lifter
    assert get_operation("spectral_envelope") is SpectralEnvelope
    assert isinstance(create_operation("cepstrum", _SAMPLING_RATE), Cepstrum)


def test_cepstrum_process_builds_lazy_shape_and_dtype_without_compute() -> None:
    signal = da_from_array(np.ones((2, 24), dtype=np.float32), chunks=(1, -1))

    with mock.patch.object(DaArray, "compute") as compute:
        result = Cepstrum(_SAMPLING_RATE, n_fft=32).process(signal)

    compute.assert_not_called()
    assert isinstance(result, da.Array)
    assert result.shape == (2, 32)
    assert result.dtype == np.float64


def test_cepstrum_matches_real_cepstrum_definition_for_padded_signal() -> None:
    time = np.arange(24, dtype=float) / _SAMPLING_RATE
    signal = (np.sin(2 * np.pi * 500 * time) + 0.25 * np.cos(2 * np.pi * 1_000 * time))[None, :]
    n_fft = 32

    result = Cepstrum(_SAMPLING_RATE, n_fft=n_fft, window="boxcar", floor=_LOG_FLOOR)._process(signal)
    magnitude = _normalized_rfft_magnitude(signal, n_fft, "boxcar")
    expected = np.fft.irfft(np.log(np.maximum(magnitude, _LOG_FLOOR)), n=n_fft, axis=-1)

    # The custom transform is defined directly by NumPy's FFT equations.
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("signal_length", "n_fft"), [(16, 16), (9, 13), (24, 16)])
def test_spectral_envelope_reconstructs_normalized_fft_magnitude(
    signal_length: int,
    n_fft: int,
) -> None:
    time = np.arange(signal_length, dtype=float) / _SAMPLING_RATE
    signal = (0.75 * np.cos(2 * np.pi * 500 * time) + 0.2)[None, :]

    cepstrum = Cepstrum(_SAMPLING_RATE, n_fft=n_fft, window="boxcar")._process(signal)
    envelope = SpectralEnvelope(_SAMPLING_RATE)._process(cepstrum)
    expected = _normalized_rfft_magnitude(signal, n_fft, "boxcar")

    # FFT/IFFT round-off is the only expected error in the analytical round trip.
    np.testing.assert_allclose(envelope.real, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(envelope.imag, np.zeros_like(envelope.imag))


def test_lifter_low_and_high_modes_partition_symmetric_cepstrum() -> None:
    coefficients = np.arange(16, dtype=float)[None, :]
    cutoff = 2 / _SAMPLING_RATE

    low = Lifter(_SAMPLING_RATE, cutoff=cutoff, mode="low")._process(coefficients)
    high = Lifter(_SAMPLING_RATE, cutoff=cutoff, mode="high")._process(coefficients)

    np.testing.assert_array_equal(low + high, coefficients)
    np.testing.assert_array_equal(np.flatnonzero(low[0]), np.array([1, 2, 14, 15]))


@pytest.mark.parametrize("invalid_n_fft", [True, 1.5])
def test_cepstrum_non_integral_n_fft_raises_type_error(invalid_n_fft: object) -> None:
    with pytest.raises(TypeError, match=r"Invalid FFT size for cepstrum"):
        Cepstrum(_SAMPLING_RATE, n_fft=invalid_n_fft)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("invalid_n_fft", [0, -1])
def test_cepstrum_non_positive_n_fft_raises_value_error(invalid_n_fft: int) -> None:
    with pytest.raises(ValueError, match=r"Invalid FFT size for cepstrum"):
        Cepstrum(_SAMPLING_RATE, n_fft=invalid_n_fft)


@pytest.mark.parametrize("invalid_floor", [0.0, -1.0, np.nan, np.inf])
def test_cepstrum_invalid_log_floor_raises_value_error(invalid_floor: float) -> None:
    with pytest.raises(ValueError, match=r"Invalid log floor for cepstrum"):
        Cepstrum(_SAMPLING_RATE, floor=invalid_floor)


def test_cepstrum_complex_input_raises_type_error() -> None:
    with pytest.raises(TypeError, match=r"real-valued input"):
        Cepstrum(_SAMPLING_RATE)._process(np.ones((1, 8), dtype=np.complex128))  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_cutoff", [0.0, -1.0, np.nan, np.inf])
def test_lifter_invalid_cutoff_raises_value_error(invalid_cutoff: float) -> None:
    with pytest.raises(ValueError, match=r"Invalid lifter cutoff"):
        Lifter(_SAMPLING_RATE, cutoff=invalid_cutoff)


def test_lifter_unknown_mode_raises_value_error() -> None:
    with pytest.raises(ValueError, match=r"Invalid lifter mode"):
        Lifter(_SAMPLING_RATE, cutoff=0.001, mode="band")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("cutoff", [0.5 / _SAMPLING_RATE, 8 / _SAMPLING_RATE])
def test_lifter_unrepresentable_cutoff_raises_value_error(cutoff: float) -> None:
    with pytest.raises(ValueError, match=r"Invalid lifter cutoff"):
        Lifter(_SAMPLING_RATE, cutoff=cutoff)._process(np.zeros((1, 16)))


def test_spectral_envelope_asymmetric_cepstrum_raises_value_error() -> None:
    coefficients = np.zeros((1, 8))
    coefficients[..., 1] = 1.0

    with pytest.raises(ValueError, match=r"symmetric real cepstral coefficients"):
        SpectralEnvelope(_SAMPLING_RATE)._process(coefficients)
