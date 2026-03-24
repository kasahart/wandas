import dask.array as da
import numpy as np
import pytest

from wandas.processing import Cepstrum, Lifter, SpectralEnvelope
from wandas.processing.base import create_operation, get_operation

_da_from_array = da.from_array  # type: ignore [unused-ignore]


class TestCepstralOperations:
    def test_cepstrum_display_names_and_registry(self) -> None:
        sr = 16000

        assert Cepstrum(sr).get_display_name() == "ceps"
        assert Lifter(sr, cutoff=0.002).get_display_name() == "lifter"
        assert SpectralEnvelope(sr).get_display_name() == "env"

        assert get_operation("cepstrum") == Cepstrum
        assert get_operation("lifter") == Lifter
        assert get_operation("spectral_envelope") == SpectralEnvelope

        assert isinstance(create_operation("cepstrum", sr), Cepstrum)
        assert isinstance(create_operation("lifter", sr, cutoff=0.002), Lifter)
        assert isinstance(create_operation("spectral_envelope", sr), SpectralEnvelope)

    def test_cepstrum_impulse_returns_zero(self) -> None:
        sr = 16000
        signal = np.zeros((1, 1024), dtype=np.float64)
        signal[0, 0] = 1.0

        result = Cepstrum(sr, n_fft=1024, window="boxcar").process_array(signal).compute()

        # Impulse -> flat spectrum with magnitude 1 -> log(1) = 0 -> zero cepstrum.
        np.testing.assert_allclose(result, np.zeros_like(result), atol=1e-12)

    def test_cepstrum_matches_direct_real_cepstrum_reference(self) -> None:
        sr = 16000
        n_fft = 1024
        t = np.arange(n_fft) / sr
        signal = np.array([np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 400 * t)], dtype=np.float64)

        result = Cepstrum(sr, n_fft=n_fft, window="boxcar").process_array(signal).compute()

        reference_spectrum = np.fft.rfft(signal, n=n_fft, axis=-1)
        reference_log_magnitude = np.log(np.maximum(np.abs(reference_spectrum), 1e-12))
        reference = np.fft.irfft(reference_log_magnitude, n=n_fft, axis=-1)

        np.testing.assert_allclose(result, reference, atol=1e-12)

    def test_lifter_low_mode_keeps_low_and_mirrored_quefrencies(self) -> None:
        sr = 1000
        data = np.zeros((1, 16), dtype=np.float64)
        data[0, 1] = 1.0
        data[0, 14] = 2.0
        data[0, 4] = 3.0

        result = Lifter(sr, cutoff=0.002, mode="low").process_array(data).compute()

        expected = np.zeros_like(data)
        expected[0, 0:3] = data[0, 0:3]
        expected[0, -2:] = data[0, -2:]
        np.testing.assert_allclose(result, expected)

    def test_spectral_envelope_zero_cepstrum_returns_unity_spectrum(self) -> None:
        sr = 16000
        cepstrum = np.zeros((1, 1024), dtype=np.float64)

        result = SpectralEnvelope(sr, n_fft=1024).process_array(cepstrum).compute()

        # Zero cepstrum -> log spectrum of 0 -> exp(0) = 1 across all bins.
        np.testing.assert_allclose(result.real, np.ones((1, 513)), atol=1e-12)
        np.testing.assert_allclose(result.imag, np.zeros((1, 513)), atol=1e-12)

    def test_cepstrum_process_preserves_lazy_output(self) -> None:
        sr = 16000
        signal = _da_from_array(np.ones((1, 512), dtype=np.float32), chunks=(1, -1))

        result = Cepstrum(sr).process(signal)

        assert isinstance(result, da.Array)
        assert result.dtype == np.float32

    def test_lifter_invalid_mode_raises_error(self) -> None:
        with pytest.raises(ValueError, match=r"Invalid lifter mode"):
            Lifter(16000, cutoff=0.002, mode="band")

    def test_lifter_too_small_cutoff_for_sampling_rate_raises_error(self) -> None:
        data = np.zeros((1, 16), dtype=np.float64)

        with pytest.raises(ValueError, match=r"Invalid lifter cutoff"):
            Lifter(1000, cutoff=0.0005).process_array(data).compute()
