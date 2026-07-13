import dask.array as da
import numpy as np
import pytest
from scipy.signal import get_window

from wandas.processing import FFT, Cepstrum, Lifter, SpectralEnvelope
from wandas.processing.base import create_operation, get_operation


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

        result = Cepstrum(sr, n_fft=1024, window="boxcar")._process(signal)

        # Impulse -> flat spectrum with magnitude 1 -> log(1) = 0 -> zero cepstrum.
        np.testing.assert_allclose(result, np.zeros_like(result), atol=1e-12)

    def test_cepstrum_matches_direct_real_cepstrum_reference(self) -> None:
        sr = 16000
        n_fft = 1024
        t = np.arange(n_fft) / sr
        signal = np.array([np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 400 * t)], dtype=np.float64)

        result = Cepstrum(sr, n_fft=n_fft, window="boxcar")._process(signal)

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

        result = Lifter(sr, cutoff=0.002, mode="low")._process(data)

        expected = np.zeros_like(data)
        expected[0, 0:3] = data[0, 0:3]
        expected[0, -2:] = data[0, -2:]
        np.testing.assert_allclose(result, expected)

    def test_spectral_envelope_zero_cepstrum_uses_fft_amplitude_scaling(self) -> None:
        sr = 16000
        cepstrum = np.zeros((1, 1024), dtype=np.float64)

        result = SpectralEnvelope(sr)._process(cepstrum)

        expected = np.full((1, 513), 2.0 / np.sum(get_window("hann", 1024)))
        expected[..., 0] *= 0.5
        expected[..., -1] *= 0.5
        np.testing.assert_allclose(result.real, expected, atol=1e-12)
        np.testing.assert_allclose(result.imag, np.zeros((1, 513)), atol=1e-12)

    def test_complete_cepstrum_reconstructs_fft_amplitude_scale(self) -> None:
        rng = np.random.default_rng(42)
        signal = rng.normal(size=(2, 64))

        cepstrum = Cepstrum(16000, n_fft=64, window="boxcar")._process(signal)
        reconstructed = SpectralEnvelope(16000, window="boxcar")._process(cepstrum)
        spectrum = FFT(16000, n_fft=64, window="boxcar")._process(signal)

        np.testing.assert_allclose(reconstructed.real, np.abs(spectrum), rtol=1e-12, atol=1e-12)

    def test_cepstrum_process_preserves_lazy_output(self) -> None:
        sr = 16000
        signal = da.from_array(np.ones((1, 512), dtype=np.float32), chunks=(1, -1))

        result = Cepstrum(sr).process(signal)

        assert isinstance(result, da.Array)
        assert result.dtype == np.float32

    @pytest.mark.parametrize(
        ("input_dtype", "expected_dtype"),
        [(np.float32, np.complex64), (np.float64, np.complex128)],
    )
    def test_spectral_envelope_process_is_lazy_with_complex_dtype(
        self,
        input_dtype: np.dtype,
        expected_dtype: np.dtype,
    ) -> None:
        data = da.from_array(np.zeros((2, 16), dtype=input_dtype), chunks=(1, -1))

        result = SpectralEnvelope(16000).process(data)

        assert isinstance(result, da.Array)
        assert result.shape == (2, 9)
        assert result.dtype == expected_dtype
        expected = np.full((2, 9), 2.0 / np.sum(get_window("hann", 16)), dtype=expected_dtype)
        expected[..., 0] *= 0.5
        expected[..., -1] *= 0.5
        np.testing.assert_allclose(result.compute(), expected)

    def test_lifter_invalid_mode_raises_error(self) -> None:
        with pytest.raises(ValueError, match=r"Invalid lifter mode"):
            Lifter(16000, cutoff=0.002, mode="band")

    @pytest.mark.parametrize("invalid_cutoff", [np.nan, np.inf, -np.inf])
    def test_lifter_rejects_non_finite_cutoff(self, invalid_cutoff: float) -> None:
        with pytest.raises(ValueError, match=r"Invalid lifter cutoff"):
            Lifter(16000, cutoff=invalid_cutoff)

    def test_lifter_too_small_cutoff_for_sampling_rate_raises_error(self) -> None:
        data = np.zeros((1, 16), dtype=np.float64)

        with pytest.raises(ValueError, match=r"Invalid lifter cutoff"):
            Lifter(1000, cutoff=0.0005)._process(data)

    @pytest.mark.parametrize("invalid_n_fft", [0, -1])
    def test_cepstrum_rejects_non_positive_n_fft(self, invalid_n_fft: int) -> None:
        with pytest.raises(ValueError, match=r"Invalid FFT size for cepstrum"):
            Cepstrum(16000, n_fft=invalid_n_fft)

    @pytest.mark.parametrize("invalid_n_fft", [True, 4.5])
    def test_cepstrum_rejects_non_integer_n_fft(self, invalid_n_fft: object) -> None:
        with pytest.raises(TypeError, match=r"n_fft must be a positive integer or None"):
            Cepstrum(16000, n_fft=invalid_n_fft)

    def test_cepstral_operations_accept_numpy_integer_n_fft(self) -> None:
        assert Cepstrum(16000, n_fft=np.int64(16)).n_fft == 16

    def test_cepstrum_rejects_non_positive_floor(self) -> None:
        with pytest.raises(ValueError, match=r"Invalid log floor for cepstrum"):
            Cepstrum(16000, floor=0)

    @pytest.mark.parametrize("invalid_floor", [np.nan, np.inf, -np.inf])
    def test_cepstrum_rejects_non_finite_floor(self, invalid_floor: float) -> None:
        with pytest.raises(ValueError, match=r"Invalid log floor for cepstrum"):
            Cepstrum(16000, floor=invalid_floor)

    def test_cepstrum_truncates_to_explicit_n_fft(self) -> None:
        signal = np.ones((1, 32), dtype=np.float32)

        result = Cepstrum(16000, n_fft=16, window="boxcar")._process(signal)

        assert result.shape == (1, 16)
        assert result.dtype == np.float32

    def test_lifter_high_mode_is_complement_of_low_mode(self) -> None:
        data = np.arange(16, dtype=np.float64).reshape(1, -1)
        low = Lifter(1000, cutoff=0.002, mode="low")._process(data)
        high = Lifter(1000, cutoff=0.002, mode="high")._process(data)

        np.testing.assert_allclose(low + high, data)

    def test_lifter_rejects_cutoff_overlapping_mirrored_region(self) -> None:
        with pytest.raises(ValueError, match=r"overlap the mirrored negative quefrency region"):
            Lifter(1000, cutoff=0.008)._process(np.zeros((1, 16)))

    @pytest.mark.parametrize(
        ("operation", "attribute", "new_value"),
        [
            (Cepstrum(16000, n_fft=16), "n_fft", 32),
            (Cepstrum(16000, window="hann"), "window", "boxcar"),
            (Cepstrum(16000, floor=1e-12), "floor", 1e-6),
            (Lifter(16000, cutoff=0.002), "cutoff", 0.003),
            (Lifter(16000, cutoff=0.002), "mode", "high"),
        ],
    )
    def test_operation_configuration_is_read_only(
        self,
        operation: object,
        attribute: str,
        new_value: object,
    ) -> None:
        with pytest.raises(AttributeError):
            setattr(operation, attribute, new_value)
