from collections.abc import Callable
from dataclasses import dataclass
from unittest import mock

import numpy as np
import pytest
from dask.array.core import Array as DaArray
from mosqito.sound_level_meter import noct_spectrum, noct_synthesis
from mosqito.sound_level_meter.noct_spectrum._center_freq import _center_freq
from scipy import signal as ss
from scipy.signal import ShortTimeFFT as ScipySTFT
from scipy.signal import get_window

import wandas.processing.spectral as spectral_module
from tests.processing_helpers import as_operation_dask, run_operation_eager, run_operation_lazy
from wandas.processing.base import create_operation, get_operation
from wandas.processing.spectral import (
    CSD,
    FFT,
    IFFT,
    ISTFT,
    STFT,
    Coherence,
    NOctSpectrum,
    NOctSynthesis,
    TransferFunction,
    Welch,
)
from wandas.utils.dask_helpers import da_from_array
from wandas.utils.types import NDArrayComplex, NDArrayReal

_SR: int = 16000


class TestGetDisplayNames:
    """Test display names for all spectral operations."""

    def test_all_display_names(self) -> None:
        """Test that all operations have appropriate display names."""
        sr = 16000
        assert FFT(sr).get_display_name() == "FFT"
        assert IFFT(sr).get_display_name() == "iFFT"
        assert STFT(sr).get_display_name() == "STFT"
        assert ISTFT(sr).get_display_name() == "iSTFT"
        assert Welch(sr).get_display_name() == "Welch"
        assert NOctSpectrum(sr, 24, 12600).get_display_name() == "Oct"
        assert NOctSynthesis(sr, 24, 12600).get_display_name() == "Octs"
        assert Coherence(sr).get_display_name() == "Coh"
        assert CSD(sr).get_display_name() == "CSD"
        assert TransferFunction(sr).get_display_name() == "H"


class TestFFTOperation:
    """FFT operation: Layer 1 + Layer 2 + Layer 3 (np.fft.rfft reference)."""

    _N_FFT: int = 1024
    _WINDOW: str = "hann"
    _FREQ: float = 500.0

    # -- Layer 1: Unit tests -----------------------------------------------

    def test_fft_init_default_params(self) -> None:
        """Test FFT default n_fft=None, window='hann'."""
        fft = FFT(_SR)
        assert fft.sampling_rate == _SR
        assert fft.n_fft is None
        assert fft.window == "hann"

    def test_fft_init_custom_params(self) -> None:
        """Test FFT stores custom n_fft and window."""
        fft = FFT(_SR, n_fft=2048, window="hamming")
        assert fft.n_fft == 2048
        assert fft.window == "hamming"

    def test_fft_to_params_returns_lineage_parameters(self) -> None:
        fft = FFT(_SR, n_fft=2048, window="hamming")

        assert fft.to_params() == {"n_fft": 2048, "window": "hamming"}

    def test_fft_registry_returns_correct_class(self) -> None:
        """Test FFT is registered as 'fft'."""
        assert get_operation("fft") == FFT
        fft_op = create_operation("fft", _SR, n_fft=512, window="hamming")
        assert isinstance(fft_op, FFT)
        assert fft_op.n_fft == 512
        assert fft_op.window == "hamming"

    def test_fft_negative_n_fft_raises(self) -> None:
        """Test negative n_fft provides WHAT/WHY/HOW error."""
        with pytest.raises(ValueError) as exc_info:
            FFT(sampling_rate=44100, n_fft=-1024)
        error_msg = str(exc_info.value)
        assert "Invalid FFT size" in error_msg
        assert "-1024" in error_msg
        assert "Positive integer" in error_msg
        assert "Common values:" in error_msg

    def test_fft_zero_n_fft_raises(self) -> None:
        """Test zero n_fft raises error."""
        with pytest.raises(ValueError, match="Invalid FFT size"):
            FFT(sampling_rate=44100, n_fft=0)

    # -- Layer 2: Domain (shape + immutability + lazy) ---------------------

    def test_fft_preserves_immutability_and_dask_type(self, pure_sine_440hz_dask: tuple[DaArray, int]) -> None:
        """Input unchanged after FFT; result is DaArray."""
        dask_input, sr = pure_sine_440hz_dask
        fft = FFT(sr, n_fft=self._N_FFT, window=self._WINDOW)
        input_copy = dask_input.compute().copy()

        result_da = fft.process(dask_input)

        assert result_da is not dask_input
        np.testing.assert_array_equal(dask_input.compute(), input_copy)
        assert isinstance(result_da, DaArray)

    def test_fft_lazy_execution_not_computed_until_needed(self, pure_sine_440hz_dask: tuple[DaArray, int]) -> None:
        """FFT builds Dask graph without computing until .compute()."""
        dask_input, sr = pure_sine_440hz_dask
        fft = FFT(sr, n_fft=self._N_FFT, window=self._WINDOW)

        with mock.patch.object(DaArray, "compute") as mock_compute:
            result = fft.process(dask_input)
            mock_compute.assert_not_called()
            assert isinstance(result, DaArray)
            _ = result.compute()
            mock_compute.assert_called_once()

    def test_fft_shape_mono(self, pure_sine_440hz_dask: tuple[DaArray, int]) -> None:
        """FFT output shape: (1, n_fft//2 + 1) for mono."""
        dask_input, sr = pure_sine_440hz_dask
        fft = FFT(sr, n_fft=self._N_FFT, window=self._WINDOW)

        result = fft.process(dask_input).compute()
        expected_freqs = self._N_FFT // 2 + 1
        assert result.shape == (1, expected_freqs)

    def test_fft_shape_stereo(self, stereo_sine_440_880hz_dask: tuple[DaArray, int]) -> None:
        """FFT output shape: (2, n_fft//2 + 1) for stereo."""
        dask_input, sr = stereo_sine_440_880hz_dask
        fft = FFT(sr, n_fft=self._N_FFT, window=self._WINDOW)

        result = fft.process(dask_input).compute()
        expected_freqs = self._N_FFT // 2 + 1
        assert result.shape == (2, expected_freqs)

    def test_fft_truncation_longer_than_n_fft(self) -> None:
        """FFT truncates signal longer than n_fft."""
        long_signal = np.random.default_rng(42).standard_normal(2048)
        fft_op = FFT(_SR, n_fft=1024)
        result = run_operation_eager(fft_op, np.array([long_signal]))
        assert result.shape == (1, 1024 // 2 + 1)

    # -- Layer 3: Theoretical / numpy reference ----------------------------

    def test_fft_peak_at_correct_frequency_bin(self) -> None:
        """FFT peak at expected frequency bin for 500 Hz sine.

        Tolerance: peak index within +-1 bin of theoretical.
        """
        t = np.linspace(0, 1, _SR, endpoint=False)
        sig = np.array([4.0 * np.sin(2 * np.pi * self._FREQ * t)])
        fft = FFT(_SR, n_fft=self._N_FFT, window=self._WINDOW)

        result = run_operation_eager(fft, sig)
        freq_bins = np.fft.rfftfreq(self._N_FFT, 1.0 / _SR)
        target_idx = np.argmin(np.abs(freq_bins - self._FREQ))
        magnitude = np.abs(result[0])
        peak_idx = np.argmax(magnitude)

        assert abs(peak_idx - target_idx) <= 1

        # Non-peak region should be ≪ peak
        mask = np.ones_like(magnitude, dtype=bool)
        lower = max(0, peak_idx - 5)
        upper = min(len(magnitude), peak_idx + 6)
        mask[lower:upper] = False
        assert np.max(magnitude[mask]) < 0.1 * magnitude[peak_idx]

    def test_fft_amplitude_matches_numpy_rfft_reference(self) -> None:
        """FFT amplitude matches manually windowed np.fft.rfft.

        Tolerance: rtol=1e-10 — float64 precision.
        """

        amp = 2.0
        t = np.linspace(0, 1, _SR, endpoint=False)
        cos_wave = amp * np.cos(2 * np.pi * self._FREQ * t)

        fft_inst = FFT(_SR, n_fft=None, window=self._WINDOW)
        fft_result = run_operation_eager(fft_inst, np.array([cos_wave]))

        win = get_window(self._WINDOW, len(cos_wave))
        scaled_cos = cos_wave * win
        scaling_factor = np.sum(win)
        expected_fft: NDArrayComplex = np.fft.rfft(scaled_cos)
        expected_fft[1:-1] *= 2.0
        expected_fft /= scaling_factor

        np.testing.assert_allclose(
            fft_result[0],
            expected_fft,
            rtol=1e-10,  # float64 FFT precision
        )

        peak_idx = np.argmax(np.abs(fft_result[0]))
        peak_mag = np.abs(fft_result[0, peak_idx])
        # float64 FFT precision — single-bin peak amplitude
        np.testing.assert_allclose(peak_mag, amp, rtol=1e-10)

    def test_fft_odd_size_scales_last_positive_frequency_bin(self) -> None:
        """Odd FFT sizes have no Nyquist bin, so every non-DC bin doubles."""
        signal = np.zeros((1, 5), dtype=np.float64)
        signal[..., 0] = 1.0

        result = FFT(_SR, n_fft=5, window="boxcar")._process(signal)

        # Unit impulse / boxcar gain gives DC=1/5 and both positive bins=2/5.
        expected = np.array([[0.2 + 0j, 0.4 + 0j, 0.4 + 0j]])
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_fft_window_function_changes_result(self) -> None:
        """Boxcar vs Hann windows produce different spectra but same peak."""
        t = np.linspace(0, 1, _SR, endpoint=False)
        sig = np.array([4.0 * np.sin(2 * np.pi * self._FREQ * t)])

        rect_result = run_operation_eager(FFT(_SR, n_fft=None, window="boxcar"), sig)
        hann_result = run_operation_eager(FFT(_SR, n_fft=None, window="hann"), sig)

        assert not np.allclose(rect_result, hann_result)
        # Both should detect the same peak amplitude (~4.0)
        # rtol=0.1: Hann window spreads energy via spectral leakage; peak amplitude approximate
        np.testing.assert_allclose(np.abs(rect_result).max(), 4, rtol=0.1)
        np.testing.assert_allclose(np.abs(hann_result).max(), 4, rtol=0.1)


class TestIFFTOperation:
    """IFFT operation: Layer 1 + Layer 2 + Layer 3 (np.fft.irfft reference)."""

    _N_FFT: int = 1024
    _WINDOW: str = "hann"

    # -- Layer 1: Unit tests -----------------------------------------------

    def test_ifft_init_default_params(self) -> None:
        """Test IFFT default n_fft=None, window='hann'."""
        ifft = IFFT(_SR)
        assert ifft.sampling_rate == _SR
        assert ifft.n_fft is None
        assert ifft.window == "hann"

    def test_ifft_init_custom_params(self) -> None:
        """Test IFFT stores custom n_fft and window."""
        ifft = IFFT(_SR, n_fft=2048, window="hamming")
        assert ifft.n_fft == 2048
        assert ifft.window == "hamming"

    def test_ifft_to_params_returns_lineage_parameters(self) -> None:
        ifft = IFFT(_SR, n_fft=2048, window="hamming")

        assert ifft.to_params() == {"n_fft": 2048, "window": "hamming"}

    def test_ifft_registry_returns_correct_class(self) -> None:
        """Test IFFT is registered as 'ifft'."""
        assert get_operation("ifft") == IFFT
        ifft_op = create_operation("ifft", _SR, n_fft=512, window="hamming")
        assert isinstance(ifft_op, IFFT)
        assert ifft_op.n_fft == 512

    # -- Layer 2: Domain (shape + immutability + lazy) ---------------------

    def test_ifft_preserves_immutability_and_dask_type(self) -> None:
        """Input unchanged after IFFT; result is DaArray."""
        spectrum = np.zeros((1, self._N_FFT // 2 + 1), dtype=complex)
        spectrum[0, 32] = 1.0
        dask_input = da_from_array(spectrum, chunks=(1, -1))
        input_copy = spectrum.copy()
        ifft = IFFT(_SR, n_fft=self._N_FFT, window=self._WINDOW)

        result_da = ifft.process(dask_input)

        assert result_da is not dask_input
        np.testing.assert_array_equal(dask_input.compute(), input_copy)
        assert isinstance(result_da, DaArray)

    def test_ifft_lazy_execution(self) -> None:
        """IFFT builds Dask graph without computing."""
        spectrum = np.zeros((1, self._N_FFT // 2 + 1), dtype=complex)
        spectrum[0, 32] = 1.0
        dask_input = da_from_array(spectrum, chunks=(1, -1))
        ifft = IFFT(_SR, n_fft=self._N_FFT, window=self._WINDOW)

        with mock.patch.object(DaArray, "compute") as mock_compute:
            result = ifft.process(dask_input)
            mock_compute.assert_not_called()
            assert isinstance(result, DaArray)
            _ = result.compute()
            mock_compute.assert_called_once()

    def test_ifft_shape_mono(self) -> None:
        """IFFT output: (1, n_fft) for mono spectrum."""
        spectrum = np.zeros((1, self._N_FFT // 2 + 1), dtype=complex)
        spectrum[0, 32] = 1.0
        ifft = IFFT(_SR, n_fft=self._N_FFT, window=self._WINDOW)
        result = run_operation_eager(ifft, spectrum)
        assert result.shape == (1, self._N_FFT)

    def test_ifft_shape_stereo(self) -> None:
        """IFFT output: (2, n_fft) for stereo spectrum."""
        spectrum = np.zeros((2, self._N_FFT // 2 + 1), dtype=complex)
        spectrum[0, 32] = 1.0
        spectrum[1, 16] = 1.0
        ifft = IFFT(_SR, n_fft=self._N_FFT, window=self._WINDOW)
        result = run_operation_eager(ifft, spectrum)
        assert result.shape == (2, self._N_FFT)

    def test_ifft_odd_size_restores_last_positive_frequency_scaling(self) -> None:
        """Odd FFT sizes require undoing amplitude doubling on the final bin."""
        spectrum = np.array([[0.2 + 0j, 0.4 + 0j, 0.4 + 0j]])

        result = IFFT(_SR, n_fft=5, window="boxcar")._process(spectrum)

        # Undoing both doubled positive bins recovers the normalized impulse.
        expected = np.array([[0.2, 0.0, 0.0, 0.0, 0.0]])
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_ifft_without_n_fft_infers_length(self) -> None:
        """IFFT with n_fft=None infers: 2*(input_length-1)."""
        spectrum = np.zeros(513, dtype=complex)
        spectrum[50] = 1.0
        ifft = IFFT(_SR, n_fft=None)
        result = run_operation_eager(ifft, np.array([spectrum]))
        assert result.shape == (1, 1024)

    def test_ifft_1d_input_reshaped(self) -> None:
        """1D-like spectrum input produces 2D output."""
        spectrum = np.zeros((1, self._N_FFT // 2 + 1), dtype=complex)
        spectrum[0, 5] = 1.0
        dask_in = da_from_array(spectrum.reshape(1, -1), chunks=(1, -1))
        ifft = IFFT(_SR, n_fft=self._N_FFT, window=self._WINDOW)

        result_da = ifft.process(dask_in)
        assert isinstance(result_da, DaArray)  # Pillar 1: Dask graph preserved
        result = result_da.compute()
        assert result.ndim == 2
        assert result.shape[0] == 1

    # -- Layer 3: Content verification -------------------------------------

    def test_ifft_output_is_real_and_frequency_preserved(self) -> None:
        """IFFT of single-frequency spectrum is real; round-trip detects same frequency.

        Tolerance: rtol=0.1 — FFT bin resolution.
        """
        freq_bins = np.fft.rfftfreq(self._N_FFT, 1.0 / _SR)
        f0 = 500.0
        target_idx = np.argmin(np.abs(freq_bins - f0))

        spectrum = np.zeros(self._N_FFT // 2 + 1, dtype=complex)
        spectrum[target_idx] = 1.0

        ifft = IFFT(_SR, n_fft=self._N_FFT, window=self._WINDOW)
        result = run_operation_eager(ifft, np.array([spectrum]))

        assert np.isrealobj(result)

        # Round-trip: FFT of IFFT result should peak at f0
        fft_of_result = np.fft.rfft(result[0])
        peak_idx = np.argmax(np.abs(fft_of_result))
        detected_freq = freq_bins[peak_idx]
        # rtol=0.1: IFFT round-trip with windowing introduces spectral leakage
        np.testing.assert_allclose(detected_freq, f0, rtol=0.1)

    def test_fft_ifft_roundtrip_preserves_frequency(self) -> None:
        """FFT->IFFT round-trip preserves frequency content.

        Note: wandas FFT applies spectral-analysis scaling (window + normalization),
        so FFT->IFFT is NOT an amplitude-preserving round-trip. Instead, verify
        that the dominant frequency is preserved through the transform pair.
        Tolerance: ±1 FFT bin — spectral leakage at bin boundaries.
        """
        t = np.linspace(0, 1, _SR, endpoint=False)
        original = np.array([np.sin(2 * np.pi * 500 * t)])

        fft = FFT(_SR, n_fft=self._N_FFT, window=self._WINDOW)
        ifft = IFFT(_SR, n_fft=self._N_FFT, window=self._WINDOW)

        spectrum = run_operation_lazy(fft, original)
        recovered = run_operation_eager(ifft, spectrum)

        # Verify frequency is preserved (not amplitude, due to analysis scaling)
        fft_of_recovered = np.fft.rfft(recovered[0])
        freq_bins = np.fft.rfftfreq(recovered.shape[1], 1.0 / _SR)
        peak_freq = freq_bins[np.argmax(np.abs(fft_of_recovered))]
        # ±1 bin tolerance for spectral leakage at bin boundaries
        assert abs(peak_freq - 500.0) <= freq_bins[1], f"Expected 500 Hz peak after FFT->IFFT, got {peak_freq:.1f} Hz"


class TestSTFTOperation:
    """STFT/ISTFT operations: Layer 1 + Layer 2 + Layer 3 (scipy ShortTimeFFT reference)."""

    _N_FFT: int = 1024
    _HOP: int = 256
    _WIN_LEN: int = 1024
    _WINDOW: str = "hann"

    def _make_mono(self) -> tuple[NDArrayReal, DaArray]:
        """1-second mono sine at 1 kHz, amplitude 4."""
        t = np.linspace(0, 1, _SR, endpoint=False)
        sig: NDArrayReal = np.array([np.sin(2 * np.pi * 1000 * t)]) * 4
        return sig, da_from_array(sig, chunks=(1, -1))

    def _make_stereo(self) -> tuple[NDArrayReal, DaArray]:
        t = np.linspace(0, 1, _SR, endpoint=False)
        sig: NDArrayReal = np.array([np.sin(2 * np.pi * 1000 * t), np.sin(2 * np.pi * 2000 * t)])
        return sig, da_from_array(sig, chunks=(1, -1))

    def _stft(self) -> STFT:
        return STFT(_SR, n_fft=self._N_FFT, hop_length=self._HOP, win_length=self._WIN_LEN, window=self._WINDOW)

    def _istft(self, length: int | None = None) -> ISTFT:
        return ISTFT(
            _SR,
            n_fft=self._N_FFT,
            hop_length=self._HOP,
            win_length=self._WIN_LEN,
            window=self._WINDOW,
            length=length,
        )

    # -- Layer 1: Unit tests -----------------------------------------------

    def test_stft_init_default_params(self) -> None:
        """STFT default parameters: n_fft=2048, hop=512, win=2048, hann."""
        stft = STFT(_SR)
        assert stft.sampling_rate == _SR
        assert stft.n_fft == 2048
        assert stft.hop_length == 512
        assert stft.win_length == 2048
        assert stft.window == "hann"
        assert not hasattr(stft, "SFT")

    def test_stft_init_custom_params(self) -> None:
        """STFT stores custom n_fft, hop_length, win_length, window."""
        stft = STFT(_SR, n_fft=1024, hop_length=256, win_length=512, window="hamming")
        assert stft.n_fft == 1024
        assert stft.hop_length == 256
        assert stft.win_length == 512
        assert stft.window == "hamming"

    def test_istft_init_default_params(self) -> None:
        """ISTFT default parameters match STFT; length defaults to None."""
        istft = ISTFT(_SR)
        assert istft.sampling_rate == _SR
        assert istft.n_fft == 2048
        assert istft.hop_length == 512
        assert istft.win_length == 2048
        assert istft.window == "hann"
        assert istft.length is None

    def test_istft_init_custom_params(self) -> None:
        """ISTFT stores custom params including explicit length."""
        istft = ISTFT(_SR, n_fft=1024, hop_length=256, win_length=512, window="hamming", length=16000)
        assert istft.n_fft == 1024
        assert istft.hop_length == 256
        assert istft.win_length == 512
        assert istft.window == "hamming"
        assert istft.length == 16000

    def test_istft_calculate_output_shape_respects_length_limit(self) -> None:
        istft = ISTFT(_SR, n_fft=256, hop_length=64, length=128)

        assert istft.calculate_output_shape((2, 129, 10)) == (2, 128)

    def test_stft_registry_returns_correct_class(self) -> None:
        """'stft' and 'istft' registry keys create correct instances."""
        assert get_operation("stft") == STFT
        assert get_operation("istft") == ISTFT
        stft_op = create_operation("stft", _SR, n_fft=512, hop_length=128)
        assert isinstance(stft_op, STFT)
        assert stft_op.n_fft == 512
        istft_op = create_operation("istft", _SR, n_fft=512, hop_length=128)
        assert isinstance(istft_op, ISTFT)
        assert istft_op.n_fft == 512

    def test_stft_negative_n_fft_raises(self) -> None:
        """Negative n_fft raises ValueError with WHAT/WHY/HOW message."""
        with pytest.raises(ValueError) as exc_info:
            STFT(sampling_rate=44100, n_fft=-2048)
        error_msg = str(exc_info.value)
        assert "Invalid FFT size for STFT" in error_msg
        assert "-2048" in error_msg
        assert "Positive integer" in error_msg

    def test_stft_zero_n_fft_raises(self) -> None:
        """Zero n_fft raises ValueError."""
        with pytest.raises(ValueError, match="Invalid FFT size for STFT"):
            STFT(sampling_rate=44100, n_fft=0)

    def test_stft_win_length_greater_than_n_fft_raises(self) -> None:
        """win_length > n_fft raises ValueError with constraint message."""
        with pytest.raises(ValueError) as exc_info:
            STFT(sampling_rate=44100, n_fft=1024, win_length=2048)
        error_msg = str(exc_info.value)
        assert "Invalid window length for STFT" in error_msg
        assert "win_length=2048" in error_msg
        assert "win_length <= n_fft" in error_msg

    def test_stft_negative_hop_length_raises(self) -> None:
        """Negative hop_length raises ValueError with WHAT/WHY/HOW message."""
        with pytest.raises(ValueError) as exc_info:
            STFT(sampling_rate=44100, n_fft=2048, hop_length=-512)
        error_msg = str(exc_info.value)
        assert "Invalid hop length for STFT" in error_msg
        assert "-512" in error_msg
        assert "Positive integer" in error_msg

    def test_stft_hop_length_greater_than_win_length_raises(self) -> None:
        """hop_length > win_length raises ValueError (would create gaps)."""
        with pytest.raises(ValueError) as exc_info:
            STFT(sampling_rate=44100, n_fft=2048, win_length=1024, hop_length=2048)
        error_msg = str(exc_info.value)
        assert "Invalid hop length for STFT" in error_msg
        assert "hop_length=2048" in error_msg
        assert "hop_length <= win_length" in error_msg
        assert "would create gaps" in error_msg

    def test_stft_negative_win_length_raises(self) -> None:
        """Negative win_length raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            STFT(sampling_rate=44100, n_fft=2048, win_length=-1024)
        error_msg = str(exc_info.value)
        assert "Invalid window length for STFT" in error_msg
        assert "Positive integer" in error_msg

    def test_stft_zero_win_length_raises(self) -> None:
        """Zero win_length raises ValueError."""
        with pytest.raises(ValueError, match="Invalid window length for STFT"):
            STFT(sampling_rate=44100, n_fft=2048, win_length=0)

    def test_stft_win_length_too_small_raises(self) -> None:
        """win_length < 4 raises ValueError (insufficient for windowing)."""
        with pytest.raises(ValueError) as exc_info:
            STFT(sampling_rate=44100, n_fft=2048, win_length=3)
        error_msg = str(exc_info.value)
        assert "Window length too small" in error_msg
        assert "win_length=3" in error_msg
        assert "win_length >= 4" in error_msg

    # -- Layer 2: Domain (shape + immutability) ----------------------------

    def test_stft_preserves_immutability_and_dask_type(self) -> None:
        """Input unchanged after STFT; result is DaArray."""
        sig, dask_sig = self._make_mono()
        input_copy = sig.copy()
        stft = self._stft()

        result_da = stft.process(dask_sig)

        assert result_da is not dask_sig
        np.testing.assert_array_equal(dask_sig.compute(), input_copy)
        assert isinstance(result_da, DaArray)

    def test_stft_shape_mono_matches_scipy(self) -> None:
        """STFT output shape matches scipy ShortTimeFFT for mono."""

        sig, _ = self._make_mono()
        stft = self._stft()
        result = run_operation_eager(stft, sig)

        assert result.ndim == 3
        sft = ScipySTFT(
            win=get_window(self._WINDOW, self._WIN_LEN),
            hop=self._HOP,
            fs=_SR,
            mfft=self._N_FFT,
            scale_to="magnitude",
        )
        expected_shape = (1, sft.f.shape[0], sft.t(sig.shape[-1]).shape[0])
        assert result.shape == expected_shape

    def test_stft_shape_stereo_matches_scipy(self) -> None:
        """STFT output shape matches scipy ShortTimeFFT for stereo."""

        sig, _ = self._make_stereo()
        stft = self._stft()
        result = run_operation_eager(stft, sig)

        assert result.ndim == 3
        sft = ScipySTFT(
            win=get_window(self._WINDOW, self._WIN_LEN),
            hop=self._HOP,
            fs=_SR,
            mfft=self._N_FFT,
            scale_to="magnitude",
        )
        expected_shape = (2, sft.f.shape[0], sft.t(sig.shape[-1]).shape[0])
        assert result.shape == expected_shape

    def test_stft_1d_input_reshaped(self) -> None:
        """1D input produces 3D output with 1 channel."""
        sig_1d = np.sin(2 * np.pi * 440 * np.linspace(0, 1, _SR, endpoint=False))
        result = run_operation_eager(self._stft(), sig_1d)
        assert result.ndim == 3
        assert result.shape[0] == 1

    def test_istft_shape_matches_original(self) -> None:
        """ISTFT output shape close to original signal length."""
        sig, _ = self._make_mono()
        stft_data = run_operation_lazy(self._stft(), sig)
        result = run_operation_eager(self._istft(), stft_data)

        assert result.ndim == 2
        assert result.shape[0] == 1
        assert abs(result.shape[1] - sig.shape[1]) < self._WIN_LEN

    def test_istft_process_rejects_2d_direct_lazy_input(self) -> None:
        """ISTFT.process requires channel-first spectrograms."""
        sig, _ = self._make_mono()
        stft_data = run_operation_eager(self._stft(), sig)
        stft_2d = stft_data[0]  # (freqs, frames)

        with pytest.raises(ValueError, match=r"AudioOperation.process requires channel-first data"):
            run_operation_eager(self._istft(), stft_2d)

    def test_istft_with_length_trims_output(self) -> None:
        """ISTFT with length parameter trims to exact output length."""
        sig, _ = self._make_mono()
        stft_data = run_operation_lazy(self._stft(), sig)
        result = run_operation_eager(self._istft(length=8000), stft_data)
        assert result.shape[1] == 8000

    # -- Layer 3: Numerical verification -----------------------------------

    def test_stft_content_matches_scipy_reference(self) -> None:
        """STFT output matches scipy ShortTimeFFT (exact reference).

        Tolerance: rtol=1e-5, atol=1e-5 — window scaling and padding.
        """

        sig, dask_sig = self._make_mono()
        stft = self._stft()
        result_da = stft.process(dask_sig)
        assert isinstance(result_da, DaArray)  # Pillar 1: Dask graph preserved
        result = result_da.compute()

        sft = ScipySTFT(
            win=get_window(self._WINDOW, self._WIN_LEN),
            hop=self._HOP,
            fs=_SR,
            mfft=self._N_FFT,
            scale_to="magnitude",
        )
        expected_raw = sft.stft(sig[0])
        expected_raw[..., 1:-1, :] *= 2.0
        expected = expected_raw.reshape(1, *expected_raw.shape)

        np.testing.assert_allclose(np.abs(result).max(), 4, rtol=1e-5)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_stft_odd_size_scales_last_positive_frequency_bin(self) -> None:
        """Odd FFT sizes have no Nyquist bin, so every positive bin doubles."""
        n_fft = 15
        hop_length = 3
        signal = np.arange(60, dtype=float)[None, :]
        result = run_operation_eager(
            STFT(
                _SR,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window="boxcar",
            ),
            signal,
        )

        scipy_stft = ScipySTFT(
            win=get_window("boxcar", n_fft),
            hop=hop_length,
            fs=_SR,
            mfft=n_fft,
            scale_to="magnitude",
        )
        expected = scipy_stft.stft(signal[0])
        expected[..., 1:, :] *= 2.0

        np.testing.assert_allclose(result[0], expected, rtol=1e-12, atol=1e-12)

    def test_stft_amplitude_scaling_matches_input(self) -> None:
        """STFT peak amplitude in middle frame matches input amplitude.

        Tolerance: rtol=1e-10 — float64 precision.
        """
        amp = 2.0
        t = np.linspace(0, 1, _SR, endpoint=False)
        cos_wave = amp * np.cos(2 * np.pi * 500 * t)
        result_da = run_operation_lazy(self._stft(), cos_wave)
        assert isinstance(result_da, DaArray)  # Pillar 1: Dask graph preserved
        result = result_da.compute()

        middle_frame = result.shape[2] // 2
        peak_idx = np.argmax(np.abs(result[0, :, middle_frame]))
        peak_mag = np.abs(result[0, peak_idx, middle_frame])
        np.testing.assert_allclose(peak_mag, amp, rtol=1e-10)

    def test_stft_istft_roundtrip_reconstruction(self) -> None:
        """STFT->ISTFT roundtrip reconstructs original signal.

        Tolerance: rtol=1e-6, atol=1e-5 — windowed overlap-add.
        Boundary: 16 samples trimmed from each end for edge effects.
        """
        sig, _ = self._make_mono()
        stft_data = run_operation_lazy(self._stft(), sig)
        istft_data = run_operation_eager(self._istft(), stft_data)

        orig_length = sig.shape[1]
        reconstructed = istft_data[:, :orig_length]
        np.testing.assert_allclose(
            reconstructed[..., 16:-16],
            sig[..., 16:-16],
            rtol=1e-6,
            atol=1e-5,
        )


@pytest.mark.parametrize(
    "operation",
    [
        NOctSpectrum(_SR, 24, 12600),
        NOctSynthesis(_SR, 24, 12600),
    ],
)
def test_direct_noct_process_preflights_dependencies(monkeypatch: pytest.MonkeyPatch, operation: object) -> None:
    original_error = ImportError('Install it with: pip install "wandas[psychoacoustic]"')
    calls: list[str] = []

    def raise_import_error(feature: str) -> None:
        calls.append(feature)
        raise original_error

    monkeypatch.setattr(spectral_module, "require_mosqito_center_freq", raise_import_error)
    monkeypatch.setattr(operation, "_process", lambda _data: pytest.fail("ran kernel before checking mosqito"))

    with pytest.raises(ImportError) as exc_info:
        getattr(operation, "process")(da_from_array(np.zeros((1, 16)), chunks=(1, -1)))

    assert exc_info.value is original_error
    assert calls == ["NOctFrame"]


def test_direct_noct_process_preflights_dependencies_for_public_process(monkeypatch: pytest.MonkeyPatch) -> None:
    original_error = ImportError('Install it with: pip install "wandas[psychoacoustic]"')

    def raise_import_error(*args: object, **kwargs: object) -> None:
        raise original_error

    operation = NOctSpectrum(_SR, 24, 12600)
    monkeypatch.setattr(spectral_module, "require_mosqito_center_freq", raise_import_error)
    monkeypatch.setattr(operation, "_process", lambda _data: pytest.fail("ran kernel before checking mosqito"))

    with pytest.raises(ImportError) as exc_info:
        operation.process(da_from_array(np.zeros((1, 16)), chunks=(1, -1)))

    assert exc_info.value is original_error


def test_direct_noct_process_continues_after_dependency_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    operation = NOctSpectrum(_SR, 24, 12600)
    kernel_result = np.zeros((1, 1))

    monkeypatch.setattr(spectral_module, "require_mosqito_center_freq", lambda feature: _center_freq)
    monkeypatch.setattr(operation, "_process", lambda _data: kernel_result)

    result = operation.process(da_from_array(np.zeros((1, 16)), chunks=(1, -1)))

    assert isinstance(result, DaArray)
    np.testing.assert_array_equal(result.compute(), kernel_result)


def _fractional_octave_noise(sample_count: int) -> tuple[NDArrayReal, NDArrayReal]:
    """Return deterministic pink and white noise with unit peak amplitude."""
    rng = np.random.default_rng(42)
    white = rng.standard_normal(sample_count)
    nonzero_frequencies = np.fft.rfftfreq(sample_count)[1:]
    spectrum = np.fft.rfft(white)
    spectrum[1:] *= 1.0 / np.sqrt(nonzero_frequencies)
    pink = np.fft.irfft(spectrum, sample_count)
    return pink / np.abs(pink).max(), white / np.abs(white).max()


@pytest.mark.parametrize(
    ("operation_type", "sampling_rate"),
    (
        pytest.param(NOctSynthesis, 48000, id="synthesis"),
        pytest.param(NOctSpectrum, 51200, id="spectrum"),
    ),
)
def test_fractional_octave_operation_stores_configuration(
    operation_type: type[NOctSynthesis] | type[NOctSpectrum],
    sampling_rate: int,
) -> None:
    """Both fractional-octave operations expose the same configuration contract."""
    operation = operation_type(sampling_rate, fmin=24.0, fmax=12600, n=3, G=10, fr=1000)

    assert operation.sampling_rate == sampling_rate
    assert operation.fmin == 24.0
    assert operation.fmax == 12600
    assert operation.n == 3
    assert operation.G == 10
    assert operation.fr == 1000


class TestNOctSynthesisOperation:
    """NOctSynthesis operation: Layer 1 + Layer 2 + Layer 3 (mosqito reference)."""

    _NOCT_SR: int = 48000
    _FMIN: float = 24.0
    _FMAX: float = 12600
    _N: int = 3
    _G: int = 10
    _FR: int = 1000

    def _op(self) -> NOctSynthesis:
        return NOctSynthesis(
            self._NOCT_SR,
            fmin=self._FMIN,
            fmax=self._FMAX,
            n=self._N,
            G=self._G,
            fr=self._FR,
        )

    # -- Layer 1: Unit tests -----------------------------------------------

    def test_registry_returns_correct_class(self) -> None:
        assert get_operation("noct_synthesis") == NOctSynthesis
        op = create_operation(
            "noct_synthesis",
            self._NOCT_SR,
            fmin=100.0,
            fmax=5000.0,
            n=1,
            G=20,
            fr=1000,
        )
        assert isinstance(op, NOctSynthesis)
        assert op.fmin == 100.0
        assert op.fmax == 5000.0

    # -- Layer 2: Domain (immutability + lazy + shapes) ---------------------

    def test_preserves_immutability_and_dask_type(self) -> None:
        """Pillar 1: input data unchanged after NOctSynthesis; result is new instance."""
        pink, _ = _fractional_octave_noise(self._NOCT_SR)
        sig = np.array([pink])
        dask_sig = da_from_array(sig, chunks=(1, 1000))
        input_copy = sig.copy()

        fft = FFT(self._NOCT_SR, n_fft=None, window="hann")
        spectrum = run_operation_eager(fft, dask_sig)
        spec_copy = spectrum.copy()

        result = run_operation_eager(self._op(), spectrum)

        np.testing.assert_array_equal(sig, input_copy)
        np.testing.assert_array_equal(spectrum, spec_copy)
        assert result is not spectrum

    def test_process_metadata_shape_uses_fractional_octave_band_count(self) -> None:
        """NOctSynthesis lazy metadata shape matches its computed band-count output."""
        spectrum = np.ones((1, 513))
        dask_spectrum = da_from_array(spectrum, chunks=(1, -1))
        _, center_freqs = _center_freq(
            fmin=self._FMIN,
            fmax=self._FMAX,
            n=self._N,
            G=self._G,
            fr=self._FR,
        )

        result_da = self._op().process(dask_spectrum)

        assert result_da.shape == (1, len(center_freqs))
        assert result_da.compute().shape == result_da.shape

    def test_delayed_execution_not_computed_early(self) -> None:
        """Pillar 1: Dask lazy evaluation preserved; no premature compute()."""
        pink, _ = _fractional_octave_noise(self._NOCT_SR)
        sig = np.array([pink])
        dask_sig = da_from_array(sig, chunks=(1, 1000))
        with mock.patch.object(DaArray, "compute") as mock_compute:
            with mock.patch("wandas.processing.spectral.noct_synthesis") as mock_noct:
                mock_noct.return_value = (np.zeros((1, self._NOCT_SR)), np.zeros(27))
                result = self._op().process(dask_sig)
                mock_compute.assert_not_called()
                _ = result.compute()
                mock_compute.assert_called_once()

    def test_odd_length_spectrum_produces_valid_output(self) -> None:
        """Odd-length spectrum exercises the else-branch in length inference."""
        fft = FFT(self._NOCT_SR, n_fft=None, window="hann")
        rng = np.random.default_rng(99)
        test_signal = rng.standard_normal(52)
        spectrum = run_operation_eager(fft, np.array([test_signal]))
        assert spectrum.shape[-1] % 2 == 1
        result_da = self._op().process(da_from_array(spectrum, chunks=(1, -1)))
        assert isinstance(result_da, DaArray)  # Pillar 1: Dask graph preserved
        result = result_da.compute()
        assert result.shape[0] == 1

    # -- Layer 3: Numerical verification (mosqito reference) ----------------

    def test_mono_matches_mosqito_noct_synthesis(self) -> None:
        """Mono result matches direct mosqito noct_synthesis call."""
        pink, _ = _fractional_octave_noise(self._NOCT_SR)
        sig = np.array([pink])
        dask_sig = da_from_array(sig, chunks=(1, 1000))

        fft = FFT(self._NOCT_SR, n_fft=None, window="hann")
        spectrum = run_operation_eager(fft, dask_sig)
        result_da = run_operation_lazy(self._op(), spectrum)
        assert isinstance(result_da, DaArray)  # Pillar 1: Dask graph preserved
        result = result_da.compute()

        n = spectrum.shape[-1]
        n = n * 2 - 1 if n % 2 == 0 else (n - 1) * 2
        freqs = np.fft.rfftfreq(n, d=1 / self._NOCT_SR)
        expected_signal, _ = noct_synthesis(
            spectrum=np.abs(spectrum).T,
            freqs=freqs,
            fmin=self._FMIN,
            fmax=self._FMAX,
            n=self._N,
            G=self._G,
            fr=self._FR,
        )
        expected_signal = expected_signal.T

        assert result.shape == expected_signal.shape
        # Wrapper equivalence: same MoSQITo noct_synthesis call; default tolerance
        np.testing.assert_allclose(result[0], expected_signal[0])

    def test_stereo_matches_mosqito_per_channel(self) -> None:
        """Stereo result matches per-channel mosqito calls.

        Tolerance: rtol=1e-5, atol=1e-5 — floating-point accumulation.
        """
        pink, white = _fractional_octave_noise(self._NOCT_SR)
        sig = np.array([pink, white])
        dask_sig = da_from_array(sig, chunks=(2, 1000))

        fft = FFT(self._NOCT_SR, n_fft=None, window="hann")
        spectrum = run_operation_eager(fft, dask_sig)
        result_da = run_operation_lazy(self._op(), spectrum)
        assert isinstance(result_da, DaArray)  # Pillar 1: Dask graph preserved
        result = result_da.compute()

        n = spectrum.shape[-1]
        n = n * 2 - 1 if n % 2 == 0 else (n - 1) * 2
        freqs = np.fft.rfftfreq(n, d=1 / self._NOCT_SR)

        exp_ch1, _ = noct_synthesis(
            spectrum=np.abs(spectrum[0:1]).T,
            freqs=freqs,
            fmin=self._FMIN,
            fmax=self._FMAX,
            n=self._N,
            G=self._G,
            fr=self._FR,
        )
        exp_ch2, _ = noct_synthesis(
            spectrum=np.abs(spectrum[1:2]).T,
            freqs=freqs,
            fmin=self._FMIN,
            fmax=self._FMAX,
            n=self._N,
            G=self._G,
            fr=self._FR,
        )

        assert result.shape == (2, exp_ch1.T.shape[1])
        # rtol/atol=1e-5: MoSQITo wrapper with window scaling rounding
        np.testing.assert_allclose(result[0], exp_ch1.T[0], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(result[1], exp_ch2.T[0], rtol=1e-5, atol=1e-5)
        assert not np.allclose(result[0], result[1])


class TestWelchOperation:
    """Welch PSD operation: Layer 1 + Layer 2 + Layer 3 (scipy reference)."""

    _N_FFT: int = 1024
    _HOP: int = 256
    _WIN_LEN: int = 1024
    _WINDOW: str = "hann"
    _AVERAGE: str = "mean"
    _DETREND: str = "constant"

    def _make_mono(self, freq: float = 1000.0) -> tuple[NDArrayReal, DaArray]:
        t = np.linspace(0, 1, _SR, endpoint=False)
        sig: NDArrayReal = np.array([np.sin(2 * np.pi * freq * t)])
        return sig, da_from_array(sig, chunks=(1, 1000))

    def _make_stereo(self) -> tuple[NDArrayReal, DaArray]:
        t = np.linspace(0, 1, _SR, endpoint=False)
        sig: NDArrayReal = np.array([np.sin(2 * np.pi * 1000 * t), np.sin(2 * np.pi * 2000 * t)])
        return sig, da_from_array(sig, chunks=(2, 1000))

    def _op(self) -> Welch:
        return Welch(
            _SR,
            n_fft=self._N_FFT,
            hop_length=self._HOP,
            win_length=self._WIN_LEN,
            window=self._WINDOW,
            average=self._AVERAGE,
            detrend=self._DETREND,
        )

    # -- Layer 1: Unit tests -----------------------------------------------

    def test_init_default_params(self) -> None:
        welch = Welch(_SR)
        assert welch.sampling_rate == _SR
        assert welch.n_fft == 2048
        assert welch.win_length == 2048
        assert welch.hop_length == 512
        assert welch.window == "hann"
        assert welch.average == "mean"
        assert welch.detrend == "constant"
        assert welch.noverlap == 1536

    def test_init_custom_params(self) -> None:
        welch = Welch(
            _SR,
            n_fft=1024,
            hop_length=256,
            win_length=512,
            window="hamming",
            average="median",
            detrend="linear",
        )
        assert welch.n_fft == 1024
        assert welch.win_length == 512
        assert welch.hop_length == 256
        assert welch.window == "hamming"
        assert welch.average == "median"
        assert welch.detrend == "linear"
        assert welch.noverlap == 256

    def test_noverlap_is_read_only(self) -> None:
        welch = self._op()

        with pytest.raises(AttributeError):
            setattr(welch, "noverlap", 0)

        assert welch.noverlap == self._WIN_LEN - self._HOP

    def test_registry_returns_correct_class(self) -> None:
        assert get_operation("welch") == Welch
        op = create_operation(
            "welch",
            _SR,
            n_fft=512,
            win_length=512,
            hop_length=128,
            window="hamming",
        )
        assert isinstance(op, Welch)
        assert op.n_fft == 512

    def test_negative_n_fft_raises(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            Welch(sampling_rate=44100, n_fft=-2048)
        error_msg = str(exc_info.value)
        assert "Invalid FFT size for Welch" in error_msg
        assert "-2048" in error_msg
        assert "Positive integer" in error_msg

    def test_win_length_greater_than_n_fft_raises(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            Welch(sampling_rate=44100, n_fft=1024, win_length=2048)
        error_msg = str(exc_info.value)
        assert "Invalid window length for Welch" in error_msg
        assert "win_length=2048" in error_msg
        assert "win_length <= n_fft" in error_msg

    def test_hop_length_greater_than_win_length_raises(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            Welch(sampling_rate=44100, n_fft=2048, win_length=1024, hop_length=2048)
        error_msg = str(exc_info.value)
        assert "Invalid hop length for Welch" in error_msg
        assert "hop_length <= win_length" in error_msg
        assert "would create gaps" in error_msg

    def test_zero_n_fft_raises(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            Welch(sampling_rate=44100, n_fft=0)
        error_msg = str(exc_info.value)
        assert "Invalid FFT size for Welch" in error_msg
        assert "Positive integer" in error_msg

    def test_negative_win_length_raises(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            Welch(sampling_rate=44100, n_fft=2048, win_length=-1024)
        error_msg = str(exc_info.value)
        assert "Invalid window length for Welch" in error_msg
        assert "Positive integer" in error_msg

    def test_zero_win_length_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid window length for Welch"):
            Welch(sampling_rate=44100, n_fft=2048, win_length=0)

    def test_win_length_too_small_raises(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            Welch(sampling_rate=44100, n_fft=2048, win_length=3)
        error_msg = str(exc_info.value)
        assert "Window length too small" in error_msg
        assert "win_length=3" in error_msg
        assert "win_length >= 4" in error_msg

    def test_non_ndarray_raises(self) -> None:
        welch = Welch(sampling_rate=_SR, n_fft=self._N_FFT)
        with pytest.raises(ValueError, match="Welch operation requires"):
            welch._process([0.0] * 100)  # ty: ignore[invalid-argument-type]

    # -- Layer 2: Domain (immutability + lazy + shapes) ---------------------

    def test_preserves_immutability_and_dask_type(self) -> None:
        sig, dask_sig = self._make_mono()
        input_copy = sig.copy()

        result = self._op().process(dask_sig)

        np.testing.assert_array_equal(dask_sig.compute(), input_copy)
        assert isinstance(result, DaArray)

    def test_delayed_execution_not_computed_early(self) -> None:
        _, dask_sig = self._make_mono()
        with mock.patch.object(DaArray, "compute") as mock_compute:
            result = self._op().process(dask_sig)
            mock_compute.assert_not_called()
            assert isinstance(result, DaArray)
            _ = result.compute()
            mock_compute.assert_called_once()

    def test_shape_mono(self) -> None:
        sig, _ = self._make_mono()
        result = run_operation_eager(self._op(), sig)
        expected_bins = self._N_FFT // 2 + 1
        assert result.shape == (1, expected_bins)

    def test_shape_stereo(self) -> None:
        sig, _ = self._make_stereo()
        result = run_operation_eager(self._op(), sig)
        expected_bins = self._N_FFT // 2 + 1
        assert result.shape == (2, expected_bins)

    # -- Layer 3: Numerical verification (scipy reference) ------------------

    def test_peak_frequency_detected_correctly(self) -> None:
        """Welch PSD peak at 1 kHz for a 1 kHz sine.

        Tolerance: rtol=0.05 — frequency bin resolution.
        """
        sig, _ = self._make_mono(freq=1000.0)
        result = run_operation_eager(self._op(), sig)
        freq_bins = np.fft.rfftfreq(self._N_FFT, 1.0 / _SR)
        detected_freq = freq_bins[np.argmax(result[0])]
        np.testing.assert_allclose(detected_freq, 1000.0, rtol=0.05)

    def test_stereo_second_channel_peak_at_2khz(self) -> None:
        """Welch PSD peak at 2 kHz for stereo second channel.

        Tolerance: rtol=0.05 — frequency bin resolution.
        """
        sig, _ = self._make_stereo()
        result = run_operation_eager(self._op(), sig)
        freq_bins = np.fft.rfftfreq(self._N_FFT, 1.0 / _SR)
        detected_freq = freq_bins[np.argmax(result[1])]
        np.testing.assert_allclose(detected_freq, 2000.0, rtol=0.05)

    def test_matches_scipy_welch_reference(self) -> None:
        """Welch output matches scipy.signal.welch with equivalent params.

        Tolerance: rtol=1e-6 — float64 precision.
        """

        sig, _ = self._make_stereo()
        result = run_operation_eager(self._op(), sig)

        _, expected = ss.welch(
            x=sig,
            fs=_SR,
            nperseg=self._WIN_LEN,
            noverlap=self._WIN_LEN - self._HOP,
            nfft=self._N_FFT,
            window=self._WINDOW,
            detrend=self._DETREND,
            scaling="spectrum",
            average=self._AVERAGE,
            axis=-1,
        )
        expected[..., 1:-1] *= 2
        expected **= 0.5

        assert result.shape == expected.shape
        # rtol=1e-6: wrapper equivalence — same scipy.signal.welch algorithm
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_amplitude_scaling_sine_wave(self) -> None:
        """Peak amplitude in Welch matches input sine amplitude.

        Tolerance: rtol=1e-10 — float64 precision.
        """
        amp = 5.0
        t = np.linspace(0, 1, _SR, endpoint=False)
        sine = amp * np.sin(2 * np.pi * 1000 * t)
        result = run_operation_eager(self._op(), np.array([sine]))

        freq_bins = np.fft.rfftfreq(self._N_FFT, 1.0 / _SR)
        peak_idx = np.argmax(result[0])
        np.testing.assert_allclose(freq_bins[peak_idx], 1000.0, rtol=1e-10)
        np.testing.assert_allclose(result[0, peak_idx], amp, rtol=1e-10)


_PAIRWISE_N_FFT = 1024
_PAIRWISE_HOP_LENGTH = 256
_PAIRWISE_WINDOW_LENGTH = 1024
_PAIRWISE_WINDOW = "hann"
_PAIRWISE_DETREND = "constant"
_PAIRWISE_SCALING = "spectrum"
_PAIRWISE_AVERAGE = "mean"

PairwiseSpectralOperation = Coherence | CSD | TransferFunction


def _cross_spectral_pair() -> NDArrayReal:
    """Return deterministic channels used by coherence and CSD tests."""
    time = np.linspace(0, 1, _SR, endpoint=False)
    return np.array(
        [
            np.sin(2 * np.pi * 1000 * time),
            np.sin(2 * np.pi * 1100 * time),
        ]
    )


def _cross_spectral_multichannel_signal() -> NDArrayReal:
    """Return three distinct channels for pairwise shape contracts."""
    time = np.linspace(0, 1, _SR, endpoint=False)
    noise = np.random.default_rng(42).standard_normal(_SR) * 0.1
    return np.array(
        [
            np.sin(2 * np.pi * 1000 * time),
            np.sin(2 * np.pi * 1100 * time),
            noise,
        ]
    )


def _transfer_pair() -> NDArrayReal:
    """Return a gain-two input/output pair with deterministic noise."""
    time = np.linspace(0, 1, _SR, endpoint=False)
    input_signal = np.sin(2 * np.pi * 1000 * time)
    output_signal = 2 * input_signal + 0.1 * np.random.default_rng(42).standard_normal(len(time))
    return np.array([input_signal, output_signal])


def _transfer_multichannel_signal() -> NDArrayReal:
    """Return two inputs and two outputs for pairwise shape contracts."""
    time = np.linspace(0, 1, _SR, endpoint=False)
    input_1 = np.sin(2 * np.pi * 1000 * time)
    input_2 = np.sin(2 * np.pi * 1500 * time)
    rng = np.random.default_rng(42)
    output_1 = 2 * input_1 + 0.5 * input_2 + 0.1 * rng.standard_normal(len(time))
    output_2 = 0.3 * input_1 + 1.5 * input_2 + 0.1 * rng.standard_normal(len(time))
    return np.array([input_1, input_2, output_1, output_2])


def _coherence_operation() -> Coherence:
    return Coherence(
        _SR,
        n_fft=_PAIRWISE_N_FFT,
        hop_length=_PAIRWISE_HOP_LENGTH,
        win_length=_PAIRWISE_WINDOW_LENGTH,
        window=_PAIRWISE_WINDOW,
        detrend=_PAIRWISE_DETREND,
    )


def _csd_operation() -> CSD:
    return CSD(
        _SR,
        n_fft=_PAIRWISE_N_FFT,
        hop_length=_PAIRWISE_HOP_LENGTH,
        win_length=_PAIRWISE_WINDOW_LENGTH,
        window=_PAIRWISE_WINDOW,
        detrend=_PAIRWISE_DETREND,
        scaling=_PAIRWISE_SCALING,
        average=_PAIRWISE_AVERAGE,
    )


def _transfer_function_operation() -> TransferFunction:
    return TransferFunction(
        _SR,
        n_fft=_PAIRWISE_N_FFT,
        hop_length=_PAIRWISE_HOP_LENGTH,
        win_length=_PAIRWISE_WINDOW_LENGTH,
        window=_PAIRWISE_WINDOW,
        detrend=_PAIRWISE_DETREND,
        scaling=_PAIRWISE_SCALING,
        average=_PAIRWISE_AVERAGE,
    )


@dataclass(frozen=True)
class PairwiseSpectralCase:
    """Inputs needed to exercise one pairwise spectral operation contract."""

    registry_name: str
    operation_type: type[PairwiseSpectralOperation]
    make_operation: Callable[[], PairwiseSpectralOperation]
    make_stereo_signal: Callable[[], NDArrayReal]
    make_multichannel_signal: Callable[[], NDArrayReal]
    expected_attributes: tuple[tuple[str, object], ...]


_PAIRWISE_SPECTRAL_CASES = (
    PairwiseSpectralCase(
        registry_name="coherence",
        operation_type=Coherence,
        make_operation=_coherence_operation,
        make_stereo_signal=_cross_spectral_pair,
        make_multichannel_signal=_cross_spectral_multichannel_signal,
        expected_attributes=(),
    ),
    PairwiseSpectralCase(
        registry_name="csd",
        operation_type=CSD,
        make_operation=_csd_operation,
        make_stereo_signal=_cross_spectral_pair,
        make_multichannel_signal=_cross_spectral_multichannel_signal,
        expected_attributes=(("scaling", _PAIRWISE_SCALING), ("average", _PAIRWISE_AVERAGE)),
    ),
    PairwiseSpectralCase(
        registry_name="transfer_function",
        operation_type=TransferFunction,
        make_operation=_transfer_function_operation,
        make_stereo_signal=_transfer_pair,
        make_multichannel_signal=_transfer_multichannel_signal,
        expected_attributes=(("scaling", _PAIRWISE_SCALING), ("average", _PAIRWISE_AVERAGE)),
    ),
)


@pytest.mark.parametrize("case", _PAIRWISE_SPECTRAL_CASES, ids=lambda case: case.registry_name)
def test_pairwise_spectral_operation_configuration_and_registry(case: PairwiseSpectralCase) -> None:
    """Every pairwise operation stores and round-trips its public configuration."""
    operation = case.make_operation()

    assert operation.sampling_rate == _SR
    assert operation.n_fft == _PAIRWISE_N_FFT
    assert operation.hop_length == _PAIRWISE_HOP_LENGTH
    assert operation.win_length == _PAIRWISE_WINDOW_LENGTH
    assert operation.window == _PAIRWISE_WINDOW
    assert operation.detrend == _PAIRWISE_DETREND
    for attribute, expected in case.expected_attributes:
        assert getattr(operation, attribute) == expected

    assert get_operation(case.registry_name) is case.operation_type
    recreated = create_operation(case.registry_name, _SR, **operation.to_params())
    assert type(recreated) is case.operation_type
    assert recreated.to_params() == operation.to_params()


@pytest.mark.parametrize("case", _PAIRWISE_SPECTRAL_CASES, ids=lambda case: case.registry_name)
def test_pairwise_spectral_operation_preserves_input_and_dask_type(case: PairwiseSpectralCase) -> None:
    """Graph construction neither mutates input samples nor leaves Dask."""
    signal = case.make_stereo_signal()
    original = signal.copy()
    dask_signal = as_operation_dask(signal)

    result = case.make_operation().process(dask_signal)

    np.testing.assert_array_equal(dask_signal.compute(), original)
    assert isinstance(result, DaArray)


@pytest.mark.parametrize("case", _PAIRWISE_SPECTRAL_CASES, ids=lambda case: case.registry_name)
def test_pairwise_spectral_operation_does_not_compute_during_graph_build(case: PairwiseSpectralCase) -> None:
    """Pairwise operations remain lazy until callers compute the result."""
    dask_signal = as_operation_dask(case.make_stereo_signal())

    with mock.patch.object(DaArray, "compute") as compute:
        result = case.make_operation().process(dask_signal)
        compute.assert_not_called()
        result.compute()
        compute.assert_called_once()


@pytest.mark.parametrize("signal_kind", ("stereo", "multichannel"))
@pytest.mark.parametrize("case", _PAIRWISE_SPECTRAL_CASES, ids=lambda case: case.registry_name)
def test_pairwise_spectral_operation_shape(case: PairwiseSpectralCase, signal_kind: str) -> None:
    """Each input channel is paired with every input channel."""
    signal = case.make_stereo_signal() if signal_kind == "stereo" else case.make_multichannel_signal()

    result = run_operation_eager(case.make_operation(), signal)

    channel_count = signal.shape[0]
    frequency_bin_count = _PAIRWISE_N_FFT // 2 + 1
    assert result.shape == (channel_count * channel_count, frequency_bin_count)


class TestCoherenceOperation:
    """Coherence-specific validation and SciPy authority checks."""

    def test_init_custom_params(self) -> None:
        operation = Coherence(
            _SR,
            n_fft=_PAIRWISE_N_FFT,
            hop_length=512,
            win_length=_PAIRWISE_WINDOW_LENGTH,
            window="hamming",
            detrend="linear",
        )

        assert operation.hop_length == 512
        assert operation.window == "hamming"
        assert operation.detrend == "linear"
        assert operation.noverlap == _PAIRWISE_WINDOW_LENGTH - 512

    def test_noverlap_is_read_only(self) -> None:
        operation = _coherence_operation()

        with pytest.raises(AttributeError):
            setattr(operation, "noverlap", 0)

        assert operation.noverlap == _PAIRWISE_WINDOW_LENGTH - _PAIRWISE_HOP_LENGTH

    def test_content_matches_scipy_coherence(self) -> None:
        """Coherence matches ``scipy.signal.coherence`` at float precision."""
        signal = _cross_spectral_pair()
        result = run_operation_eager(_coherence_operation(), signal)

        assert np.all(result >= 0)
        assert np.all(result <= 1.000001)
        np.testing.assert_allclose(result[0].mean(), 1.0, atol=1e-10)
        np.testing.assert_allclose(result[3].mean(), 1.0, atol=1e-10)
        assert 0 < np.mean(result[1]) < 1

        _, coherence = ss.coherence(
            x=signal[:, np.newaxis],
            y=signal[np.newaxis, :],
            fs=_SR,
            nperseg=_PAIRWISE_WINDOW_LENGTH,
            noverlap=_PAIRWISE_WINDOW_LENGTH - _PAIRWISE_HOP_LENGTH,
            nfft=_PAIRWISE_N_FFT,
            window=_PAIRWISE_WINDOW,
            detrend=_PAIRWISE_DETREND,
        )
        expected = coherence.reshape(-1, coherence.shape[-1])
        # rtol=1e-6: wrapper equivalence with scipy.signal.coherence.
        np.testing.assert_allclose(result, expected, rtol=1e-6)


class TestCSDOperation:
    """CSD-specific validation and SciPy authority checks."""

    def test_init_custom_params(self) -> None:
        operation = CSD(
            _SR,
            n_fft=_PAIRWISE_N_FFT,
            hop_length=512,
            win_length=_PAIRWISE_WINDOW_LENGTH,
            window="hamming",
            detrend="linear",
            scaling="density",
            average="median",
        )

        assert operation.hop_length == 512
        assert operation.window == "hamming"
        assert operation.detrend == "linear"
        assert operation.scaling == "density"
        assert operation.average == "median"

    def test_content_matches_scipy_csd(self) -> None:
        """CSD matches ``scipy.signal.csd`` at float precision."""
        signal = _cross_spectral_pair()
        result = run_operation_eager(_csd_operation(), signal)

        _, scipy_csd = ss.csd(
            x=signal[:, np.newaxis, :],
            y=signal[np.newaxis, :, :],
            fs=_SR,
            nperseg=_PAIRWISE_WINDOW_LENGTH,
            noverlap=_PAIRWISE_WINDOW_LENGTH - _PAIRWISE_HOP_LENGTH,
            nfft=_PAIRWISE_N_FFT,
            window=_PAIRWISE_WINDOW,
            detrend=_PAIRWISE_DETREND,
            scaling=_PAIRWISE_SCALING,
            average=_PAIRWISE_AVERAGE,
        )
        expected = scipy_csd.transpose(1, 0, 2).reshape(-1, scipy_csd.shape[-1])
        # rtol=1e-6: wrapper equivalence with scipy.signal.csd.
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_auto_spectrum_peaks_at_signal_frequency(self) -> None:
        signal = _cross_spectral_pair()
        result = run_operation_eager(_csd_operation(), signal)
        frequency_bins = np.fft.rfftfreq(_PAIRWISE_N_FFT, 1.0 / _SR)

        first_peak = np.argmin(np.abs(frequency_bins - 1000))
        second_peak = np.argmin(np.abs(frequency_bins - 1100))

        assert np.argmax(np.abs(result[0])) == first_peak
        assert np.argmax(np.abs(result[3])) == second_peak


class TestTransferFunctionOperation:
    """Transfer-function-specific validation and SciPy authority checks."""

    def test_init_custom_params(self) -> None:
        operation = TransferFunction(
            _SR,
            n_fft=_PAIRWISE_N_FFT,
            hop_length=512,
            win_length=_PAIRWISE_WINDOW_LENGTH,
            window="hamming",
            detrend="linear",
            scaling="density",
            average="median",
        )

        assert operation.hop_length == 512
        assert operation.window == "hamming"
        assert operation.detrend == "linear"
        assert operation.scaling == "density"
        assert operation.average == "median"

    def test_gain_at_signal_frequency(self) -> None:
        """A gain-two system produces the expected forward and inverse gain."""
        signal = _transfer_pair()
        result = run_operation_eager(_transfer_function_operation(), signal)
        frequency_bins = np.fft.rfftfreq(_PAIRWISE_N_FFT, 1.0 / _SR)
        signal_bin = np.argmin(np.abs(frequency_bins - 1000))

        # rtol=0.2: deterministic noise and spectral leakage around the target bin.
        np.testing.assert_allclose(np.abs(result[1, signal_bin]), 2.0, rtol=0.2)
        np.testing.assert_allclose(np.abs(result[0, signal_bin]), 1.0, rtol=0.2)
        np.testing.assert_allclose(np.abs(result[3, signal_bin]), 1.0, rtol=0.2)
        np.testing.assert_allclose(np.abs(result[2, signal_bin]), 0.5, rtol=0.2)

    def test_content_matches_scipy_csd_welch_ratio(self) -> None:
        """Transfer function matches a direct SciPy CSD/PSD ratio."""
        signal = _transfer_pair()
        result = run_operation_eager(_transfer_function_operation(), signal)

        _, cross_spectrum = ss.csd(
            x=signal[:, np.newaxis, :],
            y=signal[np.newaxis, :, :],
            fs=_SR,
            nperseg=_PAIRWISE_WINDOW_LENGTH,
            noverlap=_PAIRWISE_WINDOW_LENGTH - _PAIRWISE_HOP_LENGTH,
            nfft=_PAIRWISE_N_FFT,
            window=_PAIRWISE_WINDOW,
            detrend=_PAIRWISE_DETREND,
            scaling=_PAIRWISE_SCALING,
            average=_PAIRWISE_AVERAGE,
            axis=-1,
        )
        _, power_spectrum = ss.welch(
            x=signal,
            fs=_SR,
            nperseg=_PAIRWISE_WINDOW_LENGTH,
            noverlap=_PAIRWISE_WINDOW_LENGTH - _PAIRWISE_HOP_LENGTH,
            nfft=_PAIRWISE_N_FFT,
            window=_PAIRWISE_WINDOW,
            detrend=_PAIRWISE_DETREND,
            scaling=_PAIRWISE_SCALING,
            average=_PAIRWISE_AVERAGE,
            axis=-1,
        )
        transfer = cross_spectrum / power_spectrum[np.newaxis, :, :]
        expected = transfer.transpose(1, 0, 2).reshape(-1, transfer.shape[-1])
        # rtol=1e-6: wrapper equivalence with the same SciPy algorithms.
        np.testing.assert_allclose(result, expected, rtol=1e-6)


class TestNOctSpectrumOperation:
    """NOctSpectrum operation: Layer 1 + Layer 2 + Layer 3 (mosqito reference)."""

    _NOCT_SR: int = 51200
    _FMIN: float = 24.0
    _FMAX: float = 12600
    _N: int = 3
    _G: int = 10
    _FR: int = 1000

    def _op(self) -> NOctSpectrum:
        return NOctSpectrum(
            self._NOCT_SR,
            fmin=self._FMIN,
            fmax=self._FMAX,
            n=self._N,
            G=self._G,
            fr=self._FR,
        )

    # -- Layer 1: Unit tests -----------------------------------------------

    def test_registry_returns_correct_class(self) -> None:
        """'noct_spectrum' registry key creates NOctSpectrum instance."""
        assert get_operation("noct_spectrum") == NOctSpectrum
        op = create_operation(
            "noct_spectrum",
            self._NOCT_SR,
            fmin=100.0,
            fmax=5000.0,
            n=1,
            G=20,
            fr=1000,
        )
        assert isinstance(op, NOctSpectrum)
        assert op.fmin == 100.0

    # -- Layer 2: Domain (immutability + lazy + shapes) ---------------------

    def test_preserves_immutability_and_dask_type(self) -> None:
        """Pillar 1: input data unchanged after NOctSpectrum; result is DaskArray."""
        pink, _ = _fractional_octave_noise(self._NOCT_SR)
        sig = np.array([pink])
        dask_sig = da_from_array(sig, chunks=(1, -1))
        input_copy = sig.copy()

        result = self._op().process(dask_sig)

        np.testing.assert_array_equal(dask_sig.compute(), input_copy)
        assert isinstance(result, DaArray)

    def test_delayed_execution_not_computed_early(self) -> None:
        """Pillar 1: Dask lazy evaluation preserved; no premature compute()."""
        pink, _ = _fractional_octave_noise(self._NOCT_SR)
        dask_sig = da_from_array(np.array([pink]), chunks=(1, -1))
        with mock.patch.object(DaArray, "compute") as mock_compute:
            result = self._op().process(dask_sig)
            mock_compute.assert_not_called()
            _ = result.compute()
            mock_compute.assert_called_once()

    # -- Layer 3: Numerical verification (mosqito reference) ----------------

    def test_mono_matches_mosqito_noct_spectrum(self) -> None:
        """Mono result matches direct mosqito noct_spectrum call.

        Tolerance: rtol=1e-6.
        """
        pink, _ = _fractional_octave_noise(self._NOCT_SR)
        sig = np.array([pink])
        dask_sig = da_from_array(sig, chunks=(1, -1))

        result_da = self._op().process(dask_sig)
        assert isinstance(result_da, DaArray)  # Pillar 1: Dask graph preserved
        result = result_da.compute()

        expected_spectrum, _ = noct_spectrum(
            sig=sig.T,
            fs=self._NOCT_SR,
            fmin=self._FMIN,
            fmax=self._FMAX,
            n=self._N,
            G=self._G,
            fr=self._FR,
        )
        assert result.shape == (1, expected_spectrum.shape[0])
        np.testing.assert_allclose(result[0], expected_spectrum, rtol=1e-6)

        _, center_freqs = _center_freq(
            fmin=self._FMIN,
            fmax=self._FMAX,
            n=self._N,
            G=self._G,
            fr=self._FR,
        )
        assert result.shape[1] == len(center_freqs)

    def test_stereo_matches_mosqito_per_channel(self) -> None:
        """Stereo result matches per-channel mosqito calls.

        Tolerance: rtol=1e-6.
        """
        pink, white = _fractional_octave_noise(self._NOCT_SR)
        sig = np.array([pink, white])
        dask_sig = da_from_array(sig, chunks=(2, -1))

        result_da = self._op().process(dask_sig)
        assert isinstance(result_da, DaArray)  # Pillar 1: Dask graph preserved
        result = result_da.compute()

        exp_ch1, _ = noct_spectrum(
            sig=sig[0:1].T,
            fs=self._NOCT_SR,
            fmin=self._FMIN,
            fmax=self._FMAX,
            n=self._N,
            G=self._G,
            fr=self._FR,
        )
        exp_ch2, _ = noct_spectrum(
            sig=sig[1:2].T,
            fs=self._NOCT_SR,
            fmin=self._FMIN,
            fmax=self._FMAX,
            n=self._N,
            G=self._G,
            fr=self._FR,
        )

        assert result.shape == (2, exp_ch1.shape[0])
        # rtol=1e-6: wrapper equivalence — same MoSQITo noct_spectrum call per channel
        np.testing.assert_allclose(result[0], exp_ch1, rtol=1e-6)
        np.testing.assert_allclose(result[1], exp_ch2, rtol=1e-6)
        assert not np.allclose(result[0], result[1])
