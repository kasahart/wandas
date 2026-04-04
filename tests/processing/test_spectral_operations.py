from unittest import mock

import numpy as np
import pytest
from dask.array.core import Array as DaArray
from mosqito.sound_level_meter import noct_spectrum, noct_synthesis
from mosqito.sound_level_meter.noct_spectrum._center_freq import _center_freq
from scipy import signal as ss
from scipy.signal import ShortTimeFFT as ScipySTFT
from scipy.signal import get_window

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

        result = fft.process_array(dask_input.compute()).compute()
        expected_freqs = self._N_FFT // 2 + 1
        assert result.shape == (1, expected_freqs)

    def test_fft_shape_stereo(self, stereo_sine_440_880hz_dask: tuple[DaArray, int]) -> None:
        """FFT output shape: (2, n_fft//2 + 1) for stereo."""
        dask_input, sr = stereo_sine_440_880hz_dask
        fft = FFT(sr, n_fft=self._N_FFT, window=self._WINDOW)

        result = fft.process_array(dask_input.compute()).compute()
        expected_freqs = self._N_FFT // 2 + 1
        assert result.shape == (2, expected_freqs)

    def test_fft_truncation_longer_than_n_fft(self) -> None:
        """FFT truncates signal longer than n_fft."""
        long_signal = np.random.default_rng(42).standard_normal(2048)
        fft_op = FFT(_SR, n_fft=1024)
        result = fft_op.process_array(np.array([long_signal])).compute()
        assert result.shape == (1, 1024 // 2 + 1)

    # -- Layer 3: Theoretical / numpy reference ----------------------------

    def test_fft_peak_at_correct_frequency_bin(self) -> None:
        """FFT peak at expected frequency bin for 500 Hz sine.

        Tolerance: peak index within +-1 bin of theoretical.
        """
        t = np.linspace(0, 1, _SR, endpoint=False)
        sig = np.array([4.0 * np.sin(2 * np.pi * self._FREQ * t)])
        fft = FFT(_SR, n_fft=self._N_FFT, window=self._WINDOW)

        result = fft.process_array(sig).compute()
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
        fft_result = fft_inst.process_array(np.array([cos_wave])).compute()

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

    def test_fft_window_function_changes_result(self) -> None:
        """Boxcar vs Hann windows produce different spectra but same peak."""
        t = np.linspace(0, 1, _SR, endpoint=False)
        sig = np.array([4.0 * np.sin(2 * np.pi * self._FREQ * t)])

        rect_result = FFT(_SR, n_fft=None, window="boxcar").process_array(sig).compute()
        hann_result = FFT(_SR, n_fft=None, window="hann").process_array(sig).compute()

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
        result = ifft.process_array(spectrum).compute()
        assert result.shape == (1, self._N_FFT)

    def test_ifft_shape_stereo(self) -> None:
        """IFFT output: (2, n_fft) for stereo spectrum."""
        spectrum = np.zeros((2, self._N_FFT // 2 + 1), dtype=complex)
        spectrum[0, 32] = 1.0
        spectrum[1, 16] = 1.0
        ifft = IFFT(_SR, n_fft=self._N_FFT, window=self._WINDOW)
        result = ifft.process_array(spectrum).compute()
        assert result.shape == (2, self._N_FFT)

    def test_ifft_without_n_fft_infers_length(self) -> None:
        """IFFT with n_fft=None infers: 2*(input_length-1)."""
        spectrum = np.zeros(513, dtype=complex)
        spectrum[50] = 1.0
        ifft = IFFT(_SR, n_fft=None)
        result = ifft.process_array(np.array([spectrum])).compute()
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
        result = ifft.process_array(np.array([spectrum])).compute()

        assert np.isrealobj(result)

        # Round-trip: FFT of IFFT result should peak at f0
        fft_of_result = np.fft.rfft(result[0])
        peak_idx = np.argmax(np.abs(fft_of_result))
        detected_freq = freq_bins[peak_idx]
        # rtol=0.1: IFFT round-trip with windowing introduces spectral leakage
        np.testing.assert_allclose(detected_freq, f0, rtol=0.1)


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
        stft = STFT(_SR)
        assert stft.sampling_rate == _SR
        assert stft.n_fft == 2048
        assert stft.hop_length == 512
        assert stft.win_length == 2048
        assert stft.window == "hann"

    def test_stft_init_custom_params(self) -> None:
        stft = STFT(_SR, n_fft=1024, hop_length=256, win_length=512, window="hamming")
        assert stft.n_fft == 1024
        assert stft.hop_length == 256
        assert stft.win_length == 512
        assert stft.window == "hamming"

    def test_istft_init_default_params(self) -> None:
        istft = ISTFT(_SR)
        assert istft.sampling_rate == _SR
        assert istft.n_fft == 2048
        assert istft.hop_length == 512
        assert istft.win_length == 2048
        assert istft.window == "hann"
        assert istft.length is None

    def test_istft_init_custom_params(self) -> None:
        istft = ISTFT(_SR, n_fft=1024, hop_length=256, win_length=512, window="hamming", length=16000)
        assert istft.n_fft == 1024
        assert istft.hop_length == 256
        assert istft.win_length == 512
        assert istft.window == "hamming"
        assert istft.length == 16000

    def test_stft_registry_returns_correct_class(self) -> None:
        assert get_operation("stft") == STFT
        assert get_operation("istft") == ISTFT
        stft_op = create_operation("stft", _SR, n_fft=512, hop_length=128)
        assert isinstance(stft_op, STFT)
        assert stft_op.n_fft == 512
        istft_op = create_operation("istft", _SR, n_fft=512, hop_length=128)
        assert isinstance(istft_op, ISTFT)
        assert istft_op.n_fft == 512

    def test_stft_negative_n_fft_raises(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            STFT(sampling_rate=44100, n_fft=-2048)
        error_msg = str(exc_info.value)
        assert "Invalid FFT size for STFT" in error_msg
        assert "-2048" in error_msg
        assert "Positive integer" in error_msg

    def test_stft_zero_n_fft_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid FFT size for STFT"):
            STFT(sampling_rate=44100, n_fft=0)

    def test_stft_win_length_greater_than_n_fft_raises(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            STFT(sampling_rate=44100, n_fft=1024, win_length=2048)
        error_msg = str(exc_info.value)
        assert "Invalid window length for STFT" in error_msg
        assert "win_length=2048" in error_msg
        assert "win_length <= n_fft" in error_msg

    def test_stft_negative_hop_length_raises(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            STFT(sampling_rate=44100, n_fft=2048, hop_length=-512)
        error_msg = str(exc_info.value)
        assert "Invalid hop length for STFT" in error_msg
        assert "-512" in error_msg
        assert "Positive integer" in error_msg

    def test_stft_hop_length_greater_than_win_length_raises(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            STFT(sampling_rate=44100, n_fft=2048, win_length=1024, hop_length=2048)
        error_msg = str(exc_info.value)
        assert "Invalid hop length for STFT" in error_msg
        assert "hop_length=2048" in error_msg
        assert "hop_length <= win_length" in error_msg
        assert "would create gaps" in error_msg

    def test_stft_negative_win_length_raises(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            STFT(sampling_rate=44100, n_fft=2048, win_length=-1024)
        error_msg = str(exc_info.value)
        assert "Invalid window length for STFT" in error_msg
        assert "Positive integer" in error_msg

    def test_stft_zero_win_length_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid window length for STFT"):
            STFT(sampling_rate=44100, n_fft=2048, win_length=0)

    def test_stft_win_length_too_small_raises(self) -> None:
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
        result = stft.process_array(sig).compute()

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
        result = stft.process_array(sig).compute()

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
        result = self._stft().process_array(sig_1d).compute()
        assert result.ndim == 3
        assert result.shape[0] == 1

    def test_istft_shape_matches_original(self) -> None:
        """ISTFT output shape close to original signal length."""
        sig, _ = self._make_mono()
        stft_data = self._stft().process_array(sig)
        result = self._istft().process_array(stft_data).compute()

        assert result.ndim == 2
        assert result.shape[0] == 1
        assert abs(result.shape[1] - sig.shape[1]) < self._WIN_LEN

    def test_istft_2d_input_reshaped(self) -> None:
        """2D (single-channel spectrogram) produces 2D output."""
        sig, _ = self._make_mono()
        stft_data = self._stft().process_array(sig).compute()
        stft_2d = stft_data[0]  # (freqs, frames)
        result = self._istft().process_array(stft_2d).compute()
        assert result.ndim == 2
        assert result.shape[0] == 1

    def test_istft_with_length_trims_output(self) -> None:
        """ISTFT with length parameter trims to exact output length."""
        sig, _ = self._make_mono()
        stft_data = self._stft().process_array(sig)
        result = self._istft(length=8000).process_array(stft_data).compute()
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

    def test_stft_amplitude_scaling_matches_input(self) -> None:
        """STFT peak amplitude in middle frame matches input amplitude.

        Tolerance: rtol=1e-10 — float64 precision.
        """
        amp = 2.0
        t = np.linspace(0, 1, _SR, endpoint=False)
        cos_wave = amp * np.cos(2 * np.pi * 500 * t)
        result_da = self._stft().process(cos_wave)
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
        stft_data = self._stft().process_array(sig)
        istft_data = self._istft().process_array(stft_data).compute()

        orig_length = sig.shape[1]
        reconstructed = istft_data[:, :orig_length]
        np.testing.assert_allclose(
            reconstructed[..., 16:-16],
            sig[..., 16:-16],
            rtol=1e-6,
            atol=1e-5,
        )


class TestNOctSynthesisOperation:
    """NOctSynthesis operation: Layer 1 + Layer 2 + Layer 3 (mosqito reference)."""

    _NOCT_SR: int = 48000
    _FMIN: float = 24.0
    _FMAX: float = 12600
    _N: int = 3
    _G: int = 10
    _FR: int = 1000

    def _make_pink_noise(self) -> tuple[NDArrayReal, NDArrayReal]:
        """Return (pink_noise, white_noise) arrays each of length _NOCT_SR."""
        rng = np.random.default_rng(42)
        white = rng.standard_normal(self._NOCT_SR)
        k = np.fft.rfftfreq(len(white))[1:]
        X = np.fft.rfft(white)  # noqa: N806
        S = 1.0 / np.sqrt(k)  # noqa: N806
        X[1:] *= S
        pink = np.fft.irfft(X, len(white))
        return pink / np.abs(pink).max(), white / np.abs(white).max()

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

    def test_init_stores_all_params(self) -> None:
        op = self._op()
        assert op.sampling_rate == self._NOCT_SR
        assert op.fmin == self._FMIN
        assert op.fmax == self._FMAX
        assert op.n == self._N
        assert op.G == self._G
        assert op.fr == self._FR

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
        pink, _ = self._make_pink_noise()
        sig = np.array([pink])
        dask_sig = da_from_array(sig, chunks=(1, 1000))
        input_copy = sig.copy()

        fft = FFT(self._NOCT_SR, n_fft=None, window="hann")
        spectrum = fft.process(dask_sig).compute()
        spec_copy = spectrum.copy()

        result = self._op().process(spectrum).compute()

        np.testing.assert_array_equal(sig, input_copy)
        np.testing.assert_array_equal(spectrum, spec_copy)
        assert result is not spectrum

    def test_delayed_execution_not_computed_early(self) -> None:
        """Pillar 1: Dask lazy evaluation preserved; no premature compute()."""
        pink, _ = self._make_pink_noise()
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
        spectrum = fft.process(da_from_array(np.array([test_signal]), chunks=(1, -1))).compute()
        assert spectrum.shape[-1] % 2 == 1
        result_da = self._op().process(da_from_array(spectrum, chunks=(1, -1)))
        assert isinstance(result_da, DaArray)  # Pillar 1: Dask graph preserved
        result = result_da.compute()
        assert result.shape[0] == 1

    # -- Layer 3: Numerical verification (mosqito reference) ----------------

    def test_mono_matches_mosqito_noct_synthesis(self) -> None:
        """Mono result matches direct mosqito noct_synthesis call."""
        pink, _ = self._make_pink_noise()
        sig = np.array([pink])
        dask_sig = da_from_array(sig, chunks=(1, 1000))

        fft = FFT(self._NOCT_SR, n_fft=None, window="hann")
        spectrum = fft.process(dask_sig).compute()
        result_da = self._op().process(spectrum)
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
        pink, white = self._make_pink_noise()
        sig = np.array([pink, white])
        dask_sig = da_from_array(sig, chunks=(2, 1000))

        fft = FFT(self._NOCT_SR, n_fft=None, window="hann")
        spectrum = fft.process(dask_sig).compute()
        result_da = self._op().process(spectrum)
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
            welch._process_array([0.0] * 100)  # ty: ignore[invalid-argument-type]

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
        result = self._op().process_array(sig).compute()
        expected_bins = self._N_FFT // 2 + 1
        assert result.shape == (1, expected_bins)

    def test_shape_stereo(self) -> None:
        sig, _ = self._make_stereo()
        result = self._op().process_array(sig).compute()
        expected_bins = self._N_FFT // 2 + 1
        assert result.shape == (2, expected_bins)

    # -- Layer 3: Numerical verification (scipy reference) ------------------

    def test_peak_frequency_detected_correctly(self) -> None:
        """Welch PSD peak at 1 kHz for a 1 kHz sine.

        Tolerance: rtol=0.05 — frequency bin resolution.
        """
        sig, _ = self._make_mono(freq=1000.0)
        result = self._op().process_array(sig).compute()
        freq_bins = np.fft.rfftfreq(self._N_FFT, 1.0 / _SR)
        detected_freq = freq_bins[np.argmax(result[0])]
        np.testing.assert_allclose(detected_freq, 1000.0, rtol=0.05)

    def test_stereo_second_channel_peak_at_2khz(self) -> None:
        """Welch PSD peak at 2 kHz for stereo second channel.

        Tolerance: rtol=0.05 — frequency bin resolution.
        """
        sig, _ = self._make_stereo()
        result = self._op().process_array(sig).compute()
        freq_bins = np.fft.rfftfreq(self._N_FFT, 1.0 / _SR)
        detected_freq = freq_bins[np.argmax(result[1])]
        np.testing.assert_allclose(detected_freq, 2000.0, rtol=0.05)

    def test_matches_scipy_welch_reference(self) -> None:
        """Welch output matches scipy.signal.welch with equivalent params.

        Tolerance: rtol=1e-6 — float64 precision.
        """

        sig, _ = self._make_stereo()
        result = self._op().process_array(sig).compute()

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
        result = self._op().process_array(np.array([sine])).compute()

        freq_bins = np.fft.rfftfreq(self._N_FFT, 1.0 / _SR)
        peak_idx = np.argmax(result[0])
        np.testing.assert_allclose(freq_bins[peak_idx], 1000.0, rtol=1e-10)
        np.testing.assert_allclose(result[0, peak_idx], amp, rtol=1e-10)


class TestCoherenceOperation:
    """Coherence operation: Layer 1 + Layer 2 + Layer 3 (scipy reference)."""

    _N_FFT: int = 1024
    _HOP: int = 256
    _WIN_LEN: int = 1024
    _WINDOW: str = "hann"
    _DETREND: str = "constant"

    def _make_stereo(self) -> tuple[NDArrayReal, DaArray]:
        t = np.linspace(0, 1, _SR, endpoint=False)
        sig: NDArrayReal = np.array([np.sin(2 * np.pi * 1000 * t), np.sin(2 * np.pi * 1100 * t)])
        return sig, da_from_array(sig, chunks=(2, 1000))

    def _make_multi(self) -> tuple[NDArrayReal, DaArray]:
        t = np.linspace(0, 1, _SR, endpoint=False)
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(_SR) * 0.1
        sig: NDArrayReal = np.array(
            [
                np.sin(2 * np.pi * 1000 * t),
                np.sin(2 * np.pi * 1100 * t),
                noise,
            ]
        )
        return sig, da_from_array(sig, chunks=(3, 1000))

    def _op(self) -> Coherence:
        return Coherence(
            _SR,
            n_fft=self._N_FFT,
            hop_length=self._HOP,
            win_length=self._WIN_LEN,
            window=self._WINDOW,
            detrend=self._DETREND,
        )

    # -- Layer 1: Unit tests -----------------------------------------------

    def test_init_stores_params(self) -> None:
        op = self._op()
        assert op.sampling_rate == _SR
        assert op.n_fft == self._N_FFT
        assert op.hop_length == self._HOP
        assert op.win_length == self._WIN_LEN
        assert op.window == self._WINDOW
        assert op.detrend == self._DETREND

    def test_init_custom_params(self) -> None:
        op = Coherence(
            _SR,
            n_fft=self._N_FFT,
            hop_length=512,
            win_length=self._WIN_LEN,
            window="hamming",
            detrend="linear",
        )
        assert op.hop_length == 512
        assert op.window == "hamming"
        assert op.detrend == "linear"

    def test_registry_returns_correct_class(self) -> None:
        assert get_operation("coherence") == Coherence
        op = create_operation(
            "coherence",
            _SR,
            n_fft=512,
            hop_length=128,
            win_length=512,
            window="hamming",
            detrend="linear",
        )
        assert isinstance(op, Coherence)
        assert op.n_fft == 512

    # -- Layer 2: Domain (immutability + lazy + shapes) ---------------------

    def test_preserves_immutability_and_dask_type(self) -> None:
        sig, dask_sig = self._make_stereo()
        input_copy = sig.copy()
        result = self._op().process(dask_sig)
        np.testing.assert_array_equal(dask_sig.compute(), input_copy)
        assert isinstance(result, DaArray)

    def test_delayed_execution_not_computed_early(self) -> None:
        _, dask_sig = self._make_stereo()
        with mock.patch.object(DaArray, "compute") as mock_compute:
            result = self._op().process(dask_sig)
            mock_compute.assert_not_called()
            _ = result.compute()
            mock_compute.assert_called_once()

    def test_shape_stereo(self) -> None:
        sig, _ = self._make_stereo()
        result = self._op().process_array(sig).compute()
        n_ch = sig.shape[0]
        n_freqs = self._N_FFT // 2 + 1
        assert result.shape == (n_ch * n_ch, n_freqs)

    def test_shape_multi_channel(self) -> None:
        sig, _ = self._make_multi()
        result = self._op().process_array(sig).compute()
        n_ch = sig.shape[0]
        n_freqs = self._N_FFT // 2 + 1
        assert result.shape == (n_ch * n_ch, n_freqs)

    # -- Layer 3: Numerical verification (scipy reference) ------------------

    def test_content_matches_scipy_coherence(self) -> None:
        """Coherence matches scipy.signal.coherence.

        Tolerance: rtol=1e-6.
        """

        sig, _ = self._make_stereo()
        result = self._op().process_array(sig).compute()

        assert np.all(result >= 0)
        assert np.all(result <= 1.000001)

        # Self-coherence ~1
        np.testing.assert_allclose(result[0, :].mean(), 1.0, atol=1e-10)
        np.testing.assert_allclose(result[3, :].mean(), 1.0, atol=1e-10)

        # Cross-coherence between 0 and 1
        cross = np.mean(result[1, :])
        assert 0 < cross < 1

        _, coh = ss.coherence(
            x=sig[:, np.newaxis],
            y=sig[np.newaxis, :],
            fs=_SR,
            nperseg=self._WIN_LEN,
            noverlap=self._WIN_LEN - self._HOP,
            nfft=self._N_FFT,
            window=self._WINDOW,
            detrend=self._DETREND,
        )
        expected = coh.reshape(-1, coh.shape[-1])
        # rtol=1e-6: wrapper equivalence — same scipy.signal.coherence algorithm
        np.testing.assert_allclose(result, expected, rtol=1e-6)


class TestCSDOperation:
    """CSD (Cross-Spectral Density) operation: Layer 1 + Layer 2 + Layer 3 (scipy ref)."""

    _N_FFT: int = 1024
    _HOP: int = 256
    _WIN_LEN: int = 1024
    _WINDOW: str = "hann"
    _DETREND: str = "constant"
    _SCALING: str = "spectrum"
    _AVERAGE: str = "mean"

    def _make_stereo(self) -> tuple[NDArrayReal, DaArray]:
        t = np.linspace(0, 1, _SR, endpoint=False)
        sig: NDArrayReal = np.array([np.sin(2 * np.pi * 1000 * t), np.sin(2 * np.pi * 1100 * t)])
        return sig, da_from_array(sig, chunks=(2, 1000))

    def _make_multi(self) -> tuple[NDArrayReal, DaArray]:
        t = np.linspace(0, 1, _SR, endpoint=False)
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(_SR) * 0.1
        sig: NDArrayReal = np.array(
            [
                np.sin(2 * np.pi * 1000 * t),
                np.sin(2 * np.pi * 1100 * t),
                noise,
            ]
        )
        return sig, da_from_array(sig, chunks=(3, 1000))

    def _op(self) -> CSD:
        return CSD(
            _SR,
            n_fft=self._N_FFT,
            hop_length=self._HOP,
            win_length=self._WIN_LEN,
            window=self._WINDOW,
            detrend=self._DETREND,
            scaling=self._SCALING,
            average=self._AVERAGE,
        )

    # -- Layer 1: Unit tests -----------------------------------------------

    def test_init_stores_params(self) -> None:
        op = self._op()
        assert op.sampling_rate == _SR
        assert op.n_fft == self._N_FFT
        assert op.hop_length == self._HOP
        assert op.win_length == self._WIN_LEN
        assert op.window == self._WINDOW
        assert op.detrend == self._DETREND
        assert op.scaling == self._SCALING
        assert op.average == self._AVERAGE

    def test_init_custom_params(self) -> None:
        op = CSD(
            _SR,
            n_fft=self._N_FFT,
            hop_length=512,
            win_length=self._WIN_LEN,
            window="hamming",
            detrend="linear",
            scaling="density",
            average="median",
        )
        assert op.hop_length == 512
        assert op.window == "hamming"
        assert op.detrend == "linear"
        assert op.scaling == "density"
        assert op.average == "median"

    def test_registry_returns_correct_class(self) -> None:
        assert get_operation("csd") == CSD
        op = create_operation(
            "csd",
            _SR,
            n_fft=512,
            hop_length=128,
            win_length=512,
            window="hamming",
            detrend="linear",
            scaling="density",
            average="median",
        )
        assert isinstance(op, CSD)
        assert op.n_fft == 512

    # -- Layer 2: Domain (immutability + lazy + shapes) ---------------------

    def test_preserves_immutability_and_dask_type(self) -> None:
        sig, dask_sig = self._make_stereo()
        input_copy = sig.copy()
        result = self._op().process(dask_sig)
        np.testing.assert_array_equal(dask_sig.compute(), input_copy)
        assert isinstance(result, DaArray)

    def test_delayed_execution_not_computed_early(self) -> None:
        _, dask_sig = self._make_stereo()
        with mock.patch.object(DaArray, "compute") as mock_compute:
            result = self._op().process(dask_sig)
            mock_compute.assert_not_called()
            _ = result.compute()
            mock_compute.assert_called_once()

    def test_shape_stereo(self) -> None:
        sig, _ = self._make_stereo()
        result = self._op().process_array(sig).compute()
        n_ch = sig.shape[0]
        n_freqs = self._N_FFT // 2 + 1
        assert result.shape == (n_ch * n_ch, n_freqs)

    def test_shape_multi_channel(self) -> None:
        sig, _ = self._make_multi()
        result = self._op().process_array(sig).compute()
        n_ch = sig.shape[0]
        n_freqs = self._N_FFT // 2 + 1
        assert result.shape == (n_ch * n_ch, n_freqs)

    # -- Layer 3: Numerical verification (scipy reference) ------------------

    def test_content_matches_scipy_csd(self) -> None:
        """CSD matches scipy.signal.csd.

        Tolerance: rtol=1e-6.
        """

        sig, _ = self._make_stereo()
        result = self._op().process_array(sig).compute()

        _, csd_expected = ss.csd(
            x=sig[:, np.newaxis, :],
            y=sig[np.newaxis, :, :],
            fs=_SR,
            nperseg=self._WIN_LEN,
            noverlap=self._WIN_LEN - self._HOP,
            nfft=self._N_FFT,
            window=self._WINDOW,
            detrend=self._DETREND,
            scaling=self._SCALING,
            average=self._AVERAGE,
        )
        expected = csd_expected.transpose(1, 0, 2).reshape(-1, csd_expected.shape[-1])
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_auto_spectrum_peaks_at_signal_frequency(self) -> None:
        """Auto-CSD peaks at the respective signal frequencies."""
        sig, _ = self._make_stereo()
        result = self._op().process_array(sig).compute()
        freq_bins = np.fft.rfftfreq(self._N_FFT, 1.0 / _SR)

        idx_1000 = np.argmin(np.abs(freq_bins - 1000))
        idx_1100 = np.argmin(np.abs(freq_bins - 1100))

        assert np.argmax(np.abs(result[0])) == idx_1000  # ch0 auto
        assert np.argmax(np.abs(result[3])) == idx_1100  # ch1 auto


class TestTransferFunctionOperation:
    """Transfer function operation: Layer 1 + Layer 2 + Layer 3 (scipy ref)."""

    _N_FFT: int = 1024
    _HOP: int = 256
    _WIN_LEN: int = 1024
    _WINDOW: str = "hann"
    _DETREND: str = "constant"
    _SCALING: str = "spectrum"
    _AVERAGE: str = "mean"

    def _make_stereo(self) -> tuple[NDArrayReal, DaArray]:
        """Input + output pair: gain=2 with small noise."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, 1, _SR, endpoint=False)
        inp = np.sin(2 * np.pi * 1000 * t)
        out = 2 * inp + 0.1 * rng.standard_normal(len(t))
        sig: NDArrayReal = np.array([inp, out])
        return sig, da_from_array(sig, chunks=(2, 1000))

    def _make_multi(self) -> tuple[NDArrayReal, DaArray]:
        rng = np.random.default_rng(42)
        t = np.linspace(0, 1, _SR, endpoint=False)
        i1 = np.sin(2 * np.pi * 1000 * t)
        i2 = np.sin(2 * np.pi * 1500 * t)
        o1 = 2 * i1 + 0.5 * i2 + 0.1 * rng.standard_normal(len(t))
        o2 = 0.3 * i1 + 1.5 * i2 + 0.1 * rng.standard_normal(len(t))
        sig: NDArrayReal = np.array([i1, i2, o1, o2])
        return sig, da_from_array(sig, chunks=(4, 1000))

    def _op(self) -> TransferFunction:
        return TransferFunction(
            _SR,
            n_fft=self._N_FFT,
            hop_length=self._HOP,
            win_length=self._WIN_LEN,
            window=self._WINDOW,
            detrend=self._DETREND,
            scaling=self._SCALING,
            average=self._AVERAGE,
        )

    # -- Layer 1: Unit tests -----------------------------------------------

    def test_init_stores_params(self) -> None:
        op = self._op()
        assert op.sampling_rate == _SR
        assert op.n_fft == self._N_FFT
        assert op.hop_length == self._HOP
        assert op.win_length == self._WIN_LEN
        assert op.window == self._WINDOW
        assert op.detrend == self._DETREND
        assert op.scaling == self._SCALING
        assert op.average == self._AVERAGE

    def test_init_custom_params(self) -> None:
        op = TransferFunction(
            _SR,
            n_fft=self._N_FFT,
            hop_length=512,
            win_length=self._WIN_LEN,
            window="hamming",
            detrend="linear",
            scaling="density",
            average="median",
        )
        assert op.hop_length == 512
        assert op.window == "hamming"
        assert op.scaling == "density"
        assert op.average == "median"

    def test_registry_returns_correct_class(self) -> None:
        assert get_operation("transfer_function") == TransferFunction
        op = create_operation(
            "transfer_function",
            _SR,
            n_fft=512,
            hop_length=128,
            win_length=512,
            window="hamming",
            detrend="linear",
            scaling="density",
            average="median",
        )
        assert isinstance(op, TransferFunction)
        assert op.n_fft == 512

    # -- Layer 2: Domain (immutability + lazy + shapes) ---------------------

    def test_preserves_immutability_and_dask_type(self) -> None:
        sig, dask_sig = self._make_stereo()
        input_copy = sig.copy()
        result = self._op().process(dask_sig)
        np.testing.assert_array_equal(dask_sig.compute(), input_copy)
        assert isinstance(result, DaArray)

    def test_delayed_execution_not_computed_early(self) -> None:
        _, dask_sig = self._make_stereo()
        with mock.patch.object(DaArray, "compute") as mock_compute:
            result = self._op().process(dask_sig)
            mock_compute.assert_not_called()
            _ = result.compute()
            mock_compute.assert_called_once()

    def test_shape_stereo(self) -> None:
        sig, _ = self._make_stereo()
        result = self._op().process_array(sig).compute()
        n_ch = sig.shape[0]
        n_freqs = self._N_FFT // 2 + 1
        assert result.shape == (n_ch * n_ch, n_freqs)

    def test_shape_multi_channel(self) -> None:
        sig, _ = self._make_multi()
        result = self._op().process_array(sig).compute()
        n_ch = sig.shape[0]
        n_freqs = self._N_FFT // 2 + 1
        assert result.shape == (n_ch * n_ch, n_freqs)

    # -- Layer 3: Numerical verification (scipy reference) ------------------

    def test_gain_at_signal_frequency(self) -> None:
        """Transfer function gain ≈ 2 at 1 kHz for a gain-2 system.

        Tolerance: rtol=0.2 — noise in simulated system.
        """
        sig, _ = self._make_stereo()
        result = self._op().process_array(sig).compute()
        freq_bins = np.fft.rfftfreq(self._N_FFT, 1.0 / _SR)
        idx_1000 = np.argmin(np.abs(freq_bins - 1000))

        h_in_to_out = result[1, idx_1000]  # ch0 -> ch1
        # rtol=0.2: spectral leakage at non-exact FFT bin boundaries
        np.testing.assert_allclose(np.abs(h_in_to_out), 2.0, rtol=0.2)

        h_self_in = result[0, idx_1000]  # ch0 -> ch0
        np.testing.assert_allclose(np.abs(h_self_in), 1.0, rtol=0.2)

        h_self_out = result[3, idx_1000]  # ch1 -> ch1
        np.testing.assert_allclose(np.abs(h_self_out), 1.0, rtol=0.2)

        h_out_to_in = result[2, idx_1000]  # ch1 -> ch0
        np.testing.assert_allclose(np.abs(h_out_to_in), 0.5, rtol=0.2)

    def test_content_matches_scipy_csd_welch_ratio(self) -> None:
        """Transfer function H = P_yx / P_xx matches manual scipy computation.

        Tolerance: rtol=1e-6 — float64 precision.
        """

        sig, _ = self._make_stereo()
        result = self._op().process_array(sig).compute()

        _, p_yx = ss.csd(
            x=sig[:, np.newaxis, :],
            y=sig[np.newaxis, :, :],
            fs=_SR,
            nperseg=self._WIN_LEN,
            noverlap=self._WIN_LEN - self._HOP,
            nfft=self._N_FFT,
            window=self._WINDOW,
            detrend=self._DETREND,
            scaling=self._SCALING,
            average=self._AVERAGE,
            axis=-1,
        )
        _, p_xx = ss.welch(
            x=sig,
            fs=_SR,
            nperseg=self._WIN_LEN,
            noverlap=self._WIN_LEN - self._HOP,
            nfft=self._N_FFT,
            window=self._WINDOW,
            detrend=self._DETREND,
            scaling=self._SCALING,
            average=self._AVERAGE,
            axis=-1,
        )
        h_f = p_yx / p_xx[np.newaxis, :, :]
        expected = h_f.transpose(1, 0, 2).reshape(-1, h_f.shape[-1])
        # rtol=1e-6: wrapper equivalence — same scipy CSD/PSD ratio computation
        np.testing.assert_allclose(result, expected, rtol=1e-6)


class TestNOctSpectrumOperation:
    """NOctSpectrum operation: Layer 1 + Layer 2 + Layer 3 (mosqito reference)."""

    _NOCT_SR: int = 51200
    _FMIN: float = 24.0
    _FMAX: float = 12600
    _N: int = 3
    _G: int = 10
    _FR: int = 1000

    def _make_pink_noise(self) -> tuple[NDArrayReal, NDArrayReal]:
        rng = np.random.default_rng(42)
        white = rng.standard_normal(self._NOCT_SR)
        k = np.fft.rfftfreq(len(white))[1:]
        X = np.fft.rfft(white)  # noqa: N806
        S = 1.0 / np.sqrt(k)  # noqa: N806
        X[1:] *= S
        pink = np.fft.irfft(X, len(white))
        return pink / np.abs(pink).max(), white / np.abs(white).max()

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

    def test_init_stores_all_params(self) -> None:
        """All constructor parameters are stored as attributes."""
        op = self._op()
        assert op.sampling_rate == self._NOCT_SR
        assert op.fmin == self._FMIN
        assert op.fmax == self._FMAX
        assert op.n == self._N
        assert op.G == self._G
        assert op.fr == self._FR

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
        pink, _ = self._make_pink_noise()
        sig = np.array([pink])
        dask_sig = da_from_array(sig, chunks=(1, -1))
        input_copy = sig.copy()

        result = self._op().process(dask_sig)

        np.testing.assert_array_equal(dask_sig.compute(), input_copy)
        assert isinstance(result, DaArray)

    def test_delayed_execution_not_computed_early(self) -> None:
        """Pillar 1: Dask lazy evaluation preserved; no premature compute()."""
        pink, _ = self._make_pink_noise()
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
        pink, _ = self._make_pink_noise()
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
        pink, white = self._make_pink_noise()
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
