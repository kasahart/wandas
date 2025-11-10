from unittest import mock

import dask.array as da
import numpy as np
from dask.array.core import Array as DaArray
from mosqito.sound_level_meter import noct_spectrum, noct_synthesis
from mosqito.sound_level_meter.noct_spectrum._center_freq import _center_freq

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
from wandas.utils.types import NDArrayComplex, NDArrayReal

_da_from_array = da.from_array  # type: ignore [unused-ignore]


class TestGetDisplayNames:
    """Test display names for all spectral operations."""

    def test_all_display_names(self) -> None:
        """Test that all operations have appropriate display names."""
        sr = 16000
        assert FFT(sr).get_display_name() == "FFT"
        assert IFFT(sr).get_display_name() == "iFFT"
        assert STFT(sr).get_display_name() == "STFT"
        assert ISTFT(sr).get_display_name() == "iSTFT"
        assert Welch(sr).get_display_name() == "PS"
        assert NOctSpectrum(sr, 24, 12600).get_display_name() == "Oct"
        assert NOctSynthesis(sr, 24, 12600).get_display_name() == "Octs"
        assert Coherence(sr).get_display_name() == "Coh"
        assert CSD(sr).get_display_name() == "CSD"
        assert TransferFunction(sr).get_display_name() == "H"


class TestFFTOperation:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.n_fft: int = 1024
        self.window: str = "hann"
        self.fft = FFT(self.sample_rate, n_fft=self.n_fft, window=self.window)

        self.freq: float = 500
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        self.signal_mono: NDArrayReal = (
            np.array([np.sin(2 * np.pi * self.freq * t)]) * 4
        )

        self.signal_stereo: NDArrayReal = np.array(
            [
                np.sin(2 * np.pi * self.freq * t),
                np.sin(2 * np.pi * self.freq * 2 * t),
            ]
        )

        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=-1)
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=-1)

    def test_initialization(self) -> None:
        """Test FFT initialization with different parameters."""
        fft = FFT(self.sample_rate)
        assert fft.sampling_rate == self.sample_rate
        assert fft.n_fft is None
        assert fft.window == "hann"

        custom_fft = FFT(self.sample_rate, n_fft=2048, window="hamming")
        assert custom_fft.n_fft == 2048
        assert custom_fft.window == "hamming"

    def test_fft_shape(self) -> None:
        """Test FFT output shape."""
        fft_result = self.fft.process_array(self.signal_mono).compute()

        expected_freqs = self.n_fft // 2 + 1
        assert fft_result.shape == (1, expected_freqs)

        fft_result_stereo = self.fft.process_array(self.signal_stereo).compute()
        assert fft_result_stereo.shape == (2, expected_freqs)

    def test_fft_content(self) -> None:
        """Test FFT content correctness."""
        fft_result = self.fft.process_array(self.signal_mono).compute()

        freq_bins = np.fft.rfftfreq(self.n_fft, 1.0 / self.sample_rate)

        target_idx = np.argmin(np.abs(freq_bins - self.freq))

        magnitude = np.abs(fft_result[0])

        peak_idx = np.argmax(magnitude)
        assert abs(peak_idx - target_idx) <= 1

        mask = np.ones_like(magnitude, dtype=bool)
        region = 5
        lower = int(max(0, int(peak_idx - region)))
        upper = int(min(len(magnitude), int(peak_idx + region + 1)))
        mask[lower:upper] = False

        assert np.max(magnitude[mask]) < 0.1 * magnitude[peak_idx]

    def test_amplitude_scaling(self) -> None:
        """Test that FFT amplitude scaling is correct."""
        fft_inst = FFT(self.sample_rate, n_fft=None, window=self.window)
        amp = 2.0
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        cos_wave = amp * np.cos(2 * np.pi * self.freq * t)

        from scipy.signal import get_window

        win = get_window(self.window, len(cos_wave))
        scaled_cos = cos_wave * win
        scaling_factor = np.sum(win)

        fft_result = fft_inst.process_array(np.array([cos_wave])).compute()

        expected_fft: NDArrayComplex = np.fft.rfft(scaled_cos)
        expected_fft[1:-1] *= 2.0
        expected_fft /= scaling_factor

        np.testing.assert_allclose(fft_result[0], expected_fft, rtol=1e-10)

        peak_idx = np.argmax(np.abs(fft_result[0]))
        peak_mag = np.abs(fft_result[0, peak_idx])
        expected_mag = amp

        np.testing.assert_allclose(peak_mag, expected_mag, rtol=0.1)

    def test_delayed_execution(self) -> None:
        """Test that FFT operation uses dask's delayed execution."""
        with mock.patch.object(DaArray, "compute") as mock_compute:
            result = self.fft.process(self.dask_mono)
            mock_compute.assert_not_called()

            assert isinstance(result, DaArray)

            _ = result.compute()
            mock_compute.assert_called_once()

    def test_window_function_effect(self) -> None:
        """Test different window functions have different effects."""
        rect_fft = FFT(self.sample_rate, n_fft=None, window="boxcar")
        rect_result = rect_fft.process_array(self.signal_mono).compute()

        hann_fft = FFT(self.sample_rate, n_fft=None, window="hann")
        hann_result = hann_fft.process_array(self.signal_mono).compute()

        assert not np.allclose(rect_result, hann_result)

        rect_mag = np.abs(rect_result[0])
        hann_mag = np.abs(hann_result[0])

        np.testing.assert_allclose(rect_mag.max(), 4, rtol=0.1)
        np.testing.assert_allclose(hann_mag.max(), 4, rtol=0.1)

    def test_operation_registry(self) -> None:
        """Test that FFT is properly registered in the operation registry."""
        assert get_operation("fft") == FFT

        fft_op = create_operation("fft", self.sample_rate, n_fft=512, window="hamming")

        assert isinstance(fft_op, FFT)
        assert fft_op.sampling_rate == self.sample_rate
        assert fft_op.n_fft == 512
        assert fft_op.window == "hamming"

    def test_negative_n_fft_error_message(self) -> None:
        """Test that negative n_fft provides helpful error message."""
        import pytest

        with pytest.raises(ValueError) as exc_info:
            FFT(sampling_rate=44100, n_fft=-1024)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Invalid FFT size" in error_msg
        assert "-1024" in error_msg
        # Check WHY
        assert "Positive integer" in error_msg
        # Check HOW
        assert "Common values:" in error_msg
        assert "2048" in error_msg

    def test_zero_n_fft_error_message(self) -> None:
        """Test that zero n_fft provides helpful error message."""
        import pytest

        with pytest.raises(ValueError) as exc_info:
            FFT(sampling_rate=44100, n_fft=0)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Invalid FFT size" in error_msg
        # Check WHY
        assert "Positive integer" in error_msg

    def test_fft_with_truncation(self) -> None:
        """Test that FFT truncates input when it exceeds n_fft."""
        # Create signal longer than n_fft
        long_signal = np.random.randn(2048)
        fft_op = FFT(self.sample_rate, n_fft=1024)

        result = fft_op.process_array(np.array([long_signal])).compute()

        # Output should have n_fft // 2 + 1 frequency bins
        expected_bins = 1024 // 2 + 1
        assert result.shape == (1, expected_bins)


class TestIFFTOperation:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.n_fft: int = 1024
        self.window: str = "hann"

        # Create frequency domain signal
        freq_bins = np.fft.rfftfreq(self.n_fft, 1.0 / self.sample_rate)
        f0 = 500.0  # Target frequency
        target_idx = np.argmin(np.abs(freq_bins - f0))

        # Create complex spectrum with single peak at f0
        spectrum = np.zeros(self.n_fft // 2 + 1, dtype=complex)
        spectrum[target_idx] = 1.0

        # For stereo test
        spectrum2 = np.zeros(self.n_fft // 2 + 1, dtype=complex)
        spectrum2[target_idx // 2] = 1.0

        self.signal_mono: NDArrayComplex = np.array([spectrum])
        self.signal_stereo: NDArrayComplex = np.array([spectrum, spectrum2])

        # Create dask arrays
        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=-1)
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=-1)

        # Initialize IFFT
        self.ifft = IFFT(self.sample_rate, n_fft=self.n_fft, window=self.window)

    def test_initialization(self) -> None:
        """Test IFFT initialization with different parameters."""
        ifft = IFFT(self.sample_rate)
        assert ifft.sampling_rate == self.sample_rate
        assert ifft.n_fft is None
        assert ifft.window == "hann"

        custom_ifft = IFFT(self.sample_rate, n_fft=2048, window="hamming")
        assert custom_ifft.n_fft == 2048
        assert custom_ifft.window == "hamming"

    def test_ifft_shape(self) -> None:
        """Test IFFT output shape."""
        ifft_result = self.ifft.process_array(self.signal_mono).compute()

        # Expected time domain signal length
        expected_length = self.n_fft
        assert ifft_result.shape == (1, expected_length)

        ifft_result_stereo = self.ifft.process_array(self.signal_stereo).compute()
        assert ifft_result_stereo.shape == (2, expected_length)

    def test_ifft_content(self) -> None:
        """Test that IFFT properly transforms frequency domain to time domain."""
        # Process the mono signal
        ifft_result = self.ifft.process_array(self.signal_mono).compute()

        # Check that the result is real
        assert np.isrealobj(ifft_result)

        # For a single frequency component, we expect a sinusoidal time signal
        # Find the peak frequency in the time domain by FFT
        fft_of_result = np.fft.rfft(ifft_result[0])
        peak_idx = np.argmax(np.abs(fft_of_result))
        freq_bins = np.fft.rfftfreq(len(ifft_result[0]), 1.0 / self.sample_rate)
        detected_freq = freq_bins[peak_idx]

        # Check frequency matches our input
        np.testing.assert_allclose(detected_freq, 500.0, rtol=1e-1)

    def test_delayed_execution(self) -> None:
        """Test that IFFT operation uses dask's delayed execution."""
        with mock.patch.object(DaArray, "compute") as mock_compute:
            result = self.ifft.process(self.dask_mono)
            mock_compute.assert_not_called()

            assert isinstance(result, DaArray)

            _ = result.compute()
            mock_compute.assert_called_once()

    def test_1d_input_handling(self) -> None:
        """Test that 1D input is properly reshaped."""
        signal_1d = np.zeros((1, self.n_fft // 2 + 1), dtype=complex)
        signal_1d[0, 5] = 1.0  # Add a frequency component
        dask_signal_1d: DaArray = _da_from_array(signal_1d, chunks=-1)

        ifft_result = self.ifft.process(dask_signal_1d).compute()

        assert ifft_result.ndim == 2
        assert ifft_result.shape[0] == 1

    def test_operation_registry(self) -> None:
        """Test that IFFT is properly registered in the operation registry."""
        assert get_operation("ifft") == IFFT

        ifft_op = create_operation(
            "ifft", self.sample_rate, n_fft=512, window="hamming"
        )

        assert isinstance(ifft_op, IFFT)
        assert ifft_op.sampling_rate == self.sample_rate
        assert ifft_op.n_fft == 512
        assert ifft_op.window == "hamming"

    def test_ifft_without_n_fft(self) -> None:
        """Test IFFT when n_fft is None - uses input shape for calculation."""
        # Create a frequency domain signal
        spectrum = np.zeros(513, dtype=complex)
        spectrum[50] = 1.0  # Single frequency component

        # Initialize IFFT without n_fft
        ifft_no_nfft = IFFT(self.sample_rate, n_fft=None)

        result = ifft_no_nfft.process_array(np.array([spectrum])).compute()

        # Expected output length: 2 * (input_length - 1) = 2 * (513 - 1) = 1024
        assert result.shape == (1, 1024)


class TestSTFTOperation:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.n_fft: int = 1024
        self.hop_length: int = 256
        self.win_length: int = 1024
        self.window: str = "hann"
        self.boundary: str | None = "zeros"

        # Create a test signal (1 second sine wave at 440 Hz)
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        self.signal_mono: NDArrayReal = np.array([np.sin(2 * np.pi * 1000 * t)]) * 4
        self.signal_stereo: NDArrayReal = np.array(
            [np.sin(2 * np.pi * 1000 * t), np.sin(2 * np.pi * 2000 * t)]
        )

        # Create dask arrays
        self.dask_mono: DaArray = _da_from_array(self.signal_mono)
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo)

        # Initialize STFT
        self.stft = STFT(
            sampling_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
        )

        # Initialize ISTFT
        self.istft = ISTFT(
            sampling_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
        )

    def test_stft_initialization(self) -> None:
        """Test STFT initialization with different parameters."""
        # Default initialization
        stft = STFT(self.sample_rate)
        assert stft.sampling_rate == self.sample_rate
        assert stft.n_fft == 2048
        assert stft.hop_length == 512  # 2048 // 4
        assert stft.win_length == 2048
        assert stft.window == "hann"

        # Custom initialization
        custom_stft = STFT(
            sampling_rate=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            win_length=512,
            window="hamming",
        )
        assert custom_stft.n_fft == 1024
        assert custom_stft.hop_length == 256
        assert custom_stft.win_length == 512
        assert custom_stft.window == "hamming"

    def test_istft_initialization(self) -> None:
        """Test ISTFT initialization with different parameters."""
        # Default initialization
        istft = ISTFT(self.sample_rate)
        assert istft.sampling_rate == self.sample_rate
        assert istft.n_fft == 2048
        assert istft.hop_length == 512  # 2048 // 4
        assert istft.win_length == 2048
        assert istft.window == "hann"
        assert istft.length is None

        # Custom initialization
        custom_istft = ISTFT(
            sampling_rate=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            win_length=512,
            window="hamming",
            length=16000,
        )
        assert custom_istft.n_fft == 1024
        assert custom_istft.hop_length == 256
        assert custom_istft.win_length == 512
        assert custom_istft.window == "hamming"
        assert custom_istft.length == 16000

    def test_stft_shape_mono(self) -> None:
        """Test STFT output shape for mono signal."""
        from scipy.signal import ShortTimeFFT as ScipySTFT
        from scipy.signal import get_window

        # Process the mono signal
        stft_result = self.stft.process_array(self.signal_mono).compute()

        # Check the shape of the result
        assert stft_result.ndim == 3, (
            "Output should be 3D (channels, frequencies, time)"
        )

        # Expected shape: (channels, frequencies, time frames)
        sft = ScipySTFT(
            win=get_window(self.window, self.win_length),
            hop=self.hop_length,
            fs=self.sample_rate,
            mfft=self.n_fft,
            scale_to="magnitude",
        )
        expected_n_channels = 1
        expected_n_freqs = sft.f.shape[0]
        expected_n_frames = sft.t(self.signal_mono.shape[-1]).shape[0]

        expected_shape = (expected_n_channels, expected_n_freqs, expected_n_frames)
        assert stft_result.shape == expected_shape, (
            f"Expected {expected_shape}, got {stft_result.shape}"
        )

    def test_stft_shape_stereo(self) -> None:
        """Test STFT output shape for stereo signal."""
        from scipy.signal import ShortTimeFFT as ScipySTFT
        from scipy.signal import get_window

        # Process the stereo signal
        stft_result = self.stft.process_array(self.signal_stereo).compute()

        assert stft_result.ndim == 3, (
            "Output should be 3D (channels, frequencies, time)"
        )

        # Expected shape: (channels, frequencies, time frames)
        sft = ScipySTFT(
            win=get_window(self.window, self.win_length),
            hop=self.hop_length,
            fs=self.sample_rate,
            mfft=self.n_fft,
            scale_to="magnitude",
        )
        expected_n_channels = 2
        expected_n_freqs = sft.f.shape[0]
        expected_n_frames = sft.t(self.signal_mono.shape[-1]).shape[0]

        expected_shape = (expected_n_channels, expected_n_freqs, expected_n_frames)
        assert stft_result.shape == expected_shape, (
            f"Expected {expected_shape}, got {stft_result.shape}"
        )

    def test_stft_content(self) -> None:
        """Test STFT content correctness."""
        # Process the mono signal using the class under test
        stft_result = self.stft.process(self.dask_mono).compute()

        assert stft_result.ndim == 3, (
            "Output should be 3D (channels, frequencies, time)"
        )

        # Calculate the expected STFT using scipy.signal.ShortTimeFFT directly
        from scipy.signal import ShortTimeFFT as ScipySTFT
        from scipy.signal import get_window

        # Ensure parameters match the self.stft instance
        sft = ScipySTFT(
            win=get_window(self.window, self.win_length),
            hop=self.hop_length,
            fs=self.sample_rate,
            mfft=self.n_fft,
            scale_to="magnitude",
        )
        # Calculate STFT on the raw signal data (first channel)
        expected_stft_raw = sft.stft(self.signal_mono[0])
        expected_stft_raw[..., 1:-1, :] *= 2.0
        # Reshape scipy's output (freqs, time) to match class
        # output (channels, freqs, time)
        expected_stft = expected_stft_raw.reshape(1, *expected_stft_raw.shape)

        # Check the peak magnitude
        #  (should be close to the original amplitude 4 due to scale_to='magnitude')
        np.testing.assert_allclose(np.abs(stft_result).max(), 4, rtol=1e-5)
        # Compare the results from the class with the directly calculated scipy result
        np.testing.assert_allclose(stft_result, expected_stft, rtol=1e-5, atol=1e-5)

    def test_istft_shape(self) -> None:
        """Test ISTFT output shape."""
        # First get some STFT data
        stft_data = self.stft.process_array(self.signal_mono)

        # Process with ISTFT
        istft_result = self.istft.process_array(stft_data).compute()

        # Check the shape
        assert istft_result.ndim == 2, "Output should be 2D (channels, time)"

        # One channel
        assert istft_result.shape[0] == 1

        # Length should be approximately the original signal length
        expected_length = len(self.signal_mono[0])
        assert abs(istft_result.shape[1] - expected_length) < self.win_length

    def test_roundtrip_reconstruction(self) -> None:
        """Test signal reconstruction quality through STFT->ISTFT roundtrip."""
        # Process with STFT then ISTFT
        stft_data = self.stft.process_array(self.signal_mono)
        istft_data = self.istft.process_array(stft_data).compute()

        orig_length = self.signal_mono.shape[1]
        reconstructed_trimmed = istft_data[:, :orig_length]
        np.testing.assert_allclose(
            reconstructed_trimmed[..., 16:-16],
            self.signal_mono[..., 16:-16],
            rtol=1e-6,
            atol=1e-5,
        )

    def test_1d_input_handling(self) -> None:
        """Test that 1D input is properly reshaped to (1, samples)."""
        signal_1d = np.sin(
            2 * np.pi * 440 * np.linspace(0, 1, self.sample_rate, endpoint=False)
        )

        stft_result = self.stft.process_array(signal_1d).compute()

        assert stft_result.ndim == 3
        assert stft_result.shape[0] == 1

    def test_istft_2d_input_handling(self) -> None:
        """
        Test that 2D input (single channel spectrogram) is
        properly reshaped to (1, freqs, frames).
        """
        stft_data = self.stft.process_array(self.signal_mono).compute()
        stft_2d = stft_data[0]

        istft_result = self.istft.process_array(stft_2d).compute()

        assert istft_result.ndim == 2
        assert istft_result.shape[0] == 1

    def test_stft_operation_registry(self) -> None:
        """Test that STFT is properly registered in the operation registry."""
        assert get_operation("stft") == STFT
        assert get_operation("istft") == ISTFT

        stft_op = create_operation("stft", self.sample_rate, n_fft=512, hop_length=128)
        istft_op = create_operation(
            "istft", self.sample_rate, n_fft=512, hop_length=128
        )

        assert isinstance(stft_op, STFT)
        assert stft_op.n_fft == 512
        assert stft_op.hop_length == 128

        assert isinstance(istft_op, ISTFT)
        assert istft_op.n_fft == 512
        assert istft_op.hop_length == 128

    def test_negative_n_fft_error_message(self) -> None:
        """Test that negative n_fft provides helpful error message."""
        import pytest

        with pytest.raises(ValueError) as exc_info:
            STFT(sampling_rate=44100, n_fft=-2048)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Invalid FFT size for STFT" in error_msg
        assert "-2048" in error_msg
        # Check WHY
        assert "Positive integer" in error_msg
        # Check HOW
        assert "Common values:" in error_msg

    def test_win_length_greater_than_n_fft_error_message(self) -> None:
        """Test that win_length > n_fft provides helpful error message."""
        import pytest

        with pytest.raises(ValueError) as exc_info:
            STFT(sampling_rate=44100, n_fft=1024, win_length=2048)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Invalid window length for STFT" in error_msg
        assert "win_length=2048" in error_msg
        # Check WHY
        assert "win_length <= n_fft" in error_msg
        assert "1024" in error_msg
        # Check HOW
        assert (
            "win_length=1024 or smaller" in error_msg
            or "increase n_fft to 2048" in error_msg
        )

    def test_negative_hop_length_error_message(self) -> None:
        """Test that negative hop_length provides helpful error message."""
        import pytest

        with pytest.raises(ValueError) as exc_info:
            STFT(sampling_rate=44100, n_fft=2048, hop_length=-512)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Invalid hop length for STFT" in error_msg
        assert "-512" in error_msg
        # Check WHY
        assert "Positive integer" in error_msg

    def test_hop_length_greater_than_win_length_error_message(self) -> None:
        """Test that hop_length > win_length provides helpful error message."""
        import pytest

        with pytest.raises(ValueError) as exc_info:
            STFT(sampling_rate=44100, n_fft=2048, win_length=1024, hop_length=2048)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Invalid hop length for STFT" in error_msg
        assert "hop_length=2048" in error_msg
        # Check WHY
        assert "hop_length <= win_length" in error_msg
        assert "1024" in error_msg
        # Check HOW
        assert "would create gaps" in error_msg

    def test_zero_n_fft_error_message(self) -> None:
        """Test that zero n_fft provides helpful error message."""
        import pytest

        with pytest.raises(ValueError) as exc_info:
            STFT(sampling_rate=44100, n_fft=0)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Invalid FFT size for STFT" in error_msg
        assert "0" in error_msg
        # Check WHY
        assert "Positive integer" in error_msg

    def test_negative_win_length_error_message(self) -> None:
        """Test that negative win_length provides helpful error message."""
        import pytest

        with pytest.raises(ValueError) as exc_info:
            STFT(sampling_rate=44100, n_fft=2048, win_length=-1024)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Invalid window length for STFT" in error_msg
        assert "-1024" in error_msg
        # Check WHY
        assert "Positive integer" in error_msg

    def test_zero_win_length_error_message(self) -> None:
        """Test that zero win_length provides helpful error message."""
        import pytest

        with pytest.raises(ValueError) as exc_info:
            STFT(sampling_rate=44100, n_fft=2048, win_length=0)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Invalid window length for STFT" in error_msg
        # Check WHY
        assert "Positive integer" in error_msg

    def test_win_length_too_small_for_default_hop_error_message(self) -> None:
        """Test that win_length < 4 with no hop_length provides
        helpful error message."""
        import pytest

        with pytest.raises(ValueError) as exc_info:
            STFT(sampling_rate=44100, n_fft=2048, win_length=3)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Window length too small" in error_msg
        assert "win_length=3" in error_msg
        # Check WHY
        assert "win_length >= 4" in error_msg
        # Check HOW
        assert (
            "specify a larger win_length or provide hop_length explicitly" in error_msg
        )

    def test_istft_with_length_parameter(self) -> None:
        """Test ISTFT with explicit length parameter for output trimming."""
        # Create STFT data
        stft_data = self.stft.process_array(self.signal_mono)

        # Create ISTFT with specific output length
        target_length = 8000  # Half of original signal
        istft_with_length = ISTFT(
            sampling_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=target_length,
        )

        result = istft_with_length.process_array(stft_data).compute()

        # Output should be trimmed to target_length
        assert result.shape[1] == target_length


class TestNOctSynthesisOperation:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 48000
        self.fmin: float = 24.0
        self.fmax: float = 12600
        self.n: int = 3
        self.G: int = 10
        self.fr: int = 1000

        self.noct_synthesis = NOctSynthesis(
            sampling_rate=self.sample_rate,
            fmin=self.fmin,
            fmax=self.fmax,
            n=self.n,
            G=self.G,
            fr=self.fr,
        )

        # Create a test signal with pink noise
        np.random.seed(42)  # For reproducibility
        white_noise = np.random.randn(self.sample_rate)

        # Simple approximation of pink noise by filtering white noise
        k = np.fft.rfftfreq(len(white_noise))[1:]
        X = np.fft.rfft(white_noise)  # noqa: N806
        S = 1.0 / np.sqrt(k)  # noqa: N806
        X[1:] *= S
        pink_noise = np.fft.irfft(X, len(white_noise))
        pink_noise /= np.abs(pink_noise).max()  # Normalize

        self.signal_mono: NDArrayReal = np.array([pink_noise])
        self.signal_stereo: NDArrayReal = np.array(
            [pink_noise, white_noise / np.abs(white_noise).max()]
        )

        # Create dask arrays
        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=(1, 1000))
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=(2, 1000))

    def test_initialization(self) -> None:
        """Test initialization with parameters."""
        noct = NOctSynthesis(
            self.sample_rate,
            fmin=self.fmin,
            fmax=self.fmax,
            n=self.n,
            G=self.G,
            fr=self.fr,
        )
        assert noct.sampling_rate == self.sample_rate
        assert noct.fmin == self.fmin
        assert noct.fmax == self.fmax
        assert noct.n == self.n
        assert noct.G == self.G
        assert noct.fr == self.fr

    def test_noct_synthesis_integration(self) -> None:
        """外部ライブラリのnoct_synthesis関数が正確にラップされているかテスト"""
        # スペクトル計算に必要なデータを用意
        # まずfftでスペクトルを計算
        fft = FFT(self.sample_rate, n_fft=None, window="hann")
        spectrum = fft.process(self.dask_mono).compute()

        # NOctSynthesisによる合成
        result = self.noct_synthesis.process(spectrum).compute()

        # 外部ライブラリを直接呼び出した場合の結果
        # Note: NOctSynthesisのprocess_array内の処理を再現
        n = spectrum.shape[-1]
        if n % 2 == 0:
            n = n * 2 - 1
        else:
            n = (n - 1) * 2
        freqs = np.fft.rfftfreq(n, d=1 / self.sample_rate)

        expected_signal, expected_freqs = noct_synthesis(
            spectrum=np.abs(spectrum).T,
            freqs=freqs,
            fmin=self.fmin,
            fmax=self.fmax,
            n=self.n,
            G=self.G,
            fr=self.fr,
        )
        expected_signal = expected_signal.T

        # 形状が一致することを確認
        assert result.shape == expected_signal.shape

        # 結果が一致することを確認
        np.testing.assert_allclose(result[0], expected_signal[0])

    def test_noct_synthesis_stereo(self) -> None:
        """ステレオ信号でnoct_synthesis関数が正確にラップされているかテスト"""
        # FFTでスペクトル計算するための準備
        fft = FFT(self.sample_rate, n_fft=None, window="hann")

        # ステレオ信号のスペクトルを計算
        stereo_spectrum = fft.process(self.dask_stereo).compute()

        # NOctSynthesisによる合成
        result = self.noct_synthesis.process(stereo_spectrum).compute()

        # 外部ライブラリを直接呼び出した場合の結果
        n = stereo_spectrum.shape[-1]
        if n % 2 == 0:
            n = n * 2 - 1
        else:
            n = (n - 1) * 2
        freqs = np.fft.rfftfreq(n, d=1 / self.sample_rate)

        # 第1チャンネル
        expected_signal_ch1, expected_freqs = noct_synthesis(
            spectrum=np.abs(stereo_spectrum[0:1]).T,
            freqs=freqs,
            fmin=self.fmin,
            fmax=self.fmax,
            n=self.n,
            G=self.G,
            fr=self.fr,
        )
        expected_signal_ch1 = expected_signal_ch1.T

        # 第2チャンネル
        expected_signal_ch2, _ = noct_synthesis(
            spectrum=np.abs(stereo_spectrum[1:2]).T,
            freqs=freqs,
            fmin=self.fmin,
            fmax=self.fmax,
            n=self.n,
            G=self.G,
            fr=self.fr,
        )
        expected_signal_ch2 = expected_signal_ch2.T

        # 結果の形状を確認
        assert result.shape == (2, expected_signal_ch1.shape[1])

        # 各チャンネルの結果が外部ライブラリの結果と一致するか確認
        np.testing.assert_allclose(
            result[0], expected_signal_ch1[0], rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            result[1], expected_signal_ch2[0], rtol=1e-5, atol=1e-5
        )

        # ピンクノイズと白色ノイズのチャンネルで異なる結果が出ることを確認
        # 完全に異なるスペクトルから合成したので、結果も異なるはず
        assert not np.allclose(result[0], result[1])

    def test_delayed_execution(self) -> None:
        """Test that NOctSynthesis operation uses dask's delayed execution."""
        with mock.patch.object(DaArray, "compute") as mock_compute:
            with mock.patch("wandas.processing.spectral.noct_synthesis") as mock_noct:
                # Create a dummy result for the mock
                dummy_signal = np.zeros((1, self.sample_rate))
                dummy_result = (dummy_signal, np.zeros(27))
                mock_noct.return_value = dummy_result

                result = self.noct_synthesis.process(self.dask_mono)
                mock_compute.assert_not_called()

                _ = result.compute()
                mock_compute.assert_called_once()

    def test_operation_registry(self) -> None:
        """Test that NOctSynthesis is properly registered in the operation registry."""
        assert get_operation("noct_synthesis") == NOctSynthesis

        noct_op = create_operation(
            "noct_synthesis",
            self.sample_rate,
            fmin=100.0,
            fmax=5000.0,
            n=1,
            G=20,
            fr=1000,
        )

        assert isinstance(noct_op, NOctSynthesis)
        assert noct_op.sampling_rate == self.sample_rate
        assert noct_op.fmin == 100.0
        assert noct_op.fmax == 5000.0
        assert noct_op.n == 1
        assert noct_op.G == 20
        assert noct_op.fr == 1000

    def test_noct_synthesis_odd_length_spectrum(self) -> None:
        """Test NOctSynthesis with odd-length spectrum (else branch coverage)."""
        # Create spectrum with odd number of frequency bins
        # rfft of length N produces N//2 + 1 bins
        # For odd bins: need even N where N//2 + 1 is odd, so N//2 is even, N = 4k
        # Example: N=52 -> 52//2 + 1 = 27 (odd)
        fft = FFT(self.sample_rate, n_fft=None, window="hann")

        # Create a signal with even length that produces odd rfft bins
        signal_length = 52  # Will produce 27 rfft bins (odd)
        test_signal = np.random.randn(signal_length)
        spectrum = fft.process(_da_from_array(np.array([test_signal]))).compute()

        # Verify spectrum has odd length
        assert spectrum.shape[-1] % 2 == 1

        # Process through NOctSynthesis
        result = self.noct_synthesis.process(_da_from_array(spectrum)).compute()

        # Should produce valid output
        assert result.shape[0] == 1


class TestWelchOperation:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.n_fft: int = 1024
        self.hop_length: int = 256
        self.win_length: int = 1024
        self.window: str = "hann"
        self.average: str = "mean"
        self.detrend: str = "constant"

        self.welch = Welch(
            sampling_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            average=self.average,
            detrend=self.detrend,
        )

        # Create a test signal with a known frequency
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        self.freq = 1000.0
        self.signal_mono: NDArrayReal = np.array([np.sin(2 * np.pi * self.freq * t)])
        self.signal_stereo: NDArrayReal = np.array(
            [np.sin(2 * np.pi * self.freq * t), np.sin(2 * np.pi * 2000 * t)]
        )

        # Create dask arrays
        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=(1, 1000))
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=(2, 1000))

    def test_initialization(self) -> None:
        """Test initialization with different parameters."""
        # Default initialization
        welch = Welch(self.sample_rate)
        assert welch.sampling_rate == self.sample_rate
        assert welch.n_fft == 2048
        assert welch.win_length == 2048
        assert welch.hop_length == 512  # 2048 // 4
        assert welch.window == "hann"
        assert welch.average == "mean"
        assert welch.detrend == "constant"

        # Custom initialization
        custom_welch = Welch(
            sampling_rate=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            win_length=512,
            window="hamming",
            average="median",
            detrend="linear",
        )
        assert custom_welch.n_fft == 1024
        assert custom_welch.win_length == 512
        assert custom_welch.hop_length == 256
        assert custom_welch.window == "hamming"
        assert custom_welch.average == "median"
        assert custom_welch.detrend == "linear"

    def test_welch_shape(self) -> None:
        """Test Welch output shape."""
        result = self.welch.process_array(self.signal_mono).compute()

        # Expected frequency bins
        expected_bins = self.n_fft // 2 + 1
        assert result.shape == (1, expected_bins)

        # Test with stereo signal
        result_stereo = self.welch.process_array(self.signal_stereo).compute()
        assert result_stereo.shape == (2, expected_bins)

    def test_welch_content(self) -> None:
        """Test that Welch correctly identifies frequency content."""
        result = self.welch.process_array(self.signal_mono).compute()

        # Get frequency bins
        freq_bins = np.fft.rfftfreq(self.n_fft, 1.0 / self.sample_rate)

        # Find the peak frequency
        peak_idx = np.argmax(result[0])
        detected_freq = freq_bins[peak_idx]

        # Check that the detected frequency is close to the actual frequency
        np.testing.assert_allclose(detected_freq, self.freq, rtol=0.05)

        # For stereo signal, check second channel
        result_stereo = self.welch.process_array(self.signal_stereo).compute()
        peak_idx_ch2 = np.argmax(result_stereo[1])
        detected_freq_ch2 = freq_bins[peak_idx_ch2]

        # Second channel should show peak at 2000 Hz
        np.testing.assert_allclose(detected_freq_ch2, 2000.0, rtol=0.05)

    def test_delayed_execution(self) -> None:
        """Test that Welch operation uses dask's delayed execution."""
        with mock.patch.object(DaArray, "compute") as mock_compute:
            result = self.welch.process(self.dask_mono)
            mock_compute.assert_not_called()

            assert isinstance(result, DaArray)

            _ = result.compute()
            mock_compute.assert_called_once()

    def test_operation_registry(self) -> None:
        """Test that Welch is properly registered in the operation registry."""
        assert get_operation("welch") == Welch

        welch_op = create_operation(
            "welch",
            self.sample_rate,
            n_fft=512,
            win_length=512,
            hop_length=128,
            window="hamming",
        )

        assert isinstance(welch_op, Welch)
        assert welch_op.sampling_rate == self.sample_rate
        assert welch_op.n_fft == 512
        assert welch_op.win_length == 512
        assert welch_op.hop_length == 128
        assert welch_op.window == "hamming"

    def test_negative_n_fft_error_message(self) -> None:
        """Test that negative n_fft provides helpful error message."""
        import pytest

        with pytest.raises(ValueError) as exc_info:
            Welch(sampling_rate=44100, n_fft=-2048)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Invalid FFT size for Welch" in error_msg
        assert "-2048" in error_msg
        # Check WHY
        assert "Positive integer" in error_msg
        # Check HOW
        assert "Common values:" in error_msg

    def test_win_length_greater_than_n_fft_error_message(self) -> None:
        """Test that win_length > n_fft provides helpful error message."""
        import pytest

        with pytest.raises(ValueError) as exc_info:
            Welch(sampling_rate=44100, n_fft=1024, win_length=2048)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Invalid window length for Welch" in error_msg
        assert "win_length=2048" in error_msg
        # Check WHY
        assert "win_length <= n_fft" in error_msg
        # Check HOW
        assert "or increase n_fft" in error_msg

    def test_hop_length_greater_than_win_length_error_message(self) -> None:
        """Test that hop_length > win_length provides helpful error message."""
        import pytest

        with pytest.raises(ValueError) as exc_info:
            Welch(sampling_rate=44100, n_fft=2048, win_length=1024, hop_length=2048)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Invalid hop length for Welch" in error_msg
        # Check WHY
        assert "hop_length <= win_length" in error_msg
        # Check HOW
        assert "would create gaps" in error_msg

    def test_zero_n_fft_error_message(self) -> None:
        """Test that zero n_fft provides helpful error message."""
        import pytest

        with pytest.raises(ValueError) as exc_info:
            Welch(sampling_rate=44100, n_fft=0)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Invalid FFT size for Welch" in error_msg
        assert "0" in error_msg
        # Check WHY
        assert "Positive integer" in error_msg

    def test_negative_win_length_error_message(self) -> None:
        """Test that negative win_length provides helpful error message."""
        import pytest

        with pytest.raises(ValueError) as exc_info:
            Welch(sampling_rate=44100, n_fft=2048, win_length=-1024)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Invalid window length for Welch" in error_msg
        assert "-1024" in error_msg
        # Check WHY
        assert "Positive integer" in error_msg

    def test_zero_win_length_error_message(self) -> None:
        """Test that zero win_length provides helpful error message."""
        import pytest

        with pytest.raises(ValueError) as exc_info:
            Welch(sampling_rate=44100, n_fft=2048, win_length=0)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Invalid window length for Welch" in error_msg
        # Check WHY
        assert "Positive integer" in error_msg

    def test_win_length_too_small_for_default_hop_error_message(self) -> None:
        """Test that win_length < 4 with no hop_length provides
        helpful error message."""
        import pytest

        with pytest.raises(ValueError) as exc_info:
            Welch(sampling_rate=44100, n_fft=2048, win_length=3)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Window length too small" in error_msg
        assert "win_length=3" in error_msg
        # Check WHY
        assert "win_length >= 4" in error_msg
        # Check HOW
        assert (
            "specify a larger win_length or provide hop_length explicitly" in error_msg
        )


class TestCoherenceOperation:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.n_fft: int = 1024
        self.hop_length: int = 256
        self.win_length: int = 1024
        self.window: str = "hann"
        self.detrend: str = "constant"

        # Create test signals with different frequencies
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        # 2つのチャンネルを持つ信号：1つは1000Hz、もう1つは関連する1100Hz
        self.signal_stereo: NDArrayReal = np.array(
            [np.sin(2 * np.pi * 1000 * t), np.sin(2 * np.pi * 1100 * t)]
        )
        # 3チャンネルの信号（1つはノイズ）
        noise = np.random.randn(self.sample_rate) * 0.1
        self.signal_multi: NDArrayReal = np.array(
            [
                np.sin(2 * np.pi * 1000 * t),
                np.sin(2 * np.pi * 1100 * t),
                noise,
            ]
        )

        # Create dask arrays
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=(2, 1000))
        self.dask_multi: DaArray = _da_from_array(self.signal_multi, chunks=(3, 1000))

        # Initialize Coherence operation
        self.coherence = Coherence(
            sampling_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            detrend=self.detrend,
        )

    def test_initialization(self) -> None:
        """Test initialization with parameters."""
        # Default initialization already done in setup_method
        assert self.coherence.sampling_rate == self.sample_rate
        assert self.coherence.n_fft == self.n_fft
        assert self.coherence.hop_length == self.hop_length
        assert self.coherence.win_length == self.win_length
        assert self.coherence.window == self.window
        assert self.coherence.detrend == self.detrend

        # Custom initialization
        custom_hop = 512
        custom_coherence = Coherence(
            sampling_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=custom_hop,
            win_length=self.win_length,
            window="hamming",
            detrend="linear",
        )
        assert custom_coherence.sampling_rate == self.sample_rate
        assert custom_coherence.n_fft == self.n_fft
        assert custom_coherence.hop_length == custom_hop
        assert custom_coherence.win_length == self.win_length
        assert custom_coherence.window == "hamming"
        assert custom_coherence.detrend == "linear"

    def test_coherence_shape(self) -> None:
        """Test output shape for coherence."""
        # Calculate coherence for stereo signal
        result = self.coherence.process_array(self.signal_stereo).compute()

        # Expected shape: (n_channels * n_channels, n_freqs)
        n_channels = self.signal_stereo.shape[0]
        n_freqs = self.n_fft // 2 + 1
        expected_shape = (n_channels * n_channels, n_freqs)

        assert result.shape == expected_shape

        # Multi-channel test
        result_multi = self.coherence.process_array(self.signal_multi).compute()
        n_channels_multi = self.signal_multi.shape[0]
        expected_shape_multi = (n_channels_multi * n_channels_multi, n_freqs)

        assert result_multi.shape == expected_shape_multi

    def test_coherence_content(self) -> None:
        """Test coherence calculation correctness."""
        result = self.coherence.process_array(self.signal_stereo).compute()

        # Expected properties:
        # 1. Coherence values should be between 0 and 1
        assert np.all(result >= 0)
        # 小数点6桁以下を丸めて比較
        assert np.all(result <= 1.000001)

        # 2. Self-coherence (diagonal elements) should be ~1
        # For 2 channels, indices 0 and 3
        assert np.isclose(result[0, :].mean(), 1.0)
        assert np.isclose(result[3, :].mean(), 1.0)

        # 3. Cross-coherence should be less than 1 but above 0
        # For 2 channels, indices 1 and 2
        cross_coherence = np.mean(result[1, :])
        assert 0 < cross_coherence < 1, f"Cross-coherence mean: {cross_coherence}"

        # 4. Verify with scipy.signal.coherence directly
        from scipy import signal as ss

        _, coh = ss.coherence(
            x=self.signal_stereo[:, np.newaxis],
            y=self.signal_stereo[np.newaxis, :],
            fs=self.sample_rate,
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length,
            nfft=self.n_fft,
            window=self.window,
            detrend=self.detrend,
        )

        expected_result = coh.reshape(-1, coh.shape[-1])
        np.testing.assert_allclose(result, expected_result, rtol=1e-6)

    def test_delayed_execution(self) -> None:
        """Test that coherence operation uses dask's delayed execution."""
        with mock.patch.object(DaArray, "compute") as mock_compute:
            result = self.coherence.process(self.dask_stereo)
            mock_compute.assert_not_called()

            # Only when compute() is called should the computation happen
            _ = result.compute()
            mock_compute.assert_called_once()

    def test_operation_registry(self) -> None:
        """Test that Coherence is properly registered in the operation registry."""
        assert get_operation("coherence") == Coherence

        coherence_op = create_operation(
            "coherence",
            self.sample_rate,
            n_fft=512,
            hop_length=128,
            win_length=512,
            window="hamming",
            detrend="linear",
        )

        assert isinstance(coherence_op, Coherence)
        assert coherence_op.sampling_rate == self.sample_rate
        assert coherence_op.n_fft == 512
        assert coherence_op.hop_length == 128
        assert coherence_op.win_length == 512
        assert coherence_op.window == "hamming"
        assert coherence_op.detrend == "linear"


class TestCSDOperation:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.n_fft: int = 1024
        self.hop_length: int = 256
        self.win_length: int = 1024
        self.window: str = "hann"
        self.detrend: str = "constant"
        self.scaling: str = "spectrum"
        self.average: str = "mean"

        # Create test signals with different frequencies
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        # 2つのチャンネルを持つ信号：1つは1000Hz、もう1つは関連する1100Hz
        self.signal_stereo: NDArrayReal = np.array(
            [np.sin(2 * np.pi * 1000 * t), np.sin(2 * np.pi * 1100 * t)]
        )
        # 3チャンネルの信号（1つはノイズ）
        noise = np.random.randn(self.sample_rate) * 0.1
        self.signal_multi: NDArrayReal = np.array(
            [
                np.sin(2 * np.pi * 1000 * t),
                np.sin(2 * np.pi * 1100 * t),
                noise,
            ]
        )

        # Create dask arrays
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=(2, 1000))
        self.dask_multi: DaArray = _da_from_array(self.signal_multi, chunks=(3, 1000))

        # Initialize CSD operation
        self.csd = CSD(
            sampling_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            detrend=self.detrend,
            scaling=self.scaling,
            average=self.average,
        )

    def test_initialization(self) -> None:
        """Test initialization with parameters."""
        # Default initialization already done in setup_method
        assert self.csd.sampling_rate == self.sample_rate
        assert self.csd.n_fft == self.n_fft
        assert self.csd.hop_length == self.hop_length
        assert self.csd.win_length == self.win_length
        assert self.csd.window == self.window
        assert self.csd.detrend == self.detrend
        assert self.csd.scaling == self.scaling
        assert self.csd.average == self.average

        # Custom initialization
        custom_hop = 512
        custom_csd = CSD(
            sampling_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=custom_hop,
            win_length=self.win_length,
            window="hamming",
            detrend="linear",
            scaling="density",
            average="median",
        )
        assert custom_csd.sampling_rate == self.sample_rate
        assert custom_csd.n_fft == self.n_fft
        assert custom_csd.hop_length == custom_hop
        assert custom_csd.win_length == self.win_length
        assert custom_csd.window == "hamming"
        assert custom_csd.detrend == "linear"
        assert custom_csd.scaling == "density"
        assert custom_csd.average == "median"

    def test_csd_shape(self) -> None:
        """Test output shape for CSD."""
        # Calculate CSD for stereo signal
        result = self.csd.process_array(self.signal_stereo).compute()

        # Expected shape: (n_channels * n_channels, n_freqs)
        n_channels = self.signal_stereo.shape[0]
        n_freqs = self.n_fft // 2 + 1
        expected_shape = (n_channels * n_channels, n_freqs)

        assert result.shape == expected_shape

        # Multi-channel test
        result_multi = self.csd.process_array(self.signal_multi).compute()
        n_channels_multi = self.signal_multi.shape[0]
        expected_shape_multi = (n_channels_multi * n_channels_multi, n_freqs)

        assert result_multi.shape == expected_shape_multi

    def test_csd_content(self) -> None:
        """Test CSD calculation correctness."""
        result = self.csd.process_array(self.signal_stereo).compute()

        # Verify with scipy.signal.csd directly
        from scipy import signal as ss

        _, csd_expected = ss.csd(
            x=self.signal_stereo[:, np.newaxis, :],
            y=self.signal_stereo[np.newaxis, :, :],
            fs=self.sample_rate,
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length,
            nfft=self.n_fft,
            window=self.window,
            detrend=self.detrend,
            scaling=self.scaling,
            average=self.average,
        )

        expected_result = csd_expected.transpose(1, 0, 2).reshape(
            -1, csd_expected.shape[-1]
        )
        np.testing.assert_allclose(result, expected_result, rtol=1e-6)

        # CSD of a signal with itself should be real and positive
        # at the signal frequency
        # Find indices closest to our test frequencies
        freq_bins = np.fft.rfftfreq(self.n_fft, 1.0 / self.sample_rate)
        idx_1000hz = np.argmin(np.abs(freq_bins - 1000))
        idx_1100hz = np.argmin(np.abs(freq_bins - 1100))

        # Auto-spectrum at 1000Hz (channel 0 with itself) should peak at 1000Hz
        auto_ch0 = result[0]
        assert np.argmax(np.abs(auto_ch0)) == idx_1000hz

        # Auto-spectrum at 1100Hz (channel 1 with itself) should peak at 1100Hz
        auto_ch1 = result[3]  # Index 3 is the 2nd channel with itself in flattened form
        assert np.argmax(np.abs(auto_ch1)) == idx_1100hz

    def test_delayed_execution(self) -> None:
        """Test that CSD operation uses dask's delayed execution."""
        with mock.patch.object(DaArray, "compute") as mock_compute:
            result = self.csd.process(self.dask_stereo)
            mock_compute.assert_not_called()

            # Only when compute() is called should the computation happen
            _ = result.compute()
            mock_compute.assert_called_once()

    def test_operation_registry(self) -> None:
        """Test that CSD is properly registered in the operation registry."""
        assert get_operation("csd") == CSD

        csd_op = create_operation(
            "csd",
            self.sample_rate,
            n_fft=512,
            hop_length=128,
            win_length=512,
            window="hamming",
            detrend="linear",
            scaling="density",
            average="median",
        )

        assert isinstance(csd_op, CSD)
        assert csd_op.sampling_rate == self.sample_rate
        assert csd_op.n_fft == 512
        assert csd_op.hop_length == 128
        assert csd_op.win_length == 512
        assert csd_op.window == "hamming"
        assert csd_op.detrend == "linear"
        assert csd_op.scaling == "density"
        assert csd_op.average == "median"


class TestTransferFunctionOperation:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.n_fft: int = 1024
        self.hop_length: int = 256
        self.win_length: int = 1024
        self.window: str = "hann"
        self.detrend: str = "constant"
        self.scaling: str = "spectrum"
        self.average: str = "mean"

        # Create test signals with different frequencies
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        # 入力信号と出力信号のペアを作成（簡単な線形システムをシミュレート）
        input_signal = np.sin(2 * np.pi * 1000 * t)
        output_signal = 2 * input_signal + 0.1 * np.random.randn(
            len(t)
        )  # ゲイン2と少しのノイズ

        self.signal_stereo: NDArrayReal = np.array([input_signal, output_signal])

        # より複雑なシステムのシミュレーション（複数入力・出力）
        input1 = np.sin(2 * np.pi * 1000 * t)
        input2 = np.sin(2 * np.pi * 1500 * t)
        output1 = 2 * input1 + 0.5 * input2 + 0.1 * np.random.randn(len(t))
        output2 = 0.3 * input1 + 1.5 * input2 + 0.1 * np.random.randn(len(t))

        self.signal_multi: NDArrayReal = np.array([input1, input2, output1, output2])

        # Create dask arrays
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=(2, 1000))
        self.dask_multi: DaArray = _da_from_array(self.signal_multi, chunks=(4, 1000))

        # Initialize TransferFunction operation
        self.transfer_function = TransferFunction(
            sampling_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            detrend=self.detrend,
            scaling=self.scaling,
            average=self.average,
        )

    def test_initialization(self) -> None:
        """Test initialization with parameters."""
        # Default initialization already done in setup_method
        assert self.transfer_function.sampling_rate == self.sample_rate
        assert self.transfer_function.n_fft == self.n_fft
        assert self.transfer_function.hop_length == self.hop_length
        assert self.transfer_function.win_length == self.win_length
        assert self.transfer_function.window == self.window
        assert self.transfer_function.detrend == self.detrend
        assert self.transfer_function.scaling == self.scaling
        assert self.transfer_function.average == self.average

        # Custom initialization
        custom_hop = 512
        custom_tf = TransferFunction(
            sampling_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=custom_hop,
            win_length=self.win_length,
            window="hamming",
            detrend="linear",
            scaling="density",
            average="median",
        )
        assert custom_tf.sampling_rate == self.sample_rate
        assert custom_tf.n_fft == self.n_fft
        assert custom_tf.hop_length == custom_hop
        assert custom_tf.win_length == self.win_length
        assert custom_tf.window == "hamming"
        assert custom_tf.detrend == "linear"
        assert custom_tf.scaling == "density"
        assert custom_tf.average == "median"

    def test_transfer_function_shape(self) -> None:
        """Test output shape for transfer function."""
        # Calculate transfer function for stereo signal
        result = self.transfer_function.process_array(self.signal_stereo).compute()

        # Expected shape: (n_channels * n_channels, n_freqs)
        n_channels = self.signal_stereo.shape[0]
        n_freqs = self.n_fft // 2 + 1
        expected_shape = (n_channels * n_channels, n_freqs)

        assert result.shape == expected_shape

        # Multi-channel test
        result_multi = self.transfer_function.process_array(self.signal_multi).compute()
        n_channels_multi = self.signal_multi.shape[0]
        expected_shape_multi = (n_channels_multi * n_channels_multi, n_freqs)

        assert result_multi.shape == expected_shape_multi

    def test_transfer_function_content(self) -> None:
        """Test transfer function calculation correctness."""
        result = self.transfer_function.process_array(self.signal_stereo).compute()

        # 伝達関数の検証
        # シミュレーションで使用したゲインは2.0（入力から出力へ）
        # 周波数ビンの計算
        freq_bins = np.fft.rfftfreq(self.n_fft, 1.0 / self.sample_rate)
        idx_1000hz = np.argmin(np.abs(freq_bins - 1000))

        # 入力から出力への伝達関数の値（チャンネル0からチャンネル1）
        # 結果の構造：[ch0->ch0, ch0->ch1, ch1->ch0, ch1->ch1]
        h_input_to_output = result[1, idx_1000hz]  # ch0->ch1 at 1000Hz

        # ゲインは約2.0のはず（行列の形状で平坦化されていることに注意）
        assert np.isclose(np.abs(h_input_to_output), 2.0, rtol=0.2)

        # 入力から入力（自己伝達関数）と出力から出力は約1.0のはず
        h_input_to_input = result[0, idx_1000hz]  # ch0->ch0
        h_output_to_output = result[3, idx_1000hz]  # ch1->ch1

        assert np.isclose(np.abs(h_input_to_input), 1.0, rtol=0.2)
        assert np.isclose(np.abs(h_output_to_output), 1.0, rtol=0.2)

        # 出力から入力への伝達関数の値は小さいはず（因果関係が逆）
        h_output_to_input = result[2, idx_1000hz]  # ch1->ch0
        assert np.isclose(np.abs(h_output_to_input), 0.5, rtol=0.2)

        # 簡易的な手動計算による検証
        from scipy import signal as ss

        # クロススペクトル密度を計算
        f, p_yx = ss.csd(
            x=self.signal_stereo[:, np.newaxis, :],
            y=self.signal_stereo[np.newaxis, :, :],
            fs=self.sample_rate,
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length,
            nfft=self.n_fft,
            window=self.window,
            detrend=self.detrend,
            scaling=self.scaling,
            average=self.average,
            axis=-1,
        )

        # パワースペクトル密度を計算
        _, p_xx = ss.welch(
            x=self.signal_stereo,
            fs=self.sample_rate,
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length,
            nfft=self.n_fft,
            window=self.window,
            detrend=self.detrend,
            scaling=self.scaling,
            average=self.average,
            axis=-1,
        )

        # 伝達関数 H = P_yx / P_xx
        h_f = p_yx / p_xx[np.newaxis, :, :]
        expected_result = h_f.transpose(1, 0, 2).reshape(-1, h_f.shape[-1])

        np.testing.assert_allclose(result, expected_result, rtol=1e-6)

    def test_delayed_execution(self) -> None:
        """Test that transfer function operation uses dask's delayed execution."""
        with mock.patch.object(DaArray, "compute") as mock_compute:
            result = self.transfer_function.process(self.dask_stereo)
            mock_compute.assert_not_called()

            # Only when compute() is called should the computation happen
            _ = result.compute()
            mock_compute.assert_called_once()

    def test_operation_registry(self) -> None:
        """
        Test that TransferFunction is properly registered in the operation registry.
        """
        assert get_operation("transfer_function") == TransferFunction

        tf_op = create_operation(
            "transfer_function",
            self.sample_rate,
            n_fft=512,
            hop_length=128,
            win_length=512,
            window="hamming",
            detrend="linear",
            scaling="density",
            average="median",
        )

        assert isinstance(tf_op, TransferFunction)
        assert tf_op.sampling_rate == self.sample_rate
        assert tf_op.n_fft == 512
        assert tf_op.hop_length == 128
        assert tf_op.win_length == 512
        assert tf_op.window == "hamming"
        assert tf_op.detrend == "linear"
        assert tf_op.scaling == "density"
        assert tf_op.average == "median"


class TestNOctSpectrumOperation:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 51200
        self.fmin: float = 24.0
        self.fmax: float = 12600
        self.n: int = 3
        self.G: int = 10
        self.fr: int = 1000

        self.noct_spectrum = NOctSpectrum(
            sampling_rate=self.sample_rate,
            fmin=self.fmin,
            fmax=self.fmax,
            n=self.n,
            G=self.G,
            fr=self.fr,
        )

        # Create a test signal with pink noise
        np.random.seed(42)  # For reproducibility
        white_noise = np.random.randn(self.sample_rate)

        # Simple approximation of pink noise by filtering white noise
        k = np.fft.rfftfreq(len(white_noise))[1:]
        X = np.fft.rfft(white_noise)  # noqa: N806
        S = 1.0 / np.sqrt(k)  # noqa: N806
        X[1:] *= S
        pink_noise = np.fft.irfft(X, len(white_noise))
        pink_noise /= np.abs(pink_noise).max()  # Normalize

        self.signal_mono: NDArrayReal = np.array([pink_noise])
        self.signal_stereo: NDArrayReal = np.array(
            [pink_noise, white_noise / np.abs(white_noise).max()]
        )

        # Create dask arrays
        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=(1, -1))
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=(2, -1))

    def test_initialization(self) -> None:
        """Test initialization with parameters."""
        noct = NOctSpectrum(
            self.sample_rate,
            fmin=self.fmin,
            fmax=self.fmax,
            n=self.n,
            G=self.G,
            fr=self.fr,
        )
        assert noct.sampling_rate == self.sample_rate
        assert noct.fmin == self.fmin
        assert noct.fmax == self.fmax
        assert noct.n == self.n
        assert noct.G == self.G
        assert noct.fr == self.fr

    def test_noct_spectrum_shape(self) -> None:
        """外部ライブラリのnoct_spectrum関数が正確にラップされているかテスト"""
        # 実際にnoct_spectrum関数を実行
        result = self.noct_spectrum.process(self.dask_mono).compute()

        # 外部ライブラリを直接呼び出した場合の結果を取得
        expected_spectrum, expected_freqs = noct_spectrum(
            sig=self.signal_mono.T,
            fs=self.sample_rate,
            fmin=self.fmin,
            fmax=self.fmax,
            n=self.n,
            G=self.G,
            fr=self.fr,
        )

        # 形状が一致することを確認
        assert result.shape == (1, expected_spectrum.shape[0])

        # 結果が一致することを確認
        np.testing.assert_allclose(result[0], expected_spectrum, rtol=1e-6)

        # 周波数帯域数が適切かチェック
        _, center_freqs = _center_freq(
            fmin=self.fmin, fmax=self.fmax, n=self.n, G=self.G, fr=self.fr
        )
        assert result.shape[1] == len(center_freqs)

    def test_noct_spectrum_stereo(self) -> None:
        """ステレオ信号でnoct_spectrum関数が正確にラップされているかテスト"""
        # ステレオ信号用のテスト
        result = self.noct_spectrum.process(self.dask_stereo).compute()

        # 外部ライブラリを直接呼び出した場合の結果（第1チャンネル）
        expected_spectrum_ch1, expected_freqs = noct_spectrum(
            sig=self.signal_stereo[0:1].T,  # 第1チャンネルのみ
            fs=self.sample_rate,
            fmin=self.fmin,
            fmax=self.fmax,
            n=self.n,
            G=self.G,
            fr=self.fr,
        )

        # 外部ライブラリを直接呼び出した場合の結果（第2チャンネル）
        expected_spectrum_ch2, _ = noct_spectrum(
            sig=self.signal_stereo[1:2].T,  # 第2チャンネルのみ
            fs=self.sample_rate,
            fmin=self.fmin,
            fmax=self.fmax,
            n=self.n,
            G=self.G,
            fr=self.fr,
        )

        # 形状が正しいことを確認（チャンネル数 x 周波数帯域数）
        assert result.shape == (2, expected_spectrum_ch1.shape[0])

        # 第1チャンネルの結果が一致することを確認
        np.testing.assert_allclose(result[0], expected_spectrum_ch1, rtol=1e-6)

        # 第2チャンネルの結果が一致することを確認
        np.testing.assert_allclose(result[1], expected_spectrum_ch2, rtol=1e-6)

        # 白色ノイズと有色ノイズのスペクトルが異なることを確認
        # （第1チャンネルはピンクノイズ、第2チャンネルは白色ノイズ）
        assert not np.allclose(result[0], result[1])

    def test_delayed_execution(self) -> None:
        """Test that NOctSpectrum operation uses dask's delayed execution."""
        with mock.patch.object(DaArray, "compute") as mock_compute:
            result = self.noct_spectrum.process(self.dask_mono)
            mock_compute.assert_not_called()

            _ = result.compute()
            mock_compute.assert_called_once()

    def test_operation_registry(self) -> None:
        """Test that NOctSpectrum is properly registered in the operation registry."""
        assert get_operation("noct_spectrum") == NOctSpectrum

        noct_op = create_operation(
            "noct_spectrum",
            self.sample_rate,
            fmin=100.0,
            fmax=5000.0,
            n=1,
            G=20,
            fr=1000,
        )

        assert isinstance(noct_op, NOctSpectrum)
        assert noct_op.sampling_rate == self.sample_rate
        assert noct_op.fmin == 100.0
        assert noct_op.fmax == 5000.0
        assert noct_op.n == 1
        assert noct_op.G == 20
        assert noct_op.fr == 1000
