import dask.array as da
import numpy as np
from dask.array.core import Array as DaArray

from wandas.processing.base import create_operation, get_operation
from wandas.processing.temporal import (
    ReSampling,
    RmsTrend,
    Trim,
)
from wandas.utils.types import NDArrayReal

_da_from_array = da.from_array  # type: ignore [unused-ignore]


class TestReSampling:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.orig_sr: int = 16000
        self.target_sr: int = 8000
        self.resampler = ReSampling(self.orig_sr, self.target_sr)

        # Create sample signal: 1 second sine wave at 440 Hz
        t = np.linspace(0, 1, self.orig_sr, endpoint=False)
        self.signal_mono: NDArrayReal = np.array([np.sin(2 * np.pi * 440 * t)])
        self.signal_stereo: NDArrayReal = np.array(
            [
                np.sin(2 * np.pi * 440 * t),
                np.sin(2 * np.pi * 880 * t),
            ]
        )

        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=-1)
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=-1)

    def test_initialization(self) -> None:
        """Test initialization with different parameters."""
        resampler = ReSampling(self.orig_sr, self.target_sr)
        assert resampler.sampling_rate == self.orig_sr
        assert resampler.target_sr == self.target_sr

    def test_resampling_shape(self) -> None:
        """Test resampling output shape."""
        # Downsample
        result = self.resampler.process(self.dask_mono).compute()
        expected_len = int(
            np.ceil(self.signal_mono.shape[1] * (self.target_sr / self.orig_sr))
        )
        assert result.shape == (1, expected_len)

        # Upsample
        upsampler = ReSampling(self.orig_sr, self.orig_sr * 2)
        result = upsampler.process(self.dask_mono).compute()
        expected_len = int(np.ceil(self.signal_mono.shape[1] * 2))
        assert result.shape == (1, expected_len)

        # Stereo
        result = self.resampler.process(self.dask_stereo).compute()
        expected_len = int(
            np.ceil(self.signal_stereo.shape[1] * (self.target_sr / self.orig_sr))
        )
        assert result.shape == (2, expected_len)

    def test_resampling_content(self) -> None:
        """Test resampled content frequency preservation."""
        # Create test signal with a specific frequency
        freq = 440.0  # Hz
        t_orig = np.linspace(0, 1, self.orig_sr, endpoint=False)
        signal = np.array([np.sin(2 * np.pi * freq * t_orig)])
        dask_signal = _da_from_array(signal, chunks=-1)

        # Resample
        result = self.resampler.process(dask_signal).compute()

        # Check if frequency is preserved
        peak_freq_orig = self._get_peak_frequency(signal[0], self.orig_sr)
        peak_freq_resampled = self._get_peak_frequency(result[0], self.target_sr)

        # Allow a small difference due to interpolation
        np.testing.assert_allclose(peak_freq_orig, peak_freq_resampled, rtol=0.1)

    def _get_peak_frequency(self, signal: NDArrayReal, sr: int) -> float:
        """Get the peak frequency from a signal."""
        n = len(signal)
        fft_result = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(n, 1 / sr)
        peak_idx = np.argmax(fft_result)
        return float(freqs[peak_idx])

    def test_operation_registry(self) -> None:
        """Test that ReSampling is properly registered in the operation registry."""
        assert get_operation("resampling") == ReSampling

        resampling_op = create_operation("resampling", 16000, target_sr=22050)

        assert isinstance(resampling_op, ReSampling)
        assert resampling_op.sampling_rate == 16000
        assert resampling_op.target_sr == 22050


class TestTrim:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.start_time: float = 0.1  # seconds
        self.end_time: float = 0.5  # seconds
        self.trim = Trim(self.sample_rate, self.start_time, self.end_time)

        # Create sample signal: 1 second sine wave at 440 Hz
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        self.signal_mono: NDArrayReal = np.array([np.sin(2 * np.pi * 440 * t)])
        self.signal_stereo: NDArrayReal = np.array(
            [
                np.sin(2 * np.pi * 440 * t),
                np.sin(2 * np.pi * 880 * t),
            ]
        )

        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=-1)
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=-1)

    def test_initialization(self) -> None:
        """Test initialization with different parameters."""
        trim = Trim(self.sample_rate, self.start_time, self.end_time)
        assert trim.sampling_rate == self.sample_rate
        assert trim.start == self.start_time
        assert trim.end == self.end_time
        assert trim.start_sample == int(self.start_time * self.sample_rate)
        assert trim.end_sample == int(self.end_time * self.sample_rate)

    def test_trim_shape(self) -> None:
        """Test trimming output shape."""
        result = self.trim.process(self.dask_mono).compute()
        expected_samples = int(self.end_time * self.sample_rate) - int(
            self.start_time * self.sample_rate
        )
        assert result.shape == (1, expected_samples)

        result_stereo = self.trim.process(self.dask_stereo).compute()
        assert result_stereo.shape == (2, expected_samples)

    def test_trim_content(self) -> None:
        """Test trimming preserves signal content."""
        result = self.trim.process(self.dask_mono).compute()

        start_idx = int(self.start_time * self.sample_rate)
        end_idx = int(self.end_time * self.sample_rate)
        expected = self.signal_mono[:, start_idx:end_idx]

        np.testing.assert_allclose(result, expected)

    def test_operation_registry(self) -> None:
        """Test that Trim is properly registered in the operation registry."""
        assert get_operation("trim") == Trim

        trim_op = create_operation("trim", 16000, start=0.2, end=0.8)

        assert isinstance(trim_op, Trim)
        assert trim_op.sampling_rate == 16000
        assert trim_op.start == 0.2
        assert trim_op.end == 0.8


class TestRmsTrend:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.frame_length: int = 2048
        self.hop_length: int = 512
        self.rms_trend = RmsTrend(
            self.sample_rate, frame_length=self.frame_length, hop_length=self.hop_length
        )

        # Create sample signal: 1 second sine wave at 440 Hz
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        self.signal_mono: NDArrayReal = np.array([np.sin(2 * np.pi * 440 * t)])
        # Create amplitude-modulated signal
        self.am_signal = np.array(
            [np.sin(2 * np.pi * 440 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 5 * t))]
        )

        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=-1)
        self.dask_am: DaArray = _da_from_array(self.am_signal, chunks=-1)

    def test_initialization(self) -> None:
        """Test initialization with different parameters."""
        rms = RmsTrend(self.sample_rate)
        assert rms.sampling_rate == self.sample_rate
        assert rms.frame_length == 2048  # Default value
        assert rms.hop_length == 512  # Default value
        assert rms.dB is False
        assert rms.Aw is False

        custom_rms = RmsTrend(
            self.sample_rate, frame_length=1024, hop_length=256, dB=True, Aw=True
        )
        assert custom_rms.frame_length == 1024
        assert custom_rms.hop_length == 256
        assert custom_rms.dB is True
        assert custom_rms.Aw is True

    def test_rms_shape(self) -> None:
        """Test RMS calculation output shape."""
        result = self.rms_trend.process(self.dask_mono).compute()

        # Expected number of frames
        import librosa

        expected_frames = librosa.feature.rms(
            y=self.signal_mono,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        ).shape[-1]

        assert result.shape == (1, expected_frames)

    def test_rms_content(self) -> None:
        """Test RMS content correctness."""
        # For a constant amplitude sine wave, RMS should be consistent
        result = self.rms_trend.process(self.dask_mono).compute()
        expected_rms = 1 / np.sqrt(2)  # For a sine wave with amplitude 1
        np.testing.assert_allclose(np.mean(result), expected_rms, rtol=0.1)

        # For AM signal, RMS should vary
        result_am = self.rms_trend.process(self.dask_am).compute()
        assert np.std(result_am) > 0.1 * np.mean(result_am)

    def test_db_conversion(self) -> None:
        """Test dB conversion."""
        rms_db = RmsTrend(self.sample_rate, dB=True, ref=1.0)
        result = rms_db.process(self.dask_mono).compute()

        # Expected RMS value in dB
        expected_rms_linear = 1 / np.sqrt(2)
        expected_rms_db = 20 * np.log10(expected_rms_linear)

        np.testing.assert_allclose(np.mean(result), expected_rms_db, rtol=0.1)

    def test_a_weighting(self) -> None:
        """Test A-weighting effect on RMS."""
        rms_normal = RmsTrend(self.sample_rate)
        rms_aweighted = RmsTrend(self.sample_rate, Aw=True)

        # Create test signal with low frequency content (which A-weighting attenuates)
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        signal_low_freq = np.array([np.sin(2 * np.pi * 50 * t)])  # 50 Hz
        dask_low_freq = _da_from_array(signal_low_freq, chunks=-1)

        result_normal = rms_normal.process(dask_low_freq).compute()
        result_aweighted = rms_aweighted.process(dask_low_freq).compute()

        # A-weighting should attenuate low frequencies
        assert np.mean(result_aweighted) < np.mean(result_normal)

    def test_operation_registry(self) -> None:
        """Test that RmsTrend is properly registered in the operation registry."""
        assert get_operation("rms_trend") == RmsTrend

        rms_op = create_operation(
            "rms_trend", 16000, frame_length=1024, hop_length=256, dB=True
        )

        assert isinstance(rms_op, RmsTrend)
        assert rms_op.frame_length == 1024
        assert rms_op.hop_length == 256
        assert rms_op.dB is True
