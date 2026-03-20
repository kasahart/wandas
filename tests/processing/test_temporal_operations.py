import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.processing.base import create_operation, get_operation, register_operation
from wandas.processing.temporal import (
    FixLength,
    ReSampling,
    RmsTrend,
    SoundLevel,
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

        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=(1, -1))
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=(1, -1))

    def test_initialization(self) -> None:
        """Test initialization with different parameters."""
        resampler = ReSampling(self.orig_sr, self.target_sr)
        assert resampler.sampling_rate == self.orig_sr
        assert resampler.target_sr == self.target_sr

    def test_resampling_shape(self) -> None:
        """Test resampling output shape."""
        # Downsample
        result = self.resampler.process(self.dask_mono).compute()
        expected_len = int(np.ceil(self.signal_mono.shape[1] * (self.target_sr / self.orig_sr)))
        assert result.shape == (1, expected_len)

        # Upsample
        upsampler = ReSampling(self.orig_sr, self.orig_sr * 2)
        result = upsampler.process(self.dask_mono).compute()
        expected_len = int(np.ceil(self.signal_mono.shape[1] * 2))
        assert result.shape == (1, expected_len)

        # Stereo
        result = self.resampler.process(self.dask_stereo).compute()
        expected_len = int(np.ceil(self.signal_stereo.shape[1] * (self.target_sr / self.orig_sr)))
        assert result.shape == (2, expected_len)

    def test_resampling_content(self) -> None:
        """Test resampled content frequency preservation."""
        # Create test signal with a specific frequency
        freq = 440.0  # Hz
        t_orig = np.linspace(0, 1, self.orig_sr, endpoint=False)
        signal = np.array([np.sin(2 * np.pi * freq * t_orig)])
        dask_signal = _da_from_array(signal, chunks=(1, -1))

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

    def test_negative_source_sampling_rate_error_message(self) -> None:
        """Test that negative source sampling rate provides helpful error message."""
        with pytest.raises(ValueError) as exc_info:
            ReSampling(sampling_rate=-44100, target_sr=22050)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Invalid source sampling rate" in error_msg
        assert "-44100" in error_msg
        # Check WHY
        assert "Positive value" in error_msg
        # Check HOW
        assert "Common values:" in error_msg
        assert "44100" in error_msg

    def test_zero_source_sampling_rate_error_message(self) -> None:
        """Test that zero source sampling rate provides helpful error message."""
        with pytest.raises(ValueError) as exc_info:
            ReSampling(sampling_rate=0, target_sr=22050)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Invalid source sampling rate" in error_msg
        # Check WHY
        assert "Positive value" in error_msg

    def test_negative_target_sampling_rate_error_message(self) -> None:
        """Test that negative target sampling rate provides helpful error message."""
        with pytest.raises(ValueError) as exc_info:
            ReSampling(sampling_rate=44100, target_sr=-22050)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Invalid target sampling rate" in error_msg
        assert "-22050" in error_msg
        # Check WHY
        assert "Positive value" in error_msg
        # Check HOW
        assert "Common values:" in error_msg

    def test_zero_target_sampling_rate_error_message(self) -> None:
        """Test that zero target sampling rate provides helpful error message."""
        with pytest.raises(ValueError) as exc_info:
            ReSampling(sampling_rate=44100, target_sr=0)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Invalid target sampling rate" in error_msg
        # Check WHY
        assert "Positive value" in error_msg


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

        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=(1, -1))
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=(1, -1))

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
        expected_samples = int(self.end_time * self.sample_rate) - int(self.start_time * self.sample_rate)
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
        self.rms_trend = RmsTrend(self.sample_rate, frame_length=self.frame_length, hop_length=self.hop_length)

        # Create sample signal: 1 second sine wave at 440 Hz
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        self.signal_mono: NDArrayReal = np.array([np.sin(2 * np.pi * 440 * t)])
        # Create amplitude-modulated signal
        self.am_signal = np.array([np.sin(2 * np.pi * 440 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 5 * t))])

        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=(1, -1))
        self.dask_am: DaArray = _da_from_array(self.am_signal, chunks=(1, -1))

    def test_initialization(self) -> None:
        """Test initialization with different parameters."""
        rms = RmsTrend(self.sample_rate)
        assert rms.sampling_rate == self.sample_rate
        assert rms.frame_length == 2048  # Default value
        assert rms.hop_length == 512  # Default value
        assert rms.dB is False
        assert rms.Aw is False

        custom_rms = RmsTrend(self.sample_rate, frame_length=1024, hop_length=256, dB=True, Aw=True)
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

    def test_db_conversion_with_ref(self) -> None:
        """Test dB conversion with custom ref value."""
        # RMS値は1/sqrt(2)
        rms_value = 1 / np.sqrt(2)
        # ref=0.5 で dB変換
        rms_db = RmsTrend(self.sample_rate, dB=True, ref=0.5)
        result = rms_db.process(self.dask_mono).compute()
        # 期待されるdB値
        expected_rms_db = 20 * np.log10(rms_value / 0.5)
        np.testing.assert_allclose(np.mean(result), expected_rms_db, rtol=0.1)

    def test_a_weighting(self) -> None:
        """Test A-weighting effect on RMS."""
        rms_normal = RmsTrend(self.sample_rate)
        rms_aweighted = RmsTrend(self.sample_rate, Aw=True)

        # Create test signal with low frequency content (which A-weighting attenuates)
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        signal_low_freq = np.array([np.sin(2 * np.pi * 50 * t)])  # 50 Hz
        dask_low_freq = _da_from_array(signal_low_freq, chunks=(1, -1))

        result_normal = rms_normal.process(dask_low_freq).compute()
        result_aweighted = rms_aweighted.process(dask_low_freq).compute()

        # A-weighting should attenuate low frequencies
        assert np.mean(result_aweighted) < np.mean(result_normal)

    def test_operation_registry(self) -> None:
        """Test that RmsTrend is properly registered in the operation registry."""
        assert get_operation("rms_trend") == RmsTrend

        rms_op = create_operation("rms_trend", 16000, frame_length=1024, hop_length=256, dB=True)

        assert isinstance(rms_op, RmsTrend)
        assert rms_op.frame_length == 1024
        assert rms_op.hop_length == 256
        assert rms_op.dB is True


class TestFixLength:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.target_length: int = 8000  # サンプル単位での目標長
        self.target_duration: float = 0.5  # 秒単位での目標長
        self.fix_length = FixLength(self.sample_rate, length=self.target_length)

        # 1秒のサイン波（440Hz）のサンプル信号を作成
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        self.signal_mono: NDArrayReal = np.array([np.sin(2 * np.pi * 440 * t)])
        self.signal_stereo: NDArrayReal = np.array(
            [
                np.sin(2 * np.pi * 440 * t),
                np.sin(2 * np.pi * 880 * t),
            ]
        )

        # 短い信号（目標長より短い）
        t_short = np.linspace(0, 0.25, int(self.sample_rate * 0.25), endpoint=False)
        self.short_signal: NDArrayReal = np.array([np.sin(2 * np.pi * 440 * t_short)])

        # 長い信号（目標長より長い）
        t_long = np.linspace(0, 2, int(self.sample_rate * 2), endpoint=False)
        self.long_signal: NDArrayReal = np.array([np.sin(2 * np.pi * 440 * t_long)])

        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=(1, -1))
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=(1, -1))
        self.dask_short: DaArray = _da_from_array(self.short_signal, chunks=(1, -1))
        self.dask_long: DaArray = _da_from_array(self.long_signal, chunks=(1, -1))

    def test_initialization(self) -> None:
        """異なるパラメータでの初期化をテストします。"""
        # lengthで初期化
        fix_length = FixLength(self.sample_rate, length=self.target_length)
        assert fix_length.sampling_rate == self.sample_rate
        assert fix_length.target_length == self.target_length

        # durationで初期化
        fix_duration = FixLength(self.sample_rate, duration=self.target_duration)
        assert fix_duration.target_length == int(self.target_duration * self.sample_rate)

        # パラメータなしの場合はエラー
        with pytest.raises(ValueError):
            FixLength(self.sample_rate)

    def test_fix_length_shape(self) -> None:
        """出力形状をテストします。"""
        # 標準のモノラル信号
        result = self.fix_length.process(self.dask_mono).compute()
        assert result.shape == (1, self.target_length)

        # ステレオ信号
        result_stereo = self.fix_length.process(self.dask_stereo).compute()
        assert result_stereo.shape == (2, self.target_length)

        # 短い信号（パディングが必要）
        result_short = self.fix_length.process(self.dask_short).compute()
        assert result_short.shape == (1, self.target_length)

        # 長い信号（切り詰めが必要）
        result_long = self.fix_length.process(self.dask_long).compute()
        assert result_long.shape == (1, self.target_length)

    def test_fix_length_content(self) -> None:
        """処理内容をテストします。"""
        # 短い信号のパディング
        result_short = self.fix_length.process(self.dask_short).compute()
        # 元の部分は同じ
        np.testing.assert_allclose(result_short[0, : self.short_signal.shape[1]], self.short_signal[0])
        # パディング部分はゼロ
        assert np.allclose(
            result_short[0, self.short_signal.shape[1] :],
            np.zeros(self.target_length - self.short_signal.shape[1]),
        )

        # 長い信号の切り詰め
        result_long = self.fix_length.process(self.dask_long).compute()
        # 保持された部分は元のデータと同じ
        np.testing.assert_allclose(result_long[0], self.long_signal[0, : self.target_length])

    def test_operation_registry(self) -> None:
        """オペレーションレジストリにFixLengthが適切に登録されているかテストします。"""
        assert get_operation("fix_length") == FixLength

        # lengthで作成
        fix_op = create_operation("fix_length", 16000, length=12000)
        assert isinstance(fix_op, FixLength)
        assert fix_op.sampling_rate == 16000
        assert fix_op.target_length == 12000

        # durationで作成
        fix_op2 = create_operation("fix_length", 16000, duration=0.75)
        assert isinstance(fix_op2, FixLength)
        assert fix_op2.target_length == int(0.75 * 16000)


# Register FixLength in the operation registry (if not done in __init__.py)
register_operation(FixLength)


class TestRmsTrendMetadataUpdates:
    """Test metadata updates for RmsTrend operation."""

    def test_rms_trend_metadata_updates(self) -> None:
        """Test that RmsTrend returns correct metadata updates."""
        operation = RmsTrend(sampling_rate=44100, frame_length=2048, hop_length=512)

        updates = operation.get_metadata_updates()

        assert "sampling_rate" in updates
        expected_sr = 44100 / 512
        assert np.isclose(updates["sampling_rate"], expected_sr)

    def test_rms_trend_metadata_with_different_hop_length(self) -> None:
        """Test metadata updates with different hop_length values."""
        hop_length = 256
        operation = RmsTrend(sampling_rate=48000, frame_length=2048, hop_length=hop_length)

        updates = operation.get_metadata_updates()

        expected_sr = 48000 / hop_length
        assert np.isclose(updates["sampling_rate"], expected_sr)


class TestSoundLevel:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 44100
        # 1 second of 1 kHz sine wave at amplitude 1 Pa
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        self.signal_mono: NDArrayReal = np.array([np.sin(2 * np.pi * 1000 * t)])
        self.signal_stereo: NDArrayReal = np.array(
            [
                np.sin(2 * np.pi * 1000 * t),
                np.sin(2 * np.pi * 500 * t),
            ]
        )
        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=(1, -1))
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=(1, -1))

    def test_initialization_fast(self) -> None:
        """Test initialization with Fast time constant string."""
        op = SoundLevel(self.sample_rate, time_constant="F")
        assert op.tau == 0.125
        assert op.tc_label == "F"
        assert op.weighting == "A"
        assert op.dB is True

    def test_initialization_slow(self) -> None:
        """Test initialization with Slow time constant string."""
        op = SoundLevel(self.sample_rate, time_constant="S")
        assert op.tau == 1.0
        assert op.tc_label == "S"

    def test_initialization_custom_tau(self) -> None:
        """Test initialization with custom float time constant."""
        op = SoundLevel(self.sample_rate, time_constant=0.5)
        assert op.tau == 0.5
        assert op.tc_label == "0.500s"

    def test_initialization_z_weighting(self) -> None:
        """Test initialization with Z (flat) weighting."""
        op = SoundLevel(self.sample_rate, weighting="Z")
        assert op.weighting == "Z"

    def test_invalid_time_constant_string(self) -> None:
        """Test that invalid time constant string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown time constant"):
            SoundLevel(self.sample_rate, time_constant="X")

    def test_invalid_time_constant_negative(self) -> None:
        """Test that non-positive float time constant raises ValueError."""
        with pytest.raises(ValueError, match="time_constant must be positive"):
            SoundLevel(self.sample_rate, time_constant=-0.1)

    def test_invalid_weighting(self) -> None:
        """Test that invalid weighting raises ValueError."""
        with pytest.raises(ValueError, match="Unknown weighting"):
            SoundLevel(self.sample_rate, weighting="X")

    def test_output_shape_no_hop(self) -> None:
        """Test output shape without downsampling."""
        op = SoundLevel(self.sample_rate, hop_length=1)
        result = op.process(self.dask_mono).compute()
        assert result.shape == self.signal_mono.shape

    def test_output_shape_with_hop(self) -> None:
        """Test output shape with downsampling."""
        hop = 512
        op = SoundLevel(self.sample_rate, hop_length=hop)
        result = op.process(self.dask_mono).compute()
        expected_len = int(np.ceil(self.signal_mono.shape[-1] / hop))
        assert result.shape == (1, expected_len)

    def test_output_shape_stereo(self) -> None:
        """Test output shape for stereo input."""
        op = SoundLevel(self.sample_rate, hop_length=1)
        result = op.process(self.dask_stereo).compute()
        assert result.shape == self.signal_stereo.shape

    def test_db_output_range(self) -> None:
        """Test that dB output has reasonable values for 1 Pa sine wave."""
        # Use 5 seconds to allow Slow (1s) time constant to settle
        t = np.linspace(0, 5, 5 * self.sample_rate, endpoint=False)
        signal = np.array([np.sin(2 * np.pi * 1000 * t)])
        dask_signal = _da_from_array(signal, chunks=(1, -1))

        op = SoundLevel(self.sample_rate, time_constant="S", dB=True, ref=2e-5)
        result = op.process(dask_signal).compute()
        # After settling (last 20% of 5 s signal), level should be near expected
        settled = result[0, 4 * result.shape[-1] // 5 :]
        # 1 kHz sine at amplitude 1 Pa: A-weighting ~0 dB at 1 kHz, RMS = 1/sqrt(2)
        expected_db = 20 * np.log10((1 / np.sqrt(2)) / 2e-5)
        np.testing.assert_allclose(np.mean(settled), expected_db, atol=2.0)

    def test_linear_output(self) -> None:
        """Test linear (non-dB) output."""
        # Use 5 seconds for Slow time constant to settle
        t = np.linspace(0, 5, 5 * self.sample_rate, endpoint=False)
        signal = np.array([np.sin(2 * np.pi * 1000 * t)])
        dask_signal = _da_from_array(signal, chunks=(1, -1))

        op = SoundLevel(self.sample_rate, time_constant="S", dB=False, weighting="Z", ref=1.0)
        result = op.process(dask_signal).compute()
        # Should be near 1/sqrt(2) after settling
        settled = result[0, 4 * result.shape[-1] // 5 :]
        np.testing.assert_allclose(np.mean(settled), 1 / np.sqrt(2), atol=0.05)

    def test_a_weighting_attenuates_low_freq(self) -> None:
        """Test that A-weighting attenuates low frequencies."""
        t = np.linspace(0, 2, 2 * self.sample_rate, endpoint=False)
        low_freq_signal = np.array([np.sin(2 * np.pi * 50 * t)])
        dask_low = _da_from_array(low_freq_signal, chunks=(1, -1))

        op_a = SoundLevel(self.sample_rate, weighting="A", time_constant="S", dB=False)
        op_z = SoundLevel(self.sample_rate, weighting="Z", time_constant="S", dB=False)

        result_a = op_a.process(dask_low).compute()
        result_z = op_z.process(dask_low).compute()

        # A-weighting should attenuate 50 Hz
        assert np.mean(result_a) < np.mean(result_z)

    def test_fast_settles_faster_than_slow(self) -> None:
        """Test that Fast time constant settles faster than Slow."""
        # Signal switches from 0 to 1 at t=0
        t = np.linspace(0, 2, 2 * self.sample_rate, endpoint=False)
        step = np.array([np.sin(2 * np.pi * 1000 * t)])
        dask_step = _da_from_array(step, chunks=(1, -1))

        op_fast = SoundLevel(self.sample_rate, time_constant="F", dB=False, weighting="Z")
        op_slow = SoundLevel(self.sample_rate, time_constant="S", dB=False, weighting="Z")

        result_fast = op_fast.process(dask_step).compute()
        result_slow = op_slow.process(dask_step).compute()

        # At 0.25 s (2 * Fast tau), Fast should be closer to steady state
        idx_quarter = self.sample_rate // 4
        # Fast should have higher value early on (closer to settled RMS)
        assert result_fast[0, idx_quarter] > result_slow[0, idx_quarter]

    def test_metadata_updates(self) -> None:
        """Test that metadata is updated correctly."""
        hop = 100
        op = SoundLevel(self.sample_rate, hop_length=hop)
        updates = op.get_metadata_updates()
        assert "sampling_rate" in updates
        np.testing.assert_allclose(updates["sampling_rate"], self.sample_rate / hop)

    def test_operation_registry(self) -> None:
        """Test that SoundLevel is properly registered in the operation registry."""
        assert get_operation("sound_level") == SoundLevel
        op = create_operation("sound_level", self.sample_rate, time_constant="F", weighting="A")
        assert isinstance(op, SoundLevel)
        assert op.tau == 0.125
        assert op.weighting == "A"

    def test_calculate_output_shape(self) -> None:
        """Test calculate_output_shape directly."""
        op = SoundLevel(self.sample_rate, hop_length=1)
        assert op.calculate_output_shape((2, 44100)) == (2, 44100)

        op_hop = SoundLevel(self.sample_rate, hop_length=441)
        assert op_hop.calculate_output_shape((2, 44100)) == (2, 100)

    def test_display_name_db_fast(self) -> None:
        """Test display name for dB Fast A-weighting."""
        op = SoundLevel(self.sample_rate, time_constant="F", weighting="A", dB=True)
        assert op.get_display_name() == "LAF"

    def test_display_name_db_slow(self) -> None:
        """Test display name for dB Slow A-weighting."""
        op = SoundLevel(self.sample_rate, time_constant="S", weighting="A", dB=True)
        assert op.get_display_name() == "LAS"

    def test_display_name_linear(self) -> None:
        """Test display name for linear output."""
        op = SoundLevel(self.sample_rate, time_constant="F", weighting="A", dB=False)
        assert op.get_display_name() == "level_AF"
