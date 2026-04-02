from unittest import mock

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray
from scipy import signal as scipy_signal

from wandas.processing.base import create_operation, get_operation, register_operation
from wandas.processing.temporal import (
    FixLength,
    ReSampling,
    RmsTrend,
    SoundLevel,
    Trim,
)
from wandas.processing.weighting import frequency_weighting
from wandas.utils.types import NDArrayReal

_da_from_array = da.from_array


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

    def test_a_weighting_returns_tuple(self) -> None:
        """Test that A_weight returning a tuple uses the first element."""
        rms = RmsTrend(self.sample_rate, Aw=True)
        arr = self.signal_mono.copy()
        # Mock A_weight to return a tuple (array, None)
        tuple_result = (arr, None)
        with mock.patch("wandas.processing.temporal.A_weight", return_value=tuple_result):
            result = rms._process_array(arr)
        assert result.shape[0] == 1
        assert result.ndim == 2

    def test_a_weighting_returns_unexpected_type(self) -> None:
        """Test that A_weight returning an unexpected type raises ValueError."""
        rms = RmsTrend(self.sample_rate, Aw=True)
        arr = self.signal_mono.copy()
        with mock.patch("wandas.processing.temporal.A_weight", return_value=42):
            with pytest.raises(ValueError, match="A_weighting returned an unexpected type"):
                rms._process_array(arr)

    def test_ref_as_list(self) -> None:
        """Test that ref provided as a list is converted to a numpy array."""
        rms = RmsTrend(self.sample_rate, dB=True, ref=[1.0])
        assert isinstance(rms.ref, np.ndarray)
        assert rms.ref.shape == (1,)


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


class TestSoundLevel:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.duration_seconds: float = 8.0
        # Use amplitude 2.0 so squaring changes the magnitude and theoretical checks
        # can catch power/RMS mistakes more easily than with amplitude 1.0.
        self.amplitude: float = 2.0
        t = np.linspace(0, self.duration_seconds, int(self.duration_seconds * self.sample_rate), endpoint=False)
        self.low_freq: float = 50.0
        self.low_freq_signal: NDArrayReal = np.array([self.amplitude * np.sin(2 * np.pi * self.low_freq * t)])
        self.dask_low_freq: DaArray = _da_from_array(self.low_freq_signal, chunks=(1, -1))

        self.step_start = self.sample_rate
        self.step_amplitude: float = 0.5
        self.step_signal: NDArrayReal = np.zeros((1, 4 * self.sample_rate), dtype=np.float64)
        self.step_signal[0, self.step_start :] = self.step_amplitude
        self.dask_step: DaArray = _da_from_array(self.step_signal, chunks=(1, -1))

    def test_initialization(self) -> None:
        """Test initialization with different parameters."""
        op = SoundLevel(self.sample_rate, freq_weighting="A", time_weighting="Fast", ref=2e-5, dB=True)
        assert op.sampling_rate == self.sample_rate
        assert op.freq_weighting == "A"
        assert op.time_weighting == "Fast"
        assert op.dB is True

    def test_initialization_defaults_to_linear_rms_output(self) -> None:
        """Test that sound level defaults to linear time-weighted RMS output."""
        op = SoundLevel(self.sample_rate, ref=2e-5)
        assert op.dB is False

        explicit_linear = SoundLevel(self.sample_rate, ref=2e-5, dB=False)
        default_result = op.process(self.dask_low_freq).compute()
        explicit_result = explicit_linear.process(self.dask_low_freq).compute()

        np.testing.assert_allclose(default_result, explicit_result)

    def test_sound_level_shape(self) -> None:
        """Test that sound level preserves input shape."""
        op = SoundLevel(self.sample_rate, ref=1.0)
        result = op.process(self.dask_low_freq).compute()
        assert result.shape == self.low_freq_signal.shape

    def test_integer_input_returns_float64_and_matches_float_input_values(self) -> None:
        """Integer input should produce float output without truncation."""
        signal_int = np.array([[0, 1000, -1000, 500, -500, 250, -250]], dtype=np.int16)
        signal_float = signal_int.astype(np.float64)
        dask_int = _da_from_array(signal_int, chunks=(1, -1))
        dask_float = _da_from_array(signal_float, chunks=(1, -1))

        operation = SoundLevel(self.sample_rate, ref=1.0, freq_weighting="Z", time_weighting="Fast", dB=False)

        int_result_da = operation.process(dask_int)
        float_result_da = operation.process(dask_float)

        assert int_result_da.dtype == np.float64

        int_result = int_result_da.compute()
        float_result = float_result_da.compute()

        assert int_result.dtype == np.float64
        np.testing.assert_allclose(int_result, float_result)

    def test_float32_input_preserves_float32_output_dtype(self) -> None:
        """float32 input should keep float32 output metadata and computed dtype."""
        signal_f32 = self.low_freq_signal.astype(np.float32)
        dask_f32 = _da_from_array(signal_f32, chunks=(1, -1))
        operation = SoundLevel(self.sample_rate, ref=1.0, freq_weighting="Z", time_weighting="Fast", dB=False)

        result_da = operation.process(dask_f32)
        result = result_da.compute()

        assert result_da.dtype == np.float32
        assert result.dtype == np.float32

    def test_float32_db_output_with_none_weighting_and_slow_alias_preserves_float32_dtype(self) -> None:
        """float32 dB output should normalize aliases and keep float32 Dask metadata."""
        signal_f32 = self.low_freq_signal.astype(np.float32)
        dask_f32 = _da_from_array(signal_f32, chunks=(1, -1))
        float32_operation = SoundLevel(
            self.sample_rate,
            ref=1.0,
            freq_weighting=None,
            time_weighting="s",
            dB=True,
        )
        float64_reference = SoundLevel(
            self.sample_rate,
            ref=1.0,
            freq_weighting="Z",
            time_weighting="Slow",
            dB=True,
        )

        result_da = float32_operation.process(dask_f32)
        result = result_da.compute()
        # Use the same underlying samples as the float32 path, but cast to float64
        reference = float64_reference.process(dask_f32.astype(np.float64)).compute()

        assert float32_operation.freq_weighting == "Z"
        assert float32_operation.time_weighting == "Slow"
        assert result_da.dtype == np.float32
        assert result.dtype == np.float32
        # float32 dB output exercises the lower-precision lfilter/log10 path, so use a
        # tolerance appropriate for comparing it against the float64 reference result.
        # tolerance appropriate for comparing it against the float64 reference result.
        np.testing.assert_allclose(result, reference, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(("curve", "expected_gain"), [("Z", 1.0), ("A", None), ("C", None)])
    def test_sound_level_matches_theoretical_weighted_power(self, curve: str, expected_gain: float | None) -> None:
        """Test weighted sound level against theoretical steady-state power."""
        # 1. Obtain the theoretical gain (absolute value of the transfer function)
        if expected_gain is None:
            sos = frequency_weighting(self.sample_rate, curve=curve, output="sos")
            _, response = scipy_signal.freqz_sos(sos, worN=self.low_freq, fs=self.sample_rate)
            expected_gain = float(np.abs(np.asarray(response)).item())

        # 2. Calculate the theoretical expected power
        # P = (A * G / sqrt(2))^2  => squared RMS of a sine wave
        expected_rms = (self.amplitude * expected_gain) / np.sqrt(2.0)
        expected_power = expected_rms**2

        # 3. Process signal and convert dB back to linear power
        operation = SoundLevel(self.sample_rate, ref=1.0, freq_weighting=curve, time_weighting="Fast", dB=True)
        result_db = operation.process(self.dask_low_freq).compute()

        # Extract steady-state power from the latter half of the result
        steady_state_db = result_db[..., result_db.shape[-1] // 2 :]
        measured_power = np.mean(10.0 ** (steady_state_db / 10.0))

        # 4. Verification
        np.testing.assert_allclose(
            measured_power,
            expected_power,
            rtol=1e-6,
            atol=0.0,
        )

    @pytest.mark.parametrize(("curve", "expected_gain"), [("Z", 1.0), ("A", None), ("C", None)])
    def test_linear_output_matches_theoretical_weighted_rms(self, curve: str, expected_gain: float | None) -> None:
        """Test linear output against theoretical time-weighted RMS."""
        # 1. Obtain the theoretical gain (absolute value of the transfer function)
        if expected_gain is None:
            sos = frequency_weighting(self.sample_rate, curve=curve, output="sos")
            _, response = scipy_signal.freqz_sos(sos, worN=self.low_freq, fs=self.sample_rate)
            expected_gain = float(np.abs(np.asarray(response)).item())

        # 2. Calculate the theoretical expected power of the weighted sine wave
        expected_rms = (self.amplitude * expected_gain) / np.sqrt(2.0)
        expected_power = expected_rms**2

        # 3. Process signal using linear output and extract the steady-state region
        operation = SoundLevel(self.sample_rate, ref=1.0, freq_weighting=curve, time_weighting="Fast", dB=False)
        result_rms = operation.process(self.dask_low_freq).compute()
        steady_state_rms = result_rms[..., result_rms.shape[-1] // 2 :]
        measured_power = np.mean(np.square(steady_state_rms))

        # 4. Verification
        np.testing.assert_allclose(
            measured_power,
            expected_power,
            rtol=1e-6,
            atol=0.0,
        )

    @pytest.mark.parametrize(("time_weighting", "tau"), [("Fast", 0.125), ("Slow", 1.0)])
    def test_time_weighting_matches_theoretical_rc_formula(self, time_weighting: str, tau: float) -> None:
        """
        Test if the time weighting (Fast/Slow) correctly follows the
        theoretical discrete-time RC step response.
        """
        # 1. Setup the sound level operation with Z-weighting (flat)
        operation = SoundLevel(self.sample_rate, ref=1.0, freq_weighting="Z", time_weighting=time_weighting, dB=True)
        result_db = operation.process(self.dask_step).compute()

        # 2. Linearize: convert dB output back to power ratio
        measured_power = 10.0 ** (result_db / 10.0)

        # 3. Calculate theoretical parameters
        alpha = np.exp(-1.0 / (self.sample_rate * tau))
        num_samples = self.step_signal.shape[-1] - self.step_start
        n = np.arange(num_samples, dtype=np.float64)

        # 4. Generate the expected step response curve
        input_power = self.step_amplitude**2
        expected_power = input_power * (1.0 - alpha ** (n + 1.0))

        # 5. Verify that the measured power curve matches the theoretical RC response
        np.testing.assert_allclose(
            measured_power[0, self.step_start :],
            expected_power,
            rtol=1e-6,
            atol=0.0,
        )

    def test_operation_registry(self) -> None:
        """Test that SoundLevel is properly registered in the operation registry."""
        assert get_operation("sound_level") == SoundLevel

        sound_level_op = create_operation(
            "sound_level", 16000, freq_weighting="A", time_weighting="Slow", ref=2e-5, dB=True
        )

        assert isinstance(sound_level_op, SoundLevel)
        assert sound_level_op.freq_weighting == "A"
        assert sound_level_op.time_weighting == "Slow"
        assert sound_level_op.dB is True


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


class TestTemporalHelperMethods:
    """Target helper methods and edge branches for temporal operations."""

    def test_resampling_helper_methods(self) -> None:
        """Resampling helper methods should report output metadata consistently."""
        operation = ReSampling(44100, 22050)

        assert operation.get_metadata_updates() == {"sampling_rate": 22050}
        assert operation.calculate_output_shape((2, 441)) == (2, 221)
        assert operation.get_display_name() == "rs"

    def test_trim_helper_methods_cap_output_length_to_input(self) -> None:
        """Trim should cap the output length when end exceeds the input length."""
        operation = Trim(1000, start=0.1, end=2.0)

        assert operation.calculate_output_shape((2, 500)) == (2, 400)
        assert operation.get_display_name() == "trim"

    def test_fix_length_helper_methods(self) -> None:
        """FixLength helper methods should expose target length metadata."""
        operation = FixLength(16000, duration=0.25)

        assert operation.target_length == 4000
        assert operation.calculate_output_shape((2, 16000)) == (2, 4000)
        assert operation.get_display_name() == "fix"

    def test_rms_trend_helper_methods(self) -> None:
        """RmsTrend helper methods should report derived sampling information."""
        operation = RmsTrend(16000, frame_length=512, hop_length=128, dB=True)

        assert operation.get_display_name() == "RMS"
        assert operation.get_metadata_updates() == {"sampling_rate": 125.0}
        assert operation.calculate_output_shape((2, 16000)) == (2, 126)

    def test_sound_level_helper_methods_and_reference_validation(self) -> None:
        """SoundLevel helper methods should validate references and expose helpers."""
        linear_operation = SoundLevel(
            16000,
            ref=1.0,
            freq_weighting=None,
            time_weighting="fast",
            dB=False,
        )
        db_operation = SoundLevel(
            16000,
            ref=[1.0, 2.0],
            freq_weighting="c",
            time_weighting="s",
            dB=True,
        )

        assert linear_operation.calculate_output_shape((2, 16)) == (2, 16)
        assert linear_operation.time_constant == 0.125
        assert linear_operation.get_display_name() == "ZFRMS"
        assert np.array_equal(linear_operation._reference_squared(2), np.array([1.0, 1.0]))
        assert np.array_equal(SoundLevel(16000, ref=[1.0])._reference_squared(1), np.array([1.0]))
        assert linear_operation._output_dtype(np.dtype(np.float32)) == np.dtype(np.float32)
        assert linear_operation._output_dtype(np.dtype(np.int16)) == np.dtype(np.float64)

        assert db_operation.time_weighting == "Slow"
        assert db_operation.freq_weighting == "C"
        assert db_operation.time_constant == 1.0
        assert db_operation.get_display_name() == "LCS"
        assert np.array_equal(db_operation._reference_squared(2), np.array([1.0, 4.0]))

        with pytest.raises(ValueError, match="Reference count mismatch"):
            db_operation._reference_squared(3)

        with pytest.raises(ValueError, match="Invalid sound level reference"):
            SoundLevel(16000, ref=0.0)

        with pytest.raises(ValueError, match="Invalid frequency weighting"):
            SoundLevel(16000, ref=1.0, freq_weighting="B")

        with pytest.raises(ValueError, match="Invalid time weighting"):
            SoundLevel(16000, ref=1.0, time_weighting="Medium")
