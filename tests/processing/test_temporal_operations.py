from unittest import mock

import librosa
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
from wandas.utils.dask_helpers import da_from_array
from wandas.utils.types import NDArrayReal

_SR: int = 16000


class TestReSampling:
    """ReSampling operation: Layer 1 + Layer 2 + Layer 3 (scipy reference)."""

    _ORIG_SR: int = _SR
    _TARGET_SR: int = 8000

    # -- Layer 1: Unit tests -----------------------------------------------

    def test_resampling_init_stores_rates(self) -> None:
        """Test ReSampling stores source and target sampling rates."""
        r = ReSampling(self._ORIG_SR, self._TARGET_SR)
        assert r.sampling_rate == self._ORIG_SR
        assert r.target_sr == self._TARGET_SR

    def test_resampling_registry_returns_correct_class(self) -> None:
        """Test that ReSampling is registered as 'resampling'."""
        assert get_operation("resampling") == ReSampling
        r = create_operation("resampling", 16000, target_sr=22050)
        assert isinstance(r, ReSampling)
        assert r.sampling_rate == 16000
        assert r.target_sr == 22050

    def test_resampling_negative_source_sr_raises(self) -> None:
        """Test negative source sampling rate provides WHAT/WHY/HOW error."""
        with pytest.raises(ValueError) as exc_info:
            ReSampling(sampling_rate=-44100, target_sr=22050)
        error_msg = str(exc_info.value)
        assert "Invalid source sampling rate" in error_msg
        assert "-44100" in error_msg
        assert "Positive value" in error_msg
        assert "Common values:" in error_msg

    def test_resampling_zero_source_sr_raises(self) -> None:
        """Test zero source sampling rate raises error."""
        with pytest.raises(ValueError, match="Invalid source sampling rate"):
            ReSampling(sampling_rate=0, target_sr=22050)

    def test_resampling_negative_target_sr_raises(self) -> None:
        """Test negative target sampling rate provides WHAT/WHY/HOW error."""
        with pytest.raises(ValueError) as exc_info:
            ReSampling(sampling_rate=44100, target_sr=-22050)
        error_msg = str(exc_info.value)
        assert "Invalid target sampling rate" in error_msg
        assert "-22050" in error_msg
        assert "Positive value" in error_msg

    def test_resampling_zero_target_sr_raises(self) -> None:
        """Test zero target sampling rate raises error."""
        with pytest.raises(ValueError, match="Invalid target sampling rate"):
            ReSampling(sampling_rate=44100, target_sr=0)

    # -- Layer 2: Domain (shape + immutability) ----------------------------

    def test_resampling_preserves_immutability_and_dask_type(self, pure_sine_440hz_dask: tuple[DaArray, int]) -> None:
        """Input unchanged after resampling; result is DaArray."""
        dask_input, sr = pure_sine_440hz_dask
        resampler = ReSampling(sr, self._TARGET_SR)
        input_copy = dask_input.compute().copy()

        result_da = resampler.process(dask_input)

        # Pillar 1: immutability
        assert result_da is not dask_input
        np.testing.assert_array_equal(dask_input.compute(), input_copy)
        assert isinstance(result_da, DaArray)

    def test_resampling_downsample_shape_mono(self, pure_sine_440hz_dask: tuple[DaArray, int]) -> None:
        """Downsample 16 kHz -> 8 kHz halves sample count for mono."""
        dask_input, sr = pure_sine_440hz_dask
        resampler = ReSampling(sr, self._TARGET_SR)

        result = resampler.process(dask_input).compute()
        expected_len = int(np.ceil(dask_input.shape[1] * (self._TARGET_SR / sr)))
        assert result.shape == (1, expected_len)

    def test_resampling_upsample_shape_mono(self, pure_sine_440hz_dask: tuple[DaArray, int]) -> None:
        """Upsample 16 kHz -> 32 kHz doubles sample count."""
        dask_input, sr = pure_sine_440hz_dask
        upsampler = ReSampling(sr, sr * 2)

        result = upsampler.process(dask_input).compute()
        expected_len = int(np.ceil(dask_input.shape[1] * 2))
        assert result.shape == (1, expected_len)

    def test_resampling_stereo_shape(self, stereo_sine_440_880hz_dask: tuple[DaArray, int]) -> None:
        """Stereo resampling preserves channel count."""
        dask_input, sr = stereo_sine_440_880hz_dask
        resampler = ReSampling(sr, self._TARGET_SR)

        result = resampler.process(dask_input).compute()
        expected_len = int(np.ceil(dask_input.shape[1] * (self._TARGET_SR / sr)))
        assert result.shape == (2, expected_len)

    # -- Layer 3: Frequency preservation -----------------------------------

    def test_resampling_preserves_peak_frequency(self, pure_sine_440hz_dask: tuple[DaArray, int]) -> None:
        """Peak frequency preserved after resampling (FFT verification).

        Tolerance: rtol=0.1 — FFT bin resolution difference between rates.
        """
        dask_input, sr = pure_sine_440hz_dask
        resampler = ReSampling(sr, self._TARGET_SR)

        result = resampler.process(dask_input).compute()

        # Measure peak frequency in both signals
        orig = dask_input.compute()[0]
        peak_orig = self._peak_freq(orig, sr)
        peak_resampled = self._peak_freq(result[0], self._TARGET_SR)

        np.testing.assert_allclose(
            peak_orig,
            peak_resampled,
            rtol=0.1,  # FFT bin resolution difference between 16 kHz and 8 kHz
        )

    @staticmethod
    def _peak_freq(signal: NDArrayReal, sr: int) -> float:
        """Get the dominant frequency from FFT."""
        fft_result = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), 1 / sr)
        return float(freqs[np.argmax(fft_result)])


class TestTrim:
    """Trim operation: Layer 1 + Layer 2 + Layer 3 (slice equivalence)."""

    _START: float = 0.1
    _END: float = 0.5

    # -- Layer 1: Unit tests -----------------------------------------------

    def test_trim_init_stores_params(self) -> None:
        """Test Trim stores start/end times and derived sample indices."""
        trim = Trim(_SR, self._START, self._END)
        assert trim.sampling_rate == _SR
        assert trim.start == self._START
        assert trim.end == self._END
        assert trim.start_sample == int(self._START * _SR)
        assert trim.end_sample == int(self._END * _SR)

    def test_trim_registry_returns_correct_class(self) -> None:
        """Test that Trim is registered as 'trim'."""
        assert get_operation("trim") == Trim
        t = create_operation("trim", 16000, start=0.2, end=0.8)
        assert isinstance(t, Trim)
        assert t.start == 0.2
        assert t.end == 0.8

    # -- Layer 2: Domain (shape + immutability) ----------------------------

    def test_trim_preserves_immutability_and_dask_type(self, pure_sine_440hz_dask: tuple[DaArray, int]) -> None:
        """Input unchanged after trimming; result is DaArray."""
        dask_input, sr = pure_sine_440hz_dask
        trim = Trim(sr, self._START, self._END)
        input_copy = dask_input.compute().copy()

        result_da = trim.process(dask_input)

        assert result_da is not dask_input
        np.testing.assert_array_equal(dask_input.compute(), input_copy)
        assert isinstance(result_da, DaArray)

    def test_trim_shape_mono(self, pure_sine_440hz_dask: tuple[DaArray, int]) -> None:
        """Trimmed mono shape matches expected sample range."""
        dask_input, sr = pure_sine_440hz_dask
        trim = Trim(sr, self._START, self._END)

        result = trim.process(dask_input).compute()
        expected_samples = int(self._END * sr) - int(self._START * sr)
        assert result.shape == (1, expected_samples)

    def test_trim_shape_stereo(self, stereo_sine_440_880hz_dask: tuple[DaArray, int]) -> None:
        """Trimmed stereo preserves channel count."""
        dask_input, sr = stereo_sine_440_880hz_dask
        trim = Trim(sr, self._START, self._END)

        result = trim.process(dask_input).compute()
        expected_samples = int(self._END * sr) - int(self._START * sr)
        assert result.shape == (2, expected_samples)

    # -- Layer 3: Slice equivalence ----------------------------------------

    def test_trim_content_matches_numpy_slice(self, pure_sine_440hz_dask: tuple[DaArray, int]) -> None:
        """Trimmed result equals direct numpy slice (exact, no tolerance).

        Tolerance: none — slicing is exact, no numerical transformation.
        """
        dask_input, sr = pure_sine_440hz_dask
        trim = Trim(sr, self._START, self._END)

        result = trim.process(dask_input).compute()

        start_idx = int(self._START * sr)
        end_idx = int(self._END * sr)
        expected = dask_input.compute()[:, start_idx:end_idx]
        np.testing.assert_array_equal(result, expected)


class TestRmsTrend:
    """RmsTrend operation: Layer 1 + Layer 2 + Layer 3 (librosa reference)."""

    _FRAME: int = 2048
    _HOP: int = 512

    # -- Layer 1: Unit tests -----------------------------------------------

    def test_rms_trend_init_default_params(self) -> None:
        """Test RmsTrend default parameters."""
        rms = RmsTrend(_SR)
        assert rms.sampling_rate == _SR
        assert rms.frame_length == 2048
        assert rms.hop_length == 512
        assert rms.dB is False
        assert rms.Aw is False

    def test_rms_trend_init_custom_params(self) -> None:
        """Test RmsTrend stores custom parameters."""
        rms = RmsTrend(_SR, frame_length=1024, hop_length=256, dB=True, Aw=True)
        assert rms.frame_length == 1024
        assert rms.hop_length == 256
        assert rms.dB is True
        assert rms.Aw is True

    def test_rms_trend_registry_returns_correct_class(self) -> None:
        """Test that RmsTrend is registered as 'rms_trend'."""
        assert get_operation("rms_trend") == RmsTrend
        rms_op = create_operation("rms_trend", 16000, frame_length=1024, hop_length=256, dB=True)
        assert isinstance(rms_op, RmsTrend)
        assert rms_op.frame_length == 1024
        assert rms_op.hop_length == 256
        assert rms_op.dB is True

    def test_rms_trend_ref_as_list_converts_to_ndarray(self) -> None:
        """Test that ref provided as a list is converted to numpy array."""
        rms = RmsTrend(_SR, dB=True, ref=[1.0])
        assert isinstance(rms.ref, np.ndarray)
        assert rms.ref.shape == (1,)

    # -- Layer 2: Domain (shape + immutability) ----------------------------

    def test_rms_trend_preserves_immutability_and_dask_type(self, pure_sine_440hz_dask: tuple[DaArray, int]) -> None:
        """Input unchanged after RMS computation; result is DaArray."""
        dask_input, sr = pure_sine_440hz_dask
        rms = RmsTrend(sr, frame_length=self._FRAME, hop_length=self._HOP)
        input_copy = dask_input.compute().copy()

        result_da = rms.process(dask_input)

        assert result_da is not dask_input
        np.testing.assert_array_equal(dask_input.compute(), input_copy)
        assert isinstance(result_da, DaArray)

    def test_rms_trend_output_shape_matches_librosa(self, pure_sine_440hz_dask: tuple[DaArray, int]) -> None:
        """Output frame count matches librosa.feature.rms reference."""

        dask_input, sr = pure_sine_440hz_dask
        rms = RmsTrend(sr, frame_length=self._FRAME, hop_length=self._HOP)
        result = rms.process(dask_input).compute()

        expected_frames = librosa.feature.rms(
            y=dask_input.compute(),
            frame_length=self._FRAME,
            hop_length=self._HOP,
        ).shape[-1]
        assert result.shape == (1, expected_frames)

    # -- Layer 3: Numerical verification -----------------------------------

    def test_rms_trend_sine_wave_matches_theoretical_rms(self, pure_sine_440hz_dask: tuple[DaArray, int]) -> None:
        """Steady-state sine RMS approaches 1/sqrt(2).

        Tolerance: rtol=0.1 — windowed RMS frames have boundary effects.
        """
        dask_input, sr = pure_sine_440hz_dask
        rms = RmsTrend(sr, frame_length=self._FRAME, hop_length=self._HOP)
        result = rms.process(dask_input).compute()

        expected_rms = 1 / np.sqrt(2)  # Theoretical RMS for amplitude-1 sine
        np.testing.assert_allclose(
            np.mean(result),
            expected_rms,
            rtol=0.1,  # Boundary frames pull the mean slightly
        )

    def test_rms_trend_am_signal_has_high_variance(self) -> None:
        """Amplitude-modulated signal produces RMS with high variance."""
        t = np.linspace(0, 1, _SR, endpoint=False)
        am = np.array([np.sin(2 * np.pi * 440 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 5 * t))])
        dask_am = da_from_array(am, chunks=(1, -1))
        rms = RmsTrend(_SR, frame_length=self._FRAME, hop_length=self._HOP)

        result_da = rms.process(dask_am)
        assert isinstance(result_da, DaArray)  # Pillar 1: Dask graph preserved
        result = result_da.compute()
        # AM modulation should create significant RMS variation
        assert np.std(result) > 0.1 * np.mean(result)

    def test_rms_trend_db_conversion_matches_formula(self, pure_sine_440hz_dask: tuple[DaArray, int]) -> None:
        """dB output matches 20*log10(RMS/ref).

        Tolerance: rtol=0.1 — windowed boundary effects.
        """
        dask_input, sr = pure_sine_440hz_dask
        rms_db = RmsTrend(sr, dB=True, ref=1.0)
        result = rms_db.process(dask_input).compute()

        expected_rms_linear = 1 / np.sqrt(2)
        expected_rms_db = 20 * np.log10(expected_rms_linear)
        np.testing.assert_allclose(
            np.mean(result),
            expected_rms_db,
            rtol=0.1,
        )

    def test_rms_trend_db_with_custom_ref(self, pure_sine_440hz_dask: tuple[DaArray, int]) -> None:
        """dB output with custom ref matches 20*log10(RMS/ref).

        Tolerance: rtol=0.1 — windowed boundary effects.
        """
        dask_input, sr = pure_sine_440hz_dask
        rms_db = RmsTrend(sr, dB=True, ref=0.5)
        result = rms_db.process(dask_input).compute()

        rms_value = 1 / np.sqrt(2)
        expected_rms_db = 20 * np.log10(rms_value / 0.5)
        np.testing.assert_allclose(
            np.mean(result),
            expected_rms_db,
            rtol=0.1,
        )

    def test_rms_trend_a_weighting_attenuates_low_freq(self) -> None:
        """A-weighting attenuates 50 Hz RMS compared to unweighted."""
        t = np.linspace(0, 1, _SR, endpoint=False)
        low_freq = np.array([np.sin(2 * np.pi * 50 * t)])
        dask_low = da_from_array(low_freq, chunks=(1, -1))

        rms_normal = RmsTrend(_SR)
        rms_aw = RmsTrend(_SR, Aw=True)

        result_normal = rms_normal.process(dask_low)
        result_aw = rms_aw.process(dask_low)
        assert isinstance(result_normal, DaArray)  # Pillar 1: Dask graph preserved
        assert isinstance(result_aw, DaArray)

        assert np.mean(result_aw.compute()) < np.mean(result_normal.compute())

    def test_rms_trend_a_weighting_tuple_return(self) -> None:
        """A_weight returning a tuple uses the first element."""
        rms = RmsTrend(_SR, Aw=True)
        t = np.linspace(0, 1, _SR, endpoint=False)
        arr = np.array([np.sin(2 * np.pi * 440 * t)])
        tuple_result = (arr, None)
        with mock.patch("wandas.processing.temporal.A_weight", return_value=tuple_result):
            result = rms._process_array(arr)
        assert result.shape[0] == 1
        assert result.ndim == 2

    def test_rms_trend_a_weighting_unexpected_type_raises(self) -> None:
        """A_weight returning unexpected type raises ValueError."""
        rms = RmsTrend(_SR, Aw=True)
        t = np.linspace(0, 1, _SR, endpoint=False)
        arr = np.array([np.sin(2 * np.pi * 440 * t)])
        with mock.patch("wandas.processing.temporal.A_weight", return_value=42):
            with pytest.raises(ValueError, match="A_weighting returned an unexpected type"):
                rms._process_array(arr)


class TestFixLength:
    """FixLength operation: Layer 1 + Layer 2 + Layer 3 (pad/truncate verification)."""

    _TARGET: int = 8000

    # -- Layer 1: Unit tests -----------------------------------------------

    def test_fix_length_init_with_length(self) -> None:
        """Test FixLength stores target length from samples."""
        fl = FixLength(_SR, length=self._TARGET)
        assert fl.sampling_rate == _SR
        assert fl.target_length == self._TARGET

    def test_fix_length_init_with_duration(self) -> None:
        """Test FixLength computes target length from duration."""
        fl = FixLength(_SR, duration=0.5)
        assert fl.target_length == int(0.5 * _SR)

    def test_fix_length_init_no_param_raises(self) -> None:
        """Test FixLength without length or duration raises ValueError."""
        with pytest.raises(ValueError, match=r"Either length or duration must be provided"):
            FixLength(_SR)

    def test_fix_length_registry_returns_correct_class(self) -> None:
        """Test that FixLength is registered as 'fix_length'."""
        assert get_operation("fix_length") == FixLength
        fl = create_operation("fix_length", 16000, length=12000)
        assert isinstance(fl, FixLength)
        assert fl.target_length == 12000

        fl2 = create_operation("fix_length", 16000, duration=0.75)
        assert isinstance(fl2, FixLength)
        assert fl2.target_length == int(0.75 * 16000)

    # -- Layer 2: Domain (shape + immutability) ----------------------------

    def test_fix_length_preserves_immutability_and_dask_type(self, pure_sine_440hz_dask: tuple[DaArray, int]) -> None:
        """Input unchanged after fix_length; result is DaArray."""
        dask_input, sr = pure_sine_440hz_dask
        fl = FixLength(sr, length=self._TARGET)
        input_copy = dask_input.compute().copy()

        result_da = fl.process(dask_input)

        assert result_da is not dask_input
        np.testing.assert_array_equal(dask_input.compute(), input_copy)
        assert isinstance(result_da, DaArray)

    def test_fix_length_mono_shape(self, pure_sine_440hz_dask: tuple[DaArray, int]) -> None:
        """Mono output shape matches target length."""
        dask_input, sr = pure_sine_440hz_dask
        fl = FixLength(sr, length=self._TARGET)
        result = fl.process(dask_input).compute()
        assert result.shape == (1, self._TARGET)

    def test_fix_length_stereo_shape(self, stereo_sine_440_880hz_dask: tuple[DaArray, int]) -> None:
        """Stereo output shape preserves channels and matches target."""
        dask_input, sr = stereo_sine_440_880hz_dask
        fl = FixLength(sr, length=self._TARGET)
        result = fl.process(dask_input).compute()
        assert result.shape == (2, self._TARGET)

    def test_fix_length_short_signal_padded(self) -> None:
        """Short signal zero-padded to target length."""
        t_short = np.linspace(0, 0.25, int(_SR * 0.25), endpoint=False)
        short_sig = np.array([np.sin(2 * np.pi * 440 * t_short)])
        dask_short = da_from_array(short_sig, chunks=(1, -1))
        fl = FixLength(_SR, length=self._TARGET)

        result_da = fl.process(dask_short)
        assert isinstance(result_da, DaArray)  # Pillar 1: Dask graph preserved
        result = result_da.compute()
        assert result.shape == (1, self._TARGET)

    def test_fix_length_long_signal_truncated(self) -> None:
        """Long signal truncated to target length."""
        t_long = np.linspace(0, 2, int(_SR * 2), endpoint=False)
        long_sig = np.array([np.sin(2 * np.pi * 440 * t_long)])
        dask_long = da_from_array(long_sig, chunks=(1, -1))
        fl = FixLength(_SR, length=self._TARGET)

        result_da = fl.process(dask_long)
        assert isinstance(result_da, DaArray)  # Pillar 1: Dask graph preserved
        result = result_da.compute()
        assert result.shape == (1, self._TARGET)

    # -- Layer 3: Content verification -------------------------------------

    def test_fix_length_padding_content_exact(self) -> None:
        """Padded short signal: original preserved, padding is zero.

        Tolerance: none — slice + zero-pad is exact.
        """
        t_short = np.linspace(0, 0.25, int(_SR * 0.25), endpoint=False)
        short_sig = np.array([np.sin(2 * np.pi * 440 * t_short)])
        dask_short = da_from_array(short_sig, chunks=(1, -1))
        fl = FixLength(_SR, length=self._TARGET)

        result_da = fl.process(dask_short)
        assert isinstance(result_da, DaArray)  # Pillar 1: Dask graph preserved
        result = result_da.compute()

        # Original part preserved exactly
        np.testing.assert_allclose(result[0, : short_sig.shape[1]], short_sig[0])
        # Padding is zero
        np.testing.assert_array_equal(
            result[0, short_sig.shape[1] :],
            np.zeros(self._TARGET - short_sig.shape[1]),
        )

    def test_fix_length_truncation_content_exact(self) -> None:
        """Truncated long signal matches numpy slice (exact).

        Tolerance: none — slicing is exact.
        """
        t_long = np.linspace(0, 2, int(_SR * 2), endpoint=False)
        long_sig = np.array([np.sin(2 * np.pi * 440 * t_long)])
        dask_long = da_from_array(long_sig, chunks=(1, -1))
        fl = FixLength(_SR, length=self._TARGET)

        result_da = fl.process(dask_long)
        assert isinstance(result_da, DaArray)  # Pillar 1: Dask graph preserved
        result = result_da.compute()
        np.testing.assert_allclose(result[0], long_sig[0, : self._TARGET])


class TestSoundLevel:
    """SoundLevel operation: Layer 1 + Layer 2 + Layer 3 (theoretical RC + weighted power)."""

    _DURATION: float = 8.0
    _AMPLITUDE: float = 2.0
    _LOW_FREQ: float = 50.0

    # -- Layer 1: Unit tests -----------------------------------------------

    def test_sound_level_init_stores_params(self) -> None:
        """Test SoundLevel stores freq/time weighting and dB flag."""
        op = SoundLevel(_SR, freq_weighting="A", time_weighting="Fast", ref=2e-5, dB=True)
        assert op.sampling_rate == _SR
        assert op.freq_weighting == "A"
        assert op.time_weighting == "Fast"
        assert op.dB is True

    def test_sound_level_defaults_to_linear(self) -> None:
        """Test SoundLevel defaults to linear (dB=False) output."""
        op = SoundLevel(_SR, ref=2e-5)
        assert op.dB is False

    def test_sound_level_registry_returns_correct_class(self) -> None:
        """Test that SoundLevel is registered as 'sound_level'."""
        assert get_operation("sound_level") == SoundLevel
        op = create_operation("sound_level", 16000, freq_weighting="A", time_weighting="Slow", ref=2e-5, dB=True)
        assert isinstance(op, SoundLevel)
        assert op.freq_weighting == "A"
        assert op.time_weighting == "Slow"
        assert op.dB is True

    # -- Layer 2: Domain (shape + dtype + immutability) --------------------

    def test_sound_level_preserves_shape(self) -> None:
        """Output shape matches input shape."""
        t = np.linspace(0, self._DURATION, int(self._DURATION * _SR), endpoint=False)
        sig = np.array([self._AMPLITUDE * np.sin(2 * np.pi * self._LOW_FREQ * t)])
        dask_sig = da_from_array(sig, chunks=(1, -1))

        op = SoundLevel(_SR, ref=1.0)
        result_da = op.process(dask_sig)
        assert isinstance(result_da, DaArray)  # Pillar 1: Dask graph preserved
        result = result_da.compute()
        assert result.shape == sig.shape

    def test_sound_level_linear_default_matches_explicit(self) -> None:
        """Default (dB=False) equals explicit linear output."""
        t = np.linspace(0, self._DURATION, int(self._DURATION * _SR), endpoint=False)
        sig = np.array([self._AMPLITUDE * np.sin(2 * np.pi * self._LOW_FREQ * t)])
        dask_sig = da_from_array(sig, chunks=(1, -1))

        default_op = SoundLevel(_SR, ref=2e-5)
        explicit_op = SoundLevel(_SR, ref=2e-5, dB=False)
        default_da = default_op.process(dask_sig)
        explicit_da = explicit_op.process(dask_sig)
        assert isinstance(default_da, DaArray)  # Pillar 1: Dask graph preserved
        assert isinstance(explicit_da, DaArray)
        np.testing.assert_allclose(
            default_da.compute(),
            explicit_da.compute(),
        )

    def test_sound_level_int_input_returns_float64(self) -> None:
        """Integer input produces float64 output."""
        signal_int = np.array([[0, 1000, -1000, 500, -500, 250, -250]], dtype=np.int16)
        signal_float = signal_int.astype(np.float64)
        dask_int = da_from_array(signal_int, chunks=(1, -1))
        dask_float = da_from_array(signal_float, chunks=(1, -1))

        op = SoundLevel(_SR, ref=1.0, freq_weighting="Z", time_weighting="Fast", dB=False)

        int_result_da = op.process(dask_int)
        float_result_da = op.process(dask_float)

        assert int_result_da.dtype == np.float64
        int_result = int_result_da.compute()
        float_result = float_result_da.compute()
        assert int_result.dtype == np.float64
        np.testing.assert_allclose(int_result, float_result)

    def test_sound_level_float32_preserves_dtype(self) -> None:
        """float32 input keeps float32 output."""
        t = np.linspace(0, self._DURATION, int(self._DURATION * _SR), endpoint=False)
        sig_f32 = (self._AMPLITUDE * np.sin(2 * np.pi * self._LOW_FREQ * t)).astype(np.float32).reshape(1, -1)
        dask_f32 = da_from_array(sig_f32, chunks=(1, -1))

        op = SoundLevel(_SR, ref=1.0, freq_weighting="Z", time_weighting="Fast", dB=False)
        result_da = op.process(dask_f32)
        result = result_da.compute()

        assert result_da.dtype == np.float32
        assert result.dtype == np.float32

    def test_sound_level_float32_db_preserves_dtype_and_aliases(self) -> None:
        """float32 dB output normalizes aliases and preserves float32 dtype.

        Tolerance: rtol=1e-4, atol=1e-4 — float32 lfilter/log10 precision.
        """
        t = np.linspace(0, self._DURATION, int(self._DURATION * _SR), endpoint=False)
        sig_f32 = (self._AMPLITUDE * np.sin(2 * np.pi * self._LOW_FREQ * t)).astype(np.float32).reshape(1, -1)
        dask_f32 = da_from_array(sig_f32, chunks=(1, -1))

        f32_op = SoundLevel(_SR, ref=1.0, freq_weighting=None, time_weighting="s", dB=True)
        f64_ref = SoundLevel(_SR, ref=1.0, freq_weighting="Z", time_weighting="Slow", dB=True)

        result_da = f32_op.process(dask_f32)
        result = result_da.compute()
        reference = f64_ref.process(dask_f32.astype(np.float64)).compute()

        assert f32_op.freq_weighting == "Z"
        assert f32_op.time_weighting == "Slow"
        assert result_da.dtype == np.float32
        assert result.dtype == np.float32
        np.testing.assert_allclose(
            result,
            reference,
            rtol=1e-4,
            atol=1e-4,  # float32 precision limits
        )

    # -- Layer 3: Theoretical verification ---------------------------------

    @pytest.mark.parametrize(("curve", "expected_gain"), [("Z", 1.0), ("A", None), ("C", None)])
    def test_sound_level_db_matches_theoretical_weighted_power(self, curve: str, expected_gain: float | None) -> None:
        """Steady-state dB output matches theoretical (A*G/sqrt(2))^2.

        Tolerance: rtol=1e-6 — float64 IIR filter steady state.
        """
        t = np.linspace(0, self._DURATION, int(self._DURATION * _SR), endpoint=False)
        sig = np.array([self._AMPLITUDE * np.sin(2 * np.pi * self._LOW_FREQ * t)])
        dask_sig = da_from_array(sig, chunks=(1, -1))

        if expected_gain is None:
            sos = frequency_weighting(_SR, curve=curve, output="sos")
            _, response = scipy_signal.freqz_sos(sos, worN=self._LOW_FREQ, fs=_SR)
            expected_gain = float(np.abs(np.asarray(response)).item())

        expected_rms = (self._AMPLITUDE * expected_gain) / np.sqrt(2.0)
        expected_power = expected_rms**2

        op = SoundLevel(_SR, ref=1.0, freq_weighting=curve, time_weighting="Fast", dB=True)
        result_db = op.process(dask_sig).compute()

        steady_state_db = result_db[..., result_db.shape[-1] // 2 :]
        measured_power = np.mean(10.0 ** (steady_state_db / 10.0))

        np.testing.assert_allclose(
            measured_power,
            expected_power,
            rtol=1e-6,
            atol=0.0,
        )

    @pytest.mark.parametrize(("curve", "expected_gain"), [("Z", 1.0), ("A", None), ("C", None)])
    def test_sound_level_linear_matches_theoretical_weighted_rms(self, curve: str, expected_gain: float | None) -> None:
        """Linear output matches theoretical time-weighted RMS.

        Tolerance: rtol=1e-6 — float64 IIR filter steady state.
        """
        t = np.linspace(0, self._DURATION, int(self._DURATION * _SR), endpoint=False)
        sig = np.array([self._AMPLITUDE * np.sin(2 * np.pi * self._LOW_FREQ * t)])
        dask_sig = da_from_array(sig, chunks=(1, -1))

        if expected_gain is None:
            sos = frequency_weighting(_SR, curve=curve, output="sos")
            _, response = scipy_signal.freqz_sos(sos, worN=self._LOW_FREQ, fs=_SR)
            expected_gain = float(np.abs(np.asarray(response)).item())

        expected_rms = (self._AMPLITUDE * expected_gain) / np.sqrt(2.0)
        expected_power = expected_rms**2

        op = SoundLevel(_SR, ref=1.0, freq_weighting=curve, time_weighting="Fast", dB=False)
        result_rms = op.process(dask_sig).compute()
        steady_state_rms = result_rms[..., result_rms.shape[-1] // 2 :]
        measured_power = np.mean(np.square(steady_state_rms))

        np.testing.assert_allclose(
            measured_power,
            expected_power,
            rtol=1e-6,
            atol=0.0,
        )

    @pytest.mark.parametrize(("time_weighting", "tau"), [("Fast", 0.125), ("Slow", 1.0)])
    def test_sound_level_rc_step_response_matches_theory(self, time_weighting: str, tau: float) -> None:
        """Time weighting follows theoretical discrete-time RC step response.

        Tolerance: rtol=1e-6 — float64 exponential filter precision.
        """
        step_start = _SR
        step_amplitude = 0.5
        step_signal: NDArrayReal = np.zeros((1, 4 * _SR), dtype=np.float64)
        step_signal[0, step_start:] = step_amplitude
        dask_step = da_from_array(step_signal, chunks=(1, -1))

        op = SoundLevel(_SR, ref=1.0, freq_weighting="Z", time_weighting=time_weighting, dB=True)
        result_db = op.process(dask_step).compute()

        measured_power = 10.0 ** (result_db / 10.0)
        alpha = np.exp(-1.0 / (_SR * tau))
        num_samples = step_signal.shape[-1] - step_start
        n = np.arange(num_samples, dtype=np.float64)
        input_power = step_amplitude**2
        expected_power = input_power * (1.0 - alpha ** (n + 1.0))

        np.testing.assert_allclose(
            measured_power[0, step_start:],
            expected_power,
            rtol=1e-6,
            atol=0.0,
        )


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
        np.testing.assert_array_equal(linear_operation._reference_squared(2), np.array([1.0, 1.0]))
        np.testing.assert_array_equal(SoundLevel(16000, ref=[1.0])._reference_squared(1), np.array([1.0]))
        assert linear_operation._output_dtype(np.dtype(np.float32)) == np.dtype(np.float32)
        assert linear_operation._output_dtype(np.dtype(np.int16)) == np.dtype(np.float64)

        assert db_operation.time_weighting == "Slow"
        assert db_operation.freq_weighting == "C"
        assert db_operation.time_constant == 1.0
        assert db_operation.get_display_name() == "LCS"
        np.testing.assert_array_equal(db_operation._reference_squared(2), np.array([1.0, 4.0]))

        with pytest.raises(ValueError, match="Reference count mismatch"):
            db_operation._reference_squared(3)

        with pytest.raises(ValueError, match="Invalid sound level reference"):
            SoundLevel(16000, ref=0.0)

        with pytest.raises(ValueError, match="Invalid frequency weighting"):
            SoundLevel(16000, ref=1.0, freq_weighting="B")

        with pytest.raises(ValueError, match="Invalid time weighting"):
            SoundLevel(16000, ref=1.0, time_weighting="Medium")
