# tests/core/test_util.py
import librosa
import numpy as np
import pytest
from scipy.signal.windows import tukey

from wandas.utils.util import (
    amplitude_to_db,
    calculate_desired_noise_rms,
    calculate_rms,
    cut_sig,
    level_trigger,
    validate_sampling_rate,
)


class TestValidateSamplingRate:
    """Test suite for validate_sampling_rate function."""

    def test_positive_sampling_rate(self) -> None:
        """Test that positive sampling rates pass validation."""
        # Common sampling rates
        validate_sampling_rate(8000)
        validate_sampling_rate(16000)
        validate_sampling_rate(22050)
        validate_sampling_rate(44100)
        validate_sampling_rate(48000)
        validate_sampling_rate(96000)

        # Edge case: very small positive value
        validate_sampling_rate(0.001)

        # Edge case: very large value
        validate_sampling_rate(1e9)

    def test_zero_sampling_rate_raises_error(self) -> None:
        """Test that zero sampling rate raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_sampling_rate(0)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Invalid sampling_rate" in error_msg
        # Check WHY (actual vs expected)
        assert "0" in error_msg or "0.0" in error_msg
        assert "Positive value > 0" in error_msg
        # Check HOW (common values as guidance)
        assert "Common values:" in error_msg
        assert "44100" in error_msg

    def test_negative_sampling_rate_raises_error(self) -> None:
        """Test that negative sampling rate raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_sampling_rate(-44100)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Invalid sampling_rate" in error_msg
        # Check WHY (actual vs expected)
        assert "-44100" in error_msg
        assert "Positive value > 0" in error_msg
        # Check HOW
        assert "Common values:" in error_msg

    def test_custom_param_name(self) -> None:
        """Test that custom parameter name appears in error message."""
        with pytest.raises(ValueError) as exc_info:
            validate_sampling_rate(-100, "target sampling rate")

        error_msg = str(exc_info.value)
        # Custom parameter name should be in the error
        assert "target sampling rate" in error_msg
        assert "-100" in error_msg

    def test_very_small_negative_value(self) -> None:
        """Test that very small negative values are caught."""
        with pytest.raises(ValueError) as exc_info:
            validate_sampling_rate(-0.001)

        error_msg = str(exc_info.value)
        assert "Invalid sampling_rate" in error_msg
        assert "Positive value > 0" in error_msg


class TestCalculateRms:
    """Test suite for calculate_rms function — Pillar 4: theoretical value verification."""

    def test_rms_zeros_returns_zero(self) -> None:
        wave = np.zeros(10, dtype=float)
        result = calculate_rms(wave)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)  # Exact zero input yields exact zero

    def test_rms_positive_values_matches_formula(self) -> None:
        wave = np.array([3, 4], dtype=float)
        expected = np.sqrt((9 + 16) / 2)  # RMS = sqrt((3^2 + 4^2) / 2) = sqrt(12.5)
        result = calculate_rms(wave)
        np.testing.assert_allclose(result, expected)  # Exact match: simple integer arithmetic

    def test_rms_negative_values_sign_independent(self) -> None:
        wave = np.array([-3, -4], dtype=float)
        expected = np.sqrt((9 + 16) / 2)  # RMS is sign-independent: same as [3, 4]
        result = calculate_rms(wave)
        np.testing.assert_allclose(result, expected)  # Exact match: simple integer arithmetic

    def test_rms_single_value_equals_absolute(self) -> None:
        wave = np.array([5], dtype=float)
        expected = 5.0  # RMS of a single value equals its absolute value
        result = calculate_rms(wave)
        np.testing.assert_allclose(result, expected)  # Exact match: single-element trivial case

    def test_rms_full_period_sine_matches_analytical(self) -> None:
        """RMS of a full-period sine wave is 1/sqrt(2) analytically."""
        n_samples = 1000
        t = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        wave = np.sin(t)
        expected = 1.0 / np.sqrt(2)  # Analytical RMS of sin(t) over full period
        result = calculate_rms(wave)
        np.testing.assert_allclose(result, expected, rtol=1e-3)  # Discrete approximation tolerance


class TestCalculateDesiredNoiseRms:
    """Test suite for calculate_desired_noise_rms — Pillar 4: theoretical value verification."""

    def test_noise_rms_20db_snr_returns_tenth(self) -> None:
        clean_rms = np.array(1.0)
        snr = 20.0
        expected = 0.1  # noise_rms = clean_rms / 10^(snr/20) = 1.0 / 10^1 = 0.1
        result = calculate_desired_noise_rms(clean_rms, snr)
        np.testing.assert_allclose(result, expected)  # Analytical: exact power-of-10 division

    def test_noise_rms_zero_snr_equals_clean(self) -> None:
        clean_rms = np.array(0.5)
        snr = 0.0
        expected = 0.5  # At 0 dB SNR, noise_rms = clean_rms / 10^0 = clean_rms
        result = calculate_desired_noise_rms(clean_rms, snr)
        np.testing.assert_allclose(result, expected)  # Analytical: division by unity

    def test_noise_rms_negative_snr_amplifies(self) -> None:
        clean_rms = np.array(1.0)
        snr = -20.0
        expected = 10.0  # noise_rms = 1.0 / 10^(-1) = 10.0 (noise louder than signal)
        result = calculate_desired_noise_rms(clean_rms, snr)
        np.testing.assert_allclose(result, expected)  # Analytical: exact power-of-10 multiplication

    def test_noise_rms_fractional_snr_matches_formula(self) -> None:
        clean_rms = np.array(2.0)
        snr = 10.0
        expected = 2.0 / np.sqrt(10)  # noise_rms = 2.0 / 10^0.5 = 2.0 / sqrt(10)
        result = calculate_desired_noise_rms(clean_rms, snr)
        np.testing.assert_allclose(result, expected)  # Analytical: irrational but deterministic


class TestLevelTrigger:
    """Test suite for level_trigger — detects upward threshold crossings."""

    def test_trigger_basic_upward_crossings(self) -> None:
        data = np.array([0.0, 0.2, 0.6, 0.4, 0.7, 0.3, 0.9, 0.1])
        threshold = 0.5
        # sign(data - 0.5) transitions from -1 to +1 at indices 1, 3, 5
        expected = [1, 3, 5]
        result = level_trigger(data, threshold)
        assert result == expected

    def test_trigger_with_offset_shifts_indices(self) -> None:
        data = np.array([0.0, 0.2, 0.6, 0.4, 0.7, 0.3, 0.9, 0.1])
        threshold = 0.5
        offset = 10
        expected = [11, 13, 15]  # Each trigger index shifted by offset
        result = level_trigger(data, threshold, offset=offset)
        assert result == expected

    def test_trigger_with_hold_suppresses_close_events(self) -> None:
        data = np.array([0.0, 0.2, 0.6, 0.4, 0.7, 0.3, 0.9, 0.1])
        threshold = 0.5
        hold = 2  # Minimum samples between triggers
        # Raw triggers: [1, 3, 5]; hold=2 suppresses index 3 (1+2 >= 3)
        expected = [1, 5]
        result = level_trigger(data, threshold, hold=hold)
        assert result == expected

    def test_trigger_no_crossing_returns_empty(self) -> None:
        data = np.array([0.0, 0.1, 0.2, 0.3])
        threshold = 1.0  # All values below threshold
        result = level_trigger(data, threshold)
        assert len(result) == 0


class TestCutSig:
    """Test suite for cut_sig — segment extraction with windowing."""

    def test_cut_basic_valid_and_invalid_points(self) -> None:
        data = np.arange(20, dtype=float)
        cut_len = 5
        taper_rate = 0  # Rectangular window (all ones)
        point_list = [-3, 0, 10, 15, 18]  # -3 and 18 invalid (out of bounds)
        window = tukey(cut_len, taper_rate)
        expected = np.array([data[p : p + cut_len] * window for p in [0, 10, 15]])

        result = cut_sig(data, point_list, cut_len, taper_rate, dc_cut=False)
        np.testing.assert_allclose(result, expected)  # Wrapper equivalence: same window applied

    def test_cut_dc_removal_subtracts_segment_mean(self) -> None:
        data = np.arange(20, dtype=float) + 10.0  # DC offset of 10
        cut_len = 4
        taper_rate = 0
        point_list = [2, 8, 14]
        window = tukey(cut_len, taper_rate)
        expected = np.array([(data[p : p + cut_len] - data[p : p + cut_len].mean()) * window for p in point_list])

        result = cut_sig(data, point_list, cut_len, taper_rate, dc_cut=True)
        np.testing.assert_allclose(result, expected)  # Wrapper equivalence: mean removal + windowing

    def test_cut_nonzero_taper_applies_tukey_window(self) -> None:
        data = np.linspace(0, 1, 30)
        cut_len = 6
        taper_rate = 0.5  # Tukey window with 50% taper
        point_list = [0, 12, 24]  # 24+6=30, all valid
        window = tukey(cut_len, taper_rate)
        expected = np.array([data[p : p + cut_len] * window for p in point_list])

        result = cut_sig(data, point_list, cut_len, taper_rate, dc_cut=False)
        np.testing.assert_allclose(result, expected)  # Wrapper equivalence: Tukey window

    def test_cut_invalid_points_dropped_silently(self) -> None:
        data = np.arange(10, dtype=float)
        cut_len = 5
        taper_rate = 0
        point_list = [0, 6, -2]  # 6+5=11>10 and -2 invalid; only 0 valid
        window = tukey(cut_len, taper_rate)
        expected = np.array([data[0:5] * window])

        result = cut_sig(data, point_list, cut_len, taper_rate, dc_cut=False)
        np.testing.assert_allclose(result, expected)


class TestAmplitudeToDb:
    """Test suite for amplitude_to_db — Pillar 4: wrapper equivalence with librosa."""

    def test_amplitude_to_db_matches_librosa(self) -> None:
        amp = np.array([1.0, 0.5, 0.1], dtype=float)
        ref = 1.0
        result = amplitude_to_db(amp, ref)
        expected = librosa.amplitude_to_db(np.abs(amp), ref=ref, amin=1e-15, top_db=None)
        np.testing.assert_allclose(result, expected)  # Wrapper equivalence: same librosa call
