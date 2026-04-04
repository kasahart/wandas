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


def test_level_trigger_basic() -> None:
    # Data with upward crossings
    data = np.array([0.0, 0.2, 0.6, 0.4, 0.7, 0.3, 0.9, 0.1])
    threshold = 0.5
    # np.sign(data - threshold) -> [-1, -1, 1, -1, 1, -1, 1, -1]
    # diff -> [0, 2, -2, 2, -2, 2, -2] -> indices with diff > 0: [1, 3, 5]
    # For hold=1: expected triggers = [1, 3, 5]
    expected = [1, 3, 5]
    result = level_trigger(data, threshold)
    assert result == expected, f"Expected {expected} but got {result}"


def test_level_trigger_with_offset() -> None:
    data = np.array([0.0, 0.2, 0.6, 0.4, 0.7, 0.3, 0.9, 0.1])
    threshold = 0.5
    offset = 10
    # Expected triggers with offset: [1+10, 3+10, 5+10] = [11, 13, 15]
    expected = [11, 13, 15]
    result = level_trigger(data, threshold, offset=offset)
    assert result == expected, f"Expected {expected} but got {result}"


def test_level_trigger_with_hold() -> None:
    data = np.array([0.0, 0.2, 0.6, 0.4, 0.7, 0.3, 0.9, 0.1])
    threshold = 0.5
    hold = 2
    # With hold=2:
    # level_point initially: [1, 3, 5]
    # last_point starts as 1, then only 5 qualifies since (1+2)<5.
    # Expected triggers = [1, 5]
    expected = [1, 5]
    result = level_trigger(data, threshold, hold=hold)
    assert result == expected, f"Expected {expected} but got {result}"


def test_level_trigger_no_crossing() -> None:
    # Data with no upward crossing above the threshold.
    data = np.array([0.0, 0.1, 0.2, 0.3])
    threshold = 1.0
    result = level_trigger(data, threshold)

    assert len(result) == 0, "Expected no triggers"


def test_cut_sig_basic() -> None:
    # Create data array and define parameters.
    data = np.arange(20, dtype=float)
    cut_len = 5
    taper_rate = 0  # rectangular window (ones)
    dc_cut = False
    # Define point_list with valid and invalid indices.
    point_list = [-3, 0, 10, 15, 18]  # -3 and 18 are invalid: 18+5 > 20
    # Expected valid indices: 0, 10, 15.
    expected = []
    window = tukey(cut_len, taper_rate)  # should be ones when taper_rate is 0
    for p in [0, 10, 15]:
        segment = data[p : p + cut_len] * window
        expected.append(segment)
    expected_array = np.array(expected)

    result = cut_sig(data, point_list, cut_len, taper_rate, dc_cut)
    np.testing.assert_allclose(result, expected_array)


def test_cut_sig_dc_cut() -> None:
    # Create data array with a DC offset.
    data = np.arange(20, dtype=float) + 10.0
    cut_len = 4
    taper_rate = 0  # window is ones when taper_rate is 0
    dc_cut = True
    point_list = [2, 8, 14]  # all valid; 14+4=18 <=20
    expected = []
    window = tukey(cut_len, taper_rate)
    for p in point_list:
        segment = data[p : p + cut_len]
        # subtract mean from the segment
        segment_dc = segment - segment.mean()
        expected.append(segment_dc * window)
    expected_array = np.array(expected)

    result = cut_sig(data, point_list, cut_len, taper_rate, dc_cut)
    np.testing.assert_allclose(result, expected_array)


def test_cut_sig_taper_rate() -> None:
    # Test with a nonzero taper_rate.
    data = np.linspace(0, 1, 30)
    cut_len = 6
    taper_rate = 0.5  # non-rectangular window
    dc_cut = False
    point_list = [0, 12, 24]  # 24+6=30, valid indices
    expected = []
    window = tukey(cut_len, taper_rate)
    for p in point_list:
        segment = data[p : p + cut_len] * window
        expected.append(segment)
    expected_array = np.array(expected)

    result = cut_sig(data, point_list, cut_len, taper_rate, dc_cut)
    np.testing.assert_allclose(result, expected_array)


def test_cut_sig_invalid_points() -> None:
    # Points that do not yield a complete segment should be dropped.
    data = np.arange(10, dtype=float)
    cut_len = 5
    taper_rate = 0
    dc_cut = False
    # Valid point: only 0, since 6 is invalid (6+5=11>10)
    point_list = [0, 6, -2]
    window = tukey(cut_len, taper_rate)
    expected = np.array([data[0:5] * window])

    result = cut_sig(data, point_list, cut_len, taper_rate, dc_cut)
    np.testing.assert_allclose(result, expected)


def test_amplitude_to_db_basic() -> None:
    # Basic check that amplitude_to_db forwards to librosa with correct params
    amp = np.array([1.0, 0.5, 0.1], dtype=float)
    ref = 1.0
    result = amplitude_to_db(amp, ref)
    expected = librosa.amplitude_to_db(np.abs(amp), ref=ref, amin=1e-15, top_db=None)
    np.testing.assert_allclose(result, expected)
