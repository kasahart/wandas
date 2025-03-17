# tests/core/test_util.py
import numpy as np
from scipy.signal.windows import tukey

from wandas.core.util import (
    calculate_desired_noise_rms,
    calculate_rms,
    cut_sig,
    level_trigger,
)


def test_calculate_rms_zeros() -> None:
    wave = np.zeros(10, dtype=float)
    expected = 0.0
    result = calculate_rms(wave)
    np.testing.assert_almost_equal(result, expected)


def test_calculate_rms_positive() -> None:
    wave = np.array([3, 4], dtype=float)
    # RMS = sqrt((9 + 16) / 2) = sqrt(25/2)
    expected = np.sqrt((9 + 16) / 2)
    result = calculate_rms(wave)
    np.testing.assert_almost_equal(result, expected)


def test_calculate_rms_negative() -> None:
    wave = np.array([-3, -4], dtype=float)
    # RMS should be the same as for positive values
    expected = np.sqrt((9 + 16) / 2)
    result = calculate_rms(wave)
    np.testing.assert_almost_equal(result, expected)


def test_calculate_rms_single_value() -> None:
    wave = np.array([5], dtype=float)
    expected = 5.0
    result = calculate_rms(wave)
    np.testing.assert_almost_equal(result, expected)


def test_calculate_rms_random() -> None:
    np.random.seed(0)
    wave = np.random.rand(100).astype(float)
    expected = np.sqrt(np.mean(np.square(wave)))
    result = calculate_rms(wave)
    np.testing.assert_almost_equal(result, expected)


def test_calculate_desired_noise_rms_basic() -> None:
    # For a clean_rms of 1.0 and snr of 20 dB:
    # a = 20/20 = 1, so noise_rms = 1.0 / 10**1 = 0.1
    clean_rms = 1.0
    snr = 20.0
    expected = 0.1
    result = calculate_desired_noise_rms(clean_rms, snr)
    np.testing.assert_almost_equal(result, expected)


def test_calculate_desired_noise_rms_snr_zero() -> None:
    # For snr = 0 dB, a = 0 and noise_rms should equal clean_rms.
    clean_rms = 0.5
    snr = 0.0
    expected = 0.5
    result = calculate_desired_noise_rms(clean_rms, snr)
    np.testing.assert_almost_equal(result, expected)


def test_calculate_desired_noise_rms_negative_snr() -> None:
    # For a negative snr, e.g., snr = -20 dB:
    # a = -20/20 = -1, so noise_rms = clean_rms / 10**(-1) = clean_rms * 10.
    clean_rms = 1.0
    snr = -20.0
    expected = 10.0
    result = calculate_desired_noise_rms(clean_rms, snr)
    np.testing.assert_almost_equal(result, expected)


def test_calculate_desired_noise_rms_fractional() -> None:
    # For a fractional snr, e.g., snr = 10 dB:
    # a = 10/20 = 0.5, so noise_rms = clean_rms / 10**0.5 = clean_rms / sqrt(10).
    clean_rms = 2.0
    snr = 10.0
    expected = 2.0 / np.sqrt(10)
    result = calculate_desired_noise_rms(clean_rms, snr)
    np.testing.assert_almost_equal(result, expected)


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
