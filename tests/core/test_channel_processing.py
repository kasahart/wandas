from typing import Any
from unittest.mock import patch

import librosa
import numpy as np
import pytest

from wandas.core import channel_processing, util
from wandas.core.channel import Channel
from wandas.core.channel_processing import compute_rms_trend
from wandas.utils.types import NDArrayReal


def test_apply_add_basic_functionality() -> None:
    """Test that apply_add correctly combines two channels with the specified SNR."""
    # Create test channels with known values
    sampling_rate = 1000
    data1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    ch1 = Channel(data=data1, sampling_rate=sampling_rate, label="signal")
    ch2 = Channel(data=data2, sampling_rate=sampling_rate, label="noise")

    # Test with SNR = 10 dB
    snr = 10.0

    # Calculate expected values manually
    clean_rms = util.calculate_rms(data1)
    noise_rms = util.calculate_rms(data2)
    desired_noise_rms = util.calculate_desired_noise_rms(clean_rms, snr)
    gain = desired_noise_rms / noise_rms
    expected_data = data1 + data2 * gain

    # Apply the function
    result = channel_processing.apply_add(ch1, ch2, snr)

    # Check the result
    assert isinstance(result, Channel)
    assert result.sampling_rate == sampling_rate
    np.testing.assert_array_almost_equal(result.data, expected_data)


def test_apply_add_sampling_rate_mismatch() -> None:
    """Test that apply_add raises ValueError when sampling rates don't match."""
    # Create channels with different sampling rates
    data = np.array([1.0, 2.0, 3.0])
    ch1 = Channel(data=data, sampling_rate=1000, label="ch1")
    ch2 = Channel(data=data, sampling_rate=2000, label="ch2")

    with pytest.raises(
        ValueError, match="Sampling rates of the two channels are different."
    ):
        channel_processing.apply_add(ch1, ch2, 10.0)


def test_apply_add_shape_mismatch() -> None:
    """Test that apply_add raises ValueError when data shapes don't match."""
    # Create channels with different data shapes
    sampling_rate = 1000
    ch1 = Channel(
        data=np.array([1.0, 2.0, 3.0]), sampling_rate=sampling_rate, label="ch1"
    )
    ch2 = Channel(
        data=np.array([1.0, 2.0, 3.0, 4.0]), sampling_rate=sampling_rate, label="ch2"
    )

    with pytest.raises(
        ValueError, match="Data shapes of the two channels are different."
    ):
        channel_processing.apply_add(ch1, ch2, 10.0)


def test_apply_add_with_mocks() -> None:
    """
    Test apply_add using mocked utility functions for precise control.

    This test mocks the utility functions
    `calculate_rms` and `calculate_desired_noise_rms`
    to control their return values and verify that
    `apply_add` correctly uses these values
    to compute the expected result.
    """
    # Create test channels
    sampling_rate = 1000
    data1 = np.array([1.0, 2.0, 3.0])
    data2 = np.array([0.1, 0.2, 0.3])

    ch1 = Channel(data=data1, sampling_rate=sampling_rate, label="signal")
    ch2 = Channel(data=data2, sampling_rate=sampling_rate, label="noise")

    # Mock values
    mock_clean_rms = 3.0
    mock_noise_rms = 0.5
    mock_desired_noise_rms = 0.3
    expected_gain = mock_desired_noise_rms / mock_noise_rms

    # Patch the utility functions
    with (
        patch("wandas.core.util.calculate_rms") as mock_calc_rms,
        patch("wandas.core.util.calculate_desired_noise_rms") as mock_calc_desired_rms,
    ):

        def mock_rms_side_effect(data: NDArrayReal) -> float:
            if np.array_equal(data, data1):
                return mock_clean_rms
            return mock_noise_rms

        mock_calc_rms.side_effect = mock_rms_side_effect
        mock_calc_desired_rms.return_value = mock_desired_noise_rms

        # Call the function
        result = channel_processing.apply_add(ch1, ch2, 10.0)

        # Verify mocks were called correctly
        assert mock_calc_rms.call_count == 2
        mock_calc_desired_rms.assert_called_once_with(mock_clean_rms, 10.0)

        # Check the result
        expected_data = data1 + data2 * expected_gain
        np.testing.assert_array_almost_equal(result.data, expected_data)


def test_apply_add_various_snr_values() -> None:
    """Test apply_add with a range of SNR values."""
    # Create test channels
    sampling_rate = 1000
    data1 = np.ones(10)  # Signal: all ones
    data2 = np.ones(10) * 0.1  # Noise: all 0.1

    ch1 = Channel(data=data1, sampling_rate=sampling_rate, label="signal")
    ch2 = Channel(data=data2, sampling_rate=sampling_rate, label="noise")

    # Test with different SNR values
    snr_values = [-20.0, -10.0, 0.0, 10.0, 20.0]

    for snr in snr_values:
        # Calculate expected result
        clean_rms = util.calculate_rms(data1)
        noise_rms = util.calculate_rms(data2)
        desired_noise_rms = util.calculate_desired_noise_rms(clean_rms, snr)
        gain = desired_noise_rms / noise_rms
        expected_data = data1 + data2 * gain

        # Call the function
        result = channel_processing.apply_add(ch1, ch2, snr)

        # Verify result
        np.testing.assert_array_almost_equal(result.data, expected_data)


def test_apply_add_zero_signal_or_noise() -> None:
    """Test apply_add with zero signal or noise."""
    sampling_rate = 1000

    # Test with zero signal
    ch_zero = Channel(data=np.zeros(5), sampling_rate=sampling_rate, label="zero")
    ch_nonzero = Channel(data=np.ones(5), sampling_rate=sampling_rate, label="nonzero")

    # Zero signal with noise should just return scaled noise
    result = channel_processing.apply_add(ch_zero, ch_nonzero, 0.0)  # SNR 0 dB
    result_rms = util.calculate_rms(result.data)
    expected_rms = util.calculate_rms(ch_zero.data)  # Should be 0

    assert np.isclose(result_rms, expected_rms)

    # Test with zero noise - this might cause division by zero, so handle accordingly
    with pytest.raises(ValueError, match="RMS of the noise channel is zero."):
        result = channel_processing.apply_add(ch_nonzero, ch_zero, 0.0)


def test_apply_add_preserves_channel_properties() -> None:
    """Test that apply_add preserves important channel properties."""
    # Create test channels with metadata
    sampling_rate = 1000
    data1 = np.random.random(10)
    data2 = np.random.random(10)

    ch1 = Channel(data=data1, sampling_rate=sampling_rate, label="signal")
    ch2 = Channel(data=data2, sampling_rate=sampling_rate, label="noise")

    # Call the function
    result = channel_processing.apply_add(ch1, ch2, 10.0)

    # Verify important properties are preserved
    assert result.sampling_rate == sampling_rate
    # Label would typically come from ch1, but the current implementation
    # doesn't specify this - we'd need to check the actual behavior


def test_apply_hpss_percussive_basic_functionality() -> None:
    """Test that apply_hpss_percussive correctly extracts percussive components."""
    # Create a test channel with a simple waveform
    sampling_rate = 22050
    # Create a signal with both harmonic and percussive elements
    t = np.linspace(0, 1, sampling_rate)
    # Harmonic part (sine wave)
    harmonic = np.sin(2 * np.pi * 440 * t)
    # Percussive part (short impulses)
    percussive = np.zeros_like(t)
    percussive[::1000] = 1.0  # Impulses every 1000 samples
    # Combined signal
    data = harmonic + percussive

    ch = Channel(data=data, sampling_rate=sampling_rate, label="test")

    # Apply the function with default parameters
    result = channel_processing.apply_hpss_percussive(ch)

    # Check the result
    assert isinstance(result, dict)
    assert "data" in result
    assert isinstance(result["data"], np.ndarray)
    assert result["data"].shape == data.shape
    # Percussive component should be different from the original
    excepted_data = librosa.effects.percussive(data)
    assert np.array_equal(result["data"], excepted_data)


def test_apply_hpss_percussive_with_parameters() -> None:
    """Test apply_hpss_percussive with different parameter values."""
    sampling_rate = 22050
    t = np.linspace(0, 1, sampling_rate)
    data = np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 880 * t)

    ch = Channel(data=data, sampling_rate=sampling_rate, label="test")

    # Test with different margin values
    margin_values = [1.0, 2.0, 3.0]

    previous_result = None
    for margin in margin_values:
        result = channel_processing.apply_hpss_percussive(ch, margin=margin)

        # Basic checks
        assert isinstance(result, dict)
        assert "data" in result
        assert isinstance(result["data"], np.ndarray)
        assert result["data"].shape == data.shape

        # Different margin values should produce different results
        if previous_result is not None:
            # Check that results are different (at least slightly)

            assert not np.allclose(result["data"], previous_result["data"])

        previous_result = result


@patch("librosa.effects.percussive")
def test_apply_hpss_percussive_with_mocks(mock_percussive: Any) -> None:
    """Test apply_hpss_percussive using mocked librosa function for precise control."""
    # Create test data
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sampling_rate = 1000
    ch = Channel(data=data, sampling_rate=sampling_rate, label="test")

    # Set up the mock to return a known array
    mock_return = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    mock_percussive.return_value = mock_return

    # Call the function with some parameters
    kwargs = {"margin": 2.0, "kernel_size": 31}
    result = channel_processing.apply_hpss_percussive(ch, **kwargs)

    # Verify that librosa.effects.percussive was called with correct arguments
    assert mock_percussive.call_count == 1

    # Check that the result contains our mocked data
    assert "data" in result
    np.testing.assert_array_equal(result["data"], mock_return)


def test_apply_hpss_percussive_empty_data() -> None:
    """Test apply_hpss_percussive with empty data."""
    # Create a channel with empty data
    data = np.array([])
    sampling_rate = 1000
    ch = Channel(data=data, sampling_rate=sampling_rate, label="empty")

    # Apply the function
    result = channel_processing.apply_hpss_percussive(ch)

    # Check the result
    assert isinstance(result, dict)
    assert "data" in result
    assert isinstance(result["data"], np.ndarray)
    assert result["data"].size == 0


def test_apply_hpss_percussive_return_type() -> None:
    """Test that apply_hpss_percussive returns the correct type."""
    # Create a test channel
    data = np.random.random(100)
    sampling_rate = 1000
    ch = Channel(data=data, sampling_rate=sampling_rate, label="test")

    # Apply the function
    result = channel_processing.apply_hpss_percussive(ch)

    # Check the return type
    assert isinstance(result, dict)
    assert "data" in result
    assert isinstance(result["data"], np.ndarray)
    assert result["data"].dtype == np.float64 or result["data"].dtype == np.float32


def test_apply_hpss_harmonic_basic_functionality() -> None:
    """Test that apply_hpss_harmonic correctly extracts harmonic components."""
    # Create a test channel with a simple waveform
    sampling_rate = 22050
    # Create a signal with both harmonic and percussive elements
    t = np.linspace(0, 1, sampling_rate)
    # Harmonic part (sine wave)
    harmonic = np.sin(2 * np.pi * 440 * t)
    # Percussive part (short impulses)
    percussive = np.zeros_like(t)
    percussive[::1000] = 1.0  # Impulses every 1000 samples
    # Combined signal
    data = harmonic + percussive

    ch = Channel(data=data, sampling_rate=sampling_rate, label="test")

    # Apply the function with default parameters
    result = channel_processing.apply_hpss_harmonic(ch)

    # Check the result
    assert isinstance(result, dict)
    assert "data" in result
    assert isinstance(result["data"], np.ndarray)
    assert result["data"].shape == data.shape
    # Harmonic component should be different from the original
    excepted_data = librosa.effects.harmonic(data)
    assert np.array_equal(result["data"], excepted_data)


def test_apply_hpss_harmonic_with_parameters() -> None:
    """Test apply_hpss_harmonic with different parameter values."""
    sampling_rate = 22050
    t = np.linspace(0, 1, sampling_rate)
    data = np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 880 * t)

    ch = Channel(data=data, sampling_rate=sampling_rate, label="test")

    # Test with different margin values
    margin_values = [1.0, 2.0, 3.0]

    previous_result = None
    for margin in margin_values:
        result = channel_processing.apply_hpss_harmonic(ch, margin=margin)

        # Basic checks
        assert isinstance(result, dict)
        assert "data" in result
        assert isinstance(result["data"], np.ndarray)
        assert result["data"].shape == data.shape

        # Different margin values should produce different results
        if previous_result is not None:
            # Check that results are different (at least slightly)
            assert not np.allclose(result["data"], previous_result["data"])

        previous_result = result


@patch("librosa.effects.harmonic")
def test_apply_hpss_harmonic_with_mocks(mock_harmonic: Any) -> None:
    """Test apply_hpss_harmonic using mocked librosa function for precise control."""
    # Create test data
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sampling_rate = 1000
    ch = Channel(data=data, sampling_rate=sampling_rate, label="test")

    # Set up the mock to return a known array
    mock_return = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    mock_harmonic.return_value = mock_return

    # Call the function with some parameters
    kwargs = {"margin": 2.0, "kernel_size": 31}
    result = channel_processing.apply_hpss_harmonic(ch, **kwargs)

    # Verify that librosa.effects.harmonic was called with correct arguments
    assert mock_harmonic.call_count == 1

    # Check that the result contains our mocked data
    assert "data" in result
    np.testing.assert_array_equal(result["data"], mock_return)


def test_apply_hpss_harmonic_empty_data() -> None:
    """Test apply_hpss_harmonic with empty data."""
    # Create a channel with empty data
    data = np.array([])
    sampling_rate = 1000
    ch = Channel(data=data, sampling_rate=sampling_rate, label="empty")

    # Apply the function
    result = channel_processing.apply_hpss_harmonic(ch)

    # Check the result
    assert isinstance(result, dict)
    assert "data" in result
    assert isinstance(result["data"], np.ndarray)
    assert result["data"].size == 0


def test_apply_hpss_harmonic_return_type() -> None:
    """Test that apply_hpss_harmonic returns the correct type."""
    # Create a test channel
    data = np.random.random(100)
    sampling_rate = 1000
    ch = Channel(data=data, sampling_rate=sampling_rate, label="test")

    # Apply the function
    result = channel_processing.apply_hpss_harmonic(ch)

    # Check the return type
    assert isinstance(result, dict)
    assert "data" in result
    assert isinstance(result["data"], np.ndarray)
    assert result["data"].dtype == np.float64 or result["data"].dtype == np.float32


def test_compute_rms_trend_basic_usage() -> None:
    data = np.ones(2048, dtype=np.float32)
    ch = Channel(data=data, sampling_rate=2048)
    result = compute_rms_trend(ch)
    assert isinstance(result, dict)
    assert "data" in result
    assert "sampling_rate" in result
    # Check shape and sampling rate
    assert result["data"].ndim == 1
    assert result["sampling_rate"] == (ch.sampling_rate // 512)


def test_compute_rms_trend_custom_parameters() -> None:
    data = np.random.random(4096).astype(np.float32)
    ch = Channel(data=data, sampling_rate=4096)
    frame_length = 1024
    hop_length = 256
    result = compute_rms_trend(ch, frame_length=frame_length, hop_length=hop_length)
    assert result["sampling_rate"] == (ch.sampling_rate // hop_length)
    assert result["data"].shape[0] > 0


def test_compute_rms_trend_aw() -> None:
    data = np.random.random(4096).astype(np.float32)
    ch = Channel(data=data, sampling_rate=10000)
    result_aw = compute_rms_trend(ch, Aw=True)
    result_no_aw = compute_rms_trend(ch, Aw=False)
    # Ensure that A-weighted results differ from non A-weighted
    assert not np.allclose(result_aw["data"], result_no_aw["data"])
