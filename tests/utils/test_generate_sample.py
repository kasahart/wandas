# tests/utils/test_generate_sample.py

import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame
from wandas.utils.generate_sample import generate_sin, generate_sin_lazy


def test_generate_sin_single_frequency() -> None:
    freq = 1000.0
    sampling_rate = 16000
    duration = 1.0
    signal = generate_sin(
        freqs=freq, sampling_rate=sampling_rate, duration=duration, label="Test Signal"
    )

    assert isinstance(signal, ChannelFrame)
    assert signal.label == "Test Signal"
    assert len(signal) == 1

    # チャンネルのラベルを確認
    assert signal.channels[0].label == "Channel 1"

    # データ長を確認
    computed_data = signal.compute()
    assert computed_data.shape[1] == int(sampling_rate * duration)


def test_generate_sin_multiple_frequencies() -> None:
    freqs = [500.0, 800.0, 1000.0]
    sampling_rate = 16000
    duration = 1.0
    signal = generate_sin(
        freqs=freqs, sampling_rate=sampling_rate, duration=duration, label="Test Signal"
    )

    assert isinstance(signal, ChannelFrame)
    assert signal.label == "Test Signal"
    assert len(signal) == len(freqs)

    # 各チャンネルの確認
    for idx, channel in enumerate(signal.channels):
        assert channel.label == f"Channel {idx + 1}"

    # データ長を確認
    computed_data = signal.compute()
    assert computed_data.shape[1] == int(sampling_rate * duration)


def test_generate_sin_default_parameters() -> None:
    """Test generate_sin with default parameters."""
    signal = generate_sin()

    assert isinstance(signal, ChannelFrame)
    assert signal.label == "Generated Sin"
    assert len(signal) == 1
    assert signal.sampling_rate == 16000

    # Should generate 1 second of data at 16kHz = 16000 samples
    computed_data = signal.compute()
    assert computed_data.shape[1] == 16000


def test_generate_sin_custom_label() -> None:
    """Test generate_sin with custom label."""
    signal = generate_sin(freqs=440.0, label="Custom Label")

    assert signal.label == "Custom Label"


def test_generate_sin_no_label() -> None:
    """Test generate_sin without label uses default."""
    signal = generate_sin(freqs=440.0)

    assert signal.label == "Generated Sin"


def test_generate_sin_amplitude_range() -> None:
    """Test that generated sine wave has correct amplitude range [-1, 1]."""
    signal = generate_sin(freqs=440.0, sampling_rate=8000, duration=1.0)

    data = signal.compute()
    # Sine wave should be in range [-1, 1]
    assert np.max(data) <= 1.0
    assert np.min(data) >= -1.0
    # Should actually reach close to the bounds
    assert np.max(data) > 0.99
    assert np.min(data) < -0.99


def test_generate_sin_zero_frequency() -> None:
    """Test generate_sin with zero frequency produces DC signal."""
    signal = generate_sin(freqs=0.0, sampling_rate=1000, duration=1.0)

    data = signal.compute()
    # Zero frequency should produce all zeros (sin(0) = 0)
    np.testing.assert_array_almost_equal(data, np.zeros_like(data))


def test_generate_sin_high_frequency() -> None:
    """Test generate_sin with high frequency."""
    # Test frequency below Nyquist frequency
    sampling_rate = 16000
    freq = 7000.0  # Below 8000 Hz Nyquist
    signal = generate_sin(freqs=freq, sampling_rate=sampling_rate, duration=0.1)

    assert isinstance(signal, ChannelFrame)
    data = signal.compute()
    assert data.shape[1] == int(sampling_rate * 0.1)


def test_generate_sin_short_duration() -> None:
    """Test generate_sin with very short duration."""
    signal = generate_sin(freqs=1000.0, sampling_rate=16000, duration=0.001)

    # 16000 * 0.001 = 16 samples
    data = signal.compute()
    assert data.shape[1] == 16


def test_generate_sin_long_duration() -> None:
    """Test generate_sin with long duration."""
    signal = generate_sin(freqs=1000.0, sampling_rate=8000, duration=10.0)

    # 8000 * 10 = 80000 samples
    data = signal.compute()
    assert data.shape[1] == 80000


def test_generate_sin_low_sampling_rate() -> None:
    """Test generate_sin with low sampling rate."""
    signal = generate_sin(freqs=100.0, sampling_rate=1000, duration=1.0)

    assert signal.sampling_rate == 1000
    data = signal.compute()
    assert data.shape[1] == 1000


def test_generate_sin_high_sampling_rate() -> None:
    """Test generate_sin with high sampling rate."""
    signal = generate_sin(freqs=1000.0, sampling_rate=192000, duration=0.1)

    assert signal.sampling_rate == 192000
    data = signal.compute()
    assert data.shape[1] == 19200


def test_generate_sin_empty_frequency_list() -> None:
    """Test generate_sin with empty frequency list."""
    signal = generate_sin(freqs=[], sampling_rate=8000, duration=1.0)

    # Empty list should create 0 channels
    assert len(signal) == 0


def test_generate_sin_many_frequencies() -> None:
    """Test generate_sin with many frequencies."""
    freqs = [100.0 * i for i in range(1, 11)]  # 10 frequencies
    signal = generate_sin(freqs=freqs, sampling_rate=16000, duration=0.5)

    assert len(signal) == 10
    # Each channel should have correct label
    for idx in range(10):
        assert signal.channels[idx].label == f"Channel {idx + 1}"


def test_generate_sin_frequency_accuracy() -> None:
    """Test that generated signal has correct frequency content."""
    freq = 440.0
    sampling_rate = 8000
    duration = 1.0
    signal = generate_sin(freqs=freq, sampling_rate=sampling_rate, duration=duration)

    data = signal.compute()
    # Perform FFT to verify frequency
    fft_result = np.fft.rfft(data[0])
    freqs_fft = np.fft.rfftfreq(len(data[0]), 1 / sampling_rate)

    # Find peak frequency
    peak_idx = np.argmax(np.abs(fft_result))
    peak_freq = freqs_fft[peak_idx]

    # Should be close to 440 Hz
    assert abs(peak_freq - freq) < 1.0


def test_generate_sin_lazy_equivalence() -> None:
    """Test that generate_sin and generate_sin_lazy produce same results."""
    params = {
        "freqs": [440.0, 880.0],
        "sampling_rate": 8000,
        "duration": 0.5,
        "label": "Test",
    }

    signal1 = generate_sin(**params)
    signal2 = generate_sin_lazy(**params)

    # Both should produce same data
    data1 = signal1.compute()
    data2 = signal2.compute()

    np.testing.assert_array_almost_equal(data1, data2)


def test_generate_sin_invalid_freqs_type() -> None:
    """Test generate_sin with invalid freqs type raises error."""
    with pytest.raises(ValueError, match="freqs must be a float or a list of floats"):
        generate_sin(freqs="invalid", sampling_rate=8000, duration=1.0)  # type: ignore


def test_generate_sin_channel_metadata() -> None:
    """Test that generate_sin creates proper channel metadata."""
    signal = generate_sin(freqs=[100.0, 200.0, 300.0], sampling_rate=8000)

    assert len(signal.channels) == 3
    assert signal.channels[0].label == "Channel 1"
    assert signal.channels[1].label == "Channel 2"
    assert signal.channels[2].label == "Channel 3"


def test_generate_sin_fractional_samples() -> None:
    """Test generate_sin when duration * sampling_rate is not integer."""
    # 8000 * 0.001 = 8 samples
    signal = generate_sin(freqs=440.0, sampling_rate=8000, duration=0.001)

    data = signal.compute()
    assert data.shape[1] == 8


def test_generate_sin_data_type() -> None:
    """Test that generated data has correct dtype."""
    signal = generate_sin(freqs=440.0)

    data = signal.compute()
    # Should be floating point
    assert np.issubdtype(data.dtype, np.floating)
