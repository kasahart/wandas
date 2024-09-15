# tests/core/test_signal.py

import pytest
import numpy as np
from wandas.core.channel import Channel
from wandas.core.signal import Signal


def test_signal_initialization():
    data1 = np.array([0, 1, 2, 3, 4])
    data2 = np.array([5, 6, 7, 8, 9])
    sampling_rate = 1000
    channel1 = Channel(data=data1, sampling_rate=sampling_rate, label="Channel 1")
    channel2 = Channel(data=data2, sampling_rate=sampling_rate, label="Channel 2")

    signal = Signal(channels=[channel1, channel2], label="Test Signal")

    assert signal.label == "Test Signal"
    assert len(signal.channels) == 2
    assert signal.channels[0] == channel1
    assert signal.channels[1] == channel2
    assert signal.sampling_rate == sampling_rate


def test_signal_sampling_rate_mismatch():
    data1 = np.array([0, 1, 2, 3, 4])
    data2 = np.array([5, 6, 7, 8, 9])
    channel1 = Channel(data=data1, sampling_rate=1000)
    channel2 = Channel(data=data2, sampling_rate=2000)

    with pytest.raises(ValueError):
        Signal(channels=[channel1, channel2])


def test_signal_low_pass_filter():
    t = np.linspace(0, 1, 1000)
    data1 = np.sin(2 * np.pi * 50 * t)
    data2 = np.sin(2 * np.pi * 100 * t)
    sampling_rate = 1000
    channel1 = Channel(data=data1, sampling_rate=sampling_rate)
    channel2 = Channel(data=data2, sampling_rate=sampling_rate)
    signal = Signal(channels=[channel1, channel2])

    filtered_signal = signal.low_pass_filter(cutoff=30)

    # 各チャンネルがフィルタリングされていることを確認
    for original_ch, filtered_ch in zip(signal.channels, filtered_signal.channels):
        assert not np.array_equal(original_ch.data, filtered_ch.data)


def test_signal_fft():
    t = np.linspace(0, 1, 1000)
    data1 = np.sin(2 * np.pi * 50 * t)
    data2 = np.sin(2 * np.pi * 100 * t)
    sampling_rate = 1000
    channel1 = Channel(data=data1, sampling_rate=sampling_rate, label="Channel 1")
    channel2 = Channel(data=data2, sampling_rate=sampling_rate, label="Channel 2")
    signal = Signal(channels=[channel1, channel2])

    spectrum = signal.fft(n_fft=1024, window="hann")

    assert len(spectrum.channels) == 2
    for freq_ch, label, expected_freq in zip(
        spectrum.channels, ["Channel 1", "Channel 2"], [50, 100]
    ):
        assert freq_ch.label == label
        assert freq_ch.fft_params["n_fft"] == 1024
        assert freq_ch.fft_params["window"] == "hann"

        # Find the frequency bin with the maximum amplitude
        freqs = np.fft.fftfreq(1024, 1 / sampling_rate)
        fft_data = np.abs(freq_ch.data)
        peak_freq = freqs[np.argmax(fft_data)]

        # Check if the peak frequency matches the expected frequency
        assert np.isclose(peak_freq, expected_freq, atol=1)
