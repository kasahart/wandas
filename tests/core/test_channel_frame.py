# tests/core/channel_frame.py

import numpy as np
import pytest

from wandas.core.channel import Channel
from wandas.core.channel_frame import ChannelFrame


@pytest.fixture  # type: ignore [misc]
def generate_signals() -> tuple[ChannelFrame, ChannelFrame]:
    # サンプルの直流データを生成
    sampling_rate = 1000
    t = np.linspace(0, 1, sampling_rate, endpoint=False)
    data1_signal1 = np.full_like(t, 2)  # Signal 1の振幅2の直流信号
    data2_signal1 = np.full_like(t, 3)  # Signal 1の振幅3の直流信号
    data1_signal2 = np.full_like(t, 4)  # Signal 2の振幅4の直流信号
    data2_signal2 = np.full_like(t, 5)  # Signal 2の振幅5の直流信号

    ch1_signal1 = Channel(
        data=data1_signal1, sampling_rate=sampling_rate, label="Channel 1"
    )
    ch2_signal1 = Channel(
        data=data2_signal1, sampling_rate=sampling_rate, label="Channel 2"
    )
    ch1_signal2 = Channel(
        data=data1_signal2, sampling_rate=sampling_rate, label="Channel 1"
    )
    ch2_signal2 = Channel(
        data=data2_signal2, sampling_rate=sampling_rate, label="Channel 2"
    )

    signal1 = ChannelFrame(channels=[ch1_signal1, ch2_signal1], label="Signal 1")
    signal2 = ChannelFrame(channels=[ch1_signal2, ch2_signal2], label="Signal 2")

    return signal1, signal2


def test_signal_initialization() -> None:
    data1 = np.array([0, 1, 2, 3, 4])
    data2 = np.array([5, 6, 7, 8, 9])
    sampling_rate = 1000
    channel1 = Channel(data=data1, sampling_rate=sampling_rate, label="Channel 1")
    channel2 = Channel(data=data2, sampling_rate=sampling_rate, label="Channel 2")

    signal = ChannelFrame(channels=[channel1, channel2], label="Test Signal")

    assert signal.label == "Test Signal"
    assert len(signal.channels) == 2
    assert signal.channels[0] == channel1
    assert signal.channels[1] == channel2
    assert signal.sampling_rate == sampling_rate


def test_signal_sampling_rate_mismatch() -> None:
    data1 = np.array([0, 1, 2, 3, 4])
    data2 = np.array([5, 6, 7, 8, 9])
    channel1 = Channel(data=data1, sampling_rate=1000)
    channel2 = Channel(data=data2, sampling_rate=2000)

    with pytest.raises(ValueError):
        ChannelFrame(channels=[channel1, channel2])


def test_signal_low_pass_filter() -> None:
    t = np.linspace(0, 1, 1000)
    data1 = np.sin(2 * np.pi * 50 * t)
    data2 = np.sin(2 * np.pi * 100 * t)
    sampling_rate = 1000
    channel1 = Channel(data=data1, sampling_rate=sampling_rate, label="Channel 1")
    channel2 = Channel(data=data2, sampling_rate=sampling_rate, label="Channel 2")
    signal = ChannelFrame(channels=[channel1, channel2])

    filtered_signal = signal.low_pass_filter(cutoff=30)

    # 各チャンネルがフィルタリングされていることを確認
    for original_ch, filtered_ch in zip(signal.channels, filtered_signal.channels):
        assert not np.array_equal(original_ch.data, filtered_ch.data)


def test_signal_fft() -> None:
    signal_length = 1000
    t = np.linspace(0, 1, signal_length)
    data1 = np.sin(2 * np.pi * 50 * t)
    data2 = np.sin(2 * np.pi * 100 * t)
    sampling_rate = 1000
    channel1 = Channel(data=data1, sampling_rate=sampling_rate, label="Channel 1")
    channel2 = Channel(data=data2, sampling_rate=sampling_rate, label="Channel 2")
    signal = ChannelFrame(channels=[channel1, channel2])

    spectrum = signal.fft(n_fft=1024, window="hann")

    assert len(spectrum.channels) == 2
    for freq_ch, label, expected_freq in zip(
        spectrum.channels, ["Channel 1", "Channel 2"], [50, 100]
    ):
        assert freq_ch.label == label
        assert freq_ch.n_fft == 1024
        assert not np.array_equal(freq_ch.window, np.hanning(signal_length))

        # Find the frequency bin with the maximum amplitude
        freqs = np.fft.fftfreq(1024, 1 / sampling_rate)
        fft_data = np.abs(freq_ch.data)
        peak_freq = freqs[np.argmax(fft_data)]

        # Check if the peak frequency matches the expected frequency
        assert np.isclose(peak_freq, expected_freq, atol=1)


def test_signal_addition(generate_signals: tuple[ChannelFrame, ChannelFrame]) -> None:
    signal1, signal2 = generate_signals
    result_signal = signal1 + signal2

    # 各チャンネルの加算結果を確認
    for i in range(len(signal1.channels)):
        expected_data = signal1.channels[i].data + signal2.channels[i].data
        assert np.array_equal(result_signal.channels[i].data, expected_data), (
            f"Signal addition failed for channel {i + 1}."
        )


def test_signal_subtraction(
    generate_signals: tuple[ChannelFrame, ChannelFrame],
) -> None:
    signal1, signal2 = generate_signals
    result_signal = signal1 - signal2

    # 各チャンネルの減算結果を確認
    for i in range(len(signal1.channels)):
        expected_data = signal1.channels[i].data - signal2.channels[i].data
        assert np.array_equal(result_signal.channels[i].data, expected_data), (
            f"Signal subtraction failed for channel {i + 1}."
        )


def test_signal_multiplication(
    generate_signals: tuple[ChannelFrame, ChannelFrame],
) -> None:
    signal1, signal2 = generate_signals
    result_signal = signal1 * signal2

    # 各チャンネルの乗算結果を確認
    for i in range(len(signal1.channels)):
        expected_data = signal1.channels[i].data * signal2.channels[i].data
        assert np.array_equal(result_signal.channels[i].data, expected_data), (
            f"Signal multiplication failed for channel {i + 1}."
        )


def test_signal_division(generate_signals: tuple[ChannelFrame, ChannelFrame]) -> None:
    signal1, signal2 = generate_signals
    result_signal = signal1 / signal2

    # 各チャンネルの除算結果を確認
    for i in range(len(signal1.channels)):
        expected_data = signal1.channels[i].data / signal2.channels[i].data
        assert np.allclose(result_signal.channels[i].data, expected_data, atol=1e-6), (
            f"Signal division failed for channel {i + 1}."
        )
