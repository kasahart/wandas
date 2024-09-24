# tests/core/test_channel.py

import pytest
import numpy as np
from wandas.core.channel import Channel
from wandas.core.signal import ChannelFrame
import librosa


def _generate_channels():
    # サンプルの正弦波データを生成
    sampling_rate = 1000
    t = np.linspace(0, 1, sampling_rate, endpoint=False)
    data1 = np.ones_like(t) * 2
    data2 = np.ones_like(t) * 3

    ch1 = Channel(data=data1, sampling_rate=sampling_rate, label="Channel 1")
    ch2 = Channel(data=data2, sampling_rate=sampling_rate, label="Channel 2")

    return [ch1, ch2]


@pytest.fixture
def generate_channels():
    return _generate_channels()


@pytest.fixture
def generate_signal():
    return ChannelFrame(channels=_generate_channels())


def test_channel_initialization():
    data = np.array([0, 1, 2, 3, 4])
    sampling_rate = 1000
    channel = Channel(
        data=data, sampling_rate=sampling_rate, label="Test Channel", unit="V"
    )

    assert np.array_equal(channel.data, data)
    assert channel.sampling_rate == sampling_rate
    assert channel.label == "Test Channel"
    assert channel.unit == "V"
    assert channel.metadata == {}


def test_channel_low_pass_filter():
    data = np.sin(2 * np.pi * 50 * np.linspace(0, 1, 1000))
    sampling_rate = 1000
    channel = Channel(data=data, sampling_rate=sampling_rate)
    filtered_channel = channel.low_pass_filter(cutoff=30)

    # 簡易的なチェックとして、フィルタ後のデータがフィルタ前と異なることを確認
    assert not np.array_equal(channel.data, filtered_channel.data)


def test_rms_trend_signal(generate_signal):
    signal = generate_signal

    # RMS トレンドを計算
    for ch in signal:
        rms_librosa = librosa.feature.rms(
            y=ch.data, frame_length=2048, hop_length=512
        ).squeeze()
        rms = ch.rms_trend()
        assert np.array_equal(rms_librosa, rms.data)


def test_channel_plot():
    import matplotlib.pyplot as plt

    data = np.array([0, 1, 2, 3, 4])
    sampling_rate = 1000
    channel = Channel(
        data=data, sampling_rate=sampling_rate, label="Test Channel", unit="V"
    )

    fig, ax = plt.subplots()
    channel.plot(ax=ax, title="Test Plot")

    assert ax.get_xlabel() == "Time [s]"
    assert ax.get_ylabel() == "Amplitude [V]"
    assert ax.get_title() == "Test Plot"
    assert len(ax.lines) == 1
    assert np.array_equal(ax.lines[0].get_xdata(), np.arange(len(data)) / sampling_rate)
    assert np.array_equal(ax.lines[0].get_ydata(), data)


def test_channel_addition(generate_channels):
    ch1, ch2 = generate_channels
    result_channel = ch1 + ch2

    # 結果のデータを確認
    expected_data = ch1.data + ch2.data
    assert np.array_equal(
        result_channel.data, expected_data
    ), "Channel addition failed."


def test_channel_subtraction(generate_channels):
    ch1, ch2 = generate_channels
    result_channel = ch1 - ch2

    # 結果のデータを確認
    expected_data = ch1.data - ch2.data
    assert np.array_equal(
        result_channel.data, expected_data
    ), "Channel subtraction failed."


def test_channel_multiplication(generate_channels):
    ch1, ch2 = generate_channels
    result_channel = ch1 * ch2

    # 結果のデータを確認
    expected_data = ch1.data * ch2.data
    assert np.array_equal(
        result_channel.data, expected_data
    ), "Channel multiplication failed."


def test_channel_division(generate_channels):
    ch1, ch2 = generate_channels
    result_channel = ch1 / ch2

    # 結果のデータを確認
    expected_data = ch1.data / ch2.data
    assert np.allclose(
        result_channel.data, expected_data, atol=1e-6
    ), "Channel division failed."
