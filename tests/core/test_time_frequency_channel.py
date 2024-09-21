# tests/core/test_time_frequency_channel.py

import pytest
import numpy as np
from wandas.core.time_frequency_channel import TimeFrequencyChannel
from wandas.core.channel import Channel


@pytest.fixture
def generate_channel():
    sampling_rate = 1000
    t = np.linspace(0, 1, sampling_rate, endpoint=False)
    data = np.sin(2 * np.pi * 50 * t)
    return Channel(data=data, sampling_rate=sampling_rate, label="Test Channel")


@pytest.fixture
def generate_time_frequency_channel(generate_channel):
    ch = generate_channel
    return TimeFrequencyChannel.from_channel(ch)


def test_time_frequency_channel_initialization():
    data = np.random.random((1025, 44))
    sampling_rate = 1000
    n_fft = 2048
    hop_length = 512
    win_length = 2048
    window = "hann"
    center = True

    tf_channel = TimeFrequencyChannel(
        data=data,
        sampling_rate=sampling_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        label="Test TF Channel",
        unit="dB",
        metadata={"test": "metadata"},
    )

    assert np.array_equal(tf_channel.data, data)
    assert tf_channel.sampling_rate == sampling_rate
    assert tf_channel.n_fft == n_fft
    assert tf_channel.hop_length == hop_length
    assert tf_channel.win_length == win_length
    assert tf_channel.window == window
    assert tf_channel.center == center
    assert tf_channel.label == "Test TF Channel"
    assert tf_channel.unit == "dB"
    assert tf_channel.metadata == {"test": "metadata"}


def test_time_frequency_channel_from_channel(generate_channel):
    ch = generate_channel
    tf_channel = TimeFrequencyChannel.from_channel(ch)

    assert tf_channel.sampling_rate == ch.sampling_rate
    assert tf_channel.label == ch.label
    assert tf_channel.unit == ch.unit
    assert tf_channel.metadata == ch.metadata


def test_time_frequency_channel_plot(generate_time_frequency_channel):
    tf_channel = generate_time_frequency_channel

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    tf_channel.plot(ax=ax, title="Test Plot")

    assert ax.get_xlabel() == "Time [s]"
    assert ax.get_ylabel() == "Frequency [Hz]"
    assert ax.get_title() == "Test Plot"


def test_time_frequency_channel_to_db(generate_time_frequency_channel):
    tf_channel = generate_time_frequency_channel
    db_data = tf_channel._to_db()

    assert db_data.shape == tf_channel.data.shape
    assert np.max(db_data) <= 0  # dB values should be <= 0
