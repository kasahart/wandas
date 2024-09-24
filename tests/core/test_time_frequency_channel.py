# tests/core/test_time_frequency_channel.py

import pytest
import numpy as np
from wandas.core.time_frequency_channel import TimeFrequencyChannel
from wandas.core.channel import Channel


@pytest.fixture
def generate_channel():
    sampling_rate = 16000
    freq = 1000  # 周波数5Hz
    amplitude = 2.0
    data_length = 512 * 20

    sine_wave = (
        amplitude * np.sin(freq * 2.0 * np.pi * np.arange(data_length) / sampling_rate)
    ).squeeze()

    return Channel(
        data=sine_wave,
        sampling_rate=sampling_rate,
        label="Test Channel",
        unit="V",
    )


@pytest.fixture
def generate_time_frequency_channel(generate_channel):
    n_fft = 1024
    win_length = 1024
    hop_length = n_fft // 2
    window = "hann"
    ch = generate_channel
    return ch.stft(
        n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window
    )


@pytest.fixture
def generate_time_frequency_channel_boxcar(generate_channel):
    n_fft = 1024
    win_length = 1024
    hop_length = n_fft // 2
    window = "boxcar"
    ch = generate_channel
    return ch.stft(
        n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window
    )


def test_time_frequency_channel_initialization():
    data = np.random.random((1025, 44))
    sampling_rate = 16000
    n_fft = 1024
    hop_length = 512
    win_length = 2048
    window = "hann"
    # center = True

    tf_channel = TimeFrequencyChannel(
        data=data,
        sampling_rate=sampling_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        # center=center,
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
    # assert tf_channel.center == center
    assert tf_channel.label == "Test TF Channel"
    assert tf_channel.unit == "dB"
    assert tf_channel.metadata == {"test": "metadata"}


def test_time_frequency_channel_from_channel(generate_channel):
    ch = generate_channel
    tf_channel = ch.stft()

    assert tf_channel.sampling_rate == ch.sampling_rate
    assert tf_channel.label == ch.label
    assert tf_channel.unit == ch.unit
    assert tf_channel.metadata == ch.metadata


def test_stft_amplitude():
    fs = 16000
    n_fft = 1024
    win_length = 1024
    hop_length = n_fft // 2
    data_length = hop_length * 20
    window = "hann"
    freq = 1000  # 周波数5Hz

    amplitude = 2.0 * np.sqrt(2.0)
    sine_wave = (
        amplitude * np.sin(freq * 2.0 * np.pi * np.arange(data_length) / fs)
    ).squeeze()

    stft_result = TimeFrequencyChannel.stft(
        data=sine_wave,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    )

    stft_amplitude = np.abs(stft_result["data"])
    peak_amplitude = np.max(stft_amplitude)

    assert np.isclose(
        peak_amplitude, amplitude, atol=1e-5
    ), f"Expected {amplitude}, but got {peak_amplitude}"

    # ############
    # paddingした場合
    # ############
    stft_result = TimeFrequencyChannel.stft(
        data=sine_wave,
        n_fft=n_fft * 2,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    )

    stft_amplitude = np.abs(stft_result["data"])
    peak_amplitude = np.max(stft_amplitude)

    assert np.isclose(
        peak_amplitude, amplitude, atol=1e-5
    ), f"Expected {amplitude}, but got {peak_amplitude}"

    # ###########
    # 窓関数を変えてた場合
    # ############
    stft_result = TimeFrequencyChannel.stft(
        data=sine_wave,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="blackman",
    )

    stft_amplitude = np.abs(stft_result["data"])
    peak_amplitude = np.max(stft_amplitude)

    assert np.isclose(
        peak_amplitude, amplitude, atol=1e-5
    ), f"Expected {amplitude}, but got {peak_amplitude}"

    # ###########
    # 窓関数を変えてた場合
    # ############
    stft_result = TimeFrequencyChannel.stft(
        data=sine_wave,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="boxcar",
    )

    stft_amplitude = np.abs(stft_result["data"])
    peak_amplitude = np.max(stft_amplitude)

    assert np.isclose(
        peak_amplitude, amplitude, atol=1e-5
    ), f"Expected {amplitude}, but got {peak_amplitude}"


def test_time_frequency_channel_plot(generate_time_frequency_channel):
    tf_channel = generate_time_frequency_channel

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax, spec = tf_channel.plot(ax=ax, title="Test Plot")

    assert ax.get_xlabel() == "Time [s]"
    assert ax.get_ylabel() == "Frequency [Hz]"
    assert ax.get_title() == "Test Plot"

    ref = 20 * np.log10(2)
    cal = spec.max()
    assert np.isclose(
        cal, ref, atol=1e-5
    ), f"Expected {cal}, but got {ref}"  # dB values should be <= 0


def test_time_frequency_channel_plot_boxcar(generate_time_frequency_channel_boxcar):
    tf_channel = generate_time_frequency_channel_boxcar

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax, spec = tf_channel.plot(ax=ax, title="Test Plot")

    assert ax.get_xlabel() == "Time [s]"
    assert ax.get_ylabel() == "Frequency [Hz]"
    assert ax.get_title() == "Test Plot"

    ref = 20 * np.log10(2)
    cal = spec.max()
    assert np.isclose(
        cal, ref, atol=1e-5
    ), f"Expected {cal}, but got {ref}"  # dB values should be <= 0


def test_time_frequency_channel_to_db(generate_time_frequency_channel):
    tf_channel = generate_time_frequency_channel
    db_data = tf_channel._to_db()
    ref = 20 * np.log10(2)
    cal = db_data.max()
    assert db_data.shape == tf_channel.data.shape
    assert np.isclose(
        cal, ref, atol=1e-5
    ), f"Expected {cal}, but got {ref}"  # dB values should be <= 0


def test_time_frequency_channel_to_db_boxcar(generate_time_frequency_channel_boxcar):
    tf_channel = generate_time_frequency_channel_boxcar
    db_data = tf_channel._to_db()
    ref = 20 * np.log10(2)
    cal = db_data.max()
    assert db_data.shape == tf_channel.data.shape
    assert np.isclose(
        cal, ref, atol=1e-5
    ), f"Expected {cal}, but got {ref}"  # dB values should be <= 0
