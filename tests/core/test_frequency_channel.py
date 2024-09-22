# tests/core/test_frequency_channel.py
import pytest

import numpy as np
from wandas.core.channel import Channel
from wandas.core.frequency_channel import FrequencyChannel


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


def test_frequency_channel_initialization():
    data = np.array([10, 9, 8, 7, 6])
    sampling_rate = 1000
    n_fft = 1024
    window = np.hanning(5)
    label = "Test Spectrum"
    unit = "V"
    calibration_value = 1
    metadata = {"note": "Test metadata"}

    freq_channel = FrequencyChannel(
        data=data,
        sampling_rate=sampling_rate,
        n_fft=n_fft,
        window=window,
        label=label,
        unit=unit,
        metadata=metadata,
    )

    assert np.array_equal(freq_channel._data, data)
    assert freq_channel.sampling_rate == sampling_rate
    assert freq_channel.n_fft == n_fft
    assert np.array_equal(freq_channel.window, window)
    assert freq_channel.label == label
    assert freq_channel.unit == unit
    assert freq_channel.calibration_value == calibration_value
    assert freq_channel.metadata == metadata


def test_frequency_channel_from_channel():
    data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    sampling_rate = 1000
    label = "Test Channel"
    unit = "V"
    calibration_value = 1.0
    metadata = {"note": "Test metadata"}

    ch = Channel(data, sampling_rate, label, unit, calibration_value, metadata)
    n_fft = 8
    window = "hann"

    freq_channel = FrequencyChannel.from_channel(ch, n_fft=n_fft, window=window)

    assert freq_channel.sampling_rate == sampling_rate
    assert freq_channel.n_fft == n_fft
    assert freq_channel.label == label
    assert freq_channel.unit == unit
    assert freq_channel.metadata == metadata


def test_frequency_channel_data_property():
    data = np.array([10, 9, 8, 7, 6])
    freq_channel = FrequencyChannel(
        data=data,
        sampling_rate=1000,
        n_fft=1024,
        window=np.hanning(5),
    )

    expected_data = data
    assert np.array_equal(freq_channel.data, expected_data)


def test_fft_amplitude():
    fs = 16000
    nperseg = 4096
    win = "hann"
    freq = 1000  # 周波数5Hz
    amplitude = 2.0
    sine_wave = (
        amplitude * np.sin(freq * 2.0 * np.pi * np.arange(nperseg) / fs)
    ).squeeze()

    # FFTを計算
    fft_result, _, _ = FrequencyChannel._fft(sine_wave, window=win)

    # 振幅値がスペクトルの振幅と一致することを確認
    fft_amplitude = np.abs(fft_result)
    peak_amplitude = np.max(fft_amplitude)

    assert np.isclose(
        peak_amplitude, amplitude, atol=1e-5
    ), f"Expected {amplitude}, but got {peak_amplitude}"

    # ############
    # paddingした場合
    # ############
    # FFTを計算
    fft_result, _, _ = FrequencyChannel._fft(sine_wave, n_fft=nperseg * 2, window=win)

    # 振幅値がスペクトルの振幅と一致することを確認
    fft_amplitude = np.abs(fft_result)
    peak_amplitude = np.max(fft_amplitude)

    assert np.isclose(
        peak_amplitude, amplitude, atol=1e-5
    ), f"Expected {amplitude}, but got {peak_amplitude}"

    # ###########
    # 窓関数を変えてた場合
    # ############
    # FFTを計算
    fft_result, _, _ = FrequencyChannel._fft(
        sine_wave, n_fft=nperseg, window="blackman"
    )

    # 振幅値がスペクトルの振幅と一致することを確認
    fft_amplitude = np.abs(fft_result)
    peak_amplitude = np.max(fft_amplitude)

    assert np.isclose(
        peak_amplitude, amplitude, atol=1e-5
    ), f"Expected {amplitude}, but got {peak_amplitude}"

    # ###########
    # 窓関数を変えてた場合
    # ############
    # FFTを計算

    fft_result, _, _ = FrequencyChannel._fft(sine_wave, n_fft=nperseg, window="boxcar")

    # 振幅値がスペクトルの振幅と一致することを確認
    fft_amplitude = np.abs(fft_result)
    peak_amplitude = np.max(fft_amplitude)

    assert np.isclose(
        peak_amplitude, amplitude, atol=1e-5
    ), f"Expected {amplitude}, but got {peak_amplitude}"


def test_welch_amplitude(generate_channel):
    ch = generate_channel
    amplitude = 2
    n_fft = 1024
    win_length = 1024
    hop_length = n_fft // 2
    window = "hann"
    average = "mean"

    # Welch 法を計算
    welch_result = FrequencyChannel.from_channel_to_welch(
        ch,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        average=average,
    )

    # 振幅値がスペクトルの振幅と一致することを確認
    welch_amplitude = np.abs(welch_result.data)
    peak_amplitude = np.max(welch_amplitude)

    assert np.isclose(
        peak_amplitude, amplitude, atol=1e-5
    ), f"Expected {amplitude}, but got {peak_amplitude}"

    # ############
    # paddingした場合
    # ############
    # Welch 法を計算
    welch_result = FrequencyChannel.from_channel_to_welch(
        ch,
        n_fft=n_fft * 2,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        average=average,
    )

    # 振幅値がスペクトルの振幅と一致することを確認
    welch_amplitude = np.abs(welch_result.data)
    peak_amplitude = np.max(welch_amplitude)

    assert np.isclose(
        peak_amplitude, amplitude, atol=1e-5
    ), f"Expected {amplitude}, but got {peak_amplitude}"

    # ###########
    # 窓関数を変えてた場合
    # ############
    # Welch 法を計算
    welch_result = FrequencyChannel.from_channel_to_welch(
        ch,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="boxcar",
        average=average,
    )

    # 振幅値がスペクトルの振幅と一致することを確認
    welch_amplitude = np.abs(welch_result.data)
    peak_amplitude = np.max(welch_amplitude)

    assert np.isclose(
        peak_amplitude, amplitude, atol=1e-5
    ), f"Expected {amplitude}, but got {peak_amplitude}"


def test_frequency_channel_plot():
    data = np.array([10, 9, 8, 7, 6])
    freq_channel = FrequencyChannel(
        data=data,
        sampling_rate=1000,
        n_fft=8,
        window=np.hanning(8),
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    freq_channel.plot(ax=ax)
    plt.close(fig)
