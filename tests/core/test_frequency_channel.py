# tests/core/test_frequency_channel.py
import numpy as np
import pytest
from mosqito.sound_level_meter import noct_spectrum, noct_synthesis

from wandas.core.channel import Channel
from wandas.core.frequency_channel import FrequencyChannel, NOctChannel


@pytest.fixture  # type: ignore [misc, unused-ignore]
def generate_channel() -> Channel:
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


def test_frequency_channel_initialization() -> None:
    data = np.array([10, 9, 8, 7, 6])
    sampling_rate = 1000
    n_fft = 1024
    window = np.hanning(5)
    label = "Test Spectrum"
    unit = "V"
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

    assert np.array_equal(freq_channel.data, data)
    assert freq_channel.sampling_rate == sampling_rate
    assert freq_channel.n_fft == n_fft
    assert np.array_equal(freq_channel.window, window)
    assert freq_channel.label == label
    assert freq_channel.unit == unit
    assert freq_channel.metadata == metadata


def test_frequency_channel_from_channel() -> None:
    data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    sampling_rate = 1000
    label = "Test Channel"
    unit = "V"
    metadata = {"note": "Test metadata"}

    ch = Channel(data, sampling_rate, label, unit, metadata)
    n_fft = 8
    window = "hann"

    freq_channel: FrequencyChannel = ch.fft(n_fft=n_fft, window=window)

    assert freq_channel.sampling_rate == sampling_rate
    assert freq_channel.n_fft == n_fft
    assert freq_channel.label == label
    assert freq_channel.unit == unit
    assert freq_channel.metadata == metadata


def test_frequency_channel_data_property() -> None:
    data = np.array([10, 9, 8, 7, 6])
    freq_channel = FrequencyChannel(
        data=data,
        sampling_rate=1000,
        n_fft=1024,
        window=np.hanning(5),
    )

    expected_data = data
    assert np.array_equal(freq_channel.data, expected_data)


def test_fft_amplitude() -> None:
    fs = 16000
    nperseg = 4096
    win = "hann"
    freq = 1000  # 周波数5Hz
    amplitude = 2.0
    sine_wave = (
        amplitude * np.sin(freq * 2.0 * np.pi * np.arange(nperseg) / fs)
    ).squeeze()

    # FFTを計算
    result = FrequencyChannel.fft(sine_wave, window=win)

    # 振幅値がスペクトルの振幅と一致することを確認
    fft_amplitude = np.abs(result["data"])
    peak_amplitude = np.max(fft_amplitude)

    assert np.isclose(peak_amplitude, amplitude, atol=1e-5), (
        f"Expected {amplitude}, but got {peak_amplitude}"
    )

    # ############
    # paddingした場合
    # ############
    # FFTを計算
    result = FrequencyChannel.fft(sine_wave, n_fft=nperseg * 2, window=win)

    # 振幅値がスペクトルの振幅と一致することを確認
    fft_amplitude = np.abs(result["data"])
    peak_amplitude = np.max(fft_amplitude)

    assert np.isclose(peak_amplitude, amplitude, atol=1e-5), (
        f"Expected {amplitude}, but got {peak_amplitude}"
    )

    # ###########
    # 窓関数を変えてた場合
    # ############
    # FFTを計算
    result = FrequencyChannel.fft(sine_wave, n_fft=nperseg, window="blackman")

    # 振幅値がスペクトルの振幅と一致することを確認
    fft_amplitude = np.abs(result["data"])
    peak_amplitude = np.max(fft_amplitude)

    assert np.isclose(peak_amplitude, amplitude, atol=1e-5), (
        f"Expected {amplitude}, but got {peak_amplitude}"
    )

    # ###########
    # 窓関数を変えてた場合
    # ############
    # FFTを計算

    result = FrequencyChannel.fft(sine_wave, n_fft=nperseg, window="boxcar")

    # 振幅値がスペクトルの振幅と一致することを確認
    fft_amplitude = np.abs(result["data"])
    peak_amplitude = np.max(fft_amplitude)

    assert np.isclose(peak_amplitude, amplitude, atol=1e-5), (
        f"Expected {amplitude}, but got {peak_amplitude}"
    )


def test_welch_amplitude(generate_channel: Channel) -> None:
    ch = generate_channel
    amplitude = 2
    n_fft = 1024
    win_length = 1024
    hop_length = n_fft // 2
    window = "hann"
    average = "mean"

    # Welch 法を計算
    welch_result: FrequencyChannel = ch.welch(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        average=average,
    )

    # 振幅値がスペクトルの振幅と一致することを確認
    welch_amplitude = np.abs(welch_result.data)
    peak_amplitude = np.max(welch_amplitude)

    assert np.isclose(peak_amplitude, amplitude, atol=1e-5), (
        f"Expected {amplitude}, but got {peak_amplitude}"
    )

    # ############
    # paddingした場合
    # ############
    # Welch 法を計算
    welch_result: FrequencyChannel = ch.welch(  # type: ignore
        n_fft=n_fft * 2,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        average=average,
    )

    # 振幅値がスペクトルの振幅と一致することを確認
    welch_amplitude = np.abs(welch_result.data)
    peak_amplitude = np.max(welch_amplitude)

    assert np.isclose(peak_amplitude, amplitude, atol=1e-5), (
        f"Expected {amplitude}, but got {peak_amplitude}"
    )

    # ###########
    # 窓関数を変えてた場合
    # ############
    # Welch 法を計算
    welch_result: FrequencyChannel = ch.welch(  # type: ignore
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="boxcar",
        average=average,
    )

    # 振幅値がスペクトルの振幅と一致することを確認
    welch_amplitude = np.abs(welch_result.data)
    peak_amplitude = np.max(welch_amplitude)

    assert np.isclose(peak_amplitude, amplitude, atol=1e-5), (
        f"Expected {amplitude}, but got {peak_amplitude}"
    )


def test_frequency_channel_plot() -> None:
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


def test_frequency_channel_noct_spectrum() -> None:
    f = 1000
    fs = 48000
    d = 0.2
    dB = 60  # noqa: N806
    time = np.arange(0, d, 1 / fs)
    stimulus = np.sin(2 * np.pi * f * time) + 0.5 * np.sin(6 * np.pi * f * time)
    rms = np.sqrt(np.mean(np.power(stimulus, 2)))
    ampl = 0.00002 * np.power(10, dB / 20) / rms
    stimulus = stimulus * ampl
    spec, _ = noct_spectrum(stimulus, fs, fmin=90, fmax=14000)
    spec = np.squeeze(spec)

    result = NOctChannel.noct_spectrum(
        data=stimulus,
        sampling_rate=fs,
        fmin=90,
        fmax=14000,
    )

    assert np.allclose(result["data"], spec, atol=1e-5), (
        f"Expected {spec}, but got {(result['data'],)}"
    )


def test_frequency_channel_noct_synthesis() -> None:
    f = 1000
    fs = 48000
    d = 0.2
    dB = 60  # noqa: N806
    time = np.arange(0, d, 1 / fs)
    stimulus = np.sin(2 * np.pi * f * time) + 0.5 * np.sin(6 * np.pi * f * time)
    rms = np.sqrt(np.mean(np.power(stimulus, 2)))
    ampl = 0.00002 * np.power(10, dB / 20) / rms
    stimulus = stimulus * ampl

    ch = Channel(
        data=stimulus,
        sampling_rate=fs,
    )
    fch = ch.fft(n_fft=stimulus.shape[-1], window="hann")

    spec_3, _ = noct_synthesis(fch.data / np.sqrt(2), fch.freqs, fmin=90, fmax=14000)
    spec_3 = np.squeeze(spec_3)

    noch = fch.noct_synthesis(fmin=90, fmax=14000)

    assert np.argmax(noch.data) == np.argmax(spec_3)

    assert np.isclose(noch.data.max(), spec_3.max(), atol=1e-5), (
        f"Expected {spec_3.max()}, but got {noch.data.max()}"
    )
