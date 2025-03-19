# tests/core/test_time_frequency_channel.py

import librosa
import numpy as np
import pytest
from scipy import fft

from wandas.core.channel import Channel
from wandas.core.time_frequency_channel import (
    TimeFrequencyChannel,
    TimeMelFrequencyChannel,
)


@pytest.fixture()  # type: ignore [misc, unused-ignore]
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


@pytest.fixture()  # type: ignore [misc, unused-ignore]
def generate_time_frequency_channel(generate_channel: Channel) -> TimeFrequencyChannel:
    n_fft = 1024
    win_length = 1024
    hop_length = n_fft // 2
    window = "hann"
    ch = generate_channel
    return ch.stft(
        n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window
    )


@pytest.fixture()  # type: ignore [misc, unused-ignore]
def generate_time_frequency_channel_boxcar(
    generate_channel: Channel,
) -> TimeFrequencyChannel:
    n_fft = 1024
    win_length = 1024
    hop_length = n_fft // 2
    window = "boxcar"
    ch = generate_channel
    return ch.stft(
        n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window
    )


def test_time_frequency_channel_initialization() -> None:
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


def test_time_frequency_channel_from_channel(generate_channel: Channel) -> None:
    ch = generate_channel
    tf_channel = ch.stft()

    assert tf_channel.sampling_rate == ch.sampling_rate
    assert tf_channel.label == ch.label
    assert tf_channel.unit == ch.unit
    assert tf_channel.metadata == ch.metadata


def test_stft_amplitude() -> None:
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

    assert np.isclose(peak_amplitude, amplitude, atol=1e-5), (
        f"Expected {amplitude}, but got {peak_amplitude}"
    )

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

    assert np.isclose(peak_amplitude, amplitude, atol=1e-5), (
        f"Expected {amplitude}, but got {peak_amplitude}"
    )

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

    assert np.isclose(peak_amplitude, amplitude, atol=1e-5), (
        f"Expected {amplitude}, but got {peak_amplitude}"
    )

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

    assert np.isclose(peak_amplitude, amplitude, atol=1e-5), (
        f"Expected {amplitude}, but got {peak_amplitude}"
    )


def test_time_frequency_channel_plot(
    generate_time_frequency_channel: TimeFrequencyChannel,
) -> None:
    tf_channel = generate_time_frequency_channel

    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    ax, spec = tf_channel.plot(ax=ax, title="Test Plot")

    assert ax.get_xlabel() == "Time [s]"
    assert ax.get_ylabel() == "Frequency [Hz]"
    assert ax.get_title() == "Test Plot"

    ref = 20 * np.log10(2)
    cal = spec.max()
    assert np.isclose(cal, ref, atol=1e-5), (
        f"Expected {cal}, but got {ref}"
    )  # dB values should be <= 0


def test_time_frequency_channel_plot_boxcar(
    generate_time_frequency_channel_boxcar: TimeFrequencyChannel,
) -> None:
    tf_channel = generate_time_frequency_channel_boxcar

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax, spec = tf_channel.plot(ax=ax, title="Test Plot")

    assert ax.get_xlabel() == "Time [s]"
    assert ax.get_ylabel() == "Frequency [Hz]"
    assert ax.get_title() == "Test Plot"

    ref = 20 * np.log10(2)
    cal = spec.max()
    assert np.isclose(cal, ref, atol=1e-5), (
        f"Expected {cal}, but got {ref}"
    )  # dB values should be <= 0


def test_time_frequency_channel_to_db(
    generate_time_frequency_channel: TimeFrequencyChannel,
) -> None:
    tf_channel = generate_time_frequency_channel
    db_data = tf_channel._to_db()
    ref = 20 * np.log10(2)
    cal = db_data.max()
    assert db_data.shape == tf_channel.data.shape
    assert np.isclose(cal, ref, atol=1e-5), (
        f"Expected {cal}, but got {ref}"
    )  # dB values should be <= 0


def test_time_frequency_channel_to_db_boxcar(
    generate_time_frequency_channel_boxcar: TimeFrequencyChannel,
) -> None:
    tf_channel = generate_time_frequency_channel_boxcar
    db_data = tf_channel._to_db()
    ref = 20 * np.log10(2)
    cal = db_data.max()
    assert db_data.shape == tf_channel.data.shape
    assert np.isclose(cal, ref, atol=1e-5), (
        f"Expected {cal}, but got {ref}"
    )  # dB values should be <= 0


def test_time_frequency_channel_melspectrogram(
    generate_time_frequency_channel: TimeFrequencyChannel, generate_channel: Channel
) -> None:
    channel = generate_channel
    tf_channel = generate_time_frequency_channel

    n_mels = 128
    mel_spectrogram = channel.melspectrogram(
        n_mels=n_mels,
        n_fft=tf_channel.n_fft,
        hop_length=tf_channel.hop_length,
        win_length=tf_channel.win_length,
        window=tf_channel.window,
    )
    spec2mel_spec = tf_channel.melspectrogram(n_mels=n_mels)

    assert spec2mel_spec.sampling_rate == tf_channel.sampling_rate
    assert spec2mel_spec.n_fft == tf_channel.n_fft
    assert spec2mel_spec.hop_length == tf_channel.hop_length
    assert spec2mel_spec.win_length == tf_channel.win_length
    assert spec2mel_spec.window == tf_channel.window
    assert spec2mel_spec.label == tf_channel.label
    assert spec2mel_spec.unit == tf_channel.unit
    assert spec2mel_spec.metadata == tf_channel.metadata

    cal = mel_spectrogram.data.max()
    ref = spec2mel_spec.data.max()

    assert np.isclose(cal, ref, atol=1e-5), (
        f"Expected {cal}, but got {ref}"
    )  # dB values should be <= 0


def test_time_mel_frequency_channel_initialization() -> None:
    data = np.random.random((128, 44))
    sampling_rate = 16000
    n_fft = 2048
    hop_length = 512
    win_length = 2048
    window = "hann"
    n_mels = 128

    tf_mel_channel = TimeMelFrequencyChannel(
        data=data,
        sampling_rate=sampling_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        n_mels=n_mels,
        label="Test Mel TF Channel",
        unit="dB",
        metadata={"test": "metadata"},
    )

    assert np.array_equal(tf_mel_channel.data, data)
    assert tf_mel_channel.sampling_rate == sampling_rate
    assert tf_mel_channel.n_fft == n_fft
    assert tf_mel_channel.hop_length == hop_length
    assert tf_mel_channel.win_length == win_length
    assert tf_mel_channel.window == window
    assert tf_mel_channel.label == "Test Mel TF Channel"
    assert tf_mel_channel.unit == "dB"
    assert tf_mel_channel.metadata == {"test": "metadata"}


def test_time_mel_frequency_channel_melspectrogram() -> None:
    fs = 16000
    n_fft = 2048
    win_length = 2048
    hop_length = 512
    data_length = hop_length * 20
    window = "hann"
    n_mels = 128
    freq = 1000

    amplitude = 2.0 * np.sqrt(2.0)
    sine_wave = (
        amplitude * np.sin(freq * 2.0 * np.pi * np.arange(data_length) / fs)
    ).squeeze()

    ref_mel = librosa.feature.melspectrogram(
        y=sine_wave,
        sr=fs,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        n_mels=n_mels,
        power=1.0,
        norm=None,
    )

    spec = librosa.stft(
        y=sine_wave,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    )
    mel_spectrogram_result = TimeMelFrequencyChannel.spec2melspec(
        sampling_rate=fs,
        data=np.abs(spec),
        n_fft=n_fft,
        n_mels=n_mels,
    )

    assert mel_spectrogram_result["sampling_rate"] == fs
    assert mel_spectrogram_result["n_fft"] == n_fft
    assert mel_spectrogram_result["data"].shape[0] == n_mels
    assert np.allclose(mel_spectrogram_result["data"], ref_mel, atol=1e-5), (
        f"Expected {mel_spectrogram_result['data']}, but got {ref_mel}"
    )  # dB values should be <= 0


def test_time_mel_frequency_channel_plot() -> None:
    data = np.random.random((128, 44))
    sampling_rate = 16000
    n_fft = 2048
    hop_length = 512
    win_length = 2048
    window = "hann"
    n_mels = 128

    tf_mel_channel = TimeMelFrequencyChannel(
        data=data,
        sampling_rate=sampling_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        n_mels=n_mels,
        label="Test Mel TF Channel",
        unit="dB",
        metadata={"test": "metadata"},
    )

    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    ax, _ = tf_mel_channel.plot(ax=ax, title="Test Mel Plot")
    assert ax.get_xlabel() == "Time [s]"
    assert ax.get_ylabel() == "Frequency [Hz]"
    assert ax.get_title() == "Test Mel Plot"
    # Language: python


def _create_dummy_tf_channel() -> TimeFrequencyChannel:
    # Use small values so that rfftfreq returns a small array.
    n_fft = 4
    sampling_rate = 4
    # Dummy data for BaseChannel purposes; it is overwritten later.
    dummy_data = np.array([0, 1.0, 2.0, 3.0])
    # Create a TimeFrequencyChannel instance.
    tf_channel = TimeFrequencyChannel(
        data=dummy_data,
        sampling_rate=sampling_rate,
        n_fft=n_fft,
        hop_length=2,
        win_length=4,
        window="hann",
        label="Dummy TF Channel",
        unit="Pa",
        metadata={"test": "metadata"},
    )

    return tf_channel


def test_data_aw_returns_weighted_to_db_when_flag_true() -> None:
    # Instantiate dummy channel
    tf_channel = _create_dummy_tf_channel()
    # Compute frequency bins
    freqs = fft.rfftfreq(tf_channel.n_fft, 1 / tf_channel.sampling_rate)
    # Compute expected weighted power via perceptual weighting, then cast to float64
    expected_weighted = librosa.perceptual_weighting(
        np.abs(tf_channel._data) ** 2, freqs, kind="A", ref=tf_channel.ref**2
    ).astype(np.float64)

    # Call data_Aw with to_dB=True
    result = tf_channel.data_Aw(to_dB=True)

    # Assert that the result matches expected weighted values
    np.testing.assert_allclose(result, expected_weighted, rtol=1e-5)


def test_data_aw_returns_amplitude_converted_values_when_flag_false() -> None:
    # Instantiate dummy channel
    tf_channel = _create_dummy_tf_channel()
    # Compute frequency bins
    freqs = fft.rfftfreq(tf_channel.n_fft, 1 / tf_channel.sampling_rate)
    # Compute expected weighted power via perceptual weighting
    weighted = librosa.perceptual_weighting(
        np.abs(tf_channel._data) ** 2, freqs, kind="A", ref=tf_channel.ref**2
    )
    # Convert weighted dB values to amplitude
    expected_amplitude = np.asarray(librosa.db_to_amplitude(weighted), dtype=np.float64)

    # Call data_Aw with to_dB=False
    result = tf_channel.data_Aw(to_dB=False)

    # Assert that the result matches expected amplitude values
    np.testing.assert_allclose(result, expected_amplitude, rtol=1e-5)


def test_frequency_channel_hpss_harmonic(generate_channel: Channel) -> None:
    """
    Compare fc.hpss_harmonic() result with librosa.decompose.hpss.
    """

    ch = generate_channel
    n_fft = 512
    win_length = 512
    window = "hann"

    # Apply HPSS with librosa directly on time-domain data
    fc = ch.stft(n_fft=n_fft, win_length=win_length, window=window)
    # Apply HPSS in our FrequencyChannel
    harmonic_fc = fc.hpss_harmonic()
    h, _ = librosa.decompose.hpss(fc.data)
    # Check shape
    assert harmonic_fc.data.shape == h.shape, (
        f"Expected shape {h.shape}, got {harmonic_fc.data.shape}"
    )

    # Compare amplitude
    result = np.abs(harmonic_fc.data)
    expected_weighted = np.abs(h)
    np.testing.assert_allclose(result, expected_weighted)


def test_frequency_channel_hpss_percussive(generate_channel: Channel) -> None:
    """
    Compare fc.hpss_percussive() result with librosa.decompose.hpss.
    """

    ch = generate_channel
    n_fft = 512
    win_length = 512
    window = "hann"

    # Apply HPSS with librosa directly on time-domain data
    fc = ch.stft(n_fft=n_fft, win_length=win_length, window=window)
    # Apply HPSS in our FrequencyChannel
    percussive_fc = fc.hpss_percussive()
    _, p = librosa.decompose.hpss(fc.data)
    # Check shape
    assert percussive_fc.data.shape == p.shape, (
        f"Expected shape {p.shape}, got {percussive_fc.data.shape}"
    )
    # Compare amplitude
    result = np.abs(percussive_fc.data)
    expected_weighted = np.abs(p)
    np.testing.assert_allclose(result, expected_weighted)
