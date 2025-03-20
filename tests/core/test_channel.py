# tests/core/test_channel.py
from typing import Any

import ipywidgets as widgets
import librosa
import numpy as np
import pytest
from matplotlib.axes import Axes

from wandas.core import util
from wandas.core.channel import Channel
from wandas.core.channel_frame import ChannelFrame
from wandas.core.frequency_channel import FrequencyChannel, NOctChannel
from wandas.core.time_frequency_channel import TimeFrequencyChannel
from wandas.utils.types import NDArrayReal


def _generate_channels() -> list[Channel]:
    # サンプルの正弦波データを生成
    sampling_rate = 1000
    t = np.linspace(0, 1, sampling_rate, endpoint=False)
    data1 = np.ones_like(t) * 2
    data2 = np.ones_like(t) * 3

    ch1 = Channel(data=data1, sampling_rate=sampling_rate, label="Channel 1")
    ch2 = Channel(data=data2, sampling_rate=sampling_rate, label="Channel 2")

    return [ch1, ch2]


@pytest.fixture  # type: ignore [misc, unused-ignore]
def generate_channels() -> list[Channel]:
    return _generate_channels()


@pytest.fixture  # type: ignore [misc, unused-ignore]
def generate_signal() -> ChannelFrame:
    return ChannelFrame(channels=_generate_channels())


def test_channel_initialization() -> None:
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


def test_channel_low_pass_filter() -> None:
    data = np.sin(2 * np.pi * 50 * np.linspace(0, 1, 1000))
    sampling_rate = 1000
    channel = Channel(data=data, sampling_rate=sampling_rate)
    filtered_channel = channel.low_pass_filter(cutoff=30)

    # 簡易的なチェックとして、フィルタ後のデータがフィルタ前と異なることを確認
    assert not np.array_equal(channel.data, filtered_channel.data)


def test_rms_trend_signal(generate_signal: ChannelFrame) -> None:
    signal = generate_signal

    # RMS トレンドを計算
    for ch in signal:
        rms_librosa = librosa.feature.rms(
            y=ch.data, frame_length=2048, hop_length=512
        ).squeeze()
        rms = ch.rms_trend()
        assert np.array_equal(rms_librosa, rms.data)


def test_channel_plot() -> None:
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


def test_channel_addition(generate_channels: list[Channel]) -> None:
    ch1, ch2 = generate_channels
    result_channel = ch1 + ch2

    # 結果のデータを確認
    expected_data = ch1.data + ch2.data
    assert np.array_equal(result_channel.data, expected_data), (
        "Channel addition failed."
    )
    assert np.array_equal((ch1 + ch2.data).data, expected_data), (
        "Channel addition failed."
    )


def test_channel_subtraction(generate_channels: list[Channel]) -> None:
    ch1, ch2 = generate_channels
    result_channel = ch1 - ch2

    # 結果のデータを確認
    expected_data = ch1.data - ch2.data
    assert np.array_equal(result_channel.data, expected_data), (
        "Channel subtraction failed."
    )
    assert np.array_equal((ch1 - ch2.data).data, expected_data), (
        "Channel subtraction failed."
    )


def test_channel_multiplication(generate_channels: list[Channel]) -> None:
    ch1, ch2 = generate_channels
    result_channel = ch1 * ch2

    # 結果のデータを確認
    expected_data = ch1.data * ch2.data
    assert np.array_equal(result_channel.data, expected_data), (
        "Channel multiplication failed."
    )
    assert np.array_equal((ch1 * ch2.data).data, expected_data), (
        "Channel multiplication failed."
    )


def test_channel_division(generate_channels: list[Channel]) -> None:
    ch1, ch2 = generate_channels
    result_channel = ch1 / ch2

    # 結果のデータを確認
    expected_data = ch1.data / ch2.data
    assert np.allclose(result_channel.data, expected_data, atol=1e-6), (
        "Channel division failed."
    )
    assert np.array_equal((ch1 / ch2.data).data, expected_data), (
        "Channel division failed."
    )


def test_channel_high_pass_filter() -> None:
    sampling_rate: int = 1000
    t = np.linspace(0, 1, sampling_rate, endpoint=False)
    # Create a composite signal: low frequency + high frequency
    low_freq = np.sin(2 * np.pi * 10 * t)
    high_freq = np.sin(2 * np.pi * 200 * t)
    data = low_freq + high_freq

    channel: Channel = Channel(
        data=data, sampling_rate=sampling_rate, label="Composite"
    )
    hp_channel: Channel = channel.high_pass_filter(cutoff=50)
    # Check that the filtered data is different
    # from the original due to removal of low frequency
    assert not np.array_equal(channel.data, hp_channel.data)
    # Expect high frequency to remain
    # (RMS of filtered data should be closer to high_freq only)
    rms_high: float = np.sqrt(np.mean(high_freq**2))
    rms_filtered: float = np.sqrt(np.mean(hp_channel.data.astype(float) ** 2))
    assert np.isclose(rms_filtered, rms_high, atol=0.2)


def test_channel_fft() -> None:
    sampling_rate: int = 1000
    t = np.linspace(0, 1, sampling_rate, endpoint=False)
    n_fft = sampling_rate  # set n_fft equal to the number of data points
    data = np.sin(2 * np.pi * 50 * t)
    channel: Channel = Channel(data=data, sampling_rate=sampling_rate, label="Sine50")
    fft_channel: FrequencyChannel = channel.fft(n_fft=n_fft)
    # Verify that transformed channel has a data attribute and same sampling rate
    assert hasattr(fft_channel, "data")
    assert fft_channel.sampling_rate == sampling_rate


def test_channel_welch() -> None:
    sampling_rate: int = 1000
    t = np.linspace(0, 1, sampling_rate, endpoint=False)
    data = np.sin(2 * np.pi * 20 * t)
    channel: Channel = Channel(data=data, sampling_rate=sampling_rate, label="Sine20")
    welch_channel: FrequencyChannel = channel.welch(
        n_fft=256, hop_length=128, win_length=256
    )
    # Check that welch returns a channel-like object with spectral data
    assert hasattr(welch_channel, "data")
    # Sampling rate may be different because of hop_length adjustment, but exists
    assert welch_channel.sampling_rate is not None


def test_channel_stft() -> None:
    sampling_rate: int = 1000
    t = np.linspace(0, 1, sampling_rate, endpoint=False)
    data = np.sin(2 * np.pi * 100 * t)
    channel: Channel = Channel(data=data, sampling_rate=sampling_rate, label="Sine100")
    stft_channel: TimeFrequencyChannel = channel.stft(
        n_fft=256, hop_length=128, win_length=256, window="hann"
    )
    # STFT's data should be a 2D array (frequency bins x time frames)
    assert isinstance(stft_channel.data, np.ndarray)
    assert stft_channel.data.ndim == 2


def test_channel_melspectrogram() -> None:
    sampling_rate: int = 1000
    t = np.linspace(0, 1, sampling_rate, endpoint=False)
    data = np.sin(2 * np.pi * 100 * t)
    channel: Channel = Channel(data=data, sampling_rate=sampling_rate, label="Sine100")
    mel_channel: TimeFrequencyChannel = channel.melspectrogram(
        n_mels=40, n_fft=256, hop_length=128, win_length=256
    )
    # Mel spectrogram data should be a 2D array (mels x time frames)
    assert isinstance(mel_channel.data, np.ndarray)
    assert mel_channel.data.ndim == 2


def test_channel_rms_plot() -> None:
    sampling_rate: int = 1000
    t = np.linspace(0, 1, sampling_rate, endpoint=False)
    data = np.sin(2 * np.pi * 50 * t)
    channel: Channel = Channel(data=data, sampling_rate=sampling_rate, label="Sine50")
    # Call rms_plot, which should return a Channel object (with RMS trend data)
    rms_ch: Axes = channel.rms_plot(title="RMS Plot Test")
    assert isinstance(rms_ch, Axes)


def test_channel_to_audio() -> None:
    sampling_rate: int = 1000
    data = np.sin(2 * np.pi * 50 * np.linspace(0, 1, sampling_rate, endpoint=False))
    channel: Channel = Channel(data=data, sampling_rate=sampling_rate, label="Sine50")
    audio_widget: widgets.VBox = channel.to_audio(normalize=True, label=True)
    # Check that the returned widget is a VBox and it contains children
    assert isinstance(audio_widget, widgets.VBox)
    assert len(audio_widget.children) >= 1


def test_channel_describe() -> None:
    sampling_rate: int = 2048
    data = np.random.randn(sampling_rate)
    channel: Channel = Channel(data=data, sampling_rate=sampling_rate, label="Random")
    desc_widget: widgets.VBox = channel.describe()
    # Check that the description returns an ipywidgets.VBox
    assert isinstance(desc_widget, widgets.VBox)
    # VBox should contain at least one child widget
    assert len(desc_widget.children) >= 1


def test_channel_noct_spectrum_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Create a simple sine wave channel
    sampling_rate = 1000
    t = np.linspace(0, 1, sampling_rate, endpoint=False)
    data = np.sin(2 * np.pi * 50 * t)
    channel = Channel(data=data, sampling_rate=sampling_rate, label="Sine50")

    # Define a dummy noct_spectrum function returning fixed values
    def dummy_noct_spectrum(
        data: NDArrayReal,
        sampling_rate: int,
        fmin: int,
        fmax: int,
        n: int,
        G: int,  # noqa: N803
        fr: int,
    ) -> dict[str, Any]:
        return {
            "data": np.array([[1, 2, 3]]),
            "sampling_rate": sampling_rate,
            "fpref": np.array([fmin, fmax]),
            "n": n,
            "G": G,
            "fr": fr,
        }

    # Monkey-patch NOctChannel.noct_spectrum with the dummy function
    monkeypatch.setattr(NOctChannel, "noct_spectrum", staticmethod(dummy_noct_spectrum))

    # Call noct_spectrum using default parameters
    result = channel.noct_spectrum()

    # Verify that the returned object's attributes match the dummy values and defaults
    np.testing.assert_array_equal(result.data, np.array([[1, 2, 3]]))
    assert result.sampling_rate == sampling_rate
    # Default fmin, fmax, n, G, fr values are 20, 20000, 3, 10, 1000 respectively
    assert result.fpref[0] == 20
    assert result.fpref[-1] == 20000
    assert result.n == 3
    assert result.G == 10
    assert result.fr == 1000


def test_channel_add_without_snr(generate_channels: list[Channel]) -> None:
    # Test the add() method without providing an SNR (should delegate to __add__)
    ch1, ch2 = generate_channels
    result_via_add = ch1.add(ch2, snr=None)
    result_via_operator = ch1 + ch2
    np.testing.assert_array_equal(result_via_add.data, result_via_operator.data)
    assert result_via_add.label == result_via_operator.label


def test_channel_add_with_snr(monkeypatch: pytest.MonkeyPatch) -> None:
    # Create two channels with simple constant data for predictable RMS values
    sampling_rate = 1000
    clean_data = np.array([1, 2, 3, 4], dtype=float)
    noise_data = np.array([2, 2, 2, 2], dtype=float)
    ch_clean = Channel(data=clean_data, sampling_rate=sampling_rate, label="Clean")
    ch_noise = Channel(data=noise_data, sampling_rate=sampling_rate, label="Noise")

    # Define predictable implementations
    # for calculate_rms and calculate_desired_noise_rms
    # RMS is computed as the square root of the mean of the squares.
    monkeypatch.setattr(util, "calculate_rms", lambda data: np.sqrt(np.mean(data**2)))
    # For simplicity, desired noise RMS is defined as (clean_rms / snr)
    monkeypatch.setattr(
        util,
        "calculate_desired_noise_rms",
        lambda clean_rms, snr: clean_rms / snr,
    )

    # Compute expected gain manually
    clean_rms = np.sqrt(np.mean(clean_data**2))
    noise_rms = np.sqrt(np.mean(noise_data**2))
    snr_value = 2.0
    desired_noise_rms = clean_rms / snr_value
    expected_gain = desired_noise_rms / noise_rms
    expected_data = clean_data + noise_data * expected_gain

    # Use the add() method with SNR provided
    result_channel = ch_clean.add(ch_noise, snr=snr_value)

    np.testing.assert_allclose(result_channel.data, expected_data, rtol=1e-5)


def create_sequential_channel(
    num_samples: int = 1000, sampling_rate: int = 1000
) -> Channel:
    data = np.arange(num_samples, dtype=float)
    return Channel(data=data, sampling_rate=sampling_rate, label="Sequential")


def test_channel_trim_extraction() -> None:
    sampling_rate = 1000
    channel = create_sequential_channel(num_samples=1000, sampling_rate=sampling_rate)

    start_time = 0.2  # seconds
    end_time = 0.5  # seconds
    start_idx = int(start_time * sampling_rate)
    end_idx = int(end_time * sampling_rate)

    trimmed_channel = channel.trim(start_time, end_time)
    expected_data = channel.data[start_idx:end_idx]

    np.testing.assert_array_equal(trimmed_channel.data, expected_data)
    # Ensure that other attributes are preserved
    assert trimmed_channel.sampling_rate == channel.sampling_rate
    assert trimmed_channel.label == channel.label


def test_channel_trim_full_length() -> None:
    sampling_rate = 1000
    channel = create_sequential_channel(num_samples=1000, sampling_rate=sampling_rate)

    # Trimming from start to end should return the full data
    trimmed_channel = channel.trim(0.0, 1.0)
    np.testing.assert_array_equal(trimmed_channel.data, channel.data)


def test_channel_trim_edge_cases() -> None:
    sampling_rate = 1000
    num_samples = 1000
    channel = create_sequential_channel(
        num_samples=num_samples, sampling_rate=sampling_rate
    )

    # Test trimming with start time 0
    trimmed_channel_start = channel.trim(0.0, 0.3)
    expected_data_start = channel.data[: int(0.3 * sampling_rate)]
    np.testing.assert_array_equal(trimmed_channel_start.data, expected_data_start)

    # Test trimming with end time equal to duration
    duration = num_samples / sampling_rate
    trimmed_channel_end = channel.trim(0.7, duration)
    expected_data_end = channel.data[int(0.7 * sampling_rate) :]
    np.testing.assert_array_equal(trimmed_channel_end.data, expected_data_end)

    # Test trimming with start and end resulting in zero samples if applicable
    trimmed_channel_empty = channel.trim(0.5, 0.5)
    assert trimmed_channel_empty.data.size == 0


def test_channel_trigger_level_basic() -> None:
    # This test verifies that the Channel.trigger method
    # returns the expected trigger indices
    # using the "level" trigger on a known data array.

    # Create a simple signal with known upward crossings.
    # Calculation details:
    # data = [0.0, 0.2, 0.6, 0.4, 0.7, 0.3, 0.9, 0.1]
    # For threshold = 0.5, np.sign(data - 0.5) -> [-1, -1, 1, -1, 1, -1, 1, -1]
    # diff yields [0, 2, -2, 2, -2, 2, -2]. The indices where diff > 0 are [1, 3, 5].
    data = np.array([0.0, 0.2, 0.6, 0.4, 0.7, 0.3, 0.9, 0.1])
    channel = Channel(data=data, sampling_rate=1000, label="Test Channel")
    result = channel.trigger(threshold=0.5)
    expected = util.level_trigger(data, level=0.5)
    assert result == expected, f"Expected {expected}, but got {result}"


def test_channel_trigger_invalid_trigger_type() -> None:
    # This test checks that an unsupported trigger type raises a ValueError.

    data = np.array([0.0, 0.2, 0.6, 0.4, 0.7, 0.3, 0.9, 0.1])
    channel = Channel(data=data, sampling_rate=1000, label="Test Channel")

    with pytest.raises(ValueError) as excinfo:
        channel.trigger(threshold=0.5, trigger_type="unsupported")
    assert "Unsupported trigger type" in str(excinfo.value)


def test_channel_cut_basic() -> None:
    # Create a channel with simple sequential data
    data = np.arange(20, dtype=float)
    sampling_rate = 1000
    channel = Channel(data=data, sampling_rate=sampling_rate, label="CutTest")
    # Define point_list including valid and invalid indices.
    # Valid if p >= 0 and p + cut_len <= len(data)
    point_list = [0, 10, 15, -1, 16]  # Only 0, 10, and 15 are valid (16+5=21 > 20)
    cut_len = 5
    taper_rate = 0  # rectangular window; tukey returns ones
    dc_cut = False

    result_channels = channel.cut(point_list, cut_len, taper_rate, dc_cut)
    # Expected valid starting indices: [0, 10, 15]
    expected_indices = [0, 10, 15]
    expected_segments = util.cut_sig(
        data, expected_indices, cut_len, taper_rate, dc_cut
    )

    assert len(result_channels) == len(expected_segments)
    for res_ch, exp_seg in zip(result_channels, expected_segments):
        np.testing.assert_allclose(res_ch.data, exp_seg)


def test_channel_cut_dc_cut() -> None:
    # Create a channel with data having a DC offset
    data = np.arange(20, dtype=float) + 10  # Values from 10 to 29
    sampling_rate = 1000
    channel = Channel(data=data, sampling_rate=sampling_rate, label="CutDC")
    # Choose valid cut points
    point_list = [2, 8]  # Valid if 2+5 <= 20 and 8+5 <= 20
    cut_len = 5
    taper_rate = 0  # rectangular window
    dc_cut = True

    result_channels = channel.cut(point_list, cut_len, taper_rate, dc_cut)

    expected_segments = util.cut_sig(
        data=data, point_list=point_list, cut_len=cut_len, dc_cut=dc_cut
    )

    assert len(result_channels) == len(expected_segments)
    for res_ch, exp_seg in zip(result_channels, expected_segments):
        np.testing.assert_allclose(res_ch.data, exp_seg)


def test_channel_cut_taper_rate() -> None:
    # Create a channel with linearly spaced data
    data = np.linspace(0, 1, 30)
    sampling_rate = 1000
    channel = Channel(data=data, sampling_rate=sampling_rate, label="CutTaper")
    # Define valid cut points
    point_list = [0, 12]  # Both valid since 0+6<=30 and 12+6<=30
    cut_len = 6
    taper_rate = 0.5  # Nonzero taper rate produces a tapered window
    dc_cut = False

    result_channels = channel.cut(point_list, cut_len, taper_rate, dc_cut)
    expected_segments = util.cut_sig(
        data=data,
        point_list=point_list,
        cut_len=cut_len,
        taper_rate=taper_rate,
        dc_cut=dc_cut,
    )

    assert len(result_channels) == len(expected_segments)
    for res_ch, exp_seg in zip(result_channels, expected_segments):
        np.testing.assert_allclose(res_ch.data, exp_seg)


def test_channel_hpss_harmonic_basic() -> None:
    """
    Test the basic functionality of the hpss_harmonic method.
    """
    # Create a test signal with both harmonic and percussive components
    sampling_rate = 1000
    t = np.linspace(0, 1, sampling_rate, endpoint=False)
    # Harmonic component (sine wave)
    harmonic = np.sin(2 * np.pi * 10 * t)
    # Percussive component (short impulses)
    percussive = np.zeros_like(t)
    percussive[::100] = 1.0  # Impulses every 100 samples
    # Combined signal
    data = harmonic + percussive

    channel = Channel(data=data, sampling_rate=sampling_rate, label="Mixed")

    # Call hpss_harmonic with default parameters
    harmonic_channel = channel.hpss_harmonic()

    # Basic assertions
    assert isinstance(harmonic_channel, Channel)
    assert harmonic_channel.sampling_rate == sampling_rate
    assert harmonic_channel.label == channel.label
    assert len(harmonic_channel.data) == len(channel.data)
    # Verify the result is not identical to the input (should be processed)
    assert not np.array_equal(harmonic_channel.data, channel.data)


def test_channel_hpss_percussive_basic() -> None:
    """
    Test the basic functionality of the hpss_percussive method.
    """
    # Create a test signal with both harmonic and percussive components
    sampling_rate = 1000
    t = np.linspace(0, 1, sampling_rate, endpoint=False)
    # Harmonic component (sine wave)
    harmonic = np.sin(2 * np.pi * 10 * t)
    # Percussive component (short impulses)
    percussive = np.zeros_like(t)
    percussive[::100] = 1.0  # Impulses every 100 samples
    # Combined signal
    data = harmonic + percussive

    channel = Channel(data=data, sampling_rate=sampling_rate, label="Mixed")

    # Call hpss_percussive with default parameters
    percussive_channel = channel.hpss_percussive()

    # Basic assertions
    assert isinstance(percussive_channel, Channel)
    assert percussive_channel.sampling_rate == sampling_rate
    assert percussive_channel.label == channel.label
    assert len(percussive_channel.data) == len(channel.data)
    # Verify the result is not identical to the input (should be processed)
    assert not np.array_equal(percussive_channel.data, channel.data)


def test_channel_hpss_parameter_passing(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that parameters are correctly passed to the HPSS functions.
    Uses monkeypatch to replace the actual implementation and verify parameter passing.
    """
    from wandas.core import channel_processing

    # Track calls to the functions
    harmonic_calls = []
    percussive_calls = []

    # Create mock versions of the channel_processing functions
    def mock_apply_hpss_harmonic(
        ch: "Channel", **kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        harmonic_calls.append((ch, kwargs))
        return {"data": ch.data}  # Return minimal result

    def mock_apply_hpss_percussive(
        ch: "Channel", **kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        percussive_calls.append((ch, kwargs))
        return {"data": ch.data}  # Return minimal result

    # Replace the actual functions with our mocks
    monkeypatch.setattr(
        channel_processing, "apply_hpss_harmonic", mock_apply_hpss_harmonic
    )
    monkeypatch.setattr(
        channel_processing, "apply_hpss_percussive", mock_apply_hpss_percussive
    )

    # Create a simple channel
    sampling_rate = 1000
    data = np.sin(2 * np.pi * 10 * np.linspace(0, 1, sampling_rate, endpoint=False))
    channel = Channel(data=data, sampling_rate=sampling_rate, label="Test")

    # Custom parameter values to verify passing
    test_params: dict[str, Any] = {
        "kernel_size": 51,
        "power": 3.0,
        "mask": True,
        "margin": 2.0,
        "n_fft": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "window": "hamming",
        "center": False,
        "pad_mode": "reflect",
    }

    # Call the methods with the custom parameters
    channel.hpss_harmonic(**test_params)
    channel.hpss_percussive(**test_params)

    # Verify that each function was called once
    assert len(harmonic_calls) == 1
    assert len(percussive_calls) == 1

    # Verify that the channel was passed correctly
    assert harmonic_calls[0][0] is channel
    assert percussive_calls[0][0] is channel

    # Verify that all parameters were passed correctly
    for param_name, param_value in test_params.items():
        assert harmonic_calls[0][1][param_name] == param_value
        assert percussive_calls[0][1][param_name] == param_value
