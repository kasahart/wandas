# tests/core/test_frequency_channel.py

import numpy as np
from wandas.core.channel import Channel
from wandas.core.frequency_channel import FrequencyChannel


def test_frequency_channel_initialization():
    data = np.array([10, 9, 8, 7, 6])
    sampling_rate = 1000
    n_fft = 1024
    window = np.hanning(5)
    norm = "forward"
    label = "Test Spectrum"
    unit = "V"
    calibration_value = 1
    metadata = {"note": "Test metadata"}

    freq_channel = FrequencyChannel(
        data=data,
        sampling_rate=sampling_rate,
        n_fft=n_fft,
        window=window,
        norm=norm,
        label=label,
        unit=unit,
        metadata=metadata,
    )

    assert np.array_equal(freq_channel._data, data)
    assert freq_channel.sampling_rate == sampling_rate
    assert freq_channel.n_fft == n_fft
    assert np.array_equal(freq_channel.window, window)
    assert freq_channel.norm == norm
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
