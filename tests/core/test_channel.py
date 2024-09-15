# tests/core/test_channel.py

import pytest
import numpy as np
from wandas.core.channel import Channel


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
    assert channel.calibration_value is None
    assert channel.metadata == {}


def test_channel_calibration():
    data = np.array([0, 1, 2, 3, 4], dtype=float)
    sampling_rate = 1000
    calibration_value = 2.0
    channel = Channel(
        data=data, sampling_rate=sampling_rate, calibration_value=calibration_value
    )

    expected_data = data * calibration_value
    assert np.allclose(channel.data, expected_data)


def test_channel_low_pass_filter():
    data = np.sin(2 * np.pi * 50 * np.linspace(0, 1, 1000))
    sampling_rate = 1000
    channel = Channel(data=data, sampling_rate=sampling_rate)
    filtered_channel = channel.low_pass_filter(cutoff=30)

    # 簡易的なチェックとして、フィルタ後のデータがフィルタ前と異なることを確認
    assert not np.array_equal(channel.data, filtered_channel.data)


def test_channel_fft():
    data = np.sin(2 * np.pi * 50 * np.linspace(0, 1, 1000, endpoint=False))
    sampling_rate = 1000
    channel = Channel(data=data, sampling_rate=sampling_rate)
    freq_channel = channel.fft()

    # Expected frequencies and amplitudes
    expected_frequencies = np.fft.rfftfreq(len(data), 1 / sampling_rate)
    expected_amplitudes = np.abs(np.fft.rfft(data)) * 2 / len(data)

    # Check if the frequencies and amplitudes are close to the expected values
    assert np.allclose(freq_channel.frequencies, expected_frequencies)
    assert np.allclose(freq_channel.data, expected_amplitudes)

    # Check specific frequency bin (50 Hz)
    freq_bin = np.where(expected_frequencies == 50)[0][0]
    assert np.isclose(freq_channel.data[freq_bin], 1.0, atol=1e-2)


def test_channel_plot():
    import matplotlib.pyplot as plt

    data = np.array([0, 1, 2, 3, 4])
    sampling_rate = 1000
    channel = Channel(
        data=data, sampling_rate=sampling_rate, label="Test Channel", unit="V"
    )

    fig, ax = plt.subplots()
    channel.plot(ax=ax, title="Test Plot")

    assert ax.get_xlabel() == "Time (s)"
    assert ax.get_ylabel() == "Amplitude (V)"
    assert ax.get_title() == "Test Plot"
    assert len(ax.lines) == 1
    assert np.array_equal(ax.lines[0].get_xdata(), np.arange(len(data)) / sampling_rate)
    assert np.array_equal(ax.lines[0].get_ydata(), data)
