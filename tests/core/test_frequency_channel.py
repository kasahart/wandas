# tests/core/test_frequency_channel.py

import pytest
import numpy as np
from wandas.core.frequency_channel import FrequencyChannel


def test_frequency_channel_initialization():
    frequencies = np.array([0, 1, 2, 3, 4])
    amplitudes = np.array([10, 9, 8, 7, 6])
    label = "Test Spectrum"
    unit = "V/Hz"
    fft_params = {"n_fft": 1024, "window": "hann"}
    metadata = {"note": "Test metadata"}

    freq_channel = FrequencyChannel(
        frequencies=frequencies,
        data=amplitudes,
        label=label,
        unit=unit,
        fft_params=fft_params,
        metadata=metadata,
    )

    assert np.array_equal(freq_channel.frequencies, frequencies)
    assert np.array_equal(freq_channel.data, amplitudes)
    assert freq_channel.label == label
    assert freq_channel.unit == unit
    assert freq_channel.fft_params == fft_params
    assert freq_channel.metadata == metadata
