# tests/core/test_spectrum.py

import pytest
import numpy as np
from wandas.core.frequency_channel import FrequencyChannel
from wandas.core.spectrum import Spectrum


def test_spectrum_initialization():
    frequencies = np.array([0, 1, 2, 3, 4])
    amplitudes1 = np.array([10, 9, 8, 7, 6])
    amplitudes2 = np.array([5, 4, 3, 2, 1])
    freq_channel1 = FrequencyChannel(
        frequencies=frequencies, data=amplitudes1, label="Channel 1"
    )
    freq_channel2 = FrequencyChannel(
        frequencies=frequencies, data=amplitudes2, label="Channel 2"
    )

    spectrum = Spectrum(channels=[freq_channel1, freq_channel2], label="Test Spectrum")

    assert spectrum.label == "Test Spectrum"
    assert len(spectrum.channels) == 2
    assert spectrum.channels[0] == freq_channel1
    assert spectrum.channels[1] == freq_channel2
