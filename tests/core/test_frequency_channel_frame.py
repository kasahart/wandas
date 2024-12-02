# tests/core/test_frequency_channel_frame.py

import numpy as np
from wandas.core.frequency_channel import FrequencyChannel
from wandas.core.frequency_channel_frame import FrequencyChannelFrame


def test_spectrum_initialization():
    data1 = np.array([10, 9, 8, 7, 6])
    data2 = np.array([5, 4, 3, 2, 1])
    sampling_rate = 1000
    n_fft = 1024
    window = np.hanning(5)
    norm = "forward"
    label = "Test Spectrum"
    unit = "V"
    metadata = {"note": "Test metadata"}

    freq_channel1 = FrequencyChannel(
        data=data1,
        sampling_rate=sampling_rate,
        n_fft=n_fft,
        window=window,
        label=label,
        unit=unit,
        metadata=metadata,
    )
    freq_channel2 = FrequencyChannel(
        data=data2,
        sampling_rate=sampling_rate,
        n_fft=n_fft,
        window=window,
        label=label,
        unit=unit,
        metadata=metadata,
    )

    spectrum = FrequencyChannelFrame(
        channels=[freq_channel1, freq_channel2], label="Test Spectrum"
    )

    assert spectrum.label == "Test Spectrum"
    assert len(spectrum.channels) == 2
    assert spectrum.channels[0] == freq_channel1
    assert spectrum.channels[1] == freq_channel2
