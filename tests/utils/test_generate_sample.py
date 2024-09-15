# tests/utils/test_generate_sample.py

import pytest
import numpy as np
from wandas.utils.generate_sample import generate_sample
from wandas.core.signal import Signal
from wandas.core.channel import Channel


def test_generate_sample_single_frequency():
    freq = 440
    sampling_rate = 8000
    duration = 1.0
    signal = generate_sample(
        freqs=freq, sampling_rate=sampling_rate, duration=duration, label="Test Signal"
    )

    assert isinstance(signal, Signal)
    assert signal.label == "Test Signal"
    assert len(signal.channels) == 1
    channel = signal.channels[0]
    assert isinstance(channel, Channel)
    assert channel.sampling_rate == sampling_rate
    assert channel.label == "Channel 1"
    assert len(channel.data) == int(sampling_rate * duration)


def test_generate_sample_multiple_frequencies():
    freqs = [440, 880, 1760]
    sampling_rate = 8000
    duration = 1.0
    signal = generate_sample(
        freqs=freqs, sampling_rate=sampling_rate, duration=duration, label="Test Signal"
    )

    assert isinstance(signal, Signal)
    assert signal.label == "Test Signal"
    assert len(signal.channels) == len(freqs)
    for idx, channel in enumerate(signal.channels):
        assert isinstance(channel, Channel)
        assert channel.sampling_rate == sampling_rate
        assert channel.label == f"Channel {idx + 1}"
        assert len(channel.data) == int(sampling_rate * duration)
