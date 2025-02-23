# tests/utils/test_generate_sample.py

from wandas.core.channel import Channel
from wandas.core.channel_frame import ChannelFrame
from wandas.utils.generate_sample import generate_sample


def test_generate_sample_single_frequency() -> None:
    freq = 1000.0
    sampling_rate = 16000
    duration = 1.0
    signal = generate_sample(
        freqs=freq, sampling_rate=sampling_rate, duration=duration, label="Test Signal"
    )

    assert isinstance(signal, ChannelFrame)
    assert signal.label == "Test Signal"
    assert len(signal.channels) == 1
    channel = signal.channels[0]
    assert isinstance(channel, Channel)
    assert channel.sampling_rate == sampling_rate
    assert channel.label == "Channel 1"
    assert len(channel.data) == int(sampling_rate * duration)


def test_generate_sample_multiple_frequencies() -> None:
    freqs = [500.0, 800.0, 1000.0]
    sampling_rate = 16000
    duration = 1.0
    signal = generate_sample(
        freqs=freqs, sampling_rate=sampling_rate, duration=duration, label="Test Signal"
    )

    assert isinstance(signal, ChannelFrame)
    assert signal.label == "Test Signal"
    assert len(signal.channels) == len(freqs)
    for idx, channel in enumerate(signal.channels):
        assert isinstance(channel, Channel)
        assert channel.sampling_rate == sampling_rate
        assert channel.label == f"Channel {idx + 1}"
        assert len(channel.data) == int(sampling_rate * duration)
