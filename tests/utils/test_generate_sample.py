# tests/utils/test_generate_sample.py

from wandas.frames.channel import ChannelFrame
from wandas.utils.generate_sample import generate_sin


def test_generate_sin_single_frequency() -> None:
    freq = 1000.0
    sampling_rate = 16000
    duration = 1.0
    signal = generate_sin(
        freqs=freq, sampling_rate=sampling_rate, duration=duration, label="Test Signal"
    )

    assert isinstance(signal, ChannelFrame)
    assert signal.label == "Test Signal"
    assert len(signal) == 1

    # チャンネルのラベルを確認
    assert signal.channels[0].label == "Channel 1"

    # データ長を確認
    computed_data = signal.compute()
    assert computed_data.shape[1] == int(sampling_rate * duration)


def test_generate_sin_multiple_frequencies() -> None:
    freqs = [500.0, 800.0, 1000.0]
    sampling_rate = 16000
    duration = 1.0
    signal = generate_sin(
        freqs=freqs, sampling_rate=sampling_rate, duration=duration, label="Test Signal"
    )

    assert isinstance(signal, ChannelFrame)
    assert signal.label == "Test Signal"
    assert len(signal) == len(freqs)

    # 各チャンネルの確認
    for idx, channel in enumerate(signal.channels):
        assert channel.label == f"Channel {idx + 1}"

    # データ長を確認
    computed_data = signal.compute()
    assert computed_data.shape[1] == int(sampling_rate * duration)
