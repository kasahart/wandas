# tests/io/test_wav_io.py
import os
from typing import Any, cast

import numpy as np
import pytest
from scipy.io import wavfile

from wandas.core.channel import Channel
from wandas.core.channel_frame import ChannelFrame
from wandas.io import read_wav
from wandas.io.wav_io import write_wav
from wandas.utils.types import NDArrayReal


@pytest.fixture  # type: ignore [misc, unused-ignore]
def create_test_wav(tmpdir: str) -> str:
    """
    テスト用の一時的な WAV ファイルを作成するフィクスチャ。
    テスト後に自動で削除されます。
    """
    # 一時ディレクトリに WAV ファイルを作成
    filename = os.path.join(tmpdir, "test_file.wav")

    # サンプルデータを作成
    sampling_rate = 44100
    duration = 1.0  # 1秒

    # 左右に振幅差をつけた直流データを生成
    data_left = (
        np.ones(int(sampling_rate * duration)) * 0.5
    )  # 左チャンネル (直流信号、振幅0.5)
    data_right = np.ones(
        int(sampling_rate * duration)
    )  # 右チャンネル (直流信号、振幅1.0)

    stereo_data = np.column_stack((data_left, data_right))

    # WAV ファイルを書き出し
    wavfile.write(filename, sampling_rate, stereo_data)

    return filename


def test_read_wav(create_test_wav: str) -> None:
    # テスト用の WAV ファイルを読み込む
    signal = read_wav(create_test_wav)

    # チャンネル数の確認
    assert len(signal._channels) == 2

    # サンプリングレートの確認
    assert signal.sampling_rate == 44100

    # チャンネルデータの確認
    assert np.allclose(signal._channels[0].data, 0.5)
    assert np.allclose(signal._channels[1].data, 1.0)
    # tests/io/test_wav_io.py


@pytest.fixture  # type: ignore [misc, unused-ignore]
def create_stereo_wav(tmpdir: str) -> str:
    """
    Create a temporary stereo WAV file for testing.
    """
    filepath = os.path.join(tmpdir, "stereo_test.wav")
    sampling_rate = 44100
    duration = 1.0  # seconds
    num_samples = int(sampling_rate * duration)
    # Create left and right channels
    data_left = np.full(num_samples, 0.5)
    data_right = np.full(num_samples, 1.0)
    stereo_data = np.column_stack((data_left, data_right))
    wavfile.write(filepath, sampling_rate, stereo_data)
    return filepath


@pytest.fixture  # type: ignore [misc, unused-ignore]
def create_mono_wav(tmpdir: str) -> str:
    """
    Create a temporary mono WAV file for testing.
    """
    filepath = os.path.join(tmpdir, "mono_test.wav")
    sampling_rate = 22050
    duration = 1.0  # seconds
    num_samples = int(sampling_rate * duration)
    # Create mono channel data
    mono_data = np.full(num_samples, 0.75)
    wavfile.write(filepath, sampling_rate, mono_data)
    return filepath


def test_read_wav_default(create_stereo_wav: str) -> None:
    """
    Test reading a default stereo WAV file without specifying labels.
    """
    channel_frame = read_wav(create_stereo_wav)
    # Assert two channels are present
    assert len(channel_frame._channels) == 2
    # Assert sampling rate
    assert channel_frame.sampling_rate == 44100
    # Assert channel data: each channel should be an array with constant values.
    # Since data is written as full arrays, test the first value in each channel.
    np.testing.assert_allclose(channel_frame._channels[0].data[0], 0.5, rtol=1e-5)
    np.testing.assert_allclose(channel_frame._channels[1].data[0], 1.0, rtol=1e-5)


def test_read_wav_mono(create_mono_wav: str) -> None:
    """
    Test reading a mono WAV file.
    """
    channel_frame = read_wav(create_mono_wav)
    # Assert one channel is present
    assert len(channel_frame._channels) == 1
    # Assert sampling rate
    assert channel_frame.sampling_rate == 22050
    # Check that the mono channel data is as expected
    np.testing.assert_allclose(channel_frame._channels[0].data[0], 0.75, rtol=1e-5)


def test_read_wav_with_labels(tmpdir: str) -> None:
    """
    Test reading a stereo WAV file and verifying provided labels are used.
    """
    filepath = os.path.join(tmpdir, "stereo_label_test.wav")
    sampling_rate = 48000
    duration = 1.0  # seconds
    num_samples = int(sampling_rate * duration)
    # Create stereo data
    data_left = np.full(num_samples, 0.3)
    data_right = np.full(num_samples, 0.8)
    stereo_data = np.column_stack((data_left, data_right))
    wavfile.write(filepath, sampling_rate, stereo_data)

    labels = ["Left Channel", "Right Channel"]
    channel_frame = read_wav(filepath, labels=labels)
    # Assert labels are set correctly
    assert channel_frame._channels[0].label == "Left Channel"
    assert channel_frame._channels[1].label == "Right Channel"
    # tests/io/test_wav_io.py


# Dummy classes to simulate Channel and ChannelFrame
class DummyChannel(Channel):
    def __init__(self, data: NDArrayReal, sampling_rate: int, label: str = ""):
        super().__init__(data=data, sampling_rate=sampling_rate, label=label)


class DummyChannelFrame(ChannelFrame):
    def __init__(self, channels: list[Channel], sampling_rate: int, label: str = ""):
        self._channels = channels
        self.sampling_rate = sampling_rate
        self.label = label


def test_write_wav_channel(tmpdir: str) -> None:
    """
    Test write_wav using a DummyChannel.
    The channel data is scaled to 16-bit integers.
    For a constant data array 0.5, the scaling should result in maximum value (32767).
    """
    sampling_rate = 44100
    num_samples = 1000
    # Create constant data array with value 0.5
    data = np.full(num_samples, 0.5, dtype=np.float32)
    channel = DummyChannel(data=data, sampling_rate=sampling_rate, label="Test Channel")

    # Write wav file using write_wav
    out_file = os.path.join(tmpdir, "dummy_channel.wav")
    write_wav(out_file, channel)

    # Read wav file and verify sampling rate and data
    sr, wav_data = wavfile.read(out_file)
    assert sr == sampling_rate
    # Since the original max is 0.5, scaling should be: (0.5/0.5)*32767 = 32767.
    expected = np.int16(np.full(num_samples, 32767))
    np.testing.assert_array_equal(wav_data, expected)


def test_write_wav_invalid_target(tmpdir: str) -> None:
    """
    Test that write_wav raises ValueError when target is neither a Channel
    nor a ChannelFrame.
    """
    out_file = os.path.join(tmpdir, "invalid_target.wav")
    invalid_target = {"data": np.array([0.1, 0.2])}
    with pytest.raises(ValueError):
        write_wav(out_file, cast(Any, invalid_target))


def test_write_wav_channel_frame(tmpdir: str) -> None:
    """
    Test writing a ChannelFrame with multiple channels.
    Expected behavior:
    - Creates a folder named after the filename (minus extension).
    - Writes each channel to its own WAV file (named by each channel's label).
    - Scales all channel data consistently based on the same global norm.
    """
    sampling_rate = 48000
    num_samples = 1000
    data_left = np.full(num_samples, 0.3, dtype=np.float32)
    data_right = np.full(num_samples, 0.6, dtype=np.float32)
    ch_left = Channel(data=data_left, sampling_rate=sampling_rate, label="Left")
    ch_right = Channel(data=data_right, sampling_rate=sampling_rate, label="Right")

    # Create a ChannelFrame with two channels
    channel_frame = ChannelFrame(channels=[ch_left, ch_right], label="StereoFrame")

    # Define output filename, no extension
    out_filename = os.path.join(tmpdir, "test_stereoframe")
    write_wav(out_filename, channel_frame)

    # After writing, a folder named "test_stereoframe" is created
    expected_dir = os.path.splitext(out_filename)[0]
    assert os.path.isdir(expected_dir)

    # Check that each channel file is written
    left_file = os.path.join(tmpdir, "test_stereoframe", "Left.wav")
    right_file = os.path.join(tmpdir, "test_stereoframe", "Right.wav")
    assert os.path.isfile(left_file)
    assert os.path.isfile(right_file)

    # Verify sampling rate and data scaling for each channel
    sr_left, wav_left = wavfile.read(left_file)
    sr_right, wav_right = wavfile.read(right_file)
    assert sr_left == sampling_rate
    assert sr_right == sampling_rate

    # Both channels should scale using the same norm, which is max(0.3, 0.6) = 0.6
    # For data=0.3, scaled to (0.3 / 0.6)*32767 = ~16383, for data=0.6 => ~32767
    np.testing.assert_array_equal(wav_left, np.full(num_samples, 16383, dtype=np.int16))
    np.testing.assert_array_equal(
        wav_right, np.full(num_samples, 32767, dtype=np.int16)
    )
