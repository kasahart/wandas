# tests/io/test_wav_io.py
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.io import wavfile

from wandas.frames.channel import ChannelFrame
from wandas.io import read_wav, write_wav


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
    assert len(signal) == 2

    # サンプリングレートの確認
    assert signal.sampling_rate == 44100

    # チャンネルデータの確認 - 新しいAPIに合わせて変更
    computed_data = signal.compute()
    assert np.allclose(computed_data[0], 0.5)
    assert np.allclose(computed_data[1], 1.0)


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
    assert len(channel_frame) == 2
    # Assert sampling rate
    assert channel_frame.sampling_rate == 44100
    # Assert channel data: each channel should be an array with constant values.
    # Since data is written as full arrays, test the first value in each channel.
    computed_data = channel_frame.compute()
    np.testing.assert_allclose(computed_data[0][0], 0.5, rtol=1e-5)
    np.testing.assert_allclose(computed_data[1][0], 1.0, rtol=1e-5)


def test_read_wav_mono(create_mono_wav: str) -> None:
    """
    Test reading a mono WAV file.
    """
    channel_frame = read_wav(create_mono_wav)
    # Assert one channel is present
    assert len(channel_frame) == 1
    # Assert sampling rate
    assert channel_frame.sampling_rate == 22050
    # Check that the mono channel data is as expected
    computed_data = channel_frame.compute()
    np.testing.assert_allclose(computed_data[0][0], 0.75, rtol=1e-5)


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
    assert channel_frame.channels[0].label == "Left Channel"
    assert channel_frame.channels[1].label == "Right Channel"


def test_read_wav_from_url():
    """
    Test reading a WAV file from a URL.
    """
    url = "https://example.com/test.wav"

    # Create mock response for requests.get
    mock_response = MagicMock()

    # Set up mock WAV data (similar to our test WAV files)
    sampling_rate = 44100
    duration = 0.1  # 0.1 seconds to keep it small
    num_samples = int(sampling_rate * duration)
    data_left = np.full(num_samples, 0.5)
    data_right = np.full(num_samples, 1.0)
    stereo_data = np.column_stack((data_left, data_right))

    # Create a BytesIO object with the WAV data
    import io

    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, sampling_rate, stereo_data)
    wav_buffer.seek(0)  # Reset buffer position

    # Set the mock response content
    mock_response.content = wav_buffer.read()

    # Patch requests.get to return our mock response
    with patch("requests.get", return_value=mock_response):
        # Call read_wav with a URL
        channel_frame = read_wav(url)

        # Verify the result
        assert len(channel_frame) == 2  # Should have 2 channels
        assert channel_frame.sampling_rate == 44100
        computed_data = channel_frame.compute()
        np.testing.assert_allclose(computed_data[0][0], 0.5, rtol=1e-5)
        np.testing.assert_allclose(computed_data[1][0], 1.0, rtol=1e-5)
        assert (
            channel_frame.label == "test.wav"
        )  # Filename should be extracted from URL


def test_write_wav(tmpdir: str):
    """
    Test writing a ChannelFrame to a WAV file.
    """
    # Create a simple ChannelFrame
    sampling_rate = 44100
    duration = 0.1  # seconds
    num_samples = int(sampling_rate * duration)
    data = np.array([np.full(num_samples, 0.5), np.full(num_samples, 0.8)])

    channel_frame = ChannelFrame.from_numpy(
        data=data,
        sampling_rate=sampling_rate,
        label="test_frame",
        ch_labels=["Left", "Right"],
    )

    # Write to WAV file
    output_path = os.path.join(tmpdir, "output_test.wav")
    write_wav(output_path, channel_frame)

    # Verify the file was written correctly by reading it back
    sr, wav_data = wavfile.read(output_path)
    assert sr == sampling_rate
    assert wav_data.shape == (num_samples, 2)

    # Create a new ChannelFrame from the WAV file
    new_frame = read_wav(output_path)

    # Verify basic properties
    assert new_frame.sampling_rate == channel_frame.sampling_rate
    assert new_frame.shape == channel_frame.shape

    # WAV書き込みでは浮動小数点数が整数にスケーリングされるため、
    # 相対的な関係を検証する（第1チャンネルと第2チャンネルの比率）
    computed_data = new_frame.compute()

    np.testing.assert_allclose(computed_data, wav_data.T, rtol=1e-2)


def test_write_wav_invalid_input():
    """
    Test that write_wav raises an error when given invalid input.
    """
    with pytest.raises(ValueError, match="target must be a ChannelFrame object."):
        write_wav("test.wav", "not_a_channel_frame")
