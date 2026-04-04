# tests/io/test_wav_io.py
import io
import os
from typing import BinaryIO, cast
from unittest.mock import MagicMock, patch

import dask.array
import numpy as np
import pytest
from scipy.io import wavfile

from wandas.frames.channel import ChannelFrame
from wandas.io import write_wav


def test_read_wav_lazy_loading(create_test_wav) -> None:
    """Verify WAV load produces a Dask array (Pillar 1: lazy evaluation).

    After loading a WAV file, cf._data must be a dask.array.core.Array
    instance, confirming data is not eagerly loaded into memory.
    """
    wav_path = create_test_wav(sr=16000, n_channels=2, n_samples=1600)
    cf = ChannelFrame.read_wav(str(wav_path))
    assert isinstance(cf._data, dask.array.core.Array), f"Expected Dask array after WAV load, got {type(cf._data)}"


def test_read_wav_stereo_dc_signal(tmp_path) -> None:
    """Test reading a stereo WAV with known DC signals.

    Writes float64 DC signals (0.5 left, 1.0 right) via scipy and reads back
    via ChannelFrame.read_wav. Verifies channel count, sampling rate, and
    data values against the analytically known DC levels.
    """
    filepath = tmp_path / "stereo_dc.wav"
    sr = 44100
    n_samples = sr  # 1 second
    data_left = np.full(n_samples, 0.5)
    data_right = np.full(n_samples, 1.0)
    stereo_data = np.column_stack((data_left, data_right))
    wavfile.write(str(filepath), sr, stereo_data)

    cf = ChannelFrame.read_wav(str(filepath))

    assert len(cf) == 2
    assert cf.sampling_rate == sr
    computed = cf.compute()
    # DC signal values must be preserved exactly through float64 WAV round-trip
    np.testing.assert_allclose(computed[0], 0.5, rtol=1e-7, err_msg="Left channel DC level mismatch")
    np.testing.assert_allclose(computed[1], 1.0, rtol=1e-7, err_msg="Right channel DC level mismatch")


def test_read_wav_stereo_channel_count(create_test_wav) -> None:
    """Test that a stereo WAV file produces n_channels == 2."""
    wav_path = create_test_wav(sr=44100, n_channels=2, n_samples=4410)
    cf = ChannelFrame.read_wav(str(wav_path))
    assert len(cf) == 2
    assert cf.sampling_rate == 44100


def test_read_wav_mono_channel_count(create_test_wav) -> None:
    """Test that a mono WAV file produces n_channels == 1."""
    wav_path = create_test_wav(sr=22050, n_channels=1, n_samples=2205)
    cf = ChannelFrame.read_wav(str(wav_path))
    assert len(cf) == 1
    assert cf.sampling_rate == 22050


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
    channel_frame = ChannelFrame.read_wav(filepath, labels=labels)
    # Assert labels are set correctly
    assert channel_frame.channels[0].label == "Left Channel"
    assert channel_frame.channels[1].label == "Right Channel"


def test_read_wav_bytes() -> None:
    """
    Test reading a WAV file from in-memory bytes.
    """
    sampling_rate = 32000
    duration = 0.1
    num_samples = int(sampling_rate * duration)
    data_left = np.full(num_samples, 0.25, dtype=np.float32)
    data_right = np.full(num_samples, 0.75, dtype=np.float32)
    stereo_data = np.column_stack((data_left, data_right))

    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, sampling_rate, stereo_data)
    wav_bytes = wav_buffer.getvalue()

    channel_frame = ChannelFrame.read_wav(wav_bytes)

    assert channel_frame.sampling_rate == sampling_rate
    assert len(channel_frame) == 2
    computed_data = channel_frame.compute()
    np.testing.assert_allclose(computed_data[0], data_left, rtol=1e-5)
    np.testing.assert_allclose(computed_data[1], data_right, rtol=1e-5)


def test_read_wav_from_url() -> None:
    """
    Test reading a WAV file from a URL via ChannelFrame.read_wav.

    Downloads the WAV content (mocked here) and passes the bytes to
    ChannelFrame.read_wav, which is the expected usage pattern when
    loading from a URL.
    """
    url = "https://example.com/test.wav"

    sampling_rate = 44100
    duration = 0.1  # 0.1 seconds to keep it small
    num_samples = int(sampling_rate * duration)
    data_left = np.full(num_samples, 0.5)
    data_right = np.full(num_samples, 1.0)
    stereo_data = np.column_stack((data_left, data_right))

    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, sampling_rate, stereo_data)
    wav_bytes = wav_buffer.getvalue()

    # Simulate downloading the URL content then reading via ChannelFrame.read_wav
    mock_response = MagicMock()
    mock_response.content = wav_bytes
    with patch("requests.get", return_value=mock_response) as mock_get:
        import requests

        response = requests.get(url)
        channel_frame = ChannelFrame.read_wav(response.content)

    mock_get.assert_called_once_with(url)
    assert len(channel_frame) == 2
    assert channel_frame.sampling_rate == 44100
    computed_data = channel_frame.compute()
    np.testing.assert_allclose(computed_data[0][0], 0.5, rtol=1e-5)
    np.testing.assert_allclose(computed_data[1][0], 1.0, rtol=1e-5)


def test_from_file_url_wav() -> None:
    """
    Test that ChannelFrame.from_file transparently downloads a WAV URL.

    urllib.request.urlopen is mocked so no real network call is made.
    The test verifies that:
    - The URL is passed to urlopen
    - The returned bytes are read as a WAV ChannelFrame
    - Sampling rate and channel data are correct
    """
    url = "https://example.com/audio/sample.wav"

    sampling_rate = 22050
    num_samples = 100
    data_left = np.full(num_samples, 0.3, dtype=np.float32)
    data_right = np.full(num_samples, 0.7, dtype=np.float32)
    stereo_data = np.column_stack((data_left, data_right))

    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, sampling_rate, stereo_data)
    wav_bytes = wav_buffer.getvalue()

    # Build a mock response context manager that returns the WAV bytes
    mock_resp = MagicMock()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read = MagicMock(return_value=wav_bytes)

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        channel_frame = ChannelFrame.from_file(url)

    mock_urlopen.assert_called_once_with(url, timeout=10.0)
    assert channel_frame.sampling_rate == sampling_rate
    assert len(channel_frame) == 2
    computed = channel_frame.compute()
    np.testing.assert_allclose(computed[0], data_left, rtol=1e-5)
    np.testing.assert_allclose(computed[1], data_right, rtol=1e-5)
    # URL should be recorded as the source for provenance
    assert channel_frame.metadata.source_file == url
    assert channel_frame.label == "sample"


def test_from_file_url_http_scheme() -> None:
    """Test that plain http:// URLs are also handled by from_file."""
    url = "http://example.com/audio/sample.wav"

    sampling_rate = 8000
    num_samples = 50
    mono_data = np.full(num_samples, 0.5, dtype=np.float32)

    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, sampling_rate, mono_data)
    wav_bytes = wav_buffer.getvalue()

    mock_resp = MagicMock()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read = MagicMock(return_value=wav_bytes)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        channel_frame = ChannelFrame.from_file(url)

    assert channel_frame.sampling_rate == sampling_rate
    assert len(channel_frame) == 1


def test_from_file_url_with_query_string() -> None:
    """Test that URL query strings don't break extension inference."""
    url = "https://example.com/audio/sample.wav?token=abc123&foo=bar"

    sampling_rate = 16000
    num_samples = 50
    mono_data = np.full(num_samples, 0.4, dtype=np.float32)

    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, sampling_rate, mono_data)
    wav_bytes = wav_buffer.getvalue()

    mock_resp = MagicMock()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read = MagicMock(return_value=wav_bytes)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        channel_frame = ChannelFrame.from_file(url)

    assert channel_frame.sampling_rate == sampling_rate
    assert len(channel_frame) == 1


def test_from_file_url_download_failure() -> None:
    """Test that a URL download failure raises OSError with a clear message."""
    import urllib.error

    url = "https://example.com/audio/sample.wav"

    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        with pytest.raises(OSError, match=r"Failed to download audio from URL"):
            ChannelFrame.from_file(url)


def test_read_wav_stream_nonseekable() -> None:
    """Test reading a WAV file from a non-seekable stream via ChannelFrame.read_wav."""
    sampling_rate = 22050
    num_samples = 100
    data_left = np.full(num_samples, 0.1, dtype=np.float32)
    data_right = np.full(num_samples, 0.3, dtype=np.float32)
    stereo_data = np.column_stack((data_left, data_right))

    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, sampling_rate, stereo_data)
    wav_bytes = wav_buffer.getvalue()

    class NonSeekableStream:
        def __init__(self, content: bytes, name: str) -> None:
            self.name = name
            self._content = content

        def read(self, *_args: object, **_kwargs: object) -> bytes:
            return self._content

        def seek(self, *_args: object, **_kwargs: object) -> None:
            raise OSError("seek not supported")

    stream = NonSeekableStream(content=wav_bytes, name="dir/my_audio.wav")

    channel_frame = ChannelFrame.read_wav(cast(BinaryIO, stream))

    assert channel_frame.sampling_rate == sampling_rate
    assert len(channel_frame) == 2
    computed_data = channel_frame.compute()
    np.testing.assert_allclose(computed_data[0], data_left, rtol=1e-5)
    np.testing.assert_allclose(computed_data[1], data_right, rtol=1e-5)


def test_read_wav_int16_raw(tmpdir: str) -> None:
    """Test that int16 WAV data is returned cast to float32 by default.

    The default normalize=False returns scipy's raw integer samples cast to float32.
    The integer values are preserved in magnitude (e.g. 16384 becomes 16384.0).
    """
    filepath = os.path.join(tmpdir, "int16_test.wav")
    sampling_rate = 16000
    num_samples = 100
    int16_left = np.full(num_samples, 16384, dtype=np.int16)
    int16_right = np.full(num_samples, -16384, dtype=np.int16)
    stereo_data = np.column_stack((int16_left, int16_right))
    wavfile.write(filepath, sampling_rate, stereo_data)

    channel_frame = ChannelFrame.read_wav(filepath)
    computed_data = channel_frame.compute()

    # Raw int16 values are cast to float32 (magnitudes preserved)
    np.testing.assert_array_equal(computed_data[0], int16_left.astype(np.float32))
    np.testing.assert_array_equal(computed_data[1], int16_right.astype(np.float32))
    assert computed_data.dtype == np.float32


def test_read_wav_int16_normalized(tmpdir: str) -> None:
    """Test that int16 WAV data is normalized to float32 [-1.0, 1.0] with normalize=True."""
    filepath = os.path.join(tmpdir, "int16_norm_test.wav")
    sampling_rate = 16000
    num_samples = 100
    int16_left = np.full(num_samples, 16384, dtype=np.int16)  # ≈ 0.5 after normalization
    int16_right = np.full(num_samples, -16384, dtype=np.int16)  # ≈ -0.5 after normalization
    stereo_data = np.column_stack((int16_left, int16_right))
    wavfile.write(filepath, sampling_rate, stereo_data)

    channel_frame = ChannelFrame.read_wav(filepath, normalize=True)
    computed_data = channel_frame.compute()

    # After normalization (dividing by 32768), values should be ≈ [0.5, -0.5]
    np.testing.assert_allclose(computed_data[0], 0.5, rtol=1e-4)
    np.testing.assert_allclose(computed_data[1], -0.5, rtol=1e-4)
    assert computed_data.dtype == np.float32


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
    new_frame = ChannelFrame.read_wav(output_path)

    # Verify basic properties
    assert new_frame.sampling_rate == channel_frame.sampling_rate
    assert new_frame.shape == channel_frame.shape

    # WAV書き込みでは浮動小数点数が整数にスケーリングされるため、
    # 相対的な関係を検証する（第1チャンネルと第2チャンネルの比率）
    computed_data = new_frame.compute()

    np.testing.assert_allclose(computed_data, wav_data.T, rtol=1e-2)


def test_write_wav_mono_squeezes_data() -> None:
    """Test mono data is squeezed before writing."""
    sampling_rate = 8000
    num_samples = 100
    data = np.full((1, num_samples), 0.5, dtype=float)

    channel_frame = ChannelFrame.from_numpy(
        data=data,
        sampling_rate=sampling_rate,
        label="mono_frame",
        ch_labels=["Mono"],
    )

    with patch("wandas.io.wav_io.sf.write") as mock_write:
        write_wav("dummy.wav", channel_frame)

    args, kwargs = mock_write.call_args
    written_data = args[1]
    assert written_data.ndim == 1
    assert kwargs.get("subtype") == "FLOAT"


def test_write_wav_nonfloat_branch() -> None:
    """Test non-FLOAT branch when data range exceeds 1."""
    sampling_rate = 8000
    num_samples = 100
    data = np.full((2, num_samples), 1.5, dtype=np.float32)

    channel_frame = ChannelFrame.from_numpy(
        data=data,
        sampling_rate=sampling_rate,
        label="loud_frame",
        ch_labels=["Left", "Right"],
    )

    with patch("wandas.io.wav_io.sf.write") as mock_write:
        write_wav("dummy.wav", channel_frame)

    _, kwargs = mock_write.call_args
    assert "subtype" not in kwargs


def test_write_wav_float32_normalized() -> None:
    """Test that float32 data with values in [-1, 1] uses FLOAT subtype."""
    sampling_rate = 8000
    num_samples = 100
    data = np.full((2, num_samples), 0.5, dtype=np.float32)

    channel_frame = ChannelFrame.from_numpy(
        data=data,
        sampling_rate=sampling_rate,
        label="float32_frame",
        ch_labels=["Left", "Right"],
    )

    with patch("wandas.io.wav_io.sf.write") as mock_write:
        write_wav("dummy.wav", channel_frame)

    _, kwargs = mock_write.call_args
    assert kwargs.get("subtype") == "FLOAT"


def test_write_wav_invalid_input():
    """
    Test that write_wav raises an error when given invalid input.
    """
    with pytest.raises(ValueError, match="target must be a ChannelFrame object."):
        write_wav("test.wav", "not_a_channel_frame")  # ty: ignore[invalid-argument-type]
