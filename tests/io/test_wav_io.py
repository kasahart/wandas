# tests/io/test_wav_io.py
import io
import tempfile
import urllib.error
from pathlib import Path
from typing import BinaryIO, cast
from unittest.mock import MagicMock, patch

import dask.array
import numpy as np
import pytest
from scipy.io import wavfile

from tests.io_helpers import mock_urlopen_stream
from wandas.frames.channel import ChannelFrame
from wandas.io import readers as io_readers
from wandas.io import write_wav


def _make_wav_bytes(sr: int, data: np.ndarray) -> bytes:
    """Create in-memory WAV bytes from sample data (samples x channels)."""
    buf = io.BytesIO()
    wavfile.write(buf, sr, data)
    return buf.getvalue()


def test_read_wav_lazy_loading(create_test_wav) -> None:
    """Verify WAV load produces a Dask array (Pillar 1: lazy evaluation).

    After loading a WAV file, cf._data must be a dask.array.core.Array
    instance, confirming data is not eagerly loaded into memory.
    """
    wav_path = create_test_wav(sr=16000, n_channels=2, n_samples=1600)
    cf = ChannelFrame.read_wav(str(wav_path))
    assert isinstance(cf._data, dask.array.core.Array), f"Expected Dask array after WAV load, got {type(cf._data)}"


def test_wav_float_roundtrip(known_signal_frame, tmp_path) -> None:
    """Float WAV round-trip: write -> read -> compare (I/O Policy requirement).

    ChannelFrame.to_wav writes IEEE FLOAT when max(abs(data)) <= 1.
    Reading back as canonical float64 returns the exact encoded FLOAT samples.
    """
    wav_path = tmp_path / "float_roundtrip.wav"
    # Capture original data before write to verify immutability (Pillar 1)
    original_data = known_signal_frame.compute().copy()
    known_signal_frame.to_wav(str(wav_path))

    # Verify original frame is unchanged after write (Pillar 1: side-effect free)
    np.testing.assert_array_equal(
        known_signal_frame.compute(), original_data, err_msg="to_wav must not mutate original frame data"
    )

    loaded = ChannelFrame.read_wav(str(wav_path))

    assert loaded.sampling_rate == known_signal_frame.sampling_rate
    assert loaded.n_channels == known_signal_frame.n_channels
    encoded = known_signal_frame.compute().astype(np.float32).astype(np.float64)
    np.testing.assert_array_equal(loaded.compute(), encoded, err_msg="Float WAV round-trip data mismatch")


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
    # Verify Dask lazy loading (Pillar 1)
    assert isinstance(cf._data, dask.array.core.Array), "WAV load must produce Dask array"
    computed = cf.compute()
    # DC signal values must be preserved exactly through float64 WAV round-trip
    np.testing.assert_array_equal(computed[0], data_left, err_msg="Left channel DC level mismatch")
    np.testing.assert_array_equal(computed[1], data_right, err_msg="Right channel DC level mismatch")


def test_read_wav_stereo_channel_count(create_test_wav) -> None:
    """Test that a stereo WAV file produces n_channels == 2."""
    wav_path = create_test_wav(sr=44100, n_channels=2, n_samples=4410)
    cf = ChannelFrame.read_wav(str(wav_path))
    assert len(cf) == 2, f"Expected 2 channels, got {len(cf)}"
    assert cf.sampling_rate == 44100, f"SR mismatch: {cf.sampling_rate}"


def test_read_wav_mono_channel_count(create_test_wav) -> None:
    """Test that a mono WAV file produces n_channels == 1."""
    wav_path = create_test_wav(sr=22050, n_channels=1, n_samples=2205)
    cf = ChannelFrame.read_wav(str(wav_path))
    assert len(cf) == 1, f"Expected 1 channel, got {len(cf)}"
    assert cf.sampling_rate == 22050, f"SR mismatch: {cf.sampling_rate}"


def test_read_wav_custom_labels_propagated(tmp_path) -> None:
    """Test that user-provided labels are correctly assigned to channels (Pillar 2)."""
    filepath = tmp_path / "stereo_label_test.wav"
    sr = 48000
    n_samples = sr  # 1 second
    data_left = np.full(n_samples, 0.3)
    data_right = np.full(n_samples, 0.8)
    stereo_data = np.column_stack((data_left, data_right))
    wavfile.write(str(filepath), sr, stereo_data)

    labels = ["Left Channel", "Right Channel"]
    cf = ChannelFrame.read_wav(str(filepath), labels=labels)
    assert cf.channels[0].label == "Left Channel", f"Label mismatch: {cf.channels[0].label}"
    assert cf.channels[1].label == "Right Channel", f"Label mismatch: {cf.channels[1].label}"


def test_read_wav_bytes_dc_signal() -> None:
    """Test reading a WAV from in-memory bytes with known DC signal.

    Uses FLOAT WAV samples to verify byte-based reads preserve encoded values.
    """
    sr = 32000
    n_samples = 3200  # 0.1 seconds
    data_left = np.full(n_samples, 0.25, dtype=np.float32)
    data_right = np.full(n_samples, 0.75, dtype=np.float32)
    stereo_data = np.column_stack((data_left, data_right))
    wav_bytes = _make_wav_bytes(sr, stereo_data)

    cf = ChannelFrame.read_wav(wav_bytes)

    assert cf.sampling_rate == sr, f"SR mismatch: {cf.sampling_rate}"
    assert len(cf) == 2, f"Expected 2 channels, got {len(cf)}"
    computed = cf.compute()
    np.testing.assert_array_equal(computed[0], data_left)
    np.testing.assert_array_equal(computed[1], data_right)


def test_read_wav_from_url_via_requests_mock() -> None:
    """Test reading WAV bytes obtained from a mocked HTTP response.

    Simulates downloading WAV content via requests.get and passing bytes
    to ChannelFrame.read_wav. Verifies all sample values, not just [0].
    """
    url = "https://example.com/test.wav"
    sr = 44100
    n_samples = 4410  # 0.1 seconds
    data_left = np.full(n_samples, 0.5, dtype=np.float32)
    data_right = np.full(n_samples, 1.0, dtype=np.float32)
    stereo_data = np.column_stack((data_left, data_right))
    wav_bytes = _make_wav_bytes(sr, stereo_data)

    mock_response = MagicMock()
    mock_response.content = wav_bytes
    with patch("requests.get", return_value=mock_response) as mock_get:
        import requests

        response = requests.get(url)
        cf = ChannelFrame.read_wav(response.content)

    mock_get.assert_called_once_with(url)
    assert len(cf) == 2, f"Expected 2 channels, got {len(cf)}"
    assert cf.sampling_rate == sr, f"SR mismatch: {cf.sampling_rate}"
    computed = cf.compute()
    np.testing.assert_array_equal(computed[0], data_left)
    np.testing.assert_array_equal(computed[1], data_right)


def test_from_file_url_wav() -> None:
    """Test from_file with HTTPS WAV URL: data, SR, provenance metadata (Pillar 2).

    Verifies urlopen is called, data matches, and source_file/label are set.
    """
    url = "https://example.com/audio/sample.wav"
    sr = 22050
    n_samples = 100
    data_left = np.full(n_samples, 0.3, dtype=np.float32)
    data_right = np.full(n_samples, 0.7, dtype=np.float32)
    stereo_data = np.column_stack((data_left, data_right))
    wav_bytes = _make_wav_bytes(sr, stereo_data)

    with mock_urlopen_stream(wav_bytes) as mock_fn:
        cf = ChannelFrame.from_file(url)

    mock_fn.assert_called_once_with(url, timeout=10.0)
    assert cf.sampling_rate == sr, f"SR mismatch: {cf.sampling_rate}"
    assert len(cf) == 2, f"Expected 2 channels, got {len(cf)}"
    # Verify Dask lazy loading from URL path (Pillar 1)
    assert isinstance(cf._data, dask.array.core.Array), "URL load must produce Dask array"
    computed = cf.compute()
    np.testing.assert_array_equal(computed[0], data_left)
    np.testing.assert_array_equal(computed[1], data_right)
    # Provenance metadata (Pillar 2)
    assert cf.metadata["_source_file"] == url, "source file metadata not preserved"
    assert cf.label == "sample"


def test_from_file_url_wav_streams_in_chunks() -> None:
    """URL WAV loads must stream in bounded chunks instead of full-response reads."""
    url = "https://example.com/audio/sample.wav"
    sr = 22050
    n_samples = 100
    mono_data = np.full(n_samples, 0.25, dtype=np.float32)
    wav_bytes = _make_wav_bytes(sr, mono_data)

    with mock_urlopen_stream(
        wav_bytes,
        forbid_unbounded_read=True,
        expected_chunk_size=io_readers.URL_DOWNLOAD_CHUNK_SIZE,
    ):
        cf = ChannelFrame.from_file(url)

    assert cf.sampling_rate == sr
    np.testing.assert_array_equal(cf.compute()[0], mono_data)


def test_download_read_is_capped_by_remaining_budget() -> None:
    """Each streamed read is bounded by the remaining limit plus one byte."""
    wav_bytes = b"0123456789"

    with mock_urlopen_stream(wav_bytes, include_content_length=False) as mock_fn:
        with pytest.raises(OSError, match=r"Streaming audio would exceed size limit"):
            io_readers.download_url_to_temporary_file(
                "https://example.com/audio/sample.wav",
                timeout=10.0,
                resource_name="audio",
                max_bytes=5,
                chunk_size=100,
            )

    mock_fn.return_value.read.assert_called_once_with(6)


def test_from_file_url_pcm_wav_preserves_normalized_samples() -> None:
    """URL PCM WAV loads use the canonical full-scale float64 contract."""
    url = "https://example.com/audio/pcm.wav"
    sr = 8000
    pcm_data = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
    wav_bytes = _make_wav_bytes(sr, pcm_data)

    with mock_urlopen_stream(wav_bytes):
        cf = ChannelFrame.from_file(url)

    expected = pcm_data.astype(np.float64) / 32768.0
    np.testing.assert_array_equal(cf.compute()[0], expected)


def test_from_file_url_http_scheme() -> None:
    """Test that plain http:// URLs are also handled by from_file."""
    url = "http://example.com/audio/sample.wav"
    sr = 8000
    n_samples = 50
    mono_data = np.full(n_samples, 0.5, dtype=np.float32)
    wav_bytes = _make_wav_bytes(sr, mono_data)

    with mock_urlopen_stream(wav_bytes):
        cf = ChannelFrame.from_file(url)

    assert cf.sampling_rate == sr
    assert len(cf) == 1


def test_from_file_url_with_query_string() -> None:
    """Test that URL query strings don't break extension inference."""
    url = "https://example.com/audio/sample.wav?token=abc123&foo=bar"
    sr = 16000
    n_samples = 50
    mono_data = np.full(n_samples, 0.4, dtype=np.float32)
    wav_bytes = _make_wav_bytes(sr, mono_data)

    with mock_urlopen_stream(wav_bytes):
        cf = ChannelFrame.from_file(url)

    assert cf.sampling_rate == sr
    assert len(cf) == 1


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


def test_download_url_midstream_failure_cleans_partial_file(monkeypatch, tmp_path) -> None:
    """A connection failure after writing data must remove the partial download."""
    import urllib.error

    temporary_directory = tempfile.TemporaryDirectory
    temporary_directory_factory = MagicMock(side_effect=lambda: temporary_directory(dir=tmp_path))
    temporary_directory_factory.cleanup = temporary_directory.cleanup
    monkeypatch.setattr(
        io_readers.tempfile,
        "TemporaryDirectory",
        temporary_directory_factory,
    )
    mock_resp = MagicMock()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.headers = {}
    mock_resp.read = MagicMock(side_effect=[b"partial WAV data", urllib.error.URLError("connection dropped")])

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(OSError, match=r"Failed to download audio from URL"):
            io_readers.download_url_to_temporary_file(
                "https://example.com/audio/sample.wav",
                timeout=10.0,
                suffix=".wav",
                resource_name="audio",
            )

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("failure", "error", "message"),
    [
        (urllib.error.URLError("wrapper failed"), OSError, "Failed to download audio"),
        (RuntimeError("wrapper failed"), RuntimeError, "wrapper failed"),
    ],
)
def test_download_url_cleans_directory_when_owner_creation_fails(
    monkeypatch,
    tmp_path: Path,
    failure: Exception,
    error: type[Exception],
    message: str,
) -> None:
    """The temporary directory is owned even if its cleanup wrapper cannot be created."""
    temporary_directory = tempfile.TemporaryDirectory
    monkeypatch.setattr(
        io_readers.tempfile,
        "TemporaryDirectory",
        MagicMock(side_effect=lambda: temporary_directory(dir=tmp_path)),
    )
    monkeypatch.setattr(io_readers, "DownloadedTemporaryFile", MagicMock(side_effect=failure))

    with mock_urlopen_stream(b""):
        with pytest.raises(error, match=message):
            io_readers.download_url_to_temporary_file(
                "https://example.com/audio/sample.wav",
                timeout=10.0,
                suffix=".wav",
                resource_name="audio",
            )

    assert list(tmp_path.iterdir()) == []


def test_from_file_url_setup_failure_cleans_download(monkeypatch, tmp_path) -> None:
    """A frame setup error after download must remove the temporary file."""
    temporary_directory = tempfile.TemporaryDirectory
    temporary_directory_factory = MagicMock(side_effect=lambda: temporary_directory(dir=tmp_path))
    temporary_directory_factory.cleanup = temporary_directory.cleanup
    monkeypatch.setattr(io_readers.tempfile, "TemporaryDirectory", temporary_directory_factory)
    wav_bytes = _make_wav_bytes(8000, np.zeros(8, dtype=np.int16))

    with mock_urlopen_stream(wav_bytes):
        with pytest.raises(ValueError, match=r"Channel specification is out of range"):
            ChannelFrame.from_file("https://example.com/audio/sample.wav", channel=1)

    assert list(tmp_path.iterdir()) == []


def test_from_file_url_over_size_limit_raises() -> None:
    """URL WAV loads must stop when streamed bytes exceed the configured limit."""
    url = "https://example.com/audio/sample.wav"
    sr = 16000
    n_samples = 500
    mono_data = np.full(n_samples, 0.5, dtype=np.float32)
    wav_bytes = _make_wav_bytes(sr, mono_data)

    with patch.object(io_readers, "MAX_URL_DOWNLOAD_BYTES", 128):
        with mock_urlopen_stream(wav_bytes, include_content_length=False):
            with pytest.raises(OSError, match=r"Streaming audio would exceed size limit"):
                ChannelFrame.from_file(url)


def test_from_file_url_declared_size_limit_raises_before_streaming() -> None:
    """URL WAV loads must reject oversized Content-Length before streaming data."""
    url = "https://example.com/audio/sample.wav"
    sr = 16000
    n_samples = 500
    mono_data = np.full(n_samples, 0.5, dtype=np.float32)
    wav_bytes = _make_wav_bytes(sr, mono_data)

    with patch.object(io_readers, "MAX_URL_DOWNLOAD_BYTES", 128):
        with mock_urlopen_stream(wav_bytes, include_content_length=True) as mock_fn:
            with pytest.raises(OSError, match=r"Declared size of audio exceeds download limit"):
                ChannelFrame.from_file(url)

    mock_resp = mock_fn.return_value
    mock_resp.read.assert_not_called()


def test_download_url_to_temporary_file_invalid_max_bytes_raises() -> None:
    """Helper must reject non-positive max_bytes before opening the URL."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        with pytest.raises(ValueError, match=r"Download size limit must be greater than zero"):
            io_readers.download_url_to_temporary_file(
                "https://example.com/audio/sample.wav",
                timeout=10.0,
                resource_name="audio",
                max_bytes=0,
            )

    mock_urlopen.assert_not_called()


def test_download_url_to_temporary_file_invalid_chunk_size_raises() -> None:
    """Helper must reject non-positive chunk_size before opening the URL."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        with pytest.raises(ValueError, match=r"Download chunk size must be greater than zero"):
            io_readers.download_url_to_temporary_file(
                "https://example.com/audio/sample.wav",
                timeout=10.0,
                resource_name="audio",
                chunk_size=0,
            )

    mock_urlopen.assert_not_called()


def test_download_url_to_temporary_file_invalid_content_length_raises() -> None:
    """Helper must reject non-numeric Content-Length headers."""
    wav_bytes = _make_wav_bytes(8000, np.full(10, 0.25, dtype=np.float32))

    with mock_urlopen_stream(wav_bytes) as mock_fn:
        mock_fn.return_value.headers = {"Content-Length": "not-a-number"}
        with pytest.raises(OSError, match=r"Invalid Content-Length for audio download"):
            io_readers.download_url_to_temporary_file(
                "https://example.com/audio/sample.wav",
                timeout=10.0,
                resource_name="audio",
            )


def test_download_url_to_temporary_file_negative_content_length_raises() -> None:
    """Helper must reject negative Content-Length headers."""
    wav_bytes = _make_wav_bytes(8000, np.full(10, 0.25, dtype=np.float32))

    with mock_urlopen_stream(wav_bytes) as mock_fn:
        mock_fn.return_value.headers = {"Content-Length": "-1"}
        with pytest.raises(OSError, match=r"Invalid Content-Length for audio download"):
            io_readers.download_url_to_temporary_file(
                "https://example.com/audio/sample.wav",
                timeout=10.0,
                resource_name="audio",
            )


def test_downloaded_temporary_file_context_manager_cleans_up() -> None:
    """DownloadedTemporaryFile must clean up its temp directory on context exit."""
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = Path(temp_dir.name) / "download.wav"

    with io_readers.DownloadedTemporaryFile(path=temp_path, temp_dir=temp_dir) as downloaded:
        downloaded.path.write_bytes(b"wav")
        assert downloaded.path.exists()

    assert not temp_path.parent.exists()


def test_from_file_nonexistent_path_raises_file_not_found(tmp_path) -> None:
    """Verify from_file raises FileNotFoundError for non-existent path (I/O Policy)."""
    nonexistent = tmp_path / "does_not_exist.wav"
    with pytest.raises(FileNotFoundError):
        ChannelFrame.from_file(str(nonexistent))


def test_read_wav_stream_nonseekable() -> None:
    """Test reading WAV from a non-seekable stream preserves data (Pillar 4)."""
    sr = 22050
    n_samples = 100
    data_left = np.full(n_samples, 0.1, dtype=np.float32)
    data_right = np.full(n_samples, 0.3, dtype=np.float32)
    stereo_data = np.column_stack((data_left, data_right))
    wav_bytes = _make_wav_bytes(sr, stereo_data)

    class NonSeekableStream:
        def __init__(self, content: bytes, name: str) -> None:
            self.name = name
            self._content = content

        def read(self, *_args: object, **_kwargs: object) -> bytes:
            return self._content

        def seek(self, *_args: object, **_kwargs: object) -> None:
            raise OSError("seek not supported")

    stream = NonSeekableStream(content=wav_bytes, name="dir/my_audio.wav")
    cf = ChannelFrame.read_wav(cast(BinaryIO, stream))

    assert cf.sampling_rate == sr, f"SR mismatch: {cf.sampling_rate}"
    assert len(cf) == 2, f"Expected 2 channels, got {len(cf)}"
    computed = cf.compute()
    np.testing.assert_array_equal(computed[0], data_left)
    np.testing.assert_array_equal(computed[1], data_right)


def test_read_wav_int16_pcm_is_full_scale_float64(tmp_path) -> None:
    """PCM int16 is decoded with libsndfile's full-scale convention."""
    filepath = tmp_path / "int16_test.wav"
    sr = 16000
    n_samples = 100
    int16_left = np.full(n_samples, 16384, dtype=np.int16)
    int16_right = np.full(n_samples, -16384, dtype=np.int16)
    stereo_data = np.column_stack((int16_left, int16_right))
    wavfile.write(str(filepath), sr, stereo_data)

    cf = ChannelFrame.read_wav(str(filepath))
    computed = cf.compute()

    np.testing.assert_array_equal(computed[0], 0.5)
    np.testing.assert_array_equal(computed[1], -0.5)
    assert computed.dtype == np.float64


def test_write_wav_roundtrip_preserves_shape_and_sr(tmp_path) -> None:
    """Write/read WAV round-trip: shape and sampling rate are preserved.

    Uses DC signals (0.5, 0.8) within [-1, 1] to trigger IEEE FLOAT subtype.
    """
    sr = 44100
    n_samples = 4410  # 0.1 seconds
    data = np.array([np.full(n_samples, 0.5), np.full(n_samples, 0.8)])

    cf = ChannelFrame.from_numpy(
        data=data,
        sampling_rate=sr,
        label="test_frame",
        ch_labels=["Left", "Right"],
    )

    output_path = tmp_path / "output_test.wav"
    write_wav(str(output_path), cf)

    # Verify via scipy directly
    read_sr, wav_data = wavfile.read(str(output_path))
    assert read_sr == sr, f"Sampling rate mismatch: {read_sr} != {sr}"
    assert wav_data.shape == (n_samples, 2), f"WAV shape mismatch: {wav_data.shape}"

    # Verify via ChannelFrame round-trip
    loaded = ChannelFrame.read_wav(str(output_path))
    assert loaded.sampling_rate == sr, f"Loaded SR mismatch: {loaded.sampling_rate}"
    assert loaded.shape == cf.shape, f"Shape mismatch: {loaded.shape} != {cf.shape}"
    # Verify Dask lazy loading (Pillar 1)
    assert isinstance(loaded._data, dask.array.core.Array), "WAV load must produce Dask array"

    computed = loaded.compute()
    np.testing.assert_array_equal(computed, wav_data.T)


def test_write_wav_mono_data_squeezed_to_1d() -> None:
    """Mono WAV write: data is squeezed to 1D before writing (FLOAT subtype)."""
    sr = 8000
    n_samples = 100
    data = np.full((1, n_samples), 0.5, dtype=float)
    cf = ChannelFrame.from_numpy(data=data, sampling_rate=sr, label="mono_frame", ch_labels=["Mono"])

    with patch("wandas.io.wav_io.sf.write") as mock_write:
        write_wav("dummy.wav", cf)

    args, kwargs = mock_write.call_args
    written_data = args[1]
    assert written_data.ndim == 1, "Mono data must be squeezed to 1D"
    assert kwargs.get("subtype") == "FLOAT"


def test_write_wav_data_exceeds_unit_range_no_float_subtype() -> None:
    """Data with max(abs) > 1 should NOT use FLOAT subtype."""
    sr = 8000
    n_samples = 100
    data = np.full((2, n_samples), 1.5, dtype=np.float32)
    cf = ChannelFrame.from_numpy(data=data, sampling_rate=sr, label="loud_frame", ch_labels=["Left", "Right"])

    with patch("wandas.io.wav_io.sf.write") as mock_write:
        write_wav("dummy.wav", cf)

    _, kwargs = mock_write.call_args
    assert "subtype" not in kwargs, "Data exceeding [-1,1] should not use FLOAT subtype"


def test_write_wav_float32_within_unit_range_uses_float_subtype() -> None:
    """Float32 data with max(abs) <= 1 uses IEEE FLOAT subtype."""
    sr = 8000
    n_samples = 100
    data = np.full((2, n_samples), 0.5, dtype=np.float32)
    cf = ChannelFrame.from_numpy(data=data, sampling_rate=sr, label="float32_frame", ch_labels=["Left", "Right"])

    with patch("wandas.io.wav_io.sf.write") as mock_write:
        write_wav("dummy.wav", cf)

    _, kwargs = mock_write.call_args
    assert kwargs.get("subtype") == "FLOAT"


def test_write_wav_invalid_input_raises_value_error() -> None:
    """write_wav with non-ChannelFrame input raises ValueError."""
    with pytest.raises(ValueError, match="target must be a ChannelFrame object."):
        write_wav("test.wav", "not_a_channel_frame")  # ty: ignore[invalid-argument-type]
