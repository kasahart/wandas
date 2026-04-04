# tests/io/test_wav_io.py
import io
from contextlib import contextmanager
from typing import BinaryIO, cast
from unittest.mock import MagicMock, patch

import dask.array
import numpy as np
import pytest
from scipy.io import wavfile

from wandas.frames.channel import ChannelFrame
from wandas.io import write_wav


def _make_wav_bytes(sr: int, data: np.ndarray) -> bytes:
    """Create in-memory WAV bytes from sample data (samples x channels)."""
    buf = io.BytesIO()
    wavfile.write(buf, sr, data)
    return buf.getvalue()


@contextmanager
def _mock_urlopen(wav_bytes: bytes):
    """Context manager that mocks urllib.request.urlopen to return wav_bytes."""
    mock_resp = MagicMock()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read = MagicMock(return_value=wav_bytes)
    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_fn:
        yield mock_fn


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
    Reading back with normalize=True should recover data within atol=1e-6
    (windowing/format conversion tolerance).
    """
    wav_path = tmp_path / "float_roundtrip.wav"
    known_signal_frame.to_wav(str(wav_path))

    loaded = ChannelFrame.read_wav(str(wav_path), normalize=True)

    assert loaded.sampling_rate == known_signal_frame.sampling_rate
    assert loaded.n_channels == known_signal_frame.n_channels
    # Float WAV preserves data with high fidelity (atol=1e-6 for format conversion)
    np.testing.assert_allclose(
        loaded.compute(),
        known_signal_frame.compute(),
        atol=1e-6,
        err_msg="Float WAV round-trip data mismatch",
    )


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
    assert cf.channels[0].label == "Left Channel"
    assert cf.channels[1].label == "Right Channel"


def test_read_wav_bytes_dc_signal() -> None:
    """Test reading a WAV from in-memory bytes with known DC signal.

    Uses float32 DC signals to verify byte-based read path preserves
    sample values. rtol=1e-5 for float32 precision tolerance.
    """
    sr = 32000
    n_samples = 3200  # 0.1 seconds
    data_left = np.full(n_samples, 0.25, dtype=np.float32)
    data_right = np.full(n_samples, 0.75, dtype=np.float32)
    stereo_data = np.column_stack((data_left, data_right))
    wav_bytes = _make_wav_bytes(sr, stereo_data)

    cf = ChannelFrame.read_wav(wav_bytes)

    assert cf.sampling_rate == sr
    assert len(cf) == 2
    computed = cf.compute()
    # Float32 DC signal: rtol=1e-5 accounts for float32 precision
    np.testing.assert_allclose(computed[0], data_left, rtol=1e-5)
    np.testing.assert_allclose(computed[1], data_right, rtol=1e-5)


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
    assert len(cf) == 2
    assert cf.sampling_rate == sr
    computed = cf.compute()
    # Verify all samples, not just first element (rtol=1e-5 for float32)
    np.testing.assert_allclose(computed[0], data_left, rtol=1e-5)
    np.testing.assert_allclose(computed[1], data_right, rtol=1e-5)


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

    with _mock_urlopen(wav_bytes) as mock_fn:
        cf = ChannelFrame.from_file(url)

    mock_fn.assert_called_once_with(url, timeout=10.0)
    assert cf.sampling_rate == sr
    assert len(cf) == 2
    computed = cf.compute()
    # Float32 DC signal: rtol=1e-5 for float32 precision
    np.testing.assert_allclose(computed[0], data_left, rtol=1e-5)
    np.testing.assert_allclose(computed[1], data_right, rtol=1e-5)
    # Provenance metadata (Pillar 2)
    assert cf.metadata.source_file == url
    assert cf.label == "sample"


def test_from_file_url_http_scheme() -> None:
    """Test that plain http:// URLs are also handled by from_file."""
    url = "http://example.com/audio/sample.wav"
    sr = 8000
    n_samples = 50
    mono_data = np.full(n_samples, 0.5, dtype=np.float32)
    wav_bytes = _make_wav_bytes(sr, mono_data)

    with _mock_urlopen(wav_bytes):
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

    with _mock_urlopen(wav_bytes):
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

    assert cf.sampling_rate == sr
    assert len(cf) == 2
    computed = cf.compute()
    # Float32 DC signal: rtol=1e-5 for float32 precision
    np.testing.assert_allclose(computed[0], data_left, rtol=1e-5)
    np.testing.assert_allclose(computed[1], data_right, rtol=1e-5)


def test_read_wav_int16_pcm_raw_values_preserved(tmp_path) -> None:
    """PCM round-trip: int16 WAV values preserved as float32 (Pillar 4).

    When normalize=False (default), scipy's raw int16 samples are cast to
    float32 with magnitudes preserved (e.g. 16384 -> 16384.0).
    Exact match expected since this is a dtype cast, not a transform.
    """
    filepath = tmp_path / "int16_test.wav"
    sr = 16000
    n_samples = 100
    int16_left = np.full(n_samples, 16384, dtype=np.int16)
    int16_right = np.full(n_samples, -16384, dtype=np.int16)
    stereo_data = np.column_stack((int16_left, int16_right))
    wavfile.write(str(filepath), sr, stereo_data)

    cf = ChannelFrame.read_wav(str(filepath))
    computed = cf.compute()

    # Exact match: raw int16 values cast to float32 (same algorithm, no transform)
    np.testing.assert_array_equal(computed[0], int16_left.astype(np.float32))
    np.testing.assert_array_equal(computed[1], int16_right.astype(np.float32))
    assert computed.dtype == np.float32


def test_read_wav_int16_normalized_to_float_range(tmp_path) -> None:
    """PCM normalization: int16 16384 -> 0.5 after dividing by 32768 (Pillar 4).

    With normalize=True, int16 samples are divided by 32768 to produce
    float32 values in [-1.0, 1.0]. 16384/32768 = 0.5 exactly.
    Tolerance: rtol=1e-4 accounts for float32 precision.
    """
    filepath = tmp_path / "int16_norm_test.wav"
    sr = 16000
    n_samples = 100
    int16_left = np.full(n_samples, 16384, dtype=np.int16)
    int16_right = np.full(n_samples, -16384, dtype=np.int16)
    stereo_data = np.column_stack((int16_left, int16_right))
    wavfile.write(str(filepath), sr, stereo_data)

    cf = ChannelFrame.read_wav(str(filepath), normalize=True)
    computed = cf.compute()

    # Theoretical: 16384 / 32768 = 0.5 exactly; rtol=1e-4 for float32 rounding
    np.testing.assert_allclose(computed[0], 0.5, rtol=1e-4)
    np.testing.assert_allclose(computed[1], -0.5, rtol=1e-4)
    assert computed.dtype == np.float32


def test_write_wav_roundtrip_preserves_shape_and_sr(tmp_path) -> None:
    """Write/read WAV round-trip: shape and sampling rate are preserved.

    Uses DC signals (0.5, 0.8) within [-1, 1] to trigger IEEE FLOAT subtype.
    rtol=1e-2 accounts for potential PCM quantization if FLOAT is not used.
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
    assert read_sr == sr
    assert wav_data.shape == (n_samples, 2)

    # Verify via ChannelFrame round-trip
    loaded = ChannelFrame.read_wav(str(output_path))
    assert loaded.sampling_rate == sr
    assert loaded.shape == cf.shape

    computed = loaded.compute()
    # rtol=1e-2: WAV format may involve float->PCM->float conversion
    np.testing.assert_allclose(computed, wav_data.T, rtol=1e-2)


def test_write_wav_mono_squeezes_to_1d() -> None:
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


def test_write_wav_nonfloat_branch_when_exceeds_unit_range() -> None:
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
