"""Tests for WDF (Wandas Data File) I/O functionality."""

import io
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import dask.array
import h5py
import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame
from wandas.io import readers as io_readers
from wandas.io import wdf_io


@contextmanager
def _mock_urlopen(
    content_bytes: bytes,
    *,
    forbid_unbounded_read: bool = False,
    include_content_length: bool = True,
    expected_chunk_size: int | None = None,
    max_chunk_bytes: int | None = None,
):
    """Context manager that mocks urllib.request.urlopen to stream content_bytes."""
    stream = io.BytesIO(content_bytes)
    mock_resp = MagicMock()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.headers = {"Content-Length": str(len(content_bytes))} if include_content_length else {}

    def _read(size: int = -1) -> bytes:
        if forbid_unbounded_read and size < 0:
            raise AssertionError("URL reads must request bounded chunks")
        if expected_chunk_size is not None and size >= 0:
            assert size == expected_chunk_size, f"Expected chunk size {expected_chunk_size}, got {size}"
        effective_size = size
        if max_chunk_bytes is not None and size >= 0:
            effective_size = min(size, max_chunk_bytes)
        return stream.read(effective_size)

    mock_resp.read = MagicMock(side_effect=_read)
    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_fn:
        yield mock_fn


def test_wdf_roundtrip_known_signal(known_signal_frame, tmp_path: Path) -> None:
    """WDF round-trip with known_signal_frame: data, SR, labels preserved (I/O Policy).

    WDF is Wandas' native format — full metadata round-trip is mandatory.
    Uses assert_allclose with rtol=1e-6 for format conversion tolerance.
    """
    path = tmp_path / "known_signal.wdf"
    known_signal_frame.save(path)

    loaded = ChannelFrame.load(path)

    # Verify numerical data (rtol=1e-6: WDF HDF5 round-trip tolerance)
    np.testing.assert_allclose(loaded.compute(), known_signal_frame.compute(), rtol=1e-6)
    # Verify sampling rate
    assert loaded.sampling_rate == known_signal_frame.sampling_rate
    # Verify channel labels
    assert loaded._channel_metadata[0].label == "left"
    assert loaded._channel_metadata[1].label == "right"
    # Verify Dask lazy loading (Pillar 1)
    assert isinstance(loaded._data, dask.array.core.Array)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    """Test saving and loading a ChannelFrame with full metadata preservation."""
    # Seeded RNG for reproducibility (Grand Policy: no random data)
    rng = np.random.default_rng(42)
    sr = 48000
    data = rng.standard_normal((2, sr))

    # Create ChannelFrame with metadata
    cf = ChannelFrame.from_numpy(
        data,
        sr,
        label="Test Frame",
        metadata={"test_key": "test_value"},
        ch_labels=["Left", "Right"],
        ch_units=["Pa", "Pa"],
    )

    # Set additional metadata on channels
    cf._channel_metadata[0].extra["sensitivity"] = 50.0
    cf._channel_metadata[1].extra["sensitivity"] = 48.5

    # Add operation history
    cf.operation_history = [
        {"operation": "normalize", "params": {"method": "peak"}},
        {"operation": "filter", "params": {"type": "lowpass", "cutoff": 1000}},
    ]

    # Save to file
    path = tmp_path / "test_roundtrip.wdf"
    cf.save(path)

    # Reload from file
    cf2 = ChannelFrame.load(path)

    # Verify basic properties
    assert cf2.sampling_rate == sr, f"Sampling rate mismatch: {cf2.sampling_rate} != {sr}"
    assert cf2.n_channels == 2, f"Channel count mismatch: {cf2.n_channels} != 2"
    assert cf2.label == "Test Frame", f"Label mismatch: {cf2.label}"
    assert cf2.metadata.get("test_key") == "test_value", "Custom metadata key not preserved"

    # Verify operation history
    assert len(cf2.operation_history) == 2, f"Expected 2 history entries, got {len(cf2.operation_history)}"
    assert cf2.operation_history[0]["operation"] == "normalize"
    assert cf2.operation_history[0]["params"]["method"] == "peak"
    assert cf2.operation_history[1]["operation"] == "filter"
    assert cf2.operation_history[1]["params"]["type"] == "lowpass"
    assert cf2.operation_history[1]["params"]["cutoff"] == 1000

    # Verify channel data — HDF5 round-trip with same dtype: exact match expected
    np.testing.assert_array_equal(cf2.data, cf.data)

    # Verify Dask lazy loading preserved after WDF load (Pillar 1)
    assert isinstance(cf2._data, dask.array.core.Array), "WDF load must produce Dask array"

    # Verify channel metadata
    assert cf2._channel_metadata[0].label == "Left"
    assert cf2._channel_metadata[0].unit == "Pa"
    assert cf2._channel_metadata[0].extra.get("sensitivity") == 50.0

    assert cf2._channel_metadata[1].label == "Right"
    assert cf2._channel_metadata[1].unit == "Pa"
    assert cf2._channel_metadata[1].extra.get("sensitivity") == 48.5


def test_wdf_operation_history_roundtrip(known_signal_frame, tmp_path: Path) -> None:
    """WDF round-trip preserves operation_history entries (Pillar 2: metadata sync).

    Verifies that operation names and parameters survive HDF5 serialization.
    """
    known_signal_frame.operation_history = [
        {"operation": "normalize", "params": {"method": "peak"}},
        {"operation": "lowpass_filter", "params": {"cutoff": 1000, "order": 4}},
    ]
    path = tmp_path / "op_history.wdf"
    known_signal_frame.save(path)

    loaded = ChannelFrame.load(path)

    assert len(loaded.operation_history) == 2
    assert loaded.operation_history[0]["operation"] == "normalize"
    assert loaded.operation_history[0]["params"]["method"] == "peak"
    assert loaded.operation_history[1]["operation"] == "lowpass_filter"
    assert loaded.operation_history[1]["params"]["cutoff"] == 1000
    assert loaded.operation_history[1]["params"]["order"] == 4


def test_save_wdf_dtype_float32_converts_stored_data(tmp_path: Path) -> None:
    """Test saving with dtype conversion."""
    rng = np.random.default_rng(1)
    sr = 44100
    data = rng.standard_normal((1, sr)).astype(np.float64)
    cf = ChannelFrame.from_numpy(data, sr)

    # Save with float32 dtype
    path = tmp_path / "test_dtype.wdf"
    cf.save(path, dtype="float32")

    # Verify dtype in saved file
    with h5py.File(path, "r") as f:
        assert f["channels/0/data"].dtype == np.dtype("float32")


def test_save_wdf_no_compression_stores_uncompressed(tmp_path: Path) -> None:
    """Test saving without compression."""
    rng = np.random.default_rng(2)
    sr = 22050
    data = rng.standard_normal((1, sr))
    cf = ChannelFrame.from_numpy(data, sr)

    path = tmp_path / "test_no_compress.wdf"
    cf.save(path, compress=None)

    # Verify that no compression was used
    with h5py.File(path, "r") as f:
        assert f["channels/0/data"].compression is None


def test_save_wdf_existing_file_overwrite_false_raises_file_exists(tmp_path: Path) -> None:
    """Test that attempting to overwrite without overwrite=True raises an error."""
    rng = np.random.default_rng(3)
    sr = 8000
    data = rng.standard_normal((1, sr))
    cf = ChannelFrame.from_numpy(data, sr)

    path = tmp_path / "test_exists.wdf"
    cf.save(path)
    assert path.exists(), "First save must create the file"

    # Second save without overwrite should fail (I/O Policy: overwrite protection)
    with pytest.raises(FileExistsError):
        cf.save(path, overwrite=False)

    # Second save with overwrite=True should succeed
    cf.save(path, overwrite=True)
    assert path.exists(), "Overwrite must keep the file"


def test_save_wdf_no_extension_adds_wdf_suffix(tmp_path: Path) -> None:
    """Test that .wdf extension is automatically added."""
    rng = np.random.default_rng(4)
    sr = 16000
    data = rng.standard_normal((1, sr))
    cf = ChannelFrame.from_numpy(data, sr)

    path = tmp_path / "test_file"  # No extension
    cf.save(path)

    # Should have added .wdf extension
    assert (tmp_path / "test_file.wdf").exists(), ".wdf extension must be auto-appended"


def test_save_load_unsupported_format_raises_not_implemented() -> None:
    """Test that unsupported formats raise NotImplementedError."""
    rng = np.random.default_rng(5)
    sr = 16000
    data = rng.standard_normal((1, sr))
    cf = ChannelFrame.from_numpy(data, sr)

    with pytest.raises(NotImplementedError):
        cf.save("test.wdf", format="unsupported")

    with pytest.raises(NotImplementedError):
        ChannelFrame.load("test.wdf", format="unsupported")


def test_load_wdf_modified_version_still_loads(tmp_path: Path) -> None:
    """Test version handling in WDF files."""
    rng = np.random.default_rng(6)
    sr = 8000
    data = rng.standard_normal((1, sr))
    cf = ChannelFrame.from_numpy(data, sr)

    path = tmp_path / "test_version.wdf"
    cf.save(path)

    # Modify the version in the file
    with h5py.File(path, "r+") as f:
        f.attrs["version"] = "0.2"

    # Should still load but log a warning
    cf2 = ChannelFrame.load(path)
    assert cf2.n_samples == sr, f"Expected {sr} samples after version-modified load, got {cf2.n_samples}"


def test_save_wdf_non_serializable_op_history_falls_back_to_string(tmp_path: Path) -> None:
    """Non-JSON-serializable op history params fallback to str representation."""
    rng = np.random.default_rng(7)
    sr = 16000
    data = rng.standard_normal((1, sr))
    cf = ChannelFrame.from_numpy(data, sr)

    class NonSerializable:
        def __str__(self) -> str:
            return "NonSerializableObj"

    cf.operation_history = [{"op": "test", "param": NonSerializable()}]

    path = tmp_path / "test_non_serializable.wdf"
    # Should not raise, but fallback to string representation
    wdf_io.save(cf, path)

    # Verify it was saved as string fallback
    cf2 = wdf_io.load(path)
    assert cf2.operation_history[0]["param"] == "NonSerializableObj", "Non-serializable must fallback to str()"


def test_load_no_channels(tmp_path: Path) -> None:
    """Test loading a file with no channels raises ValueError."""
    path = tmp_path / "empty_channels.wdf"

    # Create a dummy HDF5 file with no channels
    with h5py.File(path, "w") as f:
        f.attrs["version"] = wdf_io.WDF_FORMAT_VERSION
        f.attrs["sampling_rate"] = 16000.0
        f.create_group("channels")  # Empty group

    with pytest.raises(ValueError, match="No channel data found"):
        wdf_io.load(path)


def test_load_json_decode_error(tmp_path: Path) -> None:
    """Test loading with JSON decode error in operation history."""
    path = tmp_path / "bad_json.wdf"

    with h5py.File(path, "w") as f:
        f.attrs["version"] = wdf_io.WDF_FORMAT_VERSION
        f.attrs["sampling_rate"] = 16000.0

        # Add channels
        ch_grp = f.create_group("channels")
        c0 = ch_grp.create_group("0")
        c0.create_dataset("data", data=np.zeros(100))

        # Add bad op history
        op_grp = f.create_group("operation_history")
        op0 = op_grp.create_group("operation_0")
        # Invalid JSON string
        op0.attrs["params"] = "{bad_json"

    # Should load and fallback to raw string
    cf = wdf_io.load(path)
    assert cf.operation_history[0]["params"] == "{bad_json"


def test_load_file_not_found(tmp_path: Path) -> None:
    """Test loading a non-existent file raises FileNotFoundError."""
    path = tmp_path / "non_existent.wdf"
    with pytest.raises(FileNotFoundError, match="File not found"):
        wdf_io.load(path)


def test_source_file_roundtrip(tmp_path: Path) -> None:
    """Test that source_file in FrameMetadata survives a save/load round-trip."""
    from wandas.core.metadata import FrameMetadata

    rng = np.random.default_rng(8)
    sr = 16000
    data = rng.standard_normal((1, sr))
    source = "/data/recordings/audio.wav"
    cf = ChannelFrame.from_numpy(
        data,
        sr,
        metadata=FrameMetadata({"key": "val"}, source_file=source),
    )
    assert isinstance(cf.metadata, FrameMetadata)
    assert cf.metadata.source_file == source

    path = tmp_path / "test_source_file.wdf"
    cf.save(path)

    cf2 = ChannelFrame.load(path)
    assert isinstance(cf2.metadata, FrameMetadata)
    assert cf2.metadata.source_file == source
    assert cf2.metadata.get("key") == "val"


def test_source_file_none_roundtrip(tmp_path: Path) -> None:
    """Test round-trip when source_file is None."""
    from wandas.core.metadata import FrameMetadata

    rng = np.random.default_rng(9)
    sr = 16000
    data = rng.standard_normal((1, sr))
    cf = ChannelFrame.from_numpy(data, sr, metadata=FrameMetadata({"only": "dict"}))
    assert cf.metadata.source_file is None

    path = tmp_path / "test_no_source_file.wdf"
    cf.save(path)

    cf2 = ChannelFrame.load(path)
    assert isinstance(cf2.metadata, FrameMetadata)
    assert cf2.metadata.source_file is None
    assert cf2.metadata.get("only") == "dict"


def test_load_wdf_from_url(tmp_path: Path) -> None:
    """Test WDF load from mocked URL preserves data and metadata (Pillar 2, 4).

    Verifies urlopen is called, numerical data matches (rtol=1e-5 for float32),
    and label/channel metadata survives the URL download path.
    """
    rng = np.random.default_rng(42)
    sr = 8000
    data = rng.standard_normal((2, sr)).astype(np.float32)
    cf = ChannelFrame.from_numpy(data, sr, label="URL Test", ch_labels=["A", "B"])
    wdf_path = tmp_path / "test_url.wdf"
    cf.save(wdf_path)
    wdf_bytes = wdf_path.read_bytes()

    url = "https://example.com/data/test_url.wdf"
    with _mock_urlopen(wdf_bytes) as mock_fn:
        cf2 = wdf_io.load(url)

    mock_fn.assert_called_once_with(url, timeout=10.0)
    assert cf2.sampling_rate == sr, f"SR mismatch: {cf2.sampling_rate}"
    assert cf2.n_channels == 2, f"Channel count mismatch: {cf2.n_channels}"
    assert cf2.label == "URL Test", f"Label mismatch: {cf2.label}"
    assert cf2._channel_metadata[0].label == "A", "Channel 0 label not preserved"
    assert cf2._channel_metadata[1].label == "B", "Channel 1 label not preserved"
    # Verify Dask lazy loading from URL path (Pillar 1)
    assert isinstance(cf2._data, dask.array.core.Array), "URL WDF load must produce Dask array"
    # Float32 HDF5 round-trip: rtol=1e-5 for float32 precision
    np.testing.assert_allclose(cf2.compute(), data, rtol=1e-5)


def test_load_wdf_from_url_streams_in_chunks(tmp_path: Path) -> None:
    """URL WDF loads must stream in bounded chunks instead of full-response reads."""
    rng = np.random.default_rng(42)
    sr = 8000
    data = rng.standard_normal((1, sr)).astype(np.float32)
    cf = ChannelFrame.from_numpy(data, sr, label="URL Stream Test", ch_labels=["A"])
    wdf_path = tmp_path / "test_stream_url.wdf"
    cf.save(wdf_path)
    wdf_bytes = wdf_path.read_bytes()

    with _mock_urlopen(
        wdf_bytes,
        forbid_unbounded_read=True,
        expected_chunk_size=io_readers.URL_DOWNLOAD_CHUNK_SIZE,
    ):
        loaded = wdf_io.load("https://example.com/data/test_stream_url.wdf")

    assert loaded.sampling_rate == sr
    np.testing.assert_allclose(loaded.compute(), data, rtol=1e-5)


def test_load_wdf_from_url_over_size_limit_without_content_length_raises(tmp_path: Path) -> None:
    """URL WDF loads must stop oversized streamed downloads without Content-Length."""
    rng = np.random.default_rng(42)
    sr = 8000
    data = rng.standard_normal((1, sr)).astype(np.float32)
    cf = ChannelFrame.from_numpy(data, sr, label="URL Stream Test", ch_labels=["A"])
    wdf_path = tmp_path / "test_stream_limit_url.wdf"
    cf.save(wdf_path)
    wdf_bytes = wdf_path.read_bytes()

    with patch.object(io_readers, "MAX_URL_DOWNLOAD_BYTES", 128):
        with _mock_urlopen(wdf_bytes, include_content_length=False):
            with pytest.raises(OSError, match=r"Streaming WDF file would exceed size limit"):
                wdf_io.load("https://example.com/data/test_stream_limit_url.wdf")


def test_load_wdf_from_url_declared_size_limit_raises_before_streaming(tmp_path: Path) -> None:
    """URL WDF loads must reject oversized Content-Length before streaming data."""
    rng = np.random.default_rng(42)
    sr = 8000
    data = rng.standard_normal((1, sr)).astype(np.float32)
    cf = ChannelFrame.from_numpy(data, sr, label="Declared Size Test", ch_labels=["A"])
    wdf_path = tmp_path / "test_declared_limit_url.wdf"
    cf.save(wdf_path)
    wdf_bytes = wdf_path.read_bytes()

    with patch.object(io_readers, "MAX_URL_DOWNLOAD_BYTES", 128):
        with _mock_urlopen(wdf_bytes, include_content_length=True) as mock_fn:
            with pytest.raises(OSError, match=r"Declared size of WDF file exceeds download limit"):
                wdf_io.load("https://example.com/data/test_declared_limit_url.wdf")

    mock_resp = mock_fn.return_value
    mock_resp.read.assert_not_called()


def test_load_wdf_from_url_handles_partial_reads(tmp_path: Path) -> None:
    """URL WDF loads must continue until EOF even when reads return small chunks."""
    rng = np.random.default_rng(42)
    sr = 8000
    data = rng.standard_normal((1, sr)).astype(np.float32)
    cf = ChannelFrame.from_numpy(data, sr, label="Partial Read Test", ch_labels=["A"])
    wdf_path = tmp_path / "test_partial_reads_url.wdf"
    cf.save(wdf_path)
    wdf_bytes = wdf_path.read_bytes()

    with _mock_urlopen(
        wdf_bytes,
        forbid_unbounded_read=True,
        include_content_length=False,
        expected_chunk_size=io_readers.URL_DOWNLOAD_CHUNK_SIZE,
        max_chunk_bytes=257,
    ):
        loaded = wdf_io.load("https://example.com/data/test_partial_reads_url.wdf")

    np.testing.assert_allclose(loaded.compute(), data, rtol=1e-5)


def test_load_wdf_from_url_download_failure() -> None:
    """Test that a URL download failure raises OSError with a clear message."""
    import urllib.error

    url = "https://example.com/data/missing.wdf"

    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        with pytest.raises(OSError, match=r"Failed to download WDF file from URL"):
            wdf_io.load(url)


def test_decode_hdf5_str_invalid_utf8() -> None:
    """_decode_hdf5_str falls back to str() for non-UTF-8 bytes (lines 39-42)."""
    from wandas.io.wdf_io import _decode_hdf5_str

    invalid_bytes = b"\xff\xfe"
    result = _decode_hdf5_str(invalid_bytes)
    # Should return str() fallback rather than raising
    assert isinstance(result, str)
    assert result == str(invalid_bytes)


def test_load_wdf_meta_json_as_bytes(tmp_path: Path) -> None:
    """Loading a WDF file where meta JSON is stored as numpy.bytes_ triggers line 229."""
    sr = 16000
    data = np.random.default_rng(0).standard_normal((1, sr)).astype(np.float32)
    # metadata= ensures the save function creates a "meta" HDF5 group
    cf = ChannelFrame.from_numpy(data, sr, label="BytesTest", metadata={"key": "val"})
    wdf_path = tmp_path / "bytes_meta.wdf"
    cf.save(wdf_path)

    # Rewrite the meta JSON attribute as numpy.bytes_ — h5py stores and returns this
    # as numpy.bytes_, which triggers the isinstance(meta_json, (bytes, np.bytes_)) branch
    with h5py.File(wdf_path, "r+") as f:
        if "meta" in f:
            meta_json_str = f["meta"].attrs.get("json", "{}")
            if isinstance(meta_json_str, str):
                del f["meta"].attrs["json"]
                f["meta"].attrs["json"] = np.bytes_(meta_json_str.encode("utf-8"))

    # Load should succeed even when meta_json is numpy.bytes_
    loaded = wdf_io.load(wdf_path)
    assert loaded.sampling_rate == sr
    assert loaded.n_channels == 1
