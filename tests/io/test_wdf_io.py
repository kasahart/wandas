"""Tests for WDF (Wandas Data File) I/O functionality."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame
from wandas.io import wdf_io


def test_save_load_roundtrip(tmp_path: Path) -> None:
    """Test saving and loading a ChannelFrame with full metadata preservation."""
    # Create test data
    sr = 48000
    data = np.random.randn(2, sr)

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
    assert cf2.sampling_rate == sr
    assert cf2.n_channels == 2
    assert cf2.label == "Test Frame"
    assert cf2.metadata.get("test_key") == "test_value"

    # Verify operation history
    assert len(cf2.operation_history) == 2
    assert cf2.operation_history[0]["operation"] == "normalize"
    assert cf2.operation_history[0]["params"]["method"] == "peak"
    assert cf2.operation_history[1]["operation"] == "filter"
    assert cf2.operation_history[1]["params"]["type"] == "lowpass"
    assert cf2.operation_history[1]["params"]["cutoff"] == 1000

    # Verify channel data
    assert np.allclose(cf2.data, cf.data)

    # Verify channel metadata
    assert cf2._channel_metadata[0].label == "Left"
    assert cf2._channel_metadata[0].unit == "Pa"
    assert cf2._channel_metadata[0].extra.get("sensitivity") == 50.0

    assert cf2._channel_metadata[1].label == "Right"
    assert cf2._channel_metadata[1].unit == "Pa"
    assert cf2._channel_metadata[1].extra.get("sensitivity") == 48.5


def test_save_with_dtype_conversion(tmp_path: Path) -> None:
    """Test saving with dtype conversion."""
    # Create test data with float64
    sr = 44100
    data = np.random.randn(1, sr).astype(np.float64)
    cf = ChannelFrame.from_numpy(data, sr)

    # Save with float32 dtype
    path = tmp_path / "test_dtype.wdf"
    cf.save(path, dtype="float32")

    # Verify dtype in saved file
    with h5py.File(path, "r") as f:
        assert f["channels/0/data"].dtype == np.dtype("float32")


def test_save_without_compression(tmp_path: Path) -> None:
    """Test saving without compression."""
    sr = 22050
    data = np.random.randn(1, sr)
    cf = ChannelFrame.from_numpy(data, sr)

    path = tmp_path / "test_no_compress.wdf"
    cf.save(path, compress=None)

    # Verify that no compression was used
    with h5py.File(path, "r") as f:
        assert f["channels/0/data"].compression is None


def test_file_exists_error(tmp_path: Path) -> None:
    """Test that attempting to overwrite without overwrite=True raises an error."""
    sr = 8000
    data = np.random.randn(1, sr)
    cf = ChannelFrame.from_numpy(data, sr)

    path = tmp_path / "test_exists.wdf"
    # First save
    cf.save(path)

    # Second save without overwrite should fail
    with pytest.raises(FileExistsError):
        cf.save(path, overwrite=False)

    # Second save with overwrite should succeed
    cf.save(path, overwrite=True)


def test_wdf_extension_added(tmp_path: Path) -> None:
    """Test that .wdf extension is automatically added."""
    sr = 16000
    data = np.random.randn(1, sr)
    cf = ChannelFrame.from_numpy(data, sr)

    path = tmp_path / "test_file"  # No extension
    cf.save(path)

    # Should have added .wdf extension
    assert (tmp_path / "test_file.wdf").exists()


def test_unsupported_format() -> None:
    """Test that unsupported formats raise NotImplementedError."""
    sr = 16000
    data = np.random.randn(1, sr)
    cf = ChannelFrame.from_numpy(data, sr)

    with pytest.raises(NotImplementedError):
        cf.save("test.wdf", format="unsupported")

    with pytest.raises(NotImplementedError):
        ChannelFrame.load("test.wdf", format="unsupported")


def test_version_compatibility(tmp_path: Path) -> None:
    """Test version handling in WDF files."""
    sr = 8000
    data = np.random.randn(1, sr)
    cf = ChannelFrame.from_numpy(data, sr)

    path = tmp_path / "test_version.wdf"
    cf.save(path)

    # Modify the version in the file
    with h5py.File(path, "r+") as f:
        f.attrs["version"] = "0.2"

    # Should still load but log a warning
    cf2 = ChannelFrame.load(path)
    assert cf2.n_samples == sr


def test_save_non_serializable_op_history(tmp_path: Path) -> None:
    """Test saving with non-JSON-serializable object in operation history."""
    sr = 16000
    data = np.random.randn(1, sr)
    cf = ChannelFrame.from_numpy(data, sr)

    class NonSerializable:
        def __str__(self) -> str:
            return "NonSerializableObj"

    cf.operation_history = [{"op": "test", "param": NonSerializable()}]

    path = tmp_path / "test_non_serializable.wdf"
    # Should not raise, but fallback to string representation
    wdf_io.save(cf, path)

    # Verify it was saved as string
    cf2 = wdf_io.load(path)
    assert cf2.operation_history[0]["param"] == "NonSerializableObj"


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
