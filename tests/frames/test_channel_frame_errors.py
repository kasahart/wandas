"""Tests for improved error messages in ChannelFrame."""

import dask.array as da
import numpy as np
import pytest
from pathlib import Path

from wandas.frames.channel import ChannelFrame


class TestChannelFrameInitErrors:
    """Test error messages for ChannelFrame.__init__."""

    def test_invalid_shape_3d_array_error_message(self) -> None:
        """Test that 3D array raises informative error."""
        data = da.zeros((2, 3, 4))  # 3D array
        with pytest.raises(ValueError) as exc_info:
            ChannelFrame(data=data, sampling_rate=16000)

        error_msg = str(exc_info.value)
        assert "Invalid data shape" in error_msg
        assert "Given shape: (2, 3, 4)" in error_msg
        assert "3D array" in error_msg
        assert "Solution" in error_msg
        assert "reshape" in error_msg
        assert "Background" in error_msg
        assert "time-series data" in error_msg

    def test_invalid_shape_4d_array_error_message(self) -> None:
        """Test that 4D array raises informative error."""
        data = da.zeros((2, 3, 4, 5))  # 4D array
        with pytest.raises(ValueError) as exc_info:
            ChannelFrame(data=data, sampling_rate=16000)

        error_msg = str(exc_info.value)
        assert "Invalid data shape" in error_msg
        assert "Given shape: (2, 3, 4, 5)" in error_msg
        assert "4D array" in error_msg
        assert "Expected: 1D array (samples,) or 2D array (channels, samples)" in error_msg

    def test_negative_sampling_rate_error_message(self) -> None:
        """Test that negative sampling rate raises informative error."""
        data = da.zeros((2, 100))
        with pytest.raises(ValueError) as exc_info:
            ChannelFrame(data=data, sampling_rate=-100)

        error_msg = str(exc_info.value)
        assert "Invalid sampling rate" in error_msg
        assert "Given: -100" in error_msg or "Given: -100.0" in error_msg
        assert "Expected: Positive number > 0 Hz" in error_msg
        assert "Solution" in error_msg
        assert "Common sampling rates" in error_msg
        assert "Background" in error_msg
        assert "samples per second" in error_msg

    def test_zero_sampling_rate_error_message(self) -> None:
        """Test that zero sampling rate raises informative error."""
        data = da.zeros((2, 100))
        with pytest.raises(ValueError) as exc_info:
            ChannelFrame(data=data, sampling_rate=0)

        error_msg = str(exc_info.value)
        assert "Invalid sampling rate" in error_msg
        assert "Given: 0" in error_msg
        assert "Positive number > 0 Hz" in error_msg

    def test_valid_1d_array_no_error(self) -> None:
        """Test that valid 1D array does not raise error."""
        data = da.zeros(100)
        frame = ChannelFrame(data=data, sampling_rate=16000)
        assert frame.n_channels == 1
        assert frame.n_samples == 100

    def test_valid_2d_array_no_error(self) -> None:
        """Test that valid 2D array does not raise error."""
        data = da.zeros((2, 100))
        frame = ChannelFrame(data=data, sampling_rate=16000)
        assert frame.n_channels == 2
        assert frame.n_samples == 100


class TestChannelFrameFromNumpyErrors:
    """Test error messages for ChannelFrame.from_numpy."""

    def test_invalid_shape_3d_array_error_message(self) -> None:
        """Test that 3D numpy array raises informative error."""
        data = np.zeros((2, 3, 4))  # 3D array
        with pytest.raises(ValueError) as exc_info:
            ChannelFrame.from_numpy(data=data, sampling_rate=16000)

        error_msg = str(exc_info.value)
        assert "Invalid data shape" in error_msg
        assert "Given shape: (2, 3, 4)" in error_msg
        assert "Solution" in error_msg
        assert "Background" in error_msg


class TestChannelFrameFromFileErrors:
    """Test error messages for ChannelFrame.from_file."""

    def test_file_not_found_error_message(self, tmp_path: Path) -> None:
        """Test that missing file raises informative error."""
        non_existent_file = tmp_path / "does_not_exist.wav"
        
        with pytest.raises(FileNotFoundError) as exc_info:
            ChannelFrame.from_file(non_existent_file)

        error_msg = str(exc_info.value)
        assert "Audio file not found" in error_msg
        assert str(non_existent_file) in error_msg
        assert "Absolute path:" in error_msg
        assert "Current directory:" in error_msg
        assert "Solution" in error_msg
        assert "Check the file path is correct" in error_msg
        assert "Tip" in error_msg
        assert ".exists()" in error_msg

    def test_file_not_found_with_relative_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test file not found error with relative path."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)
        
        with pytest.raises(FileNotFoundError) as exc_info:
            ChannelFrame.from_file("nonexistent.wav")

        error_msg = str(exc_info.value)
        assert "Audio file not found" in error_msg
        assert "Specified path: nonexistent.wav" in error_msg
