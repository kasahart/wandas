"""Tests for ChannelFrame error messages."""

import dask.array as da
import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame


class TestChannelFrameInitErrors:
    """Test error messages in ChannelFrame initialization."""

    def test_invalid_sampling_rate_zero(self) -> None:
        """Test error message when sampling rate is zero."""
        data = da.from_array(np.random.random(100), chunks=50)

        with pytest.raises(ValueError) as exc_info:
            ChannelFrame(data, sampling_rate=0)

        error_msg = str(exc_info.value)
        assert "Invalid sampling rate" in error_msg
        assert "Given: 0" in error_msg
        assert "positive number" in error_msg
        assert "Solution" in error_msg
        assert "Background" in error_msg

    def test_invalid_sampling_rate_negative(self) -> None:
        """Test error message when sampling rate is negative."""
        data = da.from_array(np.random.random(100), chunks=50)

        with pytest.raises(ValueError) as exc_info:
            ChannelFrame(data, sampling_rate=-1000)

        error_msg = str(exc_info.value)
        assert "Invalid sampling rate" in error_msg
        assert "Given: -1000" in error_msg
        assert "positive number" in error_msg
        assert "Solution" in error_msg
        assert "Background" in error_msg

    def test_invalid_data_shape_3d(self) -> None:
        """Test error message when data has 3 dimensions."""
        data = da.from_array(np.random.random((2, 3, 100)), chunks=(1, 1, 50))

        with pytest.raises(ValueError) as exc_info:
            ChannelFrame(data, sampling_rate=16000)

        error_msg = str(exc_info.value)
        assert "Invalid data shape" in error_msg
        assert "3D array" in error_msg
        assert "Expected: 1D or 2D" in error_msg
        assert "Solution" in error_msg
        assert "reshape" in error_msg
        assert "Background" in error_msg

    def test_invalid_data_shape_4d(self) -> None:
        """Test error message when data has 4 dimensions."""
        data = da.from_array(np.random.random((2, 3, 4, 100)), chunks=(1, 1, 1, 50))

        with pytest.raises(ValueError) as exc_info:
            ChannelFrame(data, sampling_rate=16000)

        error_msg = str(exc_info.value)
        assert "Invalid data shape" in error_msg
        assert "4D array" in error_msg
        assert "Expected: 1D or 2D" in error_msg
        assert "Solution" in error_msg
        assert "Background" in error_msg

    def test_valid_1d_data(self) -> None:
        """Test that 1D data is accepted and reshaped correctly."""
        data = da.from_array(np.random.random(100), chunks=50)
        cf = ChannelFrame(data, sampling_rate=16000)

        assert cf.n_channels == 1
        assert cf.n_samples == 100
        assert cf.sampling_rate == 16000

    def test_valid_2d_data(self) -> None:
        """Test that 2D data is accepted."""
        data = da.from_array(np.random.random((2, 100)), chunks=(1, 50))
        cf = ChannelFrame(data, sampling_rate=16000)

        assert cf.n_channels == 2
        assert cf.n_samples == 100
        assert cf.sampling_rate == 16000

    def test_valid_positive_sampling_rate(self) -> None:
        """Test that positive sampling rates are accepted."""
        data = da.from_array(np.random.random(100), chunks=50)

        # Test various valid sampling rates
        for sr in [8000, 16000, 44100, 48000]:
            cf = ChannelFrame(data, sampling_rate=sr)
            assert cf.sampling_rate == sr
