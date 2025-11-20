import os
import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from dask.array.core import Array as DaArray
from matplotlib.axes import Axes

import wandas as wd
from wandas.core.metadata import ChannelMetadata
from wandas.frames.channel import ChannelFrame
from wandas.utils.types import NDArrayReal

_da_from_array = da.from_array  # type: ignore [unused-ignore]


class TestChannelFrame:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        # Create a simple dask array for testing
        self.sample_rate: float = 16000
        self.data: NDArrayReal = np.random.random((2, 16000))  # 2 channels, 1 second
        self.dask_data: DaArray = _da_from_array(self.data, chunks=(1, 4000))
        self.channel_frame: ChannelFrame = ChannelFrame(
            data=self.dask_data, sampling_rate=self.sample_rate, label="test_audio"
        )

    def test_initialization(self) -> None:
        """Test that initialization doesn't compute the data."""
        with mock.patch.object(
            DaArray, "compute", return_value=self.data
        ) as mock_compute:
            # Just creating the object shouldn't call compute
            cf: ChannelFrame = ChannelFrame(self.dask_data, self.sample_rate)

            # Verify compute hasn't been called
            mock_compute.assert_not_called()

            # Check properties that don't require computation
            assert cf.sampling_rate == self.sample_rate
            assert cf.n_channels == 2
            assert cf.n_samples == 16000
            assert cf.duration == 1.0

            # Still no computation should have happened
            mock_compute.assert_not_called()

    def test_data_access_triggers_compute(self) -> None:
        """Test that accessing .data triggers computation."""
        with mock.patch.object(
            DaArray, "compute", return_value=self.data
        ) as mock_compute:
            _: NDArrayReal = self.channel_frame.data
            mock_compute.assert_called_once()

    def test_compute_method(self) -> None:
        """Test explicit compute method."""
        with mock.patch.object(
            DaArray, "compute", return_value=self.data
        ) as mock_compute:
            result: NDArrayReal = self.channel_frame.compute()
            mock_compute.assert_called_once()
            np.testing.assert_array_equal(result, self.data)

    def test_time(self) -> None:
        """Test time property."""
        with mock.patch.object(
            DaArray, "compute", return_value=self.data
        ) as mock_compute:
            time: NDArrayReal = self.channel_frame.time
            mock_compute.assert_not_called()
            expected_time = np.arange(16000) / 16000
            np.testing.assert_array_equal(time, expected_time)

    def test_operations_are_lazy(self) -> None:
        """Test that operations don't trigger immediate computation."""
        with mock.patch.object(
            DaArray, "compute", return_value=self.data
        ) as mock_compute:
            # Operations should build the graph but not compute
            result: ChannelFrame = self.channel_frame + 1
            result = result * 2
            result = result.abs()
            result = result.power(2)

            # Verify no computation happened
            mock_compute.assert_not_called()

            # Check that the result has the expected type
            assert isinstance(result, ChannelFrame)
            assert isinstance(result._data, DaArray)

    def test_operation_results(self) -> None:
        """Test that operations produce correct results when computed."""
        # Apply operations
        result: ChannelFrame = self.channel_frame + 1
        result = result * 2

        # Compute and check results
        computed: NDArrayReal = result.compute()
        expected: NDArrayReal = (self.data + 1) * 2
        np.testing.assert_array_almost_equal(computed, expected)

    def test_persist(self) -> None:
        """Test that persist triggers computation but returns a new ChannelFrame."""
        with mock.patch.object(
            DaArray, "persist", return_value=self.dask_data
        ) as mock_persist:
            result: ChannelFrame = self.channel_frame.persist()
            mock_persist.assert_called_once()
            assert isinstance(result, ChannelFrame)
            assert result is not self.channel_frame

    def test_channel_extraction(self) -> None:
        """Test extracting a channel works lazily."""
        with mock.patch.object(
            DaArray, "compute", return_value=self.data[0:1]
        ) as mock_compute:
            channel: ChannelFrame = self.channel_frame.get_channel(0)
            mock_compute.assert_not_called()

            # Check that properties are correctly set
            assert channel.n_channels == 1
            assert channel.sampling_rate == self.sample_rate

            # Access data to trigger computation
            _: NDArrayReal = channel.data
            mock_compute.assert_called_once()

    def test_get_channel_single_int(self) -> None:
        """Test get_channel with single integer index."""
        # Test positive index
        channel = self.channel_frame.get_channel(0)
        assert isinstance(channel, ChannelFrame)
        assert channel.n_channels == 1
        assert channel.channels[0].label == "ch0"
        np.testing.assert_array_equal(channel.data, self.data[0])

        channel = self.channel_frame.get_channel(1)
        assert isinstance(channel, ChannelFrame)
        assert channel.n_channels == 1
        assert channel.channels[0].label == "ch1"
        np.testing.assert_array_equal(channel.data, self.data[1])

        # Test negative index
        channel = self.channel_frame.get_channel(-1)
        assert isinstance(channel, ChannelFrame)
        assert channel.n_channels == 1
        assert channel.channels[0].label == "ch1"
        np.testing.assert_array_equal(channel.data, self.data[-1])

        channel = self.channel_frame.get_channel(-2)
        assert isinstance(channel, ChannelFrame)
        assert channel.n_channels == 1
        assert channel.channels[0].label == "ch0"
        np.testing.assert_array_equal(channel.data, self.data[-2])

    def test_get_channel_list_of_ints(self) -> None:
        """Test get_channel with list of integer indices."""
        # Test with list of multiple indices
        channels = self.channel_frame.get_channel([0, 1])
        assert isinstance(channels, ChannelFrame)
        assert channels.n_channels == 2
        assert channels.channels[0].label == "ch0"
        assert channels.channels[1].label == "ch1"
        np.testing.assert_array_equal(channels.data, self.data[[0, 1]])

        # Test with reversed order
        channels = self.channel_frame.get_channel([1, 0])
        assert isinstance(channels, ChannelFrame)
        assert channels.n_channels == 2
        assert channels.channels[0].label == "ch1"
        assert channels.channels[1].label == "ch0"
        np.testing.assert_array_equal(channels.data, self.data[[1, 0]])

        # Test with single element list
        channels = self.channel_frame.get_channel([0])
        assert isinstance(channels, ChannelFrame)
        assert channels.n_channels == 1
        assert channels.channels[0].label == "ch0"
        # Single channel .data is squeezed to 1D
        np.testing.assert_array_equal(channels.data, self.data[0])

        # Test with negative indices
        channels = self.channel_frame.get_channel([-1, -2])
        assert isinstance(channels, ChannelFrame)
        assert channels.n_channels == 2
        assert channels.channels[0].label == "ch1"
        assert channels.channels[1].label == "ch0"
        np.testing.assert_array_equal(channels.data, self.data[[-1, -2]])

    def test_get_channel_tuple_of_ints(self) -> None:
        """Test get_channel with tuple of integer indices."""
        # Test with tuple
        channels = self.channel_frame.get_channel((0, 1))
        assert isinstance(channels, ChannelFrame)
        assert channels.n_channels == 2
        assert channels.channels[0].label == "ch0"
        assert channels.channels[1].label == "ch1"
        np.testing.assert_array_equal(channels.data, self.data[[0, 1]])

        # Test with single element tuple
        channels = self.channel_frame.get_channel((1,))
        assert isinstance(channels, ChannelFrame)
        assert channels.n_channels == 1
        assert channels.channels[0].label == "ch1"
        # Single channel .data is squeezed to 1D
        np.testing.assert_array_equal(channels.data, self.data[1])

    def test_get_channel_numpy_array(self) -> None:
        """Test get_channel with numpy array of indices."""
        # Test with numpy array
        indices = np.array([0, 1])
        channels = self.channel_frame.get_channel(indices)
        assert isinstance(channels, ChannelFrame)
        assert channels.n_channels == 2
        np.testing.assert_array_equal(channels.data, self.data[[0, 1]])

        # Test with single element numpy array
        indices = np.array([0])
        channels = self.channel_frame.get_channel(indices)
        assert isinstance(channels, ChannelFrame)
        assert channels.n_channels == 1
        # Single channel .data is squeezed to 1D
        np.testing.assert_array_equal(channels.data, self.data[0])

        # Test with negative indices in numpy array
        indices = np.array([-1, -2])
        channels = self.channel_frame.get_channel(indices)
        assert isinstance(channels, ChannelFrame)
        assert channels.n_channels == 2
        np.testing.assert_array_equal(channels.data, self.data[[-1, -2]])

    def test_get_channel_with_range(self) -> None:
        """Test get_channel with range object."""
        # Create a frame with more channels
        data = np.random.random((4, 16000))
        dask_data = _da_from_array(data, chunks=(1, 4000))
        frame = ChannelFrame(
            data=dask_data, sampling_rate=self.sample_rate, label="test_audio"
        )

        # Test with range
        channels = frame.get_channel(list(range(3)))
        assert isinstance(channels, ChannelFrame)
        assert channels.n_channels == 3
        assert channels.channels[0].label == "ch0"
        assert channels.channels[1].label == "ch1"
        assert channels.channels[2].label == "ch2"
        np.testing.assert_array_equal(channels.data, data[[0, 1, 2]])

    def test_get_channel_preserves_metadata(self) -> None:
        """Test that get_channel preserves metadata correctly."""
        # Set metadata
        self.channel_frame.channels[0].label = "left"
        self.channel_frame.channels[0]["gain"] = 0.5
        self.channel_frame.channels[1].label = "right"
        self.channel_frame.channels[1]["gain"] = 0.75

        # Get single channel
        channel = self.channel_frame.get_channel(0)
        assert channel.channels[0].label == "left"
        assert channel.channels[0]["gain"] == 0.5

        # Get multiple channels
        channels = self.channel_frame.get_channel([0, 1])
        assert channels.channels[0].label == "left"
        assert channels.channels[0]["gain"] == 0.5
        assert channels.channels[1].label == "right"
        assert channels.channels[1]["gain"] == 0.75

        # Get channels in reverse order
        channels = self.channel_frame.get_channel([1, 0])
        assert channels.channels[0].label == "right"
        assert channels.channels[0]["gain"] == 0.75
        assert channels.channels[1].label == "left"
        assert channels.channels[1]["gain"] == 0.5

    def test_get_channel_is_lazy(self) -> None:
        """Test that get_channel operations remain lazy."""
        with mock.patch.object(
            DaArray, "compute", return_value=self.data
        ) as mock_compute:
            # Single channel
            _ = self.channel_frame.get_channel(0)
            mock_compute.assert_not_called()

            # Multiple channels
            channels = self.channel_frame.get_channel([0, 1])
            mock_compute.assert_not_called()

            # Only accessing .data should trigger compute
            _ = channels.data
            mock_compute.assert_called_once()

    def test_plotting_triggers_compute(self) -> None:
        """Test that plotting triggers computation."""
        with mock.patch(
            "wandas.visualization.plotting.create_operation"
        ) as mock_get_strategy:
            mock_strategy: mock.MagicMock = mock.MagicMock()
            mock_get_strategy.return_value = mock_strategy

            # Create a mock for the compute method
            with mock.patch.object(
                self.channel_frame, "compute", return_value=self.data
            ) as mock_compute:
                mock_ax: mock.MagicMock = mock.MagicMock()
                _: Axes | Any = self.channel_frame.plot(
                    plot_type="waveform", ax=mock_ax
                )

                # Verify compute was called
                mock_compute.assert_not_called()

                # Verify the strategy's plot method was called
                mock_strategy.plot.assert_called_once()

    def test_initialization_with_1d_data(self) -> None:
        """Test initialization with 1D data."""
        data_1d = np.random.random(16000)
        dask_data_1d = _da_from_array(data_1d, chunks=4000)

        cf = ChannelFrame(dask_data_1d, self.sample_rate)

        # Check that the data was reshaped
        assert cf.shape == (16000,)
        assert cf.n_channels == 1

    def test_save_method(self) -> None:
        """Test saving audio to file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name

        try:
            # Test with multi-channel data
            with mock.patch("soundfile.write") as mock_write:
                with mock.patch.object(
                    self.channel_frame, "compute", return_value=self.data
                ):
                    self.channel_frame.to_wav(temp_filename)
                    mock_write.assert_called_once()

                    # Check that data was transposed for soundfile
                    args = mock_write.call_args[0]
                    assert args[0] == temp_filename
                    np.testing.assert_array_equal(args[1], self.data.T)
                    assert args[2] == int(self.sample_rate)

            # Test with single-channel data
            with mock.patch("soundfile.write") as mock_write:
                single_channel_data = self.data[0:1]
                channel_frame = ChannelFrame(
                    _da_from_array(single_channel_data, chunks=(1, 4000)),
                    self.sample_rate,
                )

                with mock.patch.object(
                    channel_frame, "compute", return_value=single_channel_data
                ):
                    channel_frame.to_wav(temp_filename)
                    mock_write.assert_called_once()

                    # Check that data was transposed and squeezed
                    args = mock_write.call_args[0]
                    np.testing.assert_array_equal(
                        args[1], single_channel_data.T.squeeze(axis=1)
                    )
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    def test_iter_method(self) -> None:
        """Test iterating over channels."""
        channels = list(self.channel_frame)

        assert len(channels) == 2
        for i, channel in enumerate(channels):
            assert isinstance(channel, ChannelFrame)
            assert channel.n_channels == 1
            assert channel.label == "test_audio"
            assert channel.sampling_rate == self.sample_rate
            assert channel.n_samples == 16000
            assert channel.channels[0].label == f"ch{i}"

    def test_array_method(self) -> None:
        """Test __array__ method for numpy conversion."""
        with mock.patch.object(
            self.channel_frame, "compute", return_value=self.data
        ) as mock_compute:
            # Test with default dtype
            array = np.array(self.channel_frame)
            mock_compute.assert_called_once()
            np.testing.assert_array_equal(array, self.data)

            # Reset mock
            mock_compute.reset_mock()

            # Test with specified dtype
            array = np.array(self.channel_frame, dtype=np.float64)
            mock_compute.assert_called_once()
            assert array.dtype == np.float64

    def test_getitem_method(self) -> None:
        """Test __getitem__ method."""

        # Slice all channels
        result = self.channel_frame["ch0"]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 1
        assert result.n_samples == 16000
        assert result.sampling_rate == self.sample_rate
        assert result.label == "test_audio"
        assert result.channels[0].label == "ch0"
        assert result.shape == (16000,)
        assert result.data.shape == (16000,)
        np.testing.assert_array_equal(result.data, self.data[0])

        result = self.channel_frame["ch1"]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 1
        assert result.n_samples == 16000
        assert result.sampling_rate == self.sample_rate
        assert result.label == "test_audio"
        assert result.channels[0].label == "ch1"
        assert result.shape == (16000,)
        assert result.data.shape == (16000,)
        np.testing.assert_array_equal(result.data, self.data[1])

        # Single channel extraction
        result = self.channel_frame[0]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 1
        assert result.n_samples == 16000
        assert result.sampling_rate == self.sample_rate
        assert result.label == "test_audio"
        assert result.channels[0].label == "ch0"
        assert result.shape == (16000,)
        np.testing.assert_array_equal(result.data, self.data[0])

        # Two channel extraction
        result = self.channel_frame[0:2]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 2
        assert result.n_samples == 16000
        assert result.sampling_rate == self.sample_rate
        assert result.label == "test_audio"
        assert result.channels[0].label == "ch0"
        assert result.channels[1].label == "ch1"
        assert result.shape == (2, 16000)
        np.testing.assert_array_equal(result.data, self.data)

        # Time slice
        result = self.channel_frame[:, :1000]
        assert isinstance(result, ChannelFrame)
        assert result.n_samples == 1000
        assert result.n_channels == 2
        assert result.sampling_rate == self.sample_rate
        assert result.label == "test_audio"
        assert result.channels[0].label == "ch0"
        assert result.shape == (2, 1000)
        np.testing.assert_array_equal(result.data, self.data[:, :1000])

        result = self.channel_frame[0:2, :1000]
        assert isinstance(result, ChannelFrame)
        assert result.n_samples == 1000
        assert result.n_channels == 2
        assert result.sampling_rate == self.sample_rate
        assert result.label == "test_audio"
        assert result.channels[0].label == "ch0"
        assert result.channels[1].label == "ch1"
        assert result.shape == (2, 1000)
        np.testing.assert_array_equal(result.data, self.data[:, :1000])

        # Test error case
        with pytest.raises(ValueError, match="Invalid key length"):
            self.channel_frame[0, 0, 0]  # type: ignore
        # Test for invalid channel index
        with pytest.raises(IndexError):
            _ = self.channel_frame[5]
        # Test for invalid slice
        with pytest.raises(TypeError, match="Invalid key type:"):
            _ = self.channel_frame[1.5]  # type: ignore

    def test_negative_indexing(self) -> None:
        """Test negative indexing support."""
        # Test negative integer index
        result = self.channel_frame[-1]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 1
        assert result.channels[0].label == "ch1"
        np.testing.assert_array_equal(result.data, self.data[1])

        result = self.channel_frame[-2]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 1
        assert result.channels[0].label == "ch0"
        np.testing.assert_array_equal(result.data, self.data[0])

        # Test negative slice
        result = self.channel_frame[-2:]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 2
        np.testing.assert_array_equal(result.data, self.data[-2:])

        result = self.channel_frame[-1:]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 1
        # For single channel result from slice, data should be (1, 16000)
        # For integer index, data is squeezed to (16000,)
        assert result.data.shape == (16000,)
        np.testing.assert_array_equal(result.data, self.data[-1])

        # Test negative slice with end
        result = self.channel_frame[-2:-1]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 1
        assert result.channels[0].label == "ch0"
        # Single channel from slice is squeezed
        np.testing.assert_array_equal(result.data, self.data[0])

        # Test out of range negative index
        with pytest.raises(IndexError):
            _ = self.channel_frame[-3]

    def test_step_slicing(self) -> None:
        """Test slicing with step parameter."""
        # Create a frame with more channels for better testing
        data = np.random.random((4, 16000))
        dask_data = _da_from_array(data, chunks=(1, 4000))
        frame = ChannelFrame(
            data=dask_data, sampling_rate=self.sample_rate, label="test_audio"
        )

        # Test every second channel
        result = frame[::2]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 2
        assert result.channels[0].label == "ch0"
        assert result.channels[1].label == "ch2"
        np.testing.assert_array_equal(result.data, data[::2])

        # Test reverse order
        result = frame[::-1]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 4
        assert result.channels[0].label == "ch3"
        assert result.channels[1].label == "ch2"
        assert result.channels[2].label == "ch1"
        assert result.channels[3].label == "ch0"
        np.testing.assert_array_equal(result.data, data[::-1])

        # Test every third channel
        result = frame[::3]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 2
        np.testing.assert_array_equal(result.data, data[::3])

        # Test with start and step
        result = frame[1::2]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 2
        assert result.channels[0].label == "ch1"
        assert result.channels[1].label == "ch3"
        np.testing.assert_array_equal(result.data, data[1::2])

    def test_boolean_indexing(self) -> None:
        """Test boolean array indexing."""
        # Test boolean mask
        mask = np.array([True, False])
        result = self.channel_frame[mask]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 1
        assert result.channels[0].label == "ch0"
        # Single channel result is squeezed
        np.testing.assert_array_equal(result.data, self.data[0])

        mask = np.array([False, True])
        result = self.channel_frame[mask]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 1
        assert result.channels[0].label == "ch1"
        # Single channel result is squeezed
        np.testing.assert_array_equal(result.data, self.data[1])

        mask = np.array([True, True])
        result = self.channel_frame[mask]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 2
        np.testing.assert_array_equal(result.data, self.data)

        # Test error for wrong length boolean array
        mask = np.array([True, False, True])
        with pytest.raises(ValueError, match="Boolean mask length"):
            _ = self.channel_frame[mask]

    def test_integer_array_indexing(self) -> None:
        """Test integer array indexing."""
        # Test with numpy array
        indices = np.array([0, 1])
        result = self.channel_frame[indices]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 2
        np.testing.assert_array_equal(result.data, self.data)

        indices = np.array([1, 0])
        result = self.channel_frame[indices]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 2
        assert result.channels[0].label == "ch1"
        assert result.channels[1].label == "ch0"
        np.testing.assert_array_equal(result.data, self.data[[1, 0]])

        # Test with list
        result = self.channel_frame[[0]]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 1
        assert result.channels[0].label == "ch0"
        # Single channel result is squeezed
        np.testing.assert_array_equal(result.data, self.data[0])

        result = self.channel_frame[[1, 0]]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 2
        assert result.channels[0].label == "ch1"
        assert result.channels[1].label == "ch0"

        # Test negative indices in array
        indices = np.array([-1, -2])
        result = self.channel_frame[indices]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 2
        assert result.channels[0].label == "ch1"
        assert result.channels[1].label == "ch0"

        # Test out of range
        indices = np.array([0, 5])
        with pytest.raises(IndexError, match="Index is out of bounds"):
            _ = self.channel_frame[indices]

        # Test empty list
        with pytest.raises(ValueError, match="Cannot index with an empty list"):
            _ = self.channel_frame[[]]

        # Test invalid list content (mixed types)
        with pytest.raises(TypeError, match="List must contain all str or all int"):
            _ = self.channel_frame[[0, "ch1"]]  # type: ignore

    def test_label_list_indexing(self) -> None:
        """Test list of labels indexing."""
        # Test single label in list
        result = self.channel_frame[["ch0"]]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 1
        assert result.channels[0].label == "ch0"
        np.testing.assert_array_equal(result.data, self.data[0])

        # Test multiple labels
        result = self.channel_frame[["ch0", "ch1"]]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 2
        assert result.channels[0].label == "ch0"
        assert result.channels[1].label == "ch1"
        np.testing.assert_array_equal(result.data, self.data)

        # Test reversed order
        result = self.channel_frame[["ch1", "ch0"]]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 2
        assert result.channels[0].label == "ch1"
        assert result.channels[1].label == "ch0"
        np.testing.assert_array_equal(result.data, self.data[[1, 0]])

        # Test duplicate labels
        result = self.channel_frame[["ch0", "ch0"]]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 2
        assert result.channels[0].label == "ch0"
        assert result.channels[1].label == "ch0"

        # Test error for non-existent label
        with pytest.raises(KeyError, match="Channel label .* not found"):
            _ = self.channel_frame[["ch0", "ch999"]]

    def test_multidimensional_indexing(self) -> None:
        """Test multidimensional indexing (channel + time)."""
        # Single channel + time slice
        result = self.channel_frame[0, 100:200]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 1
        assert result.n_samples == 100
        assert result.channels[0].label == "ch0"
        np.testing.assert_array_equal(result.data, self.data[0, 100:200])

        # Label + time slice
        result = self.channel_frame["ch1", 500:1000]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 1
        assert result.n_samples == 500
        assert result.channels[0].label == "ch1"
        np.testing.assert_array_equal(result.data, self.data[1, 500:1000])

        # Multiple channels + time slice
        result = self.channel_frame[[0, 1], 100:200]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 2
        assert result.n_samples == 100
        np.testing.assert_array_equal(result.data, self.data[:, 100:200])

        # List of labels + time slice
        result = self.channel_frame[["ch0", "ch1"], 0:1000]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 2
        assert result.n_samples == 1000
        np.testing.assert_array_equal(result.data, self.data[:, 0:1000])

        # Slice + time slice
        result = self.channel_frame[0:2, 500:1500]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 2
        assert result.n_samples == 1000
        np.testing.assert_array_equal(result.data, self.data[:, 500:1500])

        # All channels + time slice with step
        result = self.channel_frame[:, ::2]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 2
        assert result.n_samples == 8000
        np.testing.assert_array_equal(result.data, self.data[:, ::2])

        # NumPy array + time slice
        indices = np.array([0, 1])
        result = self.channel_frame[indices, 100:200]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 2
        assert result.n_samples == 100
        np.testing.assert_array_equal(result.data, self.data[:, 100:200])

        # Boolean mask + time slice
        mask = np.array([True, False])
        result = self.channel_frame[mask, 0:1000]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 1
        assert result.n_samples == 1000
        np.testing.assert_array_equal(result.data, self.data[0, 0:1000])

    def test_binary_op_with_channel_frame(self) -> None:
        """Test binary operations with another ChannelFrame."""
        # Create another ChannelFrame
        other_data = np.random.random((2, 16000))
        other_dask_data = _da_from_array(other_data, chunks=(1, 4000))
        other_cf = ChannelFrame(other_dask_data, self.sample_rate, label="other_audio")

        # Add the two ChannelFrames
        result = self.channel_frame + other_cf

        # Check result properties
        assert isinstance(result, ChannelFrame)
        assert result.sampling_rate == self.sample_rate
        assert result.n_channels == 2
        assert result.n_samples == 16000

        # Check computation results
        computed = result.compute()
        expected = self.data + other_data
        np.testing.assert_array_almost_equal(computed, expected)

        # Test sampling rate mismatch error
        other_cf = ChannelFrame(other_dask_data, 44100, label="other_audio")
        with pytest.raises(ValueError, match=r"Sampling rate mismatch"):
            _ = self.channel_frame + other_cf

    def test_add_method(self) -> None:
        """Test add method for adding signals."""
        # 通常の加算をテスト
        # Create another ChannelFrame
        other_data = np.random.random((2, 16000))
        other_dask_data = _da_from_array(other_data, chunks=(1, -1))
        other_cf = ChannelFrame(other_dask_data, self.sample_rate, label="other_audio")

        # addメソッドを使用して加算
        result = self.channel_frame.add(other_cf)

        # Check result properties
        assert isinstance(result, ChannelFrame)
        assert result.sampling_rate == self.sample_rate
        assert result.n_channels == 2
        assert result.n_samples == 16000

        # 実際の計算結果を確認
        computed = result.compute()
        expected = self.data + other_data
        np.testing.assert_array_almost_equal(computed, expected)

        # スカラー値との加算をテスト
        scalar_value = 0.5
        result = self.channel_frame.add(scalar_value)
        assert isinstance(result, ChannelFrame)
        computed = result.compute()
        expected = self.data + scalar_value
        np.testing.assert_array_almost_equal(computed, expected)

        # NumPy配列との加算をテスト
        array_value = np.random.random((2, 16000))
        result = self.channel_frame.add(array_value)
        assert isinstance(result, ChannelFrame)
        computed = result.compute()
        expected = self.data + array_value
        np.testing.assert_array_almost_equal(computed, expected)

        # サンプリングレートが一致しない場合のエラーをテスト
        mismatch_cf = ChannelFrame(other_dask_data, 44100, label="mismatch_audio")
        with pytest.raises(ValueError, match=r"Sampling rate mismatch"):
            _ = self.channel_frame.add(mismatch_cf)

    def test_channel_metadata_label_access(self) -> None:
        """Test accessing and modifying channel labels through metadata."""
        # Check default labels
        assert self.channel_frame.channels[0].label == "ch0"
        assert self.channel_frame.channels[1].label == "ch1"

        # Set new labels
        self.channel_frame.channels[0].label = "left"
        self.channel_frame.channels[1].label = "right"

        # Verify labels were changed
        assert self.channel_frame.channels[0].label == "left"
        assert self.channel_frame.channels[1].label == "right"

        # Test error handling with invalid channel index
        with pytest.raises(IndexError):
            _ = self.channel_frame.channels[5].label

        with pytest.raises(IndexError):
            self.channel_frame.channels[5].label = "invalid"

    def test_channel_metadata_unit_access(self) -> None:
        """Test accessing and modifying channel units through metadata."""
        # Check default units (empty string)
        assert self.channel_frame.channels[0].unit == ""

        # Set new units
        self.channel_frame.channels[0].unit = "Pa"
        self.channel_frame.channels[1].unit = "V"

        # Verify units were changed
        assert self.channel_frame.channels[0].unit == "Pa"
        assert self.channel_frame.channels[1].unit == "V"

    def test_channel_metadata_arbitrary_items(self) -> None:
        """Test getting and setting arbitrary metadata items."""
        # Set arbitrary metadata items
        self.channel_frame.channels[0]["gain"] = 0.5
        self.channel_frame.channels[1]["gain"] = 0.75
        self.channel_frame.channels[0]["device"] = "microphone"

        # Get items using __getitem__
        assert self.channel_frame.channels[0]["gain"] == 0.5
        assert self.channel_frame.channels[1]["gain"] == 0.75
        assert self.channel_frame.channels[0]["device"] == "microphone"

        # Check missing item returns None
        assert self.channel_frame.channels[0]["missing"] is None

    def test_channel_metadata_collection_getitem(self) -> None:
        """Test getting ChannelMetadata objects from collection."""
        # Get metadata objects for specific channels
        ch0_metadata = self.channel_frame.channels[0]
        ch1_metadata = self.channel_frame.channels[1]

        # Verify they're ChannelMetadata objects
        assert isinstance(ch0_metadata, ChannelMetadata)
        assert isinstance(ch1_metadata, ChannelMetadata)

        # Verify they reference the correct channels
        ch0_metadata.label = "test_ch0"
        assert self.channel_frame.channels[0].label == "test_ch0"

        # Test error handling with invalid index
        with pytest.raises(IndexError):
            _ = self.channel_frame.channels[5]

    def test_channel_metadata_on_new_channel_frame(self) -> None:
        """Test metadata preservation when creating derived ChannelFrames."""
        # Set metadata on original frame
        self.channel_frame.channels[0].label = "left"
        self.channel_frame.channels[0]["gain"] = 0.5
        self.channel_frame.channels[1].label = "right"

        # Create a derived ChannelFrame through an operation
        derived_frame = self.channel_frame + 1.0

        # Verify metadata was preserved
        assert (
            derived_frame.channels[0].label == "(left + 1.0)"
        )  # Underlying label preserved
        assert derived_frame.channels[0]["gain"] == 0.5
        assert derived_frame.channels[1].label == "(right + 1.0)"

        # Test metadata in extracted channel
        channel0 = self.channel_frame.get_channel(0)
        assert channel0.channels[0].label == "left"  # Label should be preserved
        assert channel0.channels[0]["gain"] == 0.5

    def test_from_numpy(self) -> None:
        """Test from_numpy method."""
        # Create a random array
        data = np.random.random((2, 16000))
        sampling_rate = 16000
        label = "test_audio"
        ch_labels = ["left", "right"]
        ch_units = ["Pa", "V"]
        metadata = {"gain": 0.5, "device": "microphone"}
        # Create a ChannelFrame from the numpy array
        cf = wd.from_numpy(
            data,
            sampling_rate=sampling_rate,
            label=label,
            ch_labels=ch_labels,
            ch_units=ch_units,
            metadata=metadata,
        )

        # Check data
        np.testing.assert_array_equal(cf.data, data)
        np.testing.assert_array_equal(cf[0].data, data[0])
        np.testing.assert_array_equal(cf[1].data, data[1])
        np.testing.assert_array_equal(cf[:, :1000].data, data[:, :1000])
        # np.testing.assert_array_equal(cf[0, :1000].data, data[0, :1000])
        # np.testing.assert_array_equal(cf[1, :1000].data, data[1, :1000])
        np.testing.assert_array_equal(cf["left"].data, data[0])
        np.testing.assert_array_equal(cf["right"].data, data[1])

        # Check properties
        assert cf.sampling_rate == sampling_rate
        assert cf.label == label
        assert cf.n_channels == 2
        assert cf.n_samples == 16000
        assert cf.channels[0].label == "left"
        assert cf.channels[1].label == "right"
        assert cf.channels[0].unit == "Pa"
        assert cf.channels[1].unit == "V"
        assert cf.metadata["gain"] == 0.5
        assert cf.metadata["device"] == "microphone"

        # Test ndim=1
        data_1d = np.random.random(16000)
        cf_1d = ChannelFrame.from_numpy(data_1d, sampling_rate=sampling_rate)
        # Check properties
        assert cf_1d.shape == (16000,)
        assert cf_1d.n_channels == 1

        # Test 3d array
        with pytest.raises(
            ValueError, match="Data must be 1-dimensional or 2-dimensional."
        ):
            ChannelFrame.from_numpy(
                np.random.random((3, 16000, 2)),
                sampling_rate=sampling_rate,
            )

    def test_from_file_lazy_loading(self) -> None:
        """Test that loading from file is lazy."""
        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename: str = temp_file.name
            sf.write(temp_filename, np.random.random((16000, 2)), 16000)
        re_test_data, _ = sf.read(temp_filename)
        re_test_data = re_test_data.T
        try:
            # Create mock array and patch from_delayed to return it
            mock_dask_array = mock.MagicMock(spec=DaArray)
            mock_data = np.random.random((2, 16000))
            mock_dask_array.compute.return_value = mock_data
            mock_dask_array.shape = (2, 16000)
            # Add ndim property to the mock
            mock_dask_array.ndim = 2

            # Mock the rechunk method to return the same mock
            mock_dask_array.rechunk.return_value = mock_dask_array

            # Patch necessary functions
            with (
                mock.patch("wandas.io.readers.get_file_reader") as mock_get_reader,
                mock.patch("dask.array.from_delayed", return_value=mock_dask_array),
                mock.patch("dask.delayed", return_value=mock.MagicMock()),
            ):
                # Set up mock reader
                mock_reader = mock.MagicMock()
                mock_reader.get_file_info.return_value = {
                    "samplerate": 16000,
                    "channels": 2,
                    "frames": 16000,
                }
                mock_reader.get_audio_data.return_value = mock_data
                mock_get_reader.return_value = mock_reader

                # Create ChannelFrame from file
                cf: ChannelFrame = ChannelFrame.from_file(temp_filename)

                # Check file reading hasn't happened yet
                mock_reader.get_audio_data.assert_not_called()

                # Access data to trigger computation
                data: NDArrayReal = cf.data

                # Verify data is correct
                np.testing.assert_array_equal(data, re_test_data)

                # Test with channel selection parameters
                cf = ChannelFrame.from_file(
                    temp_filename, channel=0, start=0.1, end=0.5
                )
                # assert cf.metadata["channels"] == [0]
                assert cf.channels[0].label == "ch0"
                # Test with multiple channels
                cf = ChannelFrame.from_file(temp_filename, channel=[0, 1])
                # assert cf.metadata["channels"] == [0, 1]
                assert cf.channels[0].label == "ch0"
                assert cf.channels[1].label == "ch1"
                # Test error cases
                with pytest.raises(
                    ValueError, match="Channel specification is out of range"
                ):
                    ChannelFrame.from_file(temp_filename, channel=5)

                with pytest.raises(
                    ValueError, match="Channel specification is out of range"
                ):
                    ChannelFrame.from_file(temp_filename, channel=[0, 5])

                with pytest.raises(
                    TypeError,
                    match="channel must be int, list, or None",
                ):
                    ChannelFrame.from_file(temp_filename, channel="invalid")  # type: ignore

                # Test file not found
                with pytest.raises(FileNotFoundError):
                    ChannelFrame.from_file("nonexistent_file.wav")

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    def test_describe_method(self) -> None:
        """Test the describe method for visual and audio display."""
        # Mock the display and Audio functions
        with (
            mock.patch("wandas.frames.channel.display") as mock_display,
            mock.patch(
                "wandas.frames.channel.Audio", return_value="mock_audio"
            ) as mock_audio,
        ):
            # Test basic describe method
            self.channel_frame.describe()

            # Verify display was called for each channel
            assert mock_display.call_count >= 2 * self.channel_frame.n_channels
            # One call for figure, one for Audio per channel

            # Verify Audio was called with correct parameters
            assert mock_audio.call_count == self.channel_frame.n_channels
            for call in mock_audio.call_args_list:
                assert call[1].get("normalize", True) is True

    def test_describe_method_with_axis_config(self) -> None:
        """Test describe method with legacy axis_config parameter."""
        with (
            mock.patch("wandas.frames.channel.display"),
            mock.patch("wandas.frames.channel.plt.close"),
            mock.patch("wandas.frames.channel.Audio", return_value="mock_audio"),
            mock.patch("wandas.frames.channel.logger") as mock_logger,
        ):
            # 古いパラメータ形式でdescribeを呼び出す
            axis_config = {
                "time_plot": {"xlim": [0, 1], "ylim": [-1, 1]},
                "freq_plot": {"xlim": [100, 1000], "ylim": [-60, 0]},
            }

            self.channel_frame.describe(axis_config=axis_config)

            # 警告メッセージが記録されたことを確認
            mock_logger.warning.assert_called_with(
                "axis_config is retained for backward compatibility but will be deprecated in the future."  # noqa: E501
            )

    def test_describe_method_with_cbar_config(self) -> None:
        """Test describe method with legacy cbar_config parameter."""
        with (
            mock.patch("wandas.frames.channel.display"),
            mock.patch("wandas.frames.channel.plt.close"),
            mock.patch("wandas.frames.channel.Audio", return_value="mock_audio"),
            mock.patch("wandas.frames.channel.logger") as mock_logger,
        ):
            # 古いパラメータ形式でdescribeを呼び出す
            cbar_config = {"vmin": -60, "vmax": 0}

            self.channel_frame.describe(cbar_config=cbar_config)

            # 警告メッセージが記録されたことを確認
            mock_logger.warning.assert_called_with(
                "cbar_config is retained for backward compatibility but will be deprecated in the future."  # noqa: E501
            )

    def test_describe_method_with_axes(self) -> None:
        """Test describe method when plot returns axes."""
        mock_ax = mock.MagicMock(spec=Axes)
        mock_ax.figure = mock.MagicMock()
        with mock.patch(
            "wandas.frames.channel.ChannelFrame.plot",
            return_value=mock_ax,  # 1つのAxesを返す
        ):
            with (
                mock.patch("wandas.frames.channel.display"),
                mock.patch("wandas.frames.channel.plt.close"),
                mock.patch("wandas.frames.channel.Audio"),
            ):
                self.channel_frame.describe()

    def test_describe_method_with_unexpected_plot_result(self) -> None:
        """Test describe method when plot returns unexpected type."""
        with mock.patch(
            "wandas.frames.channel.ChannelFrame.plot",
            return_value="not_an_axes_or_iterator",
        ):
            with pytest.raises(TypeError, match="Unexpected type for plot result"):
                self.channel_frame.describe()

    def test_describe_with_explicit_parameters(self) -> None:
        """Test describe method with new explicit parameters."""
        with (
            mock.patch("wandas.frames.channel.display"),
            mock.patch("wandas.frames.channel.plt.close"),
            mock.patch("wandas.frames.channel.Audio", return_value="mock_audio"),
            mock.patch("wandas.frames.channel.ChannelFrame.plot") as mock_plot,
        ):
            mock_ax = mock.MagicMock(spec=Axes)
            mock_ax.figure = mock.MagicMock()
            mock_plot.return_value = mock_ax

            # Test with explicit frequency parameters
            self.channel_frame.describe(
                fmin=100,
                fmax=5000,
                cmap="viridis",
                vmin=-80,
                vmax=-20,
            )

            # Verify plot was called with correct parameters
            mock_plot.assert_called()
            call_kwargs = mock_plot.call_args[1]
            assert call_kwargs["fmin"] == 100
            assert call_kwargs["fmax"] == 5000
            assert call_kwargs["cmap"] == "viridis"
            assert call_kwargs["vmin"] == -80
            assert call_kwargs["vmax"] == -20

    def test_describe_with_axis_limits(self) -> None:
        """Test describe method with axis limit parameters."""
        with (
            mock.patch("wandas.frames.channel.display"),
            mock.patch("wandas.frames.channel.plt.close"),
            mock.patch("wandas.frames.channel.Audio", return_value="mock_audio"),
            mock.patch("wandas.frames.channel.ChannelFrame.plot") as mock_plot,
        ):
            mock_ax = mock.MagicMock(spec=Axes)
            mock_ax.figure = mock.MagicMock()
            mock_plot.return_value = mock_ax

            # Test with axis limits
            self.channel_frame.describe(
                xlim=(0, 5),
                ylim=(20, 20000),
            )

            call_kwargs = mock_plot.call_args[1]
            assert call_kwargs["xlim"] == (0, 5)
            assert call_kwargs["ylim"] == (20, 20000)

    def test_describe_with_a_weighting(self) -> None:
        """Test describe method with A-weighting parameter."""
        with (
            mock.patch("wandas.frames.channel.display"),
            mock.patch("wandas.frames.channel.plt.close"),
            mock.patch("wandas.frames.channel.Audio", return_value="mock_audio"),
            mock.patch("wandas.frames.channel.ChannelFrame.plot") as mock_plot,
        ):
            mock_ax = mock.MagicMock(spec=Axes)
            mock_ax.figure = mock.MagicMock()
            mock_plot.return_value = mock_ax

            # Test with A-weighting
            self.channel_frame.describe(Aw=True)

            call_kwargs = mock_plot.call_args[1]
            assert call_kwargs["Aw"] is True

    def test_describe_with_subplot_configs(self) -> None:
        """Test describe method with waveform and spectral configurations."""
        with (
            mock.patch("wandas.frames.channel.display"),
            mock.patch("wandas.frames.channel.plt.close"),
            mock.patch("wandas.frames.channel.Audio", return_value="mock_audio"),
            mock.patch("wandas.frames.channel.ChannelFrame.plot") as mock_plot,
        ):
            mock_ax = mock.MagicMock(spec=Axes)
            mock_ax.figure = mock.MagicMock()
            mock_plot.return_value = mock_ax

            # Test with subplot configs
            waveform_config = {"ylabel": "Sound Pressure [Pa]", "xlim": (0, 10)}
            spectral_config = {"ylabel": "SPL [dB]", "xlim": (-80, -20)}

            self.channel_frame.describe(
                waveform=waveform_config,
                spectral=spectral_config,
            )

            call_kwargs = mock_plot.call_args[1]
            assert call_kwargs["waveform"] == waveform_config
            assert call_kwargs["spectral"] == spectral_config

    def test_describe_normalize_and_close_params(self) -> None:
        """Test describe method with normalize and is_close parameters."""
        with (
            mock.patch("wandas.frames.channel.display"),
            mock.patch("wandas.frames.channel.plt.close") as mock_close,
            mock.patch("wandas.frames.channel.Audio") as mock_audio,
            mock.patch("wandas.frames.channel.ChannelFrame.plot") as mock_plot,
        ):
            mock_ax = mock.MagicMock(spec=Axes)
            mock_ax.figure = mock.MagicMock()
            mock_plot.return_value = mock_ax

            # Test with normalize=False and is_close=False
            self.channel_frame.describe(normalize=False, is_close=False)

            # Verify Audio was called with normalize=False
            for call in mock_audio.call_args_list:
                assert call[1].get("normalize") is False

            # Verify plt.close was not called
            mock_close.assert_not_called()

    def test_describe_backward_compatibility_warning(self) -> None:
        """Test that using deprecated parameters shows warning."""
        with (
            mock.patch("wandas.frames.channel.display"),
            mock.patch("wandas.frames.channel.plt.close"),
            mock.patch("wandas.frames.channel.Audio", return_value="mock_audio"),
            mock.patch("wandas.frames.channel.logger") as mock_logger,
        ):
            # Use deprecated axis_config
            self.channel_frame.describe(axis_config={"time_plot": {"ylabel": "Custom"}})

            # Verify warning was logged
            assert any(
                "backward compatibility" in str(call)
                for call in mock_logger.warning.call_args_list
            )


class TestDescribeIntegration:
    """Integration tests for describe() method with actual execution."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Create a simple test signal
        self.sample_rate = 16000
        t = np.linspace(0, 1, self.sample_rate)
        # 440Hz sine wave (A4 note)
        signal = np.sin(2 * np.pi * 440 * t)
        self.data = signal.reshape(1, -1)
        self.channel_frame = ChannelFrame.from_numpy(
            data=self.data, sampling_rate=self.sample_rate, label="test_sine"
        )

    def test_describe_integration_basic(self) -> None:
        """Test describe() actually executes with default parameters."""
        with (
            mock.patch("wandas.frames.channel.display"),
            mock.patch("wandas.frames.channel.Audio"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            # Should not raise any exceptions
            self.channel_frame.describe()

    def test_describe_integration_with_explicit_params(self) -> None:
        """Test describe() with explicit frequency parameters."""
        with (
            mock.patch("wandas.frames.channel.display"),
            mock.patch("wandas.frames.channel.Audio"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            # Test with various parameter combinations
            self.channel_frame.describe(
                fmin=100, fmax=5000, cmap="viridis", vmin=-80, vmax=-20
            )

    def test_describe_integration_with_axis_limits(self) -> None:
        """Test describe() with axis limits."""
        with (
            mock.patch("wandas.frames.channel.display"),
            mock.patch("wandas.frames.channel.Audio"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            self.channel_frame.describe(xlim=(0, 0.5), ylim=(100, 1000))

    def test_describe_integration_with_a_weighting(self) -> None:
        """Test describe() with A-weighting."""
        with (
            mock.patch("wandas.frames.channel.display"),
            mock.patch("wandas.frames.channel.Audio"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            self.channel_frame.describe(Aw=True)

    def test_describe_integration_with_subplot_configs(self) -> None:
        """Test describe() with waveform and spectral configurations."""
        with (
            mock.patch("wandas.frames.channel.display"),
            mock.patch("wandas.frames.channel.Audio"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            waveform_config = {"ylabel": "Amplitude [V]", "xlim": (0, 0.5)}
            spectral_config = {"ylabel": "Power [dB]", "xlim": (-60, 0)}

            self.channel_frame.describe(
                waveform=waveform_config, spectral=spectral_config
            )

    def test_describe_integration_combined_params(self) -> None:
        """Test describe() with multiple parameters combined."""
        with (
            mock.patch("wandas.frames.channel.display"),
            mock.patch("wandas.frames.channel.Audio"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            self.channel_frame.describe(
                fmin=20,
                fmax=8000,
                cmap="magma",
                vmin=-90,
                vmax=-10,
                xlim=(0, 0.5),
                ylim=(20, 8000),
                Aw=True,
                waveform={"ylabel": "Sound Pressure [Pa]"},
                spectral={"ylabel": "SPL [dBA]"},
                normalize=False,
                is_close=True,
            )

    def test_describe_integration_typeddict_params(self) -> None:
        """Test describe() using TypedDict configuration."""
        from wandas.visualization.types import DescribeParams

        # Create configuration using TypedDict
        config: DescribeParams = {
            "fmin": 100,
            "fmax": 5000,
            "cmap": "viridis",
            "Aw": True,
            "vmin": -80,
            "vmax": -20,
            "normalize": False,
        }

        with (
            mock.patch("wandas.frames.channel.display"),
            mock.patch("wandas.frames.channel.Audio"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            # Expand TypedDict as kwargs
            self.channel_frame.describe(**config)

    def test_describe_integration_plot_is_created(self) -> None:
        """Test that describe() actually creates plots."""
        with (
            mock.patch("wandas.frames.channel.display") as mock_display,
            mock.patch("wandas.frames.channel.Audio"),
        ):
            self.channel_frame.describe(is_close=False)

            # Verify plot was created (display was called)
            assert mock_display.call_count > 0

    def test_describe_integration_stft_computation(self) -> None:
        """Test that describe() computes STFT correctly."""
        with (
            mock.patch("wandas.frames.channel.display"),
            mock.patch("wandas.frames.channel.Audio"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            # This should trigger STFT computation
            self.channel_frame.describe(fmin=100, fmax=5000)

            # If we got here without exception, STFT worked

    def test_describe_integration_welch_computation(self) -> None:
        """Test that describe() computes Welch spectrum correctly."""
        with (
            mock.patch("wandas.frames.channel.display"),
            mock.patch("wandas.frames.channel.Audio"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            # This should trigger Welch computation
            self.channel_frame.describe(Aw=True)

            # If we got here without exception, Welch worked

    def test_describe_integration_multi_channel(self) -> None:
        """Test describe() with multi-channel signal."""
        # Create 2-channel signal
        t = np.linspace(0, 1, self.sample_rate)
        signal1 = np.sin(2 * np.pi * 440 * t)
        signal2 = np.sin(2 * np.pi * 880 * t)
        multi_data = np.vstack([signal1, signal2])

        cf_multi = ChannelFrame.from_numpy(
            data=multi_data, sampling_rate=self.sample_rate, label="test_multi"
        )

        with (
            mock.patch("wandas.frames.channel.display"),
            mock.patch("wandas.frames.channel.Audio"),
            mock.patch("matplotlib.pyplot.close"),
        ):
            cf_multi.describe(fmin=100, fmax=5000)

    def test_describe_integration_backward_compat(self) -> None:
        """Test describe() with deprecated parameters still works."""
        with (
            mock.patch("wandas.frames.channel.display"),
            mock.patch("wandas.frames.channel.Audio"),
            mock.patch("matplotlib.pyplot.close"),
            mock.patch("wandas.frames.channel.logger"),
        ):
            # Old style parameters should still work
            axis_config = {
                "time_plot": {"ylabel": "Custom"},
                "freq_plot": {"xlim": (-80, -20), "ylim": (100, 5000)},
            }
            cbar_config = {"vmin": -90, "vmax": -10}

            self.channel_frame.describe(
                axis_config=axis_config, cbar_config=cbar_config
            )

            # Should complete without errors

    def test_read_csv(self) -> None:
        """Test read_csv method."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
            temp_filename = temp_file.name
            # 時間と2つの値を持つCSVファイルを作成
            header = "time,value1,value2\n"
            data = "\n".join([f"{i / 16000},{1.1},{2.2}" for i in range(16000)])
            temp_file.write(header.encode())
            temp_file.write(data.encode())
            temp_file.flush()
            temp_file.seek(0)
            # Close the file to ensure it's written
            temp_file.close()

        try:
            # Read the CSV file into a ChannelFrame
            cf = ChannelFrame.read_csv(temp_filename)

            # Check properties
            assert cf.sampling_rate == 16000
            assert cf.n_channels == 2
            assert cf.n_samples == 16000
            assert cf.label == Path(temp_filename).stem

            # Check data
            expected_data = np.loadtxt(temp_filename, delimiter=",", skiprows=1).T
            np.testing.assert_array_equal(cf.data, expected_data[1:])

        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    def test_debug_info(self) -> None:
        """Test debug_info method."""
        with mock.patch("wandas.core.base_frame.logger") as mock_logger:
            self.channel_frame.debug_info()
            assert mock_logger.debug.call_count >= 6  # At least 6 debug messages

    def test_read_wav_class_method(self) -> None:
        """Test read_wav class method."""
        with mock.patch.object(ChannelFrame, "from_file") as mock_from_file:
            mock_from_file.return_value = self.channel_frame
            result = ChannelFrame.read_wav("test.wav", labels=["left", "right"])
            mock_from_file.assert_called_with("test.wav", ch_labels=["left", "right"])
            assert result is self.channel_frame

    def test_visualize_graph(self) -> None:
        """Test visualize_graph method."""
        # Test successful visualization - returns mock object from visualize()
        with mock.patch.object(DaArray, "visualize") as mock_visualize:
            mock_return_value = mock.MagicMock()
            mock_visualize.return_value = mock_return_value
            result = self.channel_frame.visualize_graph()
            mock_visualize.assert_called_once()
            # visualize_graph returns the result from _data.visualize()
            assert result is mock_return_value

        # Test with provided filename
        with mock.patch.object(DaArray, "visualize") as mock_visualize:
            mock_return_value = mock.MagicMock()
            mock_visualize.return_value = mock_return_value
            custom_filename = "test_graph.png"
            result = self.channel_frame.visualize_graph(filename=custom_filename)
            mock_visualize.assert_called_with(filename=custom_filename)
            # Returns the mock object from visualize()
            assert result is mock_return_value

        # Test handling of visualization error
        with mock.patch.object(
            DaArray, "visualize", side_effect=Exception("Test error")
        ):
            result = self.channel_frame.visualize_graph()
            assert result is None

    def test_rms_property(self) -> None:
        """Test RMS property calculation."""
        # Create a known signal for testing RMS
        # Channel 1: constant value of 2.0
        # Channel 2: constant value of 3.0
        data = np.array([[2.0] * 1000, [3.0] * 1000])
        dask_data = _da_from_array(data, chunks=(1, 100))
        cf = ChannelFrame(data=dask_data, sampling_rate=16000)

        # Calculate RMS
        rms_values = cf.rms

        # For a constant signal, RMS equals the absolute value
        expected_rms = np.array([2.0, 3.0])
        np.testing.assert_array_almost_equal(rms_values, expected_rms)

    def test_rms_property_with_sine_wave(self) -> None:
        """Test RMS property with sine wave signals."""
        # For a sine wave, RMS = amplitude / sqrt(2)
        sample_rate = 16000
        duration = 1.0
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        # Create two sine waves with different amplitudes
        amp1 = 1.0
        amp2 = 2.0
        freq = 440  # Hz

        channel1 = amp1 * np.sin(2 * np.pi * freq * t)
        channel2 = amp2 * np.sin(2 * np.pi * freq * t)

        data = np.vstack([channel1, channel2])
        dask_data = _da_from_array(data, chunks=(1, 1000))
        cf = ChannelFrame(data=dask_data, sampling_rate=sample_rate)

        # Calculate RMS
        rms_values = cf.rms

        # Expected RMS for sine wave: amplitude / sqrt(2)
        expected_rms = np.array([amp1 / np.sqrt(2), amp2 / np.sqrt(2)])
        np.testing.assert_array_almost_equal(rms_values, expected_rms, decimal=5)

    def test_rms_property_single_channel(self) -> None:
        """Test RMS property with single channel."""
        # Single channel with known values
        data = np.array([[1.0, 2.0, 3.0, 4.0]])
        dask_data = _da_from_array(data, chunks=(1, 2))
        cf = ChannelFrame(data=dask_data, sampling_rate=16000)

        rms_values = cf.rms

        # Calculate expected RMS: sqrt(mean([1, 4, 9, 16]))
        expected_rms = np.sqrt(np.mean([1.0, 4.0, 9.0, 16.0]))
        np.testing.assert_array_almost_equal(rms_values, [expected_rms])

    def test_rms_property_indexing(self) -> None:
        """Test using RMS property for conditional indexing."""
        # Create channels with different RMS values
        data = np.array([[1.0] * 1000, [2.0] * 1000, [3.0] * 1000])
        dask_data = _da_from_array(data, chunks=(1, 100))
        cf = ChannelFrame(data=dask_data, sampling_rate=16000)

        # Get RMS values
        rms_values = cf.rms
        assert len(rms_values) == 3

        # Create boolean mask based on RMS threshold
        threshold = 1.5
        mask = rms_values > threshold
        expected_mask = np.array([False, True, True])
        np.testing.assert_array_equal(mask, expected_mask)

        # Use mask to select channels
        filtered_cf = cf[mask]
        assert filtered_cf.n_channels == 2

        # Verify the selected channels have correct RMS values
        filtered_rms = filtered_cf.rms
        assert all(filtered_rms > threshold)

    def test_rms_property_with_sorting(self) -> None:
        """Test using RMS property to sort and select top channels."""
        # Create channels with different RMS values
        data = np.array([[1.0] * 1000, [3.0] * 1000, [2.0] * 1000])
        dask_data = _da_from_array(data, chunks=(1, 100))
        cf = ChannelFrame(data=dask_data, sampling_rate=16000)

        # Get top 2 channels by RMS
        rms_values = cf.rms
        top_n = 2
        top_indices = np.argsort(rms_values)[::-1][:top_n]

        # Select top channels
        top_channels = cf[top_indices]
        assert top_channels.n_channels == top_n

        # Verify they are indeed the top channels
        top_rms = top_channels.rms
        assert np.all(top_rms >= 2.0)  # Channels with RMS 3.0 and 2.0


class TestBaseFrameExceptionHandling:
    """BaseFrameの例外処理をテスト（ChannelFrameを通じて間接的にテスト）"""

    def setup_method(self) -> None:
        """テストフィクスチャのセットアップ"""
        self.sample_rate = 16000
        self.data = np.random.random((2, 16000))
        self.dask_data = _da_from_array(self.data, chunks=(1, 4000))
        self.channel_frame = ChannelFrame(
            data=self.dask_data, sampling_rate=self.sample_rate, label="test_audio"
        )

    def test_getitem_empty_list_error(self) -> None:
        """空のリストでインデックスするとValueErrorが発生することをテスト"""
        with pytest.raises(ValueError, match="Cannot index with an empty list"):
            _ = self.channel_frame[[]]

    def test_getitem_mixed_type_list_error(self) -> None:
        """混合型のリストでインデックスするとTypeErrorが発生することをテスト"""
        with pytest.raises(TypeError, match="List must contain all str or all int"):
            _ = self.channel_frame[[0, "ch1"]]  # type: ignore

    def test_getitem_invalid_numpy_dtype_error(self) -> None:
        """無効なdtypeのNumPy配列でIndexErrorが発生することをテスト"""
        # float型のNumPy配列
        float_array = np.array([0.5, 1.5])
        with pytest.raises(
            TypeError, match="NumPy array must be of integer or boolean type"
        ):
            _ = self.channel_frame[float_array]

    def test_handle_multidim_indexing_invalid_key_length(self) -> None:
        """多次元インデックスで無効なキー長のテスト"""
        # データは2次元なので、3次元以上のインデックスは無効
        with pytest.raises(ValueError, match="Invalid key length"):
            _ = self.channel_frame[0, 0, 0]

    def test_handle_multidim_indexing_invalid_channel_key_type(self) -> None:
        """多次元インデックスで無効なチャネルキー型のテスト"""
        # 浮動小数点数は無効なチャネルキー
        with pytest.raises(TypeError, match="Invalid channel key type in tuple"):
            _ = self.channel_frame[1.5, :]  # type: ignore

    def test_label2index_key_error(self) -> None:
        """存在しないラベルでKeyErrorが発生することをテスト"""
        with pytest.raises(KeyError, match="Channel label .* not found"):
            _ = self.channel_frame.label2index("nonexistent_label")

    def test_label2index_with_valid_label(self) -> None:
        """有効なラベルでインデックスを取得できることをテスト"""
        index = self.channel_frame.label2index("ch0")
        assert index == 0

        index = self.channel_frame.label2index("ch1")
        assert index == 1

    def test_create_new_instance_invalid_label_type(self) -> None:
        """_create_new_instanceで無効なlabel型のテスト"""
        with pytest.raises(TypeError, match="Label must be a string"):
            self.channel_frame._create_new_instance(
                data=self.dask_data,
                label=123,  # type: ignore
            )

    def test_create_new_instance_invalid_metadata_type(self) -> None:
        """_create_new_instanceで無効なmetadata型のテスト"""
        with pytest.raises(TypeError, match="Metadata must be a dictionary"):
            self.channel_frame._create_new_instance(
                data=self.dask_data,
                metadata="invalid",  # type: ignore
            )

    def test_create_new_instance_invalid_channel_metadata_type(self) -> None:
        """_create_new_instanceで無効なchannel_metadata型のテスト"""
        with pytest.raises(TypeError, match="Channel metadata must be a list"):
            self.channel_frame._create_new_instance(
                data=self.dask_data,
                channel_metadata="invalid",  # type: ignore
            )

    def test_compute_non_ndarray_result(self) -> None:
        """computeで非ndarray結果のテスト"""
        # Mock the compute method to return something other than ndarray
        with mock.patch.object(DaArray, "compute", return_value="not_an_array"):
            with pytest.raises(ValueError, match="Computed result is not a np.ndarray"):
                _ = self.channel_frame.compute()

    # Error Message Tests (Phase 1 Improvements)
    def test_invalid_data_shape_error_message(self) -> None:
        """Test that invalid data shape provides helpful error message."""
        data_3d = np.random.random((2, 3, 4))  # 3D array
        dask_data_3d = _da_from_array(data_3d)

        with pytest.raises(ValueError) as exc_info:
            ChannelFrame(data=dask_data_3d, sampling_rate=16000)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Invalid data shape" in error_msg
        assert "(2, 3, 4)" in error_msg
        assert "3D" in error_msg
        # Check WHY
        assert "Expected: 1D" in error_msg
        assert "2D" in error_msg
        # Check HOW
        assert "reshape" in error_msg.lower()
        assert "Example:" in error_msg

    def test_negative_sampling_rate_error_message(self) -> None:
        """Test that negative sampling rate provides helpful error message."""
        with pytest.raises(ValueError) as exc_info:
            ChannelFrame(data=self.dask_data, sampling_rate=-44100)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Invalid sampling_rate" in error_msg
        assert "-44100" in error_msg
        # Check WHY
        assert "Positive value" in error_msg
        # Check HOW
        assert "Common values:" in error_msg
        assert "44100" in error_msg

    def test_zero_sampling_rate_error_message(self) -> None:
        """Test that zero sampling rate provides helpful error message."""
        with pytest.raises(ValueError) as exc_info:
            ChannelFrame(data=self.dask_data, sampling_rate=0)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Invalid sampling_rate" in error_msg
        assert "Got: 0 Hz" in error_msg
        # Check WHY
        assert "Expected: Positive value > 0" in error_msg
        # Check HOW
        assert (
            "Sampling rate represents samples per second and must be positive."
            in error_msg
        )
        assert "Common values: 8000, 16000, 22050, 44100, 48000 Hz" in error_msg

    def test_file_not_found_error_message(self) -> None:
        """Test that missing file provides helpful error message."""
        fake_path = "/nonexistent/path/to/audio.wav"

        with pytest.raises(FileNotFoundError) as exc_info:
            ChannelFrame.from_file(fake_path)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Audio file not found" in error_msg
        # Cross-platform: check for key path components instead of exact path
        assert "nonexistent" in error_msg and "audio.wav" in error_msg
        # Check WHY (context)
        assert "Current directory:" in error_msg
        # Check HOW
        assert "check" in error_msg.lower()
        assert "File path is correct" in error_msg
        assert "File exists" in error_msg


class TestFadeIntegration:
    """Integration tests for fade functionality with other operations."""

    def setup_method(self) -> None:
        """Set up test fixtures for fade integration tests."""
        # Create a test signal with known properties
        self.sample_rate = 16000
        duration = 1.0  # 1 second
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        # Create a sine wave with amplitude 1.0
        freq = 440  # Hz
        signal = np.sin(2 * np.pi * freq * t)
        self.data = signal.reshape(1, -1)  # Single channel
        self.dask_data = _da_from_array(self.data, chunks=(1, 4000))
        self.channel_frame = ChannelFrame(
            data=self.dask_data, sampling_rate=self.sample_rate, label="test_sine"
        )

    def test_fade_preserves_rms_calculation(self) -> None:
        """Test that fade operation allows RMS calculation to work correctly."""
        # Apply fade
        faded = self.channel_frame.fade(fade_ms=100.0)

        # RMS should be calculable without errors
        rms_values = faded.rms
        assert len(rms_values) == 1
        assert rms_values[0] > 0  # RMS should be positive

        # Faded signal should have lower RMS than original (due to fade-in/out)
        original_rms = self.channel_frame.rms
        assert rms_values[0] < original_rms[0]

    def test_fade_with_normalize_chain(self) -> None:
        """Test fade operation chained with normalize."""
        # Chain fade and normalize operations
        processed = self.channel_frame.fade(fade_ms=50.0).normalize()

        # Should complete without errors
        assert isinstance(processed, ChannelFrame)
        assert processed.sampling_rate == self.sample_rate
        assert processed.n_channels == 1
        assert processed.n_samples == self.channel_frame.n_samples

        # Check that operation history is recorded
        assert len(processed.operation_history) == 2
        assert processed.operation_history[0]["operation"] == "fade"
        assert processed.operation_history[1]["operation"] == "normalize"

        # Normalized signal should have max amplitude of 1.0
        max_amplitude = np.max(np.abs(processed.data))
        np.testing.assert_almost_equal(max_amplitude, 1.0, decimal=6)

    def test_fade_with_filter_chain(self) -> None:
        """Test fade operation chained with filtering."""
        # Chain fade and low-pass filter
        processed = self.channel_frame.fade(fade_ms=50.0).low_pass_filter(cutoff=1000)

        # Should complete without errors
        assert isinstance(processed, ChannelFrame)
        assert processed.sampling_rate == self.sample_rate

        # Check operation history
        assert len(processed.operation_history) == 2
        assert processed.operation_history[0]["operation"] == "fade"
        assert processed.operation_history[1]["operation"] == "lowpass_filter"

    def test_fade_with_multiple_operations_chain(self) -> None:
        """Test fade in a complex operation chain."""

        # Create a more complex processing chain
        processed = (
            self.channel_frame.fade(fade_ms=25.0)
            .normalize()
            .low_pass_filter(cutoff=2000)
            .high_pass_filter(cutoff=100)
        )

        # Should complete without errors
        assert isinstance(processed, ChannelFrame)
        assert processed.sampling_rate == self.sample_rate

        # Check that all operations are recorded
        assert len(processed.operation_history) == 4
        operations = [op["operation"] for op in processed.operation_history]
        assert operations == ["fade", "normalize", "lowpass_filter", "highpass_filter"]

    def test_fade_with_channel_operations(self) -> None:
        """Test fade with channel selection and operations."""
        # Create multi-channel signal
        multi_data = np.vstack([self.data[0], self.data[0] * 0.5])  # 2 channels
        multi_dask = _da_from_array(multi_data, chunks=(1, 4000))
        multi_frame = ChannelFrame(
            data=multi_dask, sampling_rate=self.sample_rate, label="multi_test"
        )

        # Apply fade to all channels, then select one channel
        processed = multi_frame.fade(fade_ms=50.0).get_channel(0)

        # Should work correctly
        assert isinstance(processed, ChannelFrame)
        assert processed.n_channels == 1
        assert processed.operation_history[-1]["operation"] == "fade"

    def test_fade_with_arithmetic_operations(self) -> None:
        """Test fade with arithmetic operations."""
        # Apply fade, then add a constant
        processed = self.channel_frame.fade(fade_ms=50.0) + 0.1

        # Should complete without errors
        assert isinstance(processed, ChannelFrame)
        assert processed.operation_history[0]["operation"] == "fade"

        # Check that arithmetic operation is recorded
        assert "+" in str(processed.operation_history[1]["operation"])

    def test_fade_preserves_metadata_and_labels(self) -> None:
        """Test that fade preserves channel metadata and labels."""
        # Set custom labels and metadata
        self.channel_frame.channels[0].label = "test_channel"
        self.channel_frame.channels[0]["gain"] = 0.8
        self.channel_frame.metadata["test_key"] = "test_value"

        # Apply fade
        faded = self.channel_frame.fade(fade_ms=50.0)

        # Check that label is updated to reflect the operation
        assert faded.channels[0].label == "fade(test_channel)"
        # Check that other metadata is preserved
        assert faded.channels[0]["gain"] == 0.8
        assert faded.metadata["test_key"] == "test_value"

        # Check operation history
        assert faded.operation_history[0]["operation"] == "fade"
        assert faded.operation_history[0]["params"]["fade_ms"] == 50.0

    def test_fade_with_file_io_roundtrip(self) -> None:
        """Test fade operation with file save/load roundtrip."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name

        try:
            # Apply fade and save
            faded = self.channel_frame.fade(fade_ms=50.0)
            faded.to_wav(temp_filename)

            # Load back and verify
            loaded = ChannelFrame.from_file(temp_filename)

            # Should be able to load and have same basic properties
            assert loaded.sampling_rate == self.sample_rate
            assert loaded.n_channels == 1
            assert loaded.n_samples == self.channel_frame.n_samples

            # Data should be different (faded) but same shape
            assert loaded.data.shape == faded.data.shape

        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    def test_fade_with_different_fade_durations(self) -> None:
        """Test fade with different fade durations and verify effects."""
        # Test with very short fade
        short_fade = self.channel_frame.fade(fade_ms=1.0)
        short_rms = short_fade.rms[0]

        # Test with longer fade
        long_fade = self.channel_frame.fade(fade_ms=100.0)
        long_rms = long_fade.rms[0]

        # Longer fade should result in lower RMS (more signal attenuated)
        assert long_rms < short_rms

        # Both should be less than original
        original_rms = self.channel_frame.rms[0]
        assert short_rms < original_rms
        assert long_rms < original_rms

    def test_fade_with_multi_channel_signal(self) -> None:
        """Test fade with multi-channel signal."""
        # Create 3-channel signal
        multi_data = np.vstack(
            [
                self.data[0],  # Original
                self.data[0] * 0.7,  # Scaled down
                self.data[0] * 1.3,  # Scaled up
            ]
        )
        multi_dask = _da_from_array(multi_data, chunks=(1, 4000))
        multi_frame = ChannelFrame(
            data=multi_dask, sampling_rate=self.sample_rate, label="multi_test"
        )

        # Apply fade
        faded = multi_frame.fade(fade_ms=50.0)

        # Should work for all channels
        assert faded.n_channels == 3
        assert faded.operation_history[0]["operation"] == "fade"

        # Each channel should have different RMS due to different amplitudes
        rms_values = faded.rms
        assert len(rms_values) == 3
        assert rms_values[0] > rms_values[1]  # Original > scaled down
        assert rms_values[2] > rms_values[0]  # Scaled up > original

    def test_fade_lazy_evaluation_preserved(self) -> None:
        """Test that fade preserves lazy evaluation."""
        # Apply fade without computing
        faded = self.channel_frame.fade(fade_ms=50.0)

        # Should still be lazy (dask array)
        assert isinstance(faded._data, DaArray)

        # Operation history should be updated without computation
        assert len(faded.operation_history) == 1
        assert faded.operation_history[0]["operation"] == "fade"

        # Only when we access .data should computation happen
        with mock.patch.object(
            DaArray, "compute", return_value=self.data
        ) as mock_compute:
            _ = faded.data
            mock_compute.assert_called_once()

    def test_fade_with_visualization(self) -> None:
        """Test that faded signal can be visualized."""
        # Apply fade
        faded = self.channel_frame.fade(fade_ms=50.0)

        # Should be able to create plots without errors
        with (
            mock.patch("matplotlib.pyplot.figure"),
            mock.patch("matplotlib.pyplot.subplot"),
            mock.patch("matplotlib.axes.Axes.plot"),
            mock.patch("matplotlib.pyplot.tight_layout"),
            mock.patch("matplotlib.pyplot.show"),
        ):
            # Basic plot should work
            faded.plot()

            # RMS plot should work
            faded.rms_plot()

    def test_fade_error_handling_integration(self) -> None:
        """Test error handling in fade operation within processing chains."""
        # Test with invalid fade_ms (too long)
        # Create very short signal
        short_data = self.data[:, :100]  # Only 100 samples
        short_dask = _da_from_array(short_data, chunks=(1, 50))
        short_frame = ChannelFrame(
            data=short_dask, sampling_rate=self.sample_rate, label="short"
        )

        # Apply fade with duration longer than signal
        # Should not fail immediately due to lazy eval
        faded_short = short_frame.fade(fade_ms=10.0)

        # Error should occur when we try to compute the result
        with pytest.raises(ValueError, match="Fade length too long"):
            _ = faded_short.data

        # Test with negative fade_ms - fails during operation creation
        with pytest.raises(ValueError, match="fade_ms must be non-negative"):
            self.channel_frame.fade(fade_ms=-1.0)

    def test_to_numpy(self) -> None:
        """Test to_numpy method converts frame data to NumPy array."""
        # For single channel, to_numpy should return 1D array (squeezed)
        if self.data.shape[0] == 1:
            expected_result = self.data.squeeze(axis=0)
        else:
            expected_result = self.data

        # Test with mock to ensure compute is called
        with mock.patch.object(
            self.channel_frame, "compute", return_value=self.data
        ) as mock_compute:
            result = self.channel_frame.to_numpy()
            mock_compute.assert_called_once()
            np.testing.assert_array_equal(result, expected_result)
            assert isinstance(result, np.ndarray)

    def test_to_dataframe(self) -> None:
        """Test to_dataframe method converts frame data to pandas DataFrame."""
        # Test basic conversion
        df = self.channel_frame.to_dataframe()

        # Check DataFrame properties
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (
            self.data.shape[1],
            self.data.shape[0],
        )  # (n_samples, n_channels)
        assert list(df.columns) == ["ch0"]  # Single channel
        assert df.index.name == "time"

        # Check data values (transposed)
        np.testing.assert_array_equal(df.values.T, self.data)

        # Check time index
        expected_time = np.arange(self.data.shape[1]) / self.sample_rate
        np.testing.assert_array_equal(df.index.values, expected_time)

    def test_to_dataframe_with_custom_labels(self) -> None:
        """Test to_dataframe with custom channel labels."""
        # Set custom labels
        self.channel_frame.channels[0].label = "left"

        df = self.channel_frame.to_dataframe()

        # Check column names
        assert list(df.columns) == ["left"]

    def test_to_dataframe_single_channel(self) -> None:
        """Test to_dataframe with single channel."""
        # Get single channel (already single channel)
        single_channel = self.channel_frame.get_channel(0)

        df = single_channel.to_dataframe()

        # Check DataFrame properties
        assert df.shape == (self.data.shape[1], 1)  # (n_samples, 1)
        assert list(df.columns) == ["ch0"]
        assert df.index.name == "time"

        # Check data values
        np.testing.assert_array_equal(df.values.flatten(), self.data[0])

    def test_info_method_basic(self, capsys: Any) -> None:
        """Test info() method displays correct information."""
        self.channel_frame.info()

        captured = capsys.readouterr()
        output = captured.out

        # Verify all expected information is present
        assert "Channels: 1" in output
        assert f"Sampling rate: {self.sample_rate} Hz" in output
        assert "Duration: 1.0 s" in output
        assert f"Samples: {self.channel_frame.n_samples}" in output
        assert "Channel labels: ['ch0']" in output

    def test_info_method_single_channel(self, capsys: Any) -> None:
        """Test info() method with single channel."""
        single_data = self.data[0:1]
        single_frame = ChannelFrame.from_numpy(
            single_data, sampling_rate=self.sample_rate, label="single"
        )

        single_frame.info()

        captured = capsys.readouterr()
        output = captured.out

        assert "Channels: 1" in output
        assert "Channel labels: ['ch0']" in output

    def test_info_method_custom_labels(self, capsys: Any) -> None:
        """Test info() method with custom channel labels."""
        # Set custom labels
        self.channel_frame.channels[0].label = "left"

        self.channel_frame.info()

        captured = capsys.readouterr()
        output = captured.out

        assert "Channel labels: ['left']" in output

    def test_info_method_different_duration(self, capsys: Any) -> None:
        """Test info() method with different durations."""
        # Create a frame with 0.5 seconds of data
        short_data = np.random.random((1, 8000))
        short_frame = ChannelFrame.from_numpy(
            short_data, sampling_rate=self.sample_rate, label="short"
        )

        short_frame.info()

        captured = capsys.readouterr()
        output = captured.out

        assert "Duration: 0.5 s" in output
        assert "Samples: 8000" in output

    # Error Message Tests (Phase 1 Improvements)
    def test_invalid_data_shape_error_message(self) -> None:
        """Test that invalid data shape provides helpful error message."""
        data_3d = np.random.random((2, 3, 4))  # 3D array
        dask_data_3d = _da_from_array(data_3d)

        with pytest.raises(ValueError) as exc_info:
            ChannelFrame(data=dask_data_3d, sampling_rate=16000)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Invalid data shape" in error_msg
        assert "(2, 3, 4)" in error_msg
        assert "3D" in error_msg
        # Check WHY
        assert "Expected: 1D" in error_msg
        assert "2D" in error_msg
        # Check HOW
        assert "reshape" in error_msg.lower()
        assert "Example:" in error_msg

    def test_negative_sampling_rate_error_message(self) -> None:
        """Test that negative sampling rate provides helpful error message."""
        with pytest.raises(ValueError) as exc_info:
            ChannelFrame(data=self.dask_data, sampling_rate=-44100)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Invalid sampling_rate" in error_msg
        assert "Got: -44100 Hz" in error_msg
        # Check WHY
        assert "Expected: Positive value > 0" in error_msg
        # Check HOW
        assert "Common values:" in error_msg
        assert "44100" in error_msg

    def test_zero_sampling_rate_error_message(self) -> None:
        """Test that zero sampling rate provides helpful error message."""
        with pytest.raises(ValueError) as exc_info:
            ChannelFrame(data=self.dask_data, sampling_rate=0)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Invalid sampling_rate" in error_msg
        assert "Got: 0 Hz" in error_msg
        # Check WHY
        assert "Expected: Positive value > 0" in error_msg
        # Check HOW
        assert (
            "Sampling rate represents samples per second and must be positive."
            in error_msg
        )
        assert "Common values: 8000, 16000, 22050, 44100, 48000 Hz" in error_msg

    def test_file_not_found_error_message(self) -> None:
        """Test that missing file provides helpful error message."""
        fake_path = "/nonexistent/path/to/audio.wav"

        with pytest.raises(FileNotFoundError) as exc_info:
            ChannelFrame.from_file(fake_path)

        error_msg = str(exc_info.value)
        # Check WHAT
        assert "Audio file not found" in error_msg
        # Cross-platform: check for key path components instead of exact path
        assert "nonexistent" in error_msg and "audio.wav" in error_msg
        # Check WHY (context)
        assert "Current directory:" in error_msg
        # Check HOW
        assert "check" in error_msg.lower()
        assert "File path is correct" in error_msg
        assert "File exists" in error_msg
