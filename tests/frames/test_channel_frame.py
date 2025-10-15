import os
import tempfile
from pathlib import Path
from typing import Any, Union
from unittest import mock

import dask.array as da
import numpy as np
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
                _: Union[Axes, Any] = self.channel_frame.plot(
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

    def test_initialization_error_high_dim(self) -> None:
        """Test initialization with data that has too many dimensions."""
        data_3d = np.random.random((2, 16000, 3))
        dask_data_3d = _da_from_array(data_3d, chunks=(1, 4000, 3))

        with pytest.raises(
            ValueError, match="Data must be 1-dimensional or 2-dimensional."
        ):
            ChannelFrame(dask_data_3d, self.sample_rate)

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
        with pytest.raises(IndexError, match="Channel index"):
            _ = self.channel_frame[5]
        # Test for invalid slice
        with pytest.raises(TypeError, match="Invalid key type:"):
            _ = self.channel_frame[1.5]  # type: ignore

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
        with pytest.raises(
            ValueError, match="Sampling rates do not match. Cannot perform operation."
        ):
            _ = self.channel_frame + other_cf

    def test_add_method(self) -> None:
        """Test add method for adding signals."""
        # 通常の加算をテスト
        # Create another ChannelFrame
        other_data = np.random.random((2, 16000))
        other_dask_data = _da_from_array(other_data, chunks=-1)
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
        with pytest.raises(
            ValueError, match="Sampling rates do not match. Cannot perform operation."
        ):
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
        # Test successful visualization
        with mock.patch.object(DaArray, "visualize") as mock_visualize:
            filename = self.channel_frame.visualize_graph()
            mock_visualize.assert_called_once()
            assert filename is not None

        # Test with provided filename
        with mock.patch.object(DaArray, "visualize") as mock_visualize:
            custom_filename = "test_graph.png"
            filename = self.channel_frame.visualize_graph(filename=custom_filename)
            mock_visualize.assert_called_with(filename=custom_filename)
            assert filename == custom_filename

        # Test handling of visualization error
        with mock.patch.object(
            DaArray, "visualize", side_effect=Exception("Test error")
        ):
            filename = self.channel_frame.visualize_graph()
            assert filename is None
