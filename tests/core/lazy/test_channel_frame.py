import os
import tempfile
from typing import Any, Union
from unittest import mock

import dask.array as da
import numpy as np
import pytest
import soundfile as sf
from dask.array.core import Array as DaArray
from matplotlib.axes import Axes

from wandas.core.lazy.channel_frame import ChannelFrame, ChannelMetadata
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

    def test_filter_operations(self) -> None:
        """Test that filter operations are lazy."""
        with mock.patch(
            "wandas.core.lazy.time_series_operation.create_operation"
        ) as mock_create_op:
            mock_op: mock.MagicMock = mock.MagicMock()
            mock_op.process.return_value = self.dask_data
            mock_create_op.return_value = mock_op

            # Apply filter operations
            result: ChannelFrame = self.channel_frame.highpass_filter(cutoff=100)
            mock_create_op.assert_called_with(
                "highpass_filter", self.sample_rate, cutoff=100, order=4
            )

            result = self.channel_frame.lowpass_filter(cutoff=5000)
            mock_create_op.assert_called_with(
                "lowpass_filter", self.sample_rate, cutoff=5000, order=4
            )

            # No compute should have happened
            mock_op.process.assert_called()
            assert isinstance(result, ChannelFrame)

    def test_plotting_triggers_compute(self) -> None:
        """Test that plotting triggers computation."""
        with mock.patch(
            "wandas.core.lazy.channel_frame.get_plot_strategy"
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
        assert cf.shape == (1, 16000)
        assert cf.n_channels == 1

    def test_initialization_error_high_dim(self) -> None:
        """Test initialization with data that has too many dimensions."""
        data_3d = np.random.random((2, 16000, 3))
        dask_data_3d = _da_from_array(data_3d, chunks=(1, 4000, 3))

        with pytest.raises(
            ValueError, match="データは1次元または2次元である必要があります"
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
                    self.channel_frame.save(temp_filename)
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
                    channel_frame.save(temp_filename)
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
            assert channel.label == f"test_audio_ch{i}"

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
        # Single channel extraction
        with mock.patch.object(DaArray, "compute") as mock_compute:
            mock_compute.return_value = self.data[0:1]
            result = self.channel_frame[0]
            assert isinstance(result, ChannelFrame)
            assert result.n_channels == 1

        # Time slice
        with mock.patch.object(DaArray, "compute") as mock_compute:
            mock_compute.return_value = self.data[:, :1000]
            result = self.channel_frame[:, :1000]  # type: ignore
            assert isinstance(result, ChannelFrame)
            assert result.n_samples == 1000

        # Test error case
        with pytest.raises(IndexError, match="インデックスの次元が多すぎます"):
            self.channel_frame[0, 0, 0]  # type: ignore

        # Test for scalar value (should raise ValueError)
        with mock.patch.object(DaArray, "__getitem__", return_value="scalar_value"):
            with pytest.raises(ValueError, match="インデックス操作の結果が不明"):
                self.channel_frame[0, 0]  # type: ignore

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
        with pytest.raises(ValueError, match="サンプリングレートが一致していません"):
            _ = self.channel_frame + other_cf

    def test_sum_mean_methods(self) -> None:
        """Test sum() and mean() methods."""
        # Test sum method
        sum_cf = self.channel_frame.sum()
        assert isinstance(sum_cf, ChannelFrame)
        assert sum_cf.n_channels == 1

        # Compute and check results
        with mock.patch.object(
            DaArray,
            "sum",
            return_value=_da_from_array(self.data.sum(axis=0, keepdims=True)),
        ):
            sum_data = sum_cf.compute()
            expected_sum = self.data.sum(axis=0, keepdims=True)
            np.testing.assert_array_almost_equal(sum_data, expected_sum)

        # Test mean method
        mean_cf = self.channel_frame.mean()
        assert isinstance(mean_cf, ChannelFrame)
        assert mean_cf.n_channels == 1

        # Compute and check results
        with mock.patch.object(
            DaArray,
            "mean",
            return_value=_da_from_array(self.data.mean(axis=0, keepdims=True)),
        ):
            mean_data = mean_cf.compute()
            expected_mean = self.data.mean(axis=0, keepdims=True)
            np.testing.assert_array_almost_equal(mean_data, expected_mean)

    def test_channel_difference(self) -> None:
        """Test channel_difference method."""
        diff_cf = self.channel_frame.channel_difference(other_channel=0)

        assert isinstance(diff_cf, ChannelFrame)
        assert diff_cf.n_channels == self.channel_frame.n_channels

        # Compute and check results
        computed = diff_cf.compute()
        expected = self.data - self.data[0:1]
        np.testing.assert_array_almost_equal(computed, expected)

        # Test invalid channel index
        with pytest.raises(ValueError, match="チャネル指定が範囲外です"):
            self.channel_frame.channel_difference(other_channel=10)

    def test_additional_filter_operations(self) -> None:
        """Test normalize and a_weighting operations."""
        with mock.patch(
            "wandas.core.lazy.time_series_operation.create_operation"
        ) as mock_create_op:
            mock_op = mock.MagicMock()
            mock_op.process.return_value = self.dask_data
            mock_create_op.return_value = mock_op

            # Test normalize
            result = self.channel_frame.normalize(target_level=-15, channel_wise=False)
            mock_create_op.assert_called_with(
                "normalize", self.sample_rate, target_level=-15, channel_wise=False
            )
            assert isinstance(result, ChannelFrame)

            # Test a_weighting
            result = self.channel_frame.a_weighting()
            mock_create_op.assert_called_with("a_weighting", self.sample_rate)
            assert isinstance(result, ChannelFrame)

            # Test HPSS methods
            result = self.channel_frame.hpss_harmonic(kernel_size=31)
            mock_create_op.assert_called_with(
                "hpss_harmonic", self.sample_rate, kernel_size=31
            )
            assert isinstance(result, ChannelFrame)

            result = self.channel_frame.hpss_percussive(kernel_size=31)
            mock_create_op.assert_called_with(
                "hpss_percussive", self.sample_rate, kernel_size=31
            )
            assert isinstance(result, ChannelFrame)

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

    def test_read_wav_class_method(self) -> None:
        """Test read_wav class method."""
        with mock.patch.object(ChannelFrame, "from_file") as mock_from_file:
            mock_from_file.return_value = self.channel_frame
            result = ChannelFrame.read_wav("test.wav", ch_labels=["left", "right"])
            mock_from_file.assert_called_with("test.wav", ch_labels=["left", "right"])
            assert result is self.channel_frame

    def test_debug_info(self) -> None:
        """Test debug_info method."""
        with mock.patch("wandas.core.lazy.channel_frame.logger") as mock_logger:
            self.channel_frame.debug_info()
            assert mock_logger.debug.call_count >= 9  # At least 9 debug messages

    @pytest.mark.integration  # type: ignore [misc, unused-ignore]
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
                mock.patch(
                    "wandas.core.lazy.file_readers.get_file_reader"
                ) as mock_get_reader,
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
                assert cf.metadata["channels"] == [0]

                # Test with multiple channels
                cf = ChannelFrame.from_file(temp_filename, channel=[0, 1])
                assert cf.metadata["channels"] == [0, 1]

                # Test error cases
                with pytest.raises(ValueError, match="チャネル指定が範囲外です"):
                    ChannelFrame.from_file(temp_filename, channel=5)

                with pytest.raises(ValueError, match="チャネル指定が範囲外です"):
                    ChannelFrame.from_file(temp_filename, channel=[0, 5])

                with pytest.raises(
                    TypeError,
                    match="channel は int, list, または None である必要があります",
                ):
                    ChannelFrame.from_file(temp_filename, channel="invalid")  # type: ignore

                # Test file not found
                with pytest.raises(FileNotFoundError):
                    ChannelFrame.from_file("nonexistent_file.wav")

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    def test_channel_metadata_label_access(self) -> None:
        """Test accessing and modifying channel labels through metadata."""
        # Check default labels
        assert self.channel_frame.channel[0].label == "test_audio_ch0"
        assert self.channel_frame.channel[1].label == "test_audio_ch1"

        # Set new labels
        self.channel_frame.channel[0].label = "left"
        self.channel_frame.channel[1].label = "right"

        # Verify labels were changed
        assert self.channel_frame.channel[0].label == "left"
        assert self.channel_frame.channel[1].label == "right"

        # Test error handling with invalid channel index
        with pytest.raises(ValueError, match="チャネル指定が範囲外です"):
            _ = self.channel_frame.channel[5].label

        with pytest.raises(ValueError, match="チャネル指定が範囲外です"):
            self.channel_frame.channel[5].label = "invalid"

    def test_channel_metadata_unit_access(self) -> None:
        """Test accessing and modifying channel units through metadata."""
        # Check default units (empty string)
        assert self.channel_frame.channel[0].unit == ""

        # Set new units
        self.channel_frame.channel[0].unit = "Pa"
        self.channel_frame.channel[1].unit = "V"

        # Verify units were changed
        assert self.channel_frame.channel[0].unit == "Pa"
        assert self.channel_frame.channel[1].unit == "V"

    def test_channel_metadata_arbitrary_items(self) -> None:
        """Test getting and setting arbitrary metadata items."""
        # Set arbitrary metadata items
        self.channel_frame.channel[0]["gain"] = 0.5
        self.channel_frame.channel[1]["gain"] = 0.75
        self.channel_frame.channel[0]["device"] = "microphone"

        # Get items using __getitem__
        assert self.channel_frame.channel[0]["gain"] == 0.5
        assert self.channel_frame.channel[1]["gain"] == 0.75
        assert self.channel_frame.channel[0]["device"] == "microphone"

        # Check missing item returns None
        assert self.channel_frame.channel[0]["missing"] is None

        # Test get() with default value
        assert self.channel_frame.channel[0].get("gain") == 0.5
        assert self.channel_frame.channel[0].get("missing") is None
        assert self.channel_frame.channel[0].get("missing", "default") == "default"

        # Test set() method
        self.channel_frame.channel[0].set("new_key", "new_value")
        assert self.channel_frame.channel[0]["new_key"] == "new_value"

    def test_channel_metadata_all_method(self) -> None:
        """Test getting all metadata for a channel."""
        # Set up some metadata
        self.channel_frame.channel[0].label = "left"
        self.channel_frame.channel[0].unit = "Pa"
        self.channel_frame.channel[0]["gain"] = 0.5

        # Get all metadata
        all_metadata = self.channel_frame.channel[0].all()

        # Check it's a dict with expected values
        assert isinstance(all_metadata, dict)
        assert all_metadata["label"] == "left"
        assert all_metadata["unit"] == "Pa"
        assert all_metadata["gain"] == 0.5

        # Verify that modifying the returned dict doesn't affect the original
        all_metadata["label"] = "modified"
        assert self.channel_frame.channel[0].label == "left"  # Original unchanged

    def test_channel_metadata_collection_getitem(self) -> None:
        """Test getting ChannelMetadata objects from collection."""
        # Get metadata objects for specific channels
        ch0_metadata = self.channel_frame.channel[0]
        ch1_metadata = self.channel_frame.channel[1]

        # Verify they're ChannelMetadata objects
        assert isinstance(ch0_metadata, ChannelMetadata)
        assert isinstance(ch1_metadata, ChannelMetadata)

        # Verify they reference the correct channels
        ch0_metadata.label = "test_ch0"
        assert self.channel_frame.channel[0].label == "test_ch0"

        # Test error handling with invalid index
        with pytest.raises(ValueError, match="チャネル指定が範囲外です"):
            _ = self.channel_frame.channel[5]

    def test_channel_metadata_collection_get_all(self) -> None:
        """Test getting all channel metadata."""
        # Set up distinct metadata for each channel
        self.channel_frame.channel[0].label = "left"
        self.channel_frame.channel[1].label = "right"
        self.channel_frame.channel[0]["gain"] = 0.5
        self.channel_frame.channel[1]["gain"] = 0.75

        # Get all channel metadata
        all_metadata = self.channel_frame.channel.get_all()

        # Verify structure and contents
        assert isinstance(all_metadata, dict)
        assert len(all_metadata) == 2
        assert 0 in all_metadata
        assert 1 in all_metadata

        assert all_metadata[0]["label"] == "left"
        assert all_metadata[0]["gain"] == 0.5
        assert all_metadata[1]["label"] == "right"
        assert all_metadata[1]["gain"] == 0.75

        # Verify that modifying the returned dict doesn't affect the original
        all_metadata[0]["label"] = "modified"
        assert self.channel_frame.channel[0].label == "left"  # Original unchanged

    def test_channel_metadata_collection_set_all(self) -> None:
        """Test setting metadata for all channels."""
        # Create new metadata to set
        new_metadata = {
            0: {"label": "new_left", "unit": "Pa", "custom": "value1"},
            1: {"label": "new_right", "unit": "V", "custom": "value2"},
        }

        # Set all metadata
        self.channel_frame.channel.set_all(new_metadata)

        # Verify metadata was set correctly
        assert self.channel_frame.channel[0].label == "new_left"
        assert self.channel_frame.channel[0].unit == "Pa"
        assert self.channel_frame.channel[0]["custom"] == "value1"

        assert self.channel_frame.channel[1].label == "new_right"
        assert self.channel_frame.channel[1].unit == "V"
        assert self.channel_frame.channel[1]["custom"] == "value2"

        # Test setting metadata for only one channel
        partial_metadata = {0: {"label": "updated_left"}}
        self.channel_frame.channel.set_all(partial_metadata)
        assert self.channel_frame.channel[0].label == "updated_left"
        assert self.channel_frame.channel[1].label == "new_right"  # Unchanged

        # Test with invalid channel indices (should be ignored)
        invalid_metadata = {5: {"label": "invalid"}}
        self.channel_frame.channel.set_all(invalid_metadata)
        # No exception should be raised, invalid indices are ignored

    def test_channel_metadata_on_new_channel_frame(self) -> None:
        """Test metadata preservation when creating derived ChannelFrames."""
        # Set metadata on original frame
        self.channel_frame.channel[0].label = "left"
        self.channel_frame.channel[0]["gain"] = 0.5
        self.channel_frame.channel[1].label = "right"

        # Create a derived ChannelFrame through an operation
        derived_frame = self.channel_frame + 1.0

        # Verify metadata was preserved
        assert (
            derived_frame.channel[0].label == "(left + 1.0)"
        )  # Underlying label preserved
        assert derived_frame.channel[0]["gain"] == 0.5
        assert derived_frame.channel[1].label == "(right + 1.0)"

        # Test metadata in extracted channel
        channel0 = self.channel_frame.get_channel(0)
        assert channel0.channel[0].label == "left"  # Label should be preserved
        assert channel0.channel[0]["gain"] == 0.5
