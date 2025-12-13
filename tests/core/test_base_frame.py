from typing import Any
from unittest import mock

import dask.array as da
import numpy as np
import pandas as pd
import pytest
from dask.array.core import Array as DaArray

import wandas as wd
from wandas.core.base_frame import BaseFrame
from wandas.core.metadata import ChannelMetadata
from wandas.frames.channel import ChannelFrame

_da_from_array = da.from_array  # type: ignore [unused-ignore]


class TestBaseFrameArithmeticOperations:
    """Test arithmetic operations in BaseFrame."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.data = np.random.random((2, 16000))
        self.dask_data: DaArray = _da_from_array(self.data, chunks=(1, -1))
        self.channel_frame = ChannelFrame(
            data=self.dask_data, sampling_rate=self.sample_rate, label="test_audio"
        )

    def test_pow_operator_with_scalar(self) -> None:
        """Test __pow__ operator with scalar values."""
        # Test squaring
        result = self.channel_frame**2

        # Check result properties
        assert isinstance(result, ChannelFrame)
        assert result.sampling_rate == self.sample_rate
        assert result.n_channels == 2
        assert result.n_samples == 16000

        # Check computation result
        computed = result.compute()
        expected = self.data**2
        np.testing.assert_array_equal(computed, expected)

        # Check operation history
        assert len(result.operation_history) == 1
        assert result.operation_history[0]["operation"] == "**"
        assert result.operation_history[0]["with"] == "2"

    def test_pow_operator_with_channel_frame(self) -> None:
        """Test __pow__ operator with another ChannelFrame."""
        # Create another ChannelFrame with exponent values
        exponent_data = np.full((2, 16000), 3.0)  # Raise to power of 3
        exponent_dask = _da_from_array(exponent_data, chunks=(1, -1))
        exponent_frame = ChannelFrame(
            data=exponent_dask, sampling_rate=self.sample_rate, label="exponent"
        )

        # Apply power operation
        result = self.channel_frame**exponent_frame

        # Check result properties
        assert isinstance(result, ChannelFrame)
        assert result.sampling_rate == self.sample_rate
        assert result.n_channels == 2
        assert result.n_samples == 16000

        # Check computation result
        computed = result.compute()
        expected = self.data**exponent_data
        np.testing.assert_array_equal(computed, expected)

        # Check operation history
        assert len(result.operation_history) == 1
        assert result.operation_history[0]["operation"] == "**"
        assert result.operation_history[0]["with"] == "exponent"

    def test_pow_operator_with_numpy_array(self) -> None:
        """Test __pow__ operator with NumPy array."""
        # Create exponent array
        exponent_array = np.full((2, 16000), 1.5)

        # Apply power operation
        result = self.channel_frame**exponent_array

        # Check result properties
        assert isinstance(result, ChannelFrame)
        assert result.sampling_rate == self.sample_rate

        # Check computation result
        computed = result.compute()
        expected = self.data**exponent_array
        np.testing.assert_array_equal(computed, expected)

        # Check operation history
        assert len(result.operation_history) == 1
        assert result.operation_history[0]["operation"] == "**"
        assert "ndarray" in result.operation_history[0]["with"]

    def test_pow_operator_with_dask_array(self) -> None:
        """Test __pow__ operator with Dask array."""
        # Create exponent dask array
        exponent_data = np.full((2, 16000), 0.5)
        exponent_dask = _da_from_array(exponent_data, chunks=(1, -1))

        # Apply power operation
        result = self.channel_frame**exponent_dask

        # Check result properties
        assert isinstance(result, ChannelFrame)
        assert result.sampling_rate == self.sample_rate

        # Check computation result
        computed = result.compute()
        expected = self.data**exponent_data
        np.testing.assert_array_equal(computed, expected)

        # Check operation history
        assert len(result.operation_history) == 1
        assert result.operation_history[0]["operation"] == "**"
        assert "dask.array" in result.operation_history[0]["with"]

    def test_pow_operator_preserves_metadata(self) -> None:
        """Test that __pow__ preserves channel metadata and labels."""
        # Set custom metadata
        self.channel_frame.channels[0].label = "left"
        self.channel_frame.channels[0]["gain"] = 0.8
        self.channel_frame.metadata["test_key"] = "test_value"

        # Apply power operation
        result = self.channel_frame**2

        # Check metadata preservation
        assert result.channels[0].label == "(left ** 2)"
        assert result.channels[0]["gain"] == 0.8  # Arbitrary metadata preserved
        assert result.metadata["test_key"] == "test_value"

    def test_pow_operator_sampling_rate_mismatch(self) -> None:
        """Test __pow__ with mismatched sampling rates raises error."""
        # Create ChannelFrame with different sampling rate
        other_data = np.random.random((2, 16000))
        other_dask = _da_from_array(other_data, chunks=(1, -1))
        other_frame = ChannelFrame(data=other_dask, sampling_rate=44100, label="other")

        # Should raise ValueError
        with pytest.raises(ValueError, match=r"Sampling rate mismatch"):
            _ = self.channel_frame**other_frame

    def test_pow_operator_lazy_evaluation(self) -> None:
        """Test that __pow__ preserves lazy evaluation."""
        # Apply operation without computing
        result = self.channel_frame**2

        # Should still be lazy
        assert isinstance(result._data, DaArray)

        # No computation should have happened yet
        with mock.patch.object(
            DaArray, "compute", return_value=self.data
        ) as mock_compute:
            # Just accessing properties shouldn't trigger compute
            _ = result.sampling_rate
            _ = result.n_channels
            _ = result.operation_history
            mock_compute.assert_not_called()

            # Only when accessing data should compute happen
            _ = result.data
            mock_compute.assert_called_once()
            mock_compute.assert_called_once()

    def test_pow_operator_mathematical_correctness(self) -> None:
        """Test mathematical correctness of power operations."""
        # Test with known values
        known_data = np.array([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]])
        known_dask = _da_from_array(known_data, chunks=(1, -1))
        known_frame = ChannelFrame(
            data=known_dask, sampling_rate=self.sample_rate, label="known"
        )

        # Test squaring
        squared = known_frame**2
        computed_squared = squared.compute()
        expected_squared = known_data**2
        np.testing.assert_array_equal(computed_squared, expected_squared)

        # Test square root (power of 0.5)
        sqrt_result = known_frame**0.5
        computed_sqrt = sqrt_result.compute()
        expected_sqrt = np.sqrt(known_data)
        np.testing.assert_array_equal(computed_sqrt, expected_sqrt)

        # Test cube root (power of 1/3)
        cuberoot_result = known_frame ** (1.0 / 3.0)
        computed_cuberoot = cuberoot_result.compute()
        expected_cuberoot = known_data ** (1.0 / 3.0)
        np.testing.assert_array_equal(computed_cuberoot, expected_cuberoot)

    def test_pow_operator_with_zero_and_negative(self) -> None:
        """Test __pow__ with edge cases like zero and negative exponents."""
        # Test with positive data and zero exponent (should give 1)
        positive_data = np.abs(self.data) + 0.1  # Ensure positive
        positive_dask = _da_from_array(positive_data, chunks=(1, -1))
        positive_frame = ChannelFrame(
            data=positive_dask, sampling_rate=self.sample_rate, label="positive"
        )

        zero_power = positive_frame**0
        computed_zero = zero_power.compute()
        expected_zero = np.ones_like(positive_data)
        np.testing.assert_array_equal(computed_zero, expected_zero)

        # Test with negative exponent
        negative_power = positive_frame ** (-1)
        computed_negative = negative_power.compute()
        expected_negative = positive_data ** (-1)
        np.testing.assert_array_equal(computed_negative, expected_negative)

    def test_pow_operator_chaining(self) -> None:
        """Test chaining power operations with other operations."""
        # Chain power with other operations
        result = (self.channel_frame**2) + 1

        # Check that operations are recorded correctly
        assert len(result.operation_history) == 2
        assert result.operation_history[0]["operation"] == "**"
        assert result.operation_history[1]["operation"] == "+"

        # Check mathematical correctness
        computed = result.compute()
        expected = (self.data**2) + 1
        np.testing.assert_array_equal(computed, expected)

    def test_pow_operator_single_channel(self) -> None:
        """Test __pow__ with single channel frame."""
        # Get single channel
        single_channel = self.channel_frame.get_channel(0)

        # Apply power operation
        result = single_channel**3

        # Check result
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 1
        assert result.sampling_rate == self.sample_rate

        # Check computation
        computed = result.compute()
        expected = self.data[0:1] ** 3
        np.testing.assert_array_equal(computed, expected)

    def test_pow_operator_complex_expressions(self) -> None:
        """Test __pow__ in complex mathematical expressions."""
        # Test expression like: sqrt(x**2 + y**2) for magnitude calculation
        x_data = np.sin(np.linspace(0, 4 * np.pi, 16000))
        y_data = np.cos(np.linspace(0, 4 * np.pi, 16000))
        vector_data = np.vstack([x_data, y_data])
        vector_dask = _da_from_array(vector_data, chunks=(1, -1))
        vector_frame = ChannelFrame(
            data=vector_dask, sampling_rate=self.sample_rate, label="vector"
        )

        # Calculate magnitude: sqrt(x**2 + y**2)
        x_squared = vector_frame[0] ** 2
        y_squared = vector_frame[1] ** 2
        sum_squares = x_squared + y_squared
        magnitude = sum_squares**0.5  # Using power of 0.5 instead of sqrt

        # Check result
        assert isinstance(magnitude, ChannelFrame)
        computed_magnitude = magnitude.compute()
        expected_magnitude = np.sqrt(x_data**2 + y_data**2)
        np.testing.assert_array_equal(computed_magnitude.squeeze(), expected_magnitude)


class TestBaseFrameChannelMetadata:
    """Test channel metadata validation and handling."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.data = np.random.random((2, 16000))
        self.dask_data: DaArray = _da_from_array(self.data, chunks=(1, -1))

    def test_channel_metadata_with_dict(self) -> None:
        """Test initialization with dict channel metadata."""
        metadata_dicts = [
            {"label": "ch0", "unit": "V", "extra": {}},
            {"label": "ch1", "unit": "A", "extra": {}},
        ]
        frame = ChannelFrame(
            data=self.dask_data,
            sampling_rate=self.sample_rate,
            channel_metadata=metadata_dicts,
        )
        assert frame.channels[0].label == "ch0"
        assert frame.channels[0].unit == "V"
        assert frame.channels[1].label == "ch1"
        assert frame.channels[1].unit == "A"

    def test_channel_metadata_with_channel_metadata_objects(self) -> None:
        """Test initialization with ChannelMetadata objects."""
        metadata_objs = [
            ChannelMetadata(label="left", unit="Pa", extra={"gain": 0.8}),
            ChannelMetadata(label="right", unit="Pa", extra={"gain": 0.9}),
        ]
        frame = ChannelFrame(
            data=self.dask_data,
            sampling_rate=self.sample_rate,
            channel_metadata=metadata_objs,
        )
        assert frame.channels[0].label == "left"
        assert frame.channels[0].unit == "Pa"
        assert frame.channels[0]["gain"] == 0.8
        assert frame.channels[1].label == "right"

    def test_channel_metadata_validation_error(self) -> None:
        """Test that invalid dict raises ValueError with validation error."""
        invalid_metadata = [
            {"label": "ch0", "unit": "V", "extra": {}},
            {"label": 123, "unit": "A", "extra": {}},  # Invalid: label must be str
        ]
        with pytest.raises(ValueError, match="Invalid channel_metadata at index 1"):
            ChannelFrame(
                data=self.dask_data,
                sampling_rate=self.sample_rate,
                channel_metadata=invalid_metadata,
            )

    def test_channel_metadata_type_error(self) -> None:
        """Test that invalid type raises TypeError."""
        invalid_metadata = [
            {"label": "ch0", "unit": "V", "extra": {}},
            "invalid_string",  # Invalid: must be dict or ChannelMetadata
        ]
        with pytest.raises(TypeError, match="Invalid type in channel_metadata"):
            ChannelFrame(
                data=self.dask_data,
                sampling_rate=self.sample_rate,
                channel_metadata=invalid_metadata,  # type: ignore
            )


class TestBaseFrameSpecialMethods:
    """Test special methods like __len__, __iter__, __array__."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.data = np.random.random((3, 16000))
        self.dask_data: DaArray = _da_from_array(self.data, chunks=(1, -1))
        self.channel_frame = ChannelFrame(
            data=self.dask_data, sampling_rate=self.sample_rate, label="test_audio"
        )

    def test_len(self) -> None:
        """Test __len__ returns number of channels."""
        assert len(self.channel_frame) == 3

    def test_iter(self) -> None:
        """Test __iter__ yields individual channels."""
        channels = list(self.channel_frame)
        assert len(channels) == 3
        for ch in channels:
            assert isinstance(ch, ChannelFrame)
            assert ch.n_channels == 1

    def test_array_no_dtype(self) -> None:
        """Test __array__ implicit conversion to NumPy array."""
        arr = np.array(self.channel_frame)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 16000)
        np.testing.assert_array_equal(arr, self.data)

    def test_array_with_dtype(self) -> None:
        """Test __array__ with dtype conversion."""
        arr = np.array(self.channel_frame, dtype=np.float32)
        assert arr.dtype == np.float32
        np.testing.assert_array_almost_equal(arr, self.data.astype(np.float32))

    def test_to_numpy(self) -> None:
        """Test to_numpy method."""
        arr = self.channel_frame.to_numpy()
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, self.channel_frame.data)

    def test_to_dataframe(self) -> None:
        """Test to_dataframe method."""
        df = self.channel_frame.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 16000  # n_samples
        assert df.shape[1] == 3  # n_channels

    def test_labels_property(self) -> None:
        """Test labels property returns list of channel labels."""
        labels = self.channel_frame.labels
        assert isinstance(labels, list)
        assert len(labels) == 3
        assert all(isinstance(label, str) for label in labels)

    def test_shape_property_single_channel(self) -> None:
        """Test shape property for single channel frame."""
        single_ch = self.channel_frame[0]
        assert single_ch.shape == (16000,)  # 1D for single channel

    def test_shape_property_multi_channel(self) -> None:
        """Test shape property for multi-channel frame."""
        assert self.channel_frame.shape == (3, 16000)


class TestBaseFrameErrorCases:
    """Test error handling in BaseFrame."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.data = np.random.random((2, 16000))
        self.dask_data: DaArray = _da_from_array(self.data, chunks=(1, -1))
        self.channel_frame = ChannelFrame(
            data=self.dask_data, sampling_rate=self.sample_rate, label="test_audio"
        )

    def test_label2index_key_error(self) -> None:
        """Test label2index raises KeyError for non-existent label."""
        with pytest.raises(KeyError, match="Channel label 'nonexistent' not found"):
            self.channel_frame.label2index("nonexistent")

    def test_create_new_instance_invalid_label_type(self) -> None:
        """Test _create_new_instance raises TypeError for invalid label."""
        with pytest.raises(TypeError, match="Label must be a string"):
            self.channel_frame._create_new_instance(
                data=self.dask_data,
                label=123,  # type: ignore
            )

    def test_create_new_instance_invalid_metadata_type(self) -> None:
        """Test _create_new_instance raises TypeError for invalid metadata."""
        with pytest.raises(TypeError, match="Metadata must be a dictionary"):
            self.channel_frame._create_new_instance(
                data=self.dask_data,
                metadata="invalid",  # type: ignore
            )

    def test_create_new_instance_invalid_channel_metadata_type(self) -> None:
        """Test _create_new_instance raises TypeError for invalid channel_metadata."""
        with pytest.raises(TypeError, match="Channel metadata must be a list"):
            self.channel_frame._create_new_instance(
                data=self.dask_data,
                channel_metadata="invalid",  # type: ignore
            )


class TestBaseFrameUtilityMethods:
    """Test utility methods like print_operation_history, persist, etc."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.data = np.random.random((2, 16000))
        self.dask_data: DaArray = _da_from_array(self.data, chunks=(1, -1))
        self.channel_frame = ChannelFrame(
            data=self.dask_data, sampling_rate=self.sample_rate, label="test_audio"
        )

    def test_print_operation_history_empty(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test print_operation_history with no operations."""
        self.channel_frame.print_operation_history()
        captured = capsys.readouterr()
        assert "Operation history: <empty>" in captured.out

    def test_print_operation_history_with_operations(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test print_operation_history with operations."""
        # Add some operations
        result = self.channel_frame + 1
        result = result * 2
        result.print_operation_history()
        captured = capsys.readouterr()
        assert "Operation history (2):" in captured.out
        assert "1: +" in captured.out
        assert "2: *" in captured.out

    def test_persist(self) -> None:
        """Test persist method."""
        persisted = self.channel_frame.persist()
        assert isinstance(persisted, ChannelFrame)
        assert persisted.sampling_rate == self.sample_rate
        assert persisted.n_channels == 2

    def test_visualize_graph_with_filename(self) -> None:
        """Test visualize_graph with custom filename."""
        result = self.channel_frame.visualize_graph(filename="test_graph.png")
        # Result can be Image, Widget, or None depending on environment
        assert result is not None

    def test_visualize_graph_without_filename(self) -> None:
        """Test visualize_graph without filename (auto-generated)."""
        result = self.channel_frame.visualize_graph()
        # Result can be Image, Widget, or None depending on environment
        assert result is not None

    def test_visualize_graph_exception_handling(self) -> None:
        """Test visualize_graph handles exceptions gracefully."""
        # Mock visualize to raise an exception
        with mock.patch.object(
            DaArray, "visualize", side_effect=Exception("test error")
        ):
            result = self.channel_frame.visualize_graph()
            assert result is None  # Should return None on exception

    def test_previous_property(self) -> None:
        """Test previous property."""
        assert self.channel_frame.previous is None
        result = self.channel_frame + 1
        assert result.previous is self.channel_frame

    def test_n_channels_property(self) -> None:
        """Test n_channels property."""
        assert self.channel_frame.n_channels == 2

    def test_channels_property(self) -> None:
        """Test channels property."""
        channels = self.channel_frame.channels
        assert isinstance(channels, list)
        assert len(channels) == 2
        assert all(isinstance(ch, ChannelMetadata) for ch in channels)


class TestBaseFrameIndexing:
    """Test advanced indexing features."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.data = np.random.random((4, 16000))
        self.dask_data: DaArray = _da_from_array(self.data, chunks=(1, -1))
        metadata_objs = [
            ChannelMetadata(label="ch0", unit="V", extra={}),
            ChannelMetadata(label="ch1", unit="A", extra={}),
            ChannelMetadata(label="ch2", unit="Pa", extra={}),
            ChannelMetadata(label="ch3", unit="W", extra={}),
        ]
        self.channel_frame = ChannelFrame(
            data=self.dask_data,
            sampling_rate=self.sample_rate,
            label="test_audio",
            channel_metadata=metadata_objs,
        )

    def test_getitem_with_string_label(self) -> None:
        """Test __getitem__ with string label."""
        result = self.channel_frame["ch1"]
        assert isinstance(result, ChannelFrame)
        assert result.n_channels == 1
        assert result.channels[0].label == "ch1"

    def test_getitem_with_list_of_strings(self) -> None:
        """Test __getitem__ with list of string labels."""
        result = self.channel_frame[["ch0", "ch2"]]
        assert result.n_channels == 2
        assert result.channels[0].label == "ch0"
        assert result.channels[1].label == "ch2"

    def test_getitem_with_list_of_ints(self) -> None:
        """Test __getitem__ with list of integers."""
        result = self.channel_frame[[0, 2, 3]]
        assert result.n_channels == 3
        np.testing.assert_array_equal(result.compute(), self.data[[0, 2, 3]])

    def test_getitem_with_empty_list_error(self) -> None:
        """Test __getitem__ with empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot index with an empty list"):
            _ = self.channel_frame[[]]

    def test_getitem_with_mixed_list_error(self) -> None:
        """Test __getitem__ with mixed type list raises TypeError."""
        with pytest.raises(TypeError, match="List must contain all str or all int"):
            _ = self.channel_frame[["ch0", 1]]  # type: ignore

    def test_getitem_with_numpy_integer_array(self) -> None:
        """Test __getitem__ with NumPy integer array."""
        indices = np.array([0, 2])
        result = self.channel_frame[indices]
        assert result.n_channels == 2
        np.testing.assert_array_equal(result.compute(), self.data[[0, 2]])

    def test_getitem_with_numpy_bool_array(self) -> None:
        """Test __getitem__ with NumPy boolean array."""
        mask = np.array([True, False, True, False])
        result = self.channel_frame[mask]
        assert result.n_channels == 2
        np.testing.assert_array_equal(result.compute(), self.data[[0, 2]])

    def test_getitem_with_bool_mask_wrong_length(self) -> None:
        """Test __getitem__ with wrong length boolean mask raises ValueError."""
        mask = np.array([True, False])  # Wrong length
        with pytest.raises(ValueError, match="Boolean mask length .* does not match"):
            _ = self.channel_frame[mask]

    def test_getitem_with_numpy_float_array_error(self) -> None:
        """Test __getitem__ with float array raises TypeError."""
        indices = np.array([0.0, 1.0])
        with pytest.raises(
            TypeError, match="NumPy array must be of integer or boolean type"
        ):
            _ = self.channel_frame[indices]

    def test_getitem_with_slice(self) -> None:
        """Test __getitem__ with slice."""
        result = self.channel_frame[1:3]
        assert result.n_channels == 2
        np.testing.assert_array_equal(result.compute(), self.data[1:3])

    def test_getitem_with_tuple_channel_and_time(self) -> None:
        """Test __getitem__ with tuple for multidimensional indexing."""
        # Select channel 0, samples 100:200
        result = self.channel_frame[0, 100:200]
        assert result.n_channels == 1
        assert result.n_samples == 100
        np.testing.assert_array_equal(result.compute(), self.data[0:1, 100:200])

    def test_getitem_with_tuple_list_and_time(self) -> None:
        """Test __getitem__ with list of channels and time slice."""
        result = self.channel_frame[[0, 2], ::2]
        assert result.n_channels == 2
        assert result.n_samples == 8000  # Every 2nd sample
        np.testing.assert_array_equal(result.compute(), self.data[[0, 2], ::2])

    def test_getitem_with_tuple_invalid_length(self) -> None:
        """Test __getitem__ with tuple of invalid length raises ValueError."""
        with pytest.raises(ValueError, match="Invalid key length"):
            _ = self.channel_frame[0, 100:200, 0]  # 3D indexing on 2D data

    def test_getitem_with_invalid_key_type(self) -> None:
        """Test __getitem__ with invalid key type raises TypeError."""
        with pytest.raises(TypeError, match="Invalid key type"):
            _ = self.channel_frame[{"key": "value"}]  # type: ignore

    def test_getitem_with_tuple_string_and_time(self) -> None:
        """Test __getitem__ with string label and time slice."""
        result = self.channel_frame["ch1", 500:1000]
        assert result.n_channels == 1
        assert result.n_samples == 500

    def test_getitem_single_channel_returns_1d_shape(self) -> None:
        """Test that single channel returns shape without channel dimension."""
        result = self.channel_frame[0]
        assert result.shape == (16000,)

    def test_get_channel_with_negative_index(self) -> None:
        """Test get_channel with negative index."""
        result = self.channel_frame.get_channel(-1)
        assert result.n_channels == 1
        np.testing.assert_array_equal(result.compute(), self.data[-1:])

    def test_get_channel_with_tuple(self) -> None:
        """Test get_channel with tuple of indices."""
        result = self.channel_frame.get_channel((0, 2))
        assert result.n_channels == 2


class TestBaseFrameInitialization:
    """Test BaseFrame initialization edge cases."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_rate = 16000

    def test_init_with_1d_data(self) -> None:
        """Test initialization with 1D data reshapes to 2D."""
        data_1d = np.random.random(16000)
        # Use channel-wise chunks for single channel: reshape to (1, n)
        dask_data_1d: DaArray = _da_from_array(data_1d.reshape(1, -1), chunks=(1, -1))
        frame = ChannelFrame(data=dask_data_1d, sampling_rate=self.sample_rate)
        assert frame.n_channels == 1
        assert frame.shape == (16000,)
        # Ensure internal data is 2D
        assert frame._data.ndim == 2
        assert frame._data.shape == (1, 16000)

    def test_init_without_channel_metadata(self) -> None:
        """Test that default channel metadata is created."""
        data = np.random.random((3, 16000))
        dask_data: DaArray = _da_from_array(data, chunks=(1, -1))
        frame = ChannelFrame(data=dask_data, sampling_rate=self.sample_rate)
        assert len(frame.channels) == 3
        assert frame.channels[0].label == "ch0"
        assert frame.channels[1].label == "ch1"
        assert frame.channels[2].label == "ch2"

    def test_init_with_operation_history(self) -> None:
        """Test initialization with operation history."""
        data = np.random.random((2, 16000))
        dask_data: DaArray = _da_from_array(data, chunks=(1, -1))
        history = [{"operation": "test", "param": "value"}]
        frame = ChannelFrame(
            data=dask_data,
            sampling_rate=self.sample_rate,
            operation_history=history,
        )
        assert len(frame.operation_history) == 1
        assert frame.operation_history[0]["operation"] == "test"

    def test_init_with_metadata(self) -> None:
        """Test initialization with custom metadata."""
        data = np.random.random((2, 16000))
        dask_data: DaArray = _da_from_array(data, chunks=(1, -1))
        metadata = {"custom_key": "custom_value"}
        frame = ChannelFrame(
            data=dask_data, sampling_rate=self.sample_rate, metadata=metadata
        )
        assert frame.metadata["custom_key"] == "custom_value"

    def test_data_property_single_channel_squeezes(self) -> None:
        """Test that data property squeezes single channel."""
        data = np.random.random(16000)
        # Use channel-wise chunks for single channel: reshape to (1, n)
        dask_data: DaArray = _da_from_array(data.reshape(1, -1), chunks=(1, -1))
        frame = ChannelFrame(data=dask_data, sampling_rate=self.sample_rate)
        result_data = frame.data
        assert result_data.ndim == 1  # Squeezed
        assert result_data.shape == (16000,)

    def test_data_property_multi_channel_not_squeezed(self) -> None:
        """Test that data property doesn't squeeze multi-channel."""
        data = np.random.random((3, 16000))
        dask_data: DaArray = _da_from_array(data, chunks=(1, -1))
        frame = ChannelFrame(data=dask_data, sampling_rate=self.sample_rate)
        result_data = frame.data
        assert result_data.ndim == 2
        assert result_data.shape == (3, 16000)


class TestBaseFrameRelabelChannels:
    """Test _relabel_channels method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.data = np.random.random((2, 16000))
        self.dask_data: DaArray = _da_from_array(self.data, chunks=(1, -1))
        metadata_objs = [
            ChannelMetadata(label="left", unit="V", extra={}),
            ChannelMetadata(label="right", unit="V", extra={}),
        ]
        self.channel_frame = ChannelFrame(
            data=self.dask_data,
            sampling_rate=self.sample_rate,
            channel_metadata=metadata_objs,
        )

    def test_relabel_channels_basic(self) -> None:
        """Test _relabel_channels with operation name."""
        new_metadata = self.channel_frame._relabel_channels("normalize")
        assert len(new_metadata) == 2
        assert new_metadata[0].label == "normalize(left)"
        assert new_metadata[1].label == "normalize(right)"

    def test_relabel_channels_with_display_name(self) -> None:
        """Test _relabel_channels with custom display name."""
        new_metadata = self.channel_frame._relabel_channels(
            "low_pass_filter", display_name="lpf"
        )
        assert new_metadata[0].label == "lpf(left)"
        assert new_metadata[1].label == "lpf(right)"


class TestBaseFrameDebugMethods:
    """Test debug and info methods."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.data = np.random.random((2, 16000))
        self.dask_data: DaArray = _da_from_array(self.data, chunks=(1, -1))
        self.channel_frame = ChannelFrame(
            data=self.dask_data, sampling_rate=self.sample_rate, label="test_audio"
        )

    def test_debug_info(self) -> None:
        """Test debug_info method runs without error."""
        # This method logs debug info, just ensure it doesn't crash
        self.channel_frame.debug_info()

    def test_print_operation_history_method(self) -> None:
        """Test _print_operation_history internal method."""
        # Test with empty history
        self.channel_frame._print_operation_history()
        # Add operations and test again
        result = self.channel_frame + 1
        result._print_operation_history()


class TestBaseFrameEdgeCases:
    """Test edge cases and rarely-used code paths."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_rate = 16000

    def test_getitem_with_tuple_invalid_channel_key_type(self) -> None:
        """Test __getitem__ tuple with invalid channel key type."""
        data = np.random.random((2, 16000))
        dask_data: DaArray = _da_from_array(data, chunks=(1, -1))
        frame = ChannelFrame(data=dask_data, sampling_rate=self.sample_rate)
        with pytest.raises(TypeError, match="Invalid channel key type in tuple"):
            _ = frame[{"invalid": "key"}, 100:200]  # type: ignore

    def test_compute_invalid_result_type(self) -> None:
        """Test compute() with invalid result type raises ValueError."""
        # Create a mock that returns non-ndarray
        data = np.random.random((2, 16000))
        dask_data: DaArray = _da_from_array(data, chunks=(1, -1))
        frame = ChannelFrame(data=dask_data, sampling_rate=self.sample_rate)

        with mock.patch.object(DaArray, "compute", return_value="not_an_array"):
            with pytest.raises(ValueError, match="Computed result is not a np.ndarray"):
                frame.compute()

    def test_slice_returns_single_channel_as_list(self) -> None:
        """Test that slicing a single channel still returns metadata as list."""
        data = np.random.random((3, 16000))
        dask_data: DaArray = _da_from_array(data, chunks=(1, -1))
        frame = ChannelFrame(data=dask_data, sampling_rate=self.sample_rate)

        # Slice to get single channel
        result = frame[1:2]
        assert isinstance(result.channels, list)
        assert len(result.channels) == 1

    def test_multidim_indexing_with_array_channels(self) -> None:
        """Test multidimensional indexing with array of channels."""
        data = np.random.random((4, 16000))
        dask_data: DaArray = _da_from_array(data, chunks=(1, -1))
        frame = ChannelFrame(data=dask_data, sampling_rate=self.sample_rate)

        # Use numpy array for channel selection with time slice
        indices = np.array([0, 2])
        result = frame[indices, 100:200]
        assert result.n_channels == 2
        assert result.n_samples == 100

    def test_multidim_indexing_channel_only(self) -> None:
        """Test multidimensional indexing with only channel selection (no time)."""
        data = np.random.random((4, 16000))
        dask_data: DaArray = _da_from_array(data, chunks=(1, -1))
        frame = ChannelFrame(data=dask_data, sampling_rate=self.sample_rate)

        # Tuple with only channel selection (time_keys will be empty)
        result = frame[(0,)]
        assert result.n_channels == 1
        assert result.n_samples == 16000

    def test_slice_single_channel_converts_to_list(self) -> None:
        """Test that slicing returning single ChannelMetadata converts to list."""
        data = np.random.random((3, 16000))
        dask_data: DaArray = _da_from_array(data, chunks=(1, -1))
        frame = ChannelFrame(data=dask_data, sampling_rate=self.sample_rate)

        # This should trigger the isinstance check on line 369
        result = frame[1:2]
        assert isinstance(result.channels, list)
        assert len(result.channels) == 1


class TestBaseFrameSingleChannelMetadata:
    """Test single channel metadata handling in BaseFrame."""

    def test_single_channel_metadata_wrapping(self) -> None:
        """Test that single ChannelMetadata is wrapped in list."""
        # Create a frame
        signal = wd.generate_sin(freqs=[440], duration=0.1, sampling_rate=16000)

        # Access by index to trigger line 378
        # Single channel access should still work
        single_channel = signal[0]
        assert single_channel.n_channels == 1


class TestDebugLoggingExceptionHandling:
    """Test debug logging exception handling."""

    def test_frame_repr_works(self) -> None:
        """Test that frame repr works without errors."""
        # Create a frame
        signal = wd.generate_sin(freqs=[440], duration=0.1, sampling_rate=16000)

        # Call repr which may trigger debugging code
        # Lines 153-154 handle exceptions in debug logging
        repr_str = repr(signal)
        assert "ChannelFrame" in repr_str or "Frame" in repr_str


class TestBaseFrameInfoAndDataframe:
    """Test info() and to_dataframe() methods."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.data = np.random.random((1, 16000))
        self.dask_data: DaArray = _da_from_array(self.data, chunks=(1, -1))
        self.channel_frame = ChannelFrame(
            data=self.dask_data, sampling_rate=self.sample_rate, label="test_audio"
        )

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


class TestBaseFrameSummary:
    def setup_method(self) -> None:
        self.sample_rate = 16000
        self.data = np.random.random((1, 16000))
        self.dask_data: DaArray = _da_from_array(self.data, chunks=(1, -1))
        self.channel_frame = ChannelFrame(
            data=self.dask_data, sampling_rate=self.sample_rate, label="test_audio"
        )

    def test_summary_contains_expected_keys(self) -> None:
        s = self.channel_frame.summary()
        assert s["type"] == "ChannelFrame"
        assert s["label"] == "test_audio"
        assert s["sampling_rate"] == self.sample_rate
        assert s["n_channels"] == 1
        assert s["shape"] == self.channel_frame.shape
        assert "metadata_keys" in s
        assert "chunks" in s

    def test_summary_does_not_compute(self) -> None:
        with mock.patch.object(self.channel_frame._data, "compute") as m:
            _ = self.channel_frame.summary()
            m.assert_not_called()

    def test_info_uses_summary_and_does_not_compute(self, capsys: Any) -> None:
        with mock.patch.object(self.channel_frame._data, "compute") as m:
            BaseFrame.info(self.channel_frame)
            m.assert_not_called()

        out = capsys.readouterr().out
        assert "ChannelFrame" in out
        assert "sampling_rate" in out
        assert "n_channels" in out


class TestBaseFrameCoverage:
    """Additional tests for BaseFrame coverage."""

    def setup_method(self) -> None:
        self.sample_rate = 16000
        self.data = np.random.random((2, 16000))
        self.dask_data: DaArray = _da_from_array(self.data, chunks=(1, -1))
        self.channel_frame = ChannelFrame(
            data=self.dask_data, sampling_rate=self.sample_rate, label="test_audio"
        )

    def test_init_rechunk_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that initialization continues even if rechunking fails."""
        # Fail first call, succeed second call (fallback)
        side_effect = [Exception("Rechunk failed"), self.dask_data]
        with mock.patch.object(DaArray, "rechunk", side_effect=side_effect):
            frame = ChannelFrame(data=self.dask_data, sampling_rate=self.sample_rate)
            # Should still be created
            assert isinstance(frame, ChannelFrame)
            # Should log warning
            assert "Rechunk failed" in caplog.text

    def test_init_debug_info_failure(self) -> None:
        """Test that initialization continues even if _debug_info_impl fails."""
        # We need to subclass to mock _debug_info_impl effectively or patch it on class
        with mock.patch(
            "wandas.frames.channel.ChannelFrame._debug_info_impl",
            side_effect=Exception("Debug info failed"),
        ):
            frame = ChannelFrame(data=self.dask_data, sampling_rate=self.sample_rate)
            assert isinstance(frame, ChannelFrame)

    def test_get_channel_invalid_type(self) -> None:
        """Test get_channel with invalid type raises TypeError."""
        with pytest.raises(TypeError):
            self.channel_frame.get_channel(1.5)  # type: ignore

    def test_handle_multidim_indexing_too_many_dims(self) -> None:
        """Test _handle_multidim_indexing with too many dimensions."""
        # 2D data, try 3D index
        key = (0, slice(None), 0)
        with pytest.raises(ValueError, match="Invalid key length"):
            self.channel_frame._handle_multidim_indexing(key)

    def test_handle_multidim_indexing_invalid_channel_key(self) -> None:
        """Test _handle_multidim_indexing with invalid channel key type."""
        # Pass a float as channel key
        key = (1.5, slice(None))
        with pytest.raises(TypeError, match="Invalid channel key type"):
            self.channel_frame._handle_multidim_indexing(key)  # type: ignore
