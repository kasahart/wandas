from unittest import mock

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.frames.channel import ChannelFrame

_da_from_array = da.from_array  # type: ignore [unused-ignore]


class TestBaseFrameArithmeticOperations:
    """Test arithmetic operations in BaseFrame."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_rate = 16000
        self.data = np.random.random((2, 16000))
        self.dask_data: DaArray = _da_from_array(self.data, chunks=(1, 4000))
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
        exponent_dask = _da_from_array(exponent_data, chunks=(1, 4000))
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
        exponent_dask = _da_from_array(exponent_data, chunks=(1, 4000))

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
        other_dask = _da_from_array(other_data, chunks=(1, 4000))
        other_frame = ChannelFrame(data=other_dask, sampling_rate=44100, label="other")

        # Should raise ValueError
        with pytest.raises(ValueError, match="Sampling rates do not match"):
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
        known_dask = _da_from_array(known_data, chunks=(1, 2))
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
        positive_dask = _da_from_array(positive_data, chunks=(1, 4000))
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
        vector_dask = _da_from_array(vector_data, chunks=(1, 4000))
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
