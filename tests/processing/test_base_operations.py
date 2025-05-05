from unittest import mock

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.processing.base import (
    _OPERATION_REGISTRY,
    AudioOperation,
    create_operation,
    get_operation,
    register_operation,
)
from wandas.utils.types import NDArrayReal

_da_from_array = da.from_array  # type: ignore [unused-ignore]


class TestOperationRegistry:
    """Test registry-related functions."""

    def test_get_operation_normal(self) -> None:
        """Test get_operation returns a registered operation."""
        # Test for existing operations
        assert "highpass_filter" in _OPERATION_REGISTRY
        assert "lowpass_filter" in _OPERATION_REGISTRY

    def test_get_operation_error(self) -> None:
        """Test get_operation raises ValueError for unknown operations."""
        with pytest.raises(ValueError, match="Unknown operation type:"):
            get_operation("nonexistent_operation")

    def test_register_operation_normal(self) -> None:
        """Test registering a valid operation."""

        # Create a test operation class
        class TestOperation(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "test_register_op"

            def calculate_output_shape(
                self, input_shape: tuple[int, ...]
            ) -> tuple[int, ...]:
                return input_shape

            def _process_array(self, x: NDArrayReal) -> NDArrayReal:
                return x

        # Register and verify
        register_operation(TestOperation)
        assert get_operation("test_register_op") == TestOperation

        # Clean up
        if "test_register_op" in _OPERATION_REGISTRY:
            del _OPERATION_REGISTRY["test_register_op"]

    def test_register_operation_error(self) -> None:
        """Test registering an invalid class raises TypeError."""

        # Create a non-AudioOperation class
        class InvalidClass:
            pass

        with pytest.raises(
            TypeError, match="Strategy class must inherit from AudioOperation."
        ):
            register_operation(InvalidClass)  # type: ignore [unused-ignore]

    def test_create_operation_with_different_types(self) -> None:
        """Test creating operations of different types."""
        # Create a highpass filter operation
        hpf_op = create_operation("highpass_filter", 16000, cutoff=150.0, order=6)
        from wandas.processing.filters import HighPassFilter

        assert isinstance(hpf_op, HighPassFilter)
        assert hpf_op.cutoff == 150.0
        assert hpf_op.order == 6


class TestAudioOperation:
    """Test AudioOperation base class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""

        # Create a simple test implementation
        class TestOp(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "test_op"

            def _process_array(self, x: NDArrayReal) -> NDArrayReal:
                return x * 2

            def calculate_output_shape(
                self, input_shape: tuple[int, ...]
            ) -> tuple[int, ...]:
                return input_shape

        self.TestOp = TestOp
        self.op = TestOp(16000)
        self.data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        # 修正: DaArray.from_array を da.from_array に変更
        self.dask_data = _da_from_array(self.data, chunks=(1, -1))

    def test_process(self) -> None:
        """Test the process method."""
        # Process the data
        result = self.op.process(self.dask_data)

        # Check that the result is a Dask array
        assert isinstance(result, DaArray)

        # Compute and check the result
        computed = result.compute()
        expected = self.data * 2
        np.testing.assert_array_equal(computed, expected)

    def test_delayed_execution(self) -> None:
        """Test that processing is delayed until compute is called."""
        with mock.patch.object(DaArray, "compute") as mock_compute:
            # Just processing shouldn't trigger computation
            result = self.op.process(self.dask_data)
            mock_compute.assert_not_called()

            # Only when compute() is called
            _ = result.compute()
            mock_compute.assert_called_once()

    def test_validate_params(self) -> None:
        """Test parameter validation."""

        # Create a subclass with parameter validation
        class ValidatedOp(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "validated_op"

            def __init__(self, sampling_rate: float, value: int):
                self.value = value
                super().__init__(sampling_rate, value=value)

            def validate_params(self) -> None:
                if self.value < 0:
                    raise ValueError("Value must be non-negative")

            def _process_array(self, x: NDArrayReal) -> NDArrayReal:
                return x

            def calculate_output_shape(
                self, input_shape: tuple[int, ...]
            ) -> tuple[int, ...]:
                return input_shape

        # Invalid parameters should raise during initialization
        with pytest.raises(ValueError, match="Value must be non-negative"):
            _ = ValidatedOp(16000, -1)
