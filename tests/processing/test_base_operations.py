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

    def test_pure_parameter_default(self) -> None:
        """Test that pure parameter defaults to True."""
        op = self.TestOp(16000)
        assert op.pure is True

    def test_pure_parameter_explicit_true(self) -> None:
        """Test that pure parameter can be explicitly set to True."""
        op = self.TestOp(16000, pure=True)
        assert op.pure is True

    def test_pure_parameter_explicit_false(self) -> None:
        """Test that pure parameter can be explicitly set to False."""
        op = self.TestOp(16000, pure=False)
        assert op.pure is False

    def test_pure_parameter_used_in_delayed(self) -> None:
        """Test that pure parameter is passed to dask.delayed."""

        # Test with pure=True
        op_true = self.TestOp(16000, pure=True)
        with mock.patch("wandas.processing.base.delayed") as mock_delayed:
            mock_delayed.return_value = lambda x: mock.MagicMock()
            test_array = np.array([[1.0, 2.0, 3.0]])
            op_true.process_array(test_array)

            # Verify delayed was called with pure=True
            mock_delayed.assert_called_once()
            _, kwargs = mock_delayed.call_args
            assert kwargs["pure"] is True

        # Test with pure=False
        op_false = self.TestOp(16000, pure=False)
        with mock.patch("wandas.processing.base.delayed") as mock_delayed:
            mock_delayed.return_value = lambda x: mock.MagicMock()
            op_false.process_array(test_array)

            # Verify delayed was called with pure=False
            mock_delayed.assert_called_once()
            _, kwargs = mock_delayed.call_args
            assert kwargs["pure"] is False

    def test_get_metadata_updates_default(self) -> None:
        """Test that get_metadata_updates returns empty dict by default."""
        op = self.TestOp(16000)
        assert op.get_metadata_updates() == {}

    def test_get_display_name_default(self) -> None:
        """Test that get_display_name returns None by default."""
        op = self.TestOp(16000)
        assert op.get_display_name() is None

    def test_process_array_not_implemented(self) -> None:
        """Test that _process_array raises NotImplementedError if not overridden."""

        class IncompleteOp(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "incomplete_op"

        op = IncompleteOp(16000)
        test_array = np.array([[1.0, 2.0, 3.0]])

        with pytest.raises(NotImplementedError, match="Subclasses must implement"):
            op._process_array(test_array)

    def test_calculate_output_shape_default_implementation(self) -> None:
        """Test default calculate_output_shape implementation."""
        op = self.TestOp(16000)

        # Test normal case
        input_shape = (2, 100)
        output_shape = op.calculate_output_shape(input_shape)
        assert output_shape == input_shape

    def test_calculate_output_shape_empty_input(self) -> None:
        """Test calculate_output_shape with empty input shape."""

        # Use an operation that doesn't override calculate_output_shape
        class NoOverrideOp(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "no_override_op"

            def _process_array(self, x: NDArrayReal) -> NDArrayReal:
                return x * 2

        op = NoOverrideOp(16000)

        # Test empty shape - should trigger early return path
        empty_shape: tuple[int, ...] = ()
        result = op.calculate_output_shape(empty_shape)
        assert result == empty_shape

    def test_calculate_output_shape_non_ndarray_output(self) -> None:
        """Test calculate_output_shape when _process_array returns non-ndarray."""

        # Use an operation that doesn't override calculate_output_shape
        class NonArrayOp(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "non_array_op"

            def _process_array(self, x: NDArrayReal) -> NDArrayReal:
                # Return a pure Python object (not an ndarray)
                # to test fallback path
                return "not_an_array"  # type: ignore

        op = NonArrayOp(16000)
        input_shape = (2, 100)
        # Should fall back to returning input_shape
        output_shape = op.calculate_output_shape(input_shape)
        assert output_shape == input_shape

    def test_calculate_output_shape_failure(self) -> None:
        """Test calculate_output_shape when _process_array fails."""

        class FailingOp(AudioOperation[NDArrayReal, NDArrayReal]):
            name = "failing_op"

            def _process_array(self, x: NDArrayReal) -> NDArrayReal:
                raise RuntimeError("Processing failed")

        op = FailingOp(16000)

        with pytest.raises(NotImplementedError, match="must implement"):
            op.calculate_output_shape((2, 100))

    def test_register_operation_abstract_class(self) -> None:
        """Test that registering an abstract class raises TypeError."""
        import abc

        class AbstractOp(AudioOperation[NDArrayReal, NDArrayReal], abc.ABC):
            name = "abstract_op"

            @abc.abstractmethod
            def abstract_method(self) -> None:
                pass

        with pytest.raises(TypeError, match="Cannot register abstract"):
            register_operation(AbstractOp)
