"""Tests for FrameTransformOperation class."""

import dask.array as da
import numpy as np
import pytest

from wandas.processing.frame_transform import FrameTransformOperation

_da_from_array = da.from_array  # type: ignore [unused-ignore]


class TestFrameTransformOperation:
    """Tests for FrameTransformOperation."""

    def test_basic_transform_with_explicit_shape_func(self) -> None:
        """Test basic transform with explicit output_shape_func."""
        data = np.array([[1.0, 2.0, 3.0, 4.0]])
        dask_data = _da_from_array(data, chunks=(1, -1))

        def double_values(x: np.ndarray) -> np.ndarray:
            return x * 2

        def output_shape(input_shape: tuple[int, ...]) -> tuple[int, ...]:
            return input_shape

        op = FrameTransformOperation(
            sampling_rate=16000,
            func=double_values,
            output_shape_func=output_shape,
        )

        result = op.process(dask_data)
        assert result.shape == data.shape
        np.testing.assert_array_equal(result.compute(), data * 2)

    def test_transform_with_shape_change(self) -> None:
        """Test transform that changes the output shape."""
        data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
        dask_data = _da_from_array(data, chunks=(1, -1))

        def downsample(x: np.ndarray) -> np.ndarray:
            return x[:, ::2]

        def output_shape(input_shape: tuple[int, ...]) -> tuple[int, ...]:
            return (input_shape[0], input_shape[1] // 2)

        op = FrameTransformOperation(
            sampling_rate=16000,
            func=downsample,
            output_shape_func=output_shape,
        )

        result = op.process(dask_data)
        assert result.shape == (1, 4)
        expected = data[:, ::2]
        np.testing.assert_array_equal(result.compute(), expected)

    def test_transform_with_dry_run_inference(self) -> None:
        """Test transform with automatic shape inference via dry-run."""
        data = np.array([[1.0, 2.0, 3.0, 4.0]])
        dask_data = _da_from_array(data, chunks=(1, -1))

        def triple_values(x: np.ndarray) -> np.ndarray:
            return x * 3

        op = FrameTransformOperation(
            sampling_rate=16000,
            func=triple_values,
            infer_output_shape=True,
        )

        result = op.process(dask_data)
        assert result.shape == data.shape
        np.testing.assert_array_equal(result.compute(), data * 3)

    def test_transform_with_dtype_change(self) -> None:
        """Test transform that changes dtype (e.g., real to complex)."""
        data = np.array([[1.0, 2.0, 3.0, 4.0]])
        dask_data = _da_from_array(data, chunks=(1, -1))

        def to_complex(x: np.ndarray) -> np.ndarray:
            return x.astype(np.complex128)

        def output_shape(input_shape: tuple[int, ...]) -> tuple[int, ...]:
            return input_shape

        op = FrameTransformOperation(
            sampling_rate=16000,
            func=to_complex,
            output_shape_func=output_shape,
            output_dtype=np.complex128,
        )

        result = op.process(dask_data)
        assert result.shape == data.shape
        assert result.dtype == np.complex128
        np.testing.assert_array_equal(result.compute(), data.astype(np.complex128))

    def test_transform_with_params(self) -> None:
        """Test transform with additional parameters."""
        data = np.array([[1.0, 2.0, 3.0, 4.0]])
        dask_data = _da_from_array(data, chunks=(1, -1))

        def scale_add(x: np.ndarray, scale: float, offset: float) -> np.ndarray:
            return x * scale + offset

        def output_shape(input_shape: tuple[int, ...]) -> tuple[int, ...]:
            return input_shape

        op = FrameTransformOperation(
            sampling_rate=16000,
            func=scale_add,
            output_shape_func=output_shape,
            scale=2.0,
            offset=1.0,
        )

        result = op.process(dask_data)
        expected = data * 2.0 + 1.0
        np.testing.assert_array_equal(result.compute(), expected)

    def test_dry_run_inference_with_custom_input_shape(self) -> None:
        """Test dry-run inference with custom infer_input_shape."""
        data = np.array([[1.0, 2.0, 3.0, 4.0]])
        dask_data = _da_from_array(data, chunks=(1, -1))

        def process(x: np.ndarray) -> np.ndarray:
            return x + 1

        op = FrameTransformOperation(
            sampling_rate=16000,
            func=process,
            infer_output_shape=True,
            infer_input_shape=(1, 10),  # Custom shape for dry-run
        )

        result = op.process(dask_data)
        assert result.shape == data.shape
        np.testing.assert_array_equal(result.compute(), data + 1)

    def test_dry_run_failure_raises_informative_error(self) -> None:
        """Test that dry-run failure raises RuntimeError with helpful message."""
        data = np.array([[1.0, 2.0, 3.0, 4.0]])
        dask_data = _da_from_array(data, chunks=(1, -1))

        def failing_func(x: np.ndarray) -> np.ndarray:
            raise ValueError("Test error")

        op = FrameTransformOperation(
            sampling_rate=16000,
            func=failing_func,
            infer_output_shape=True,
        )

        with pytest.raises(RuntimeError, match=r"Failed to infer output shape/dtype"):
            op.process(dask_data)

    def test_infer_output_shape_false_uses_input_shape(self) -> None:
        """Test that infer_output_shape=False uses input shape as output shape."""
        data = np.array([[1.0, 2.0, 3.0, 4.0]])
        dask_data = _da_from_array(data, chunks=(1, -1))

        def identity(x: np.ndarray) -> np.ndarray:
            return x

        op = FrameTransformOperation(
            sampling_rate=16000,
            func=identity,
            infer_output_shape=False,
        )

        result = op.process(dask_data)
        assert result.shape == data.shape
        np.testing.assert_array_equal(result.compute(), data)

    def test_invalid_dimension_in_input_shape_raises_value_error(self) -> None:
        """Test that invalid dimension in input_shape raises ValueError."""
        data = np.array([[1.0, 2.0, 3.0, 4.0]])
        dask_data = _da_from_array(data, chunks=(1, -1))

        def process(x: np.ndarray) -> np.ndarray:
            return x

        # Create an array with invalid dimension type for testing
        # We'll simulate this by patching the shape during dry-run inference
        op = FrameTransformOperation(
            sampling_rate=16000,
            func=process,
            infer_output_shape=True,
            infer_input_shape=(1, "invalid"),  # type: ignore [arg-type]
        )

        with pytest.raises(ValueError, match=r"Invalid dimension in input_shape"):
            op.process(dask_data)

    def test_get_display_name_uses_func_name(self) -> None:
        """Test that get_display_name returns the function name."""

        def my_transform(x: np.ndarray) -> np.ndarray:
            return x

        op = FrameTransformOperation(
            sampling_rate=16000,
            func=my_transform,
        )

        assert op.get_display_name() == "my_transform"

    def test_get_display_name_fallback_for_callable_object(self) -> None:
        """Test get_display_name fallback for callable objects without __name__."""

        class TransformCallable:
            def __call__(self, x: np.ndarray) -> np.ndarray:
                return x

        op = FrameTransformOperation(
            sampling_rate=16000,
            func=TransformCallable(),
        )

        # Should fall back to the operation name
        assert op.get_display_name() == "frame_transform"

    def test_multichannel_transform(self) -> None:
        """Test transform on multi-channel data."""
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        dask_data = _da_from_array(data, chunks=(1, -1))

        def add_offset(x: np.ndarray, offset: float) -> np.ndarray:
            return x + offset

        def output_shape(input_shape: tuple[int, ...]) -> tuple[int, ...]:
            return input_shape

        op = FrameTransformOperation(
            sampling_rate=16000,
            func=add_offset,
            output_shape_func=output_shape,
            offset=10.0,
        )

        result = op.process(dask_data)
        assert result.shape == data.shape
        expected = data + 10.0
        np.testing.assert_array_equal(result.compute(), expected)

    def test_complex_output_with_explicit_dtype(self) -> None:
        """Test complex output with explicitly specified output_dtype."""
        data = np.array([[1.0, 2.0, 3.0, 4.0]])
        dask_data = _da_from_array(data, chunks=(1, -1))

        def make_complex(x: np.ndarray) -> np.ndarray:
            return x + 1j * x

        def output_shape(input_shape: tuple[int, ...]) -> tuple[int, ...]:
            return input_shape

        op = FrameTransformOperation(
            sampling_rate=16000,
            func=make_complex,
            output_shape_func=output_shape,
            output_dtype=np.complex128,
        )

        result = op.process(dask_data)
        assert result.dtype == np.complex128
        expected = data + 1j * data
        np.testing.assert_array_almost_equal(result.compute(), expected)
