"""Test helper functions for reshaping data in plotting strategies."""

import numpy as np

from wandas.visualization.plotting import _reshape_spectrogram_data, _reshape_to_2d


class TestReshapeHelpers:
    """Test reshape helper functions."""

    def test_reshape_to_2d_1d_input(self) -> None:
        """Test _reshape_to_2d with 1D input."""
        # Create 1D data
        data_1d = np.random.rand(100)
        assert data_1d.ndim == 1

        # Reshape to 2D
        result = _reshape_to_2d(data_1d)

        # Check that data is now 2D
        assert result.ndim == 2
        assert result.shape == (1, 100)

        # Check that data values are preserved
        np.testing.assert_array_equal(result[0], data_1d)

    def test_reshape_to_2d_2d_input(self) -> None:
        """Test _reshape_to_2d with 2D input (should remain unchanged)."""
        # Create 2D data
        data_2d = np.random.rand(2, 100)
        assert data_2d.ndim == 2

        # Apply reshape
        result = _reshape_to_2d(data_2d)

        # Check that data remains 2D and unchanged
        assert result.ndim == 2
        assert result.shape == (2, 100)
        np.testing.assert_array_equal(result, data_2d)

    def test_reshape_to_2d_3d_input(self) -> None:
        """Test _reshape_to_2d with 3D input (should remain unchanged)."""
        # Create 3D data
        data_3d = np.random.rand(2, 100, 50)
        assert data_3d.ndim == 3

        # Apply reshape
        result = _reshape_to_2d(data_3d)

        # Check that data remains 3D and unchanged
        assert result.ndim == 3
        assert result.shape == (2, 100, 50)
        np.testing.assert_array_equal(result, data_3d)

    def test_reshape_spectrogram_data_1d_input(self) -> None:
        """Test _reshape_spectrogram_data with 1D input."""
        # Create 1D data (frequency bins)
        data_1d = np.random.rand(513)  # Typical frequency bins for n_fft=1024
        assert data_1d.ndim == 1

        # Reshape to 3D
        result = _reshape_spectrogram_data(data_1d)

        # Check that data is now 3D with shape (1, freqs, 1)
        assert result.ndim == 3
        assert result.shape == (1, 513, 1)

        # Check that data values are preserved
        np.testing.assert_array_equal(result[0, :, 0], data_1d)

    def test_reshape_spectrogram_data_2d_input(self) -> None:
        """Test _reshape_spectrogram_data with 2D input."""
        # Create 2D data (freqs, time)
        data_2d = np.random.rand(513, 100)  # 513 freq bins, 100 time frames
        assert data_2d.ndim == 2

        # Reshape to 3D
        result = _reshape_spectrogram_data(data_2d)

        # Check that data is now 3D with shape (1, freqs, time)
        assert result.ndim == 3
        assert result.shape == (1, 513, 100)

        # Check that data values are preserved
        np.testing.assert_array_equal(result[0], data_2d)

    def test_reshape_spectrogram_data_3d_input(self) -> None:
        """Test _reshape_spectrogram_data with 3D input (should remain unchanged)."""
        # Create 3D data (channels, freqs, time)
        data_3d = np.random.rand(2, 513, 100)
        assert data_3d.ndim == 3

        # Apply reshape
        result = _reshape_spectrogram_data(data_3d)

        # Check that data remains 3D and unchanged
        assert result.ndim == 3
        assert result.shape == (2, 513, 100)
        np.testing.assert_array_equal(result, data_3d)

    def test_reshape_spectrogram_data_1d_edge_cases(self) -> None:
        """Test _reshape_spectrogram_data with 1D edge cases."""
        # Test with small array
        data_small = np.array([1.0, 2.0, 3.0])
        result_small = _reshape_spectrogram_data(data_small)
        assert result_small.shape == (1, 3, 1)
        np.testing.assert_array_equal(result_small[0, :, 0], data_small)

        # Test with single element
        data_single = np.array([5.0])
        result_single = _reshape_spectrogram_data(data_single)
        assert result_single.shape == (1, 1, 1)
        assert result_single[0, 0, 0] == 5.0

    def test_reshape_functions_preserve_dtype(self) -> None:
        """Test that reshape functions preserve data type."""
        # Test with different dtypes
        dtypes = [np.float32, np.float64, np.complex64, np.complex128]

        for dtype in dtypes:
            # Test _reshape_to_2d
            data_1d = np.array([1, 2, 3], dtype=dtype)
            result_2d = _reshape_to_2d(data_1d)
            assert result_2d.dtype == dtype

            # Test _reshape_spectrogram_data
            result_spec = _reshape_spectrogram_data(data_1d)
            assert result_spec.dtype == dtype

    def test_reshape_functions_with_complex_data(self) -> None:
        """Test reshape functions with complex data (typical for spectrogram)."""
        # Create complex 1D data
        real_part = np.random.rand(256)
        imag_part = np.random.rand(256)
        complex_data = real_part + 1j * imag_part

        # Test _reshape_to_2d with complex data
        result_2d = _reshape_to_2d(complex_data)
        assert result_2d.shape == (1, 256)
        assert np.iscomplexobj(result_2d)
        np.testing.assert_array_equal(result_2d[0], complex_data)

        # Test _reshape_spectrogram_data with complex data
        result_spec = _reshape_spectrogram_data(complex_data)
        assert result_spec.shape == (1, 256, 1)
        assert np.iscomplexobj(result_spec)
        np.testing.assert_array_equal(result_spec[0, :, 0], complex_data)

    def test_reshape_consistency_between_functions(self) -> None:
        """Test that reshape functions provide consistent behavior."""
        # Create test data
        data_1d = np.random.rand(100)

        # Both functions should handle 1D data appropriately
        result_2d = _reshape_to_2d(data_1d)
        result_spec = _reshape_spectrogram_data(data_1d)

        # Both should preserve the original data
        assert result_2d.ndim == 2
        assert result_spec.ndim == 3

        # The 2D result should be extractable from the 3D result
        assert result_spec.shape[2] == 1  # Single time frame
        np.testing.assert_array_equal(result_2d[0], result_spec[0, :, 0])
