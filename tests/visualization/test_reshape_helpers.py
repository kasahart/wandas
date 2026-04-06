"""Test helper functions for reshaping data in plotting strategies.

All test data uses deterministic arrays (no randomness) so that
reshape correctness is analytically verifiable.
"""

import numpy as np
import pytest

from wandas.visualization.plotting import _reshape_spectrogram_data, _reshape_to_2d

# ---------------------------------------------------------------------------
# Constants — eliminate magic numbers
# ---------------------------------------------------------------------------
_N_SAMPLES = 100  # generic sample count for 1‑D / 2‑D tests
_N_FREQ_BINS = 513  # n_fft=1024 → 513 frequency bins (N/2+1)
_N_TIME_FRAMES = 50  # spectrogram time axis length
_N_CHANNELS = 2


class TestReshapeTo2D:
    """Tests for _reshape_to_2d."""

    def test_1d_input_becomes_row_vector(self) -> None:
        """1‑D array is reshaped to (1, N) — single‑channel row."""
        data = np.arange(_N_SAMPLES, dtype=np.float64)
        result = _reshape_to_2d(data)

        assert result.ndim == 2
        assert result.shape == (1, _N_SAMPLES)
        np.testing.assert_array_equal(result[0], data)

    def test_2d_input_unchanged(self) -> None:
        """2‑D array passes through without copy or shape change."""
        data = np.arange(_N_CHANNELS * _N_SAMPLES, dtype=np.float64).reshape(_N_CHANNELS, _N_SAMPLES)
        result = _reshape_to_2d(data)

        assert result.ndim == 2
        assert result.shape == (_N_CHANNELS, _N_SAMPLES)
        np.testing.assert_array_equal(result, data)

    def test_3d_input_unchanged(self) -> None:
        """3‑D array passes through — no reshape needed."""
        data = np.arange(_N_CHANNELS * _N_SAMPLES * _N_TIME_FRAMES, dtype=np.float64).reshape(
            _N_CHANNELS, _N_SAMPLES, _N_TIME_FRAMES
        )
        result = _reshape_to_2d(data)

        assert result.ndim == 3
        assert result.shape == (_N_CHANNELS, _N_SAMPLES, _N_TIME_FRAMES)
        np.testing.assert_array_equal(result, data)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_dtype_preserved(self, dtype: np.dtype) -> None:
        """Data type must survive the reshape operation."""
        data = np.array([1, 2, 3], dtype=dtype)
        result = _reshape_to_2d(data)
        assert result.dtype == dtype


class TestReshapeSpectrogramData:
    """Tests for _reshape_spectrogram_data."""

    def test_1d_input_becomes_single_channel_single_frame(self) -> None:
        """1‑D freq data → (1, n_freq, 1)."""
        data = np.arange(_N_FREQ_BINS, dtype=np.float64)
        result = _reshape_spectrogram_data(data)

        assert result.ndim == 3
        assert result.shape == (1, _N_FREQ_BINS, 1)
        np.testing.assert_array_equal(result[0, :, 0], data)

    def test_2d_input_adds_channel_axis(self) -> None:
        """2‑D (freq, time) → (1, freq, time)."""
        data = np.arange(_N_FREQ_BINS * _N_TIME_FRAMES, dtype=np.float64).reshape(_N_FREQ_BINS, _N_TIME_FRAMES)
        result = _reshape_spectrogram_data(data)

        assert result.ndim == 3
        assert result.shape == (1, _N_FREQ_BINS, _N_TIME_FRAMES)
        np.testing.assert_array_equal(result[0], data)

    def test_3d_input_unchanged(self) -> None:
        """Already 3‑D data passes through unchanged."""
        data = np.arange(_N_CHANNELS * _N_FREQ_BINS * _N_TIME_FRAMES, dtype=np.float64).reshape(
            _N_CHANNELS, _N_FREQ_BINS, _N_TIME_FRAMES
        )
        result = _reshape_spectrogram_data(data)

        assert result.ndim == 3
        assert result.shape == (_N_CHANNELS, _N_FREQ_BINS, _N_TIME_FRAMES)
        np.testing.assert_array_equal(result, data)

    def test_1d_edge_case_small_array(self) -> None:
        """Small 1‑D arrays are reshaped correctly."""
        data = np.array([1.0, 2.0, 3.0])
        result = _reshape_spectrogram_data(data)
        assert result.shape == (1, 3, 1)
        np.testing.assert_array_equal(result[0, :, 0], data)

    def test_1d_edge_case_single_element(self) -> None:
        """Single-element array produces (1, 1, 1)."""
        data = np.array([5.0])
        result = _reshape_spectrogram_data(data)
        assert result.shape == (1, 1, 1)
        assert result[0, 0, 0] == 5.0

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    def test_dtype_preserved(self, dtype: np.dtype) -> None:
        """Data type must survive the reshape operation."""
        data = np.array([1, 2, 3], dtype=dtype)
        result = _reshape_spectrogram_data(data)
        assert result.dtype == dtype

    def test_complex_data_preserved(self) -> None:
        """Complex spectrogram data values must survive reshape."""
        n = 256
        real_part = np.arange(n, dtype=np.float64)
        imag_part = np.arange(n, dtype=np.float64) * 0.5
        complex_data = real_part + 1j * imag_part

        result = _reshape_spectrogram_data(complex_data)
        assert result.shape == (1, n, 1)
        assert np.iscomplexobj(result)
        np.testing.assert_array_equal(result[0, :, 0], complex_data)


class TestReshapeConsistency:
    """Cross-function consistency checks."""

    def test_1d_data_consistent_between_functions(self) -> None:
        """Both helpers preserve the same 1‑D data after reshape."""
        data = np.arange(_N_SAMPLES, dtype=np.float64)
        result_2d = _reshape_to_2d(data)
        result_spec = _reshape_spectrogram_data(data)

        assert result_2d.ndim == 2
        assert result_spec.ndim == 3
        # The 2D row should equal the spectrogram single-frame slice
        np.testing.assert_array_equal(result_2d[0], result_spec[0, :, 0])
