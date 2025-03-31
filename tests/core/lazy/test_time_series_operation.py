from typing import Callable

# filepath: /workspaces/wandas/tests/core/lazy/test_time_series_operation.py
import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.core.lazy.time_series_operation import (
    _OPERATION_REGISTRY,
    AudioOperation,
    HighPassFilter,
    LowPassFilter,
    create_operation,
    get_operation,
    register_operation,
)
from wandas.utils.types import NDArrayReal

_da_from_array = da.from_array  # type: ignore [unused-ignore]


# class TestNormalizeOperation:
#     def setup_method(self) -> None:
#         """Set up test fixtures for each test."""
#         self.sample_rate: int = 16000
#         self.mono_data: NDArrayReal = np.array(
#             [[0.5, 0.5, 0.5, 0.5]]
#         )  # 1 channel with constant value
#         self.stereo_data: NDArrayReal = np.array(
#             [[0.5, 0.5, 0.5, 0.5], [0.25, 0.25, 0.25, 0.25]]
#         )  # 2 channels
#         self.dask_mono: DaArray = _da_from_array(self.mono_data, chunks=(1, 2))
#         self.dask_stereo: DaArray = _da_from_array(self.stereo_data, chunks=(1, 2))
#         self.normalize: Normalize = Normalize(self.sample_rate)

#     def test_initialization(self) -> None:
#         """Test initialization with different parameters."""
#         # Default initialization
#         norm = Normalize(self.sample_rate)
#         assert norm.sampling_rate == self.sample_rate
#         assert norm.target_level == -20
#         assert norm.channel_wise is True
#         assert abs(norm.target_rms - 0.1) < 1e-10  # target_rms = 10^(-20/20) ≈ 0.1

#         # Custom parameters
#         norm = Normalize(self.sample_rate, target_level=-15, channel_wise=False)
#         assert norm.target_level == -15
#         assert norm.channel_wise is False
#         assert abs(norm.target_rms - 0.1778) < 1e-4
#         # target_rms = 10^(-15/20) ≈ 0.1778

#     def test_normalize_mono_channel(self) -> None:
#         """Test normalization of mono channel data."""
#         # Create data with RMS = 0.5
#         data: NDArrayReal = np.array([[0.5, 0.5, 0.5, 0.5]])  # RMS = 0.5
#         norm: Normalize = Normalize(
#             self.sample_rate, target_level=-20
#         )  # target_rms ≈ 0.1

#         # Get the processor function
#         processor: Callable[[NDArrayReal], NDArrayReal] = norm._create_processor()
#         result: NDArrayReal = processor(data)

#         # Expected: data * (0.1 / 0.5) = data * 0.2
#         expected: NDArrayReal = data * 0.2
#         np.testing.assert_array_almost_equal(result, expected)

#     def test_normalize_multi_channel_channel_wise(self) -> None:
#         """Test channel-wise normalization of multi-channel data."""
#         # Channel 1 has RMS = 0.5, Channel 2 has RMS = 0.25
#         data: NDArrayReal = np.array([[0.5, 0.5, 0.5, 0.5], [0.25, 0.25, 0.25, 0.25]])
#         norm: Normalize = Normalize(
#             self.sample_rate, target_level=-20
#         )  # target_rms ≈ 0.1

#         processor: Callable[[NDArrayReal], NDArrayReal] = norm._create_processor()
#         result: NDArrayReal = processor(data)

#         # Expected: Channel 1 * (0.1 / 0.5) = Channel 1 * 0.2
#         #           Channel 2 * (0.1 / 0.25) = Channel 2 * 0.4
#         expected: NDArrayReal = np.array([[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]])
#         np.testing.assert_array_almost_equal(result, expected)

#     def test_normalize_multi_channel_global(self) -> None:
#         """Test global normalization of multi-channel data."""
#         # Channel 1 has RMS = 0.5, Channel 2 has RMS = 0.25
#         # Combined RMS = sqrt((0.5^2 + 0.25^2)/2) ≈ 0.3953
#         data: NDArrayReal = np.array([[0.5, 0.5, 0.5, 0.5], [0.25, 0.25, 0.25, 0.25]])
#         norm: Normalize = Normalize(
#             self.sample_rate, target_level=-20, channel_wise=False
#         )  # target_rms ≈ 0.1

#         processor: Callable[[NDArrayReal], NDArrayReal] = norm._create_processor()
#         result: NDArrayReal = processor(data)

#         # Expected: data * (0.1 / 0.3953) ≈ data * 0.2529
#         expected: NDArrayReal = data * (0.1 / np.sqrt((0.5**2 + 0.25**2) / 2))
#         np.testing.assert_array_almost_equal(result, expected)

#     def test_normalize_silent_data(self) -> None:
#         """Test normalization of silent data."""
#         data: NDArrayReal = np.zeros((2, 4))  # Silent data
#         norm: Normalize = Normalize(self.sample_rate)

#         processor: Callable[[NDArrayReal], NDArrayReal] = norm._create_processor()
#         result: NDArrayReal = processor(data)

#         # Silent data should remain silent
#         np.testing.assert_array_equal(result, data)

#     def test_normalize_near_silent_channel(self) -> None:
#         """Test normalization of a near-silent channel."""
#         # One normal channel, one very quiet channel
#         data: NDArrayReal = np.array(
#             [[0.5, 0.5, 0.5, 0.5], [1e-11, 1e-11, 1e-11, 1e-11]]
#         )
#         norm: Normalize = Normalize(self.sample_rate)

#         processor: Callable[[NDArrayReal], NDArrayReal] = norm._create_processor()
#         result: NDArrayReal = processor(data)

#         # Channel 1 should be normalized, Channel 2 should remain unchanged
#         expected: NDArrayReal = np.array(
#             [[0.1, 0.1, 0.1, 0.1], [1e-11, 1e-11, 1e-11, 1e-11]]
#         )
#         np.testing.assert_array_almost_equal(result, expected)

#     def test_normalize_empty_data(self) -> None:
#         """Test normalization of empty data."""
#         data: NDArrayReal = np.array([]).reshape(0, 0)
#         norm: Normalize = Normalize(self.sample_rate)

#         processor: Callable[[NDArrayReal], NDArrayReal] = norm._create_processor()
#         result: NDArrayReal = processor(data)

#         # Empty data should remain empty
#         np.testing.assert_array_equal(result, data)

#     def test_process_with_dask_array(self) -> None:
#         """Test process() method with dask arrays."""
#         # Mock _da_map_blocks to verify it's called correctly
#         with mock.patch(
#             "wandas.core.lazy.time_series_operation._da_map_blocks"
#         ) as mock_map_blocks:
#             # Set up the mock to return the input data
#             mock_map_blocks.return_value = self.dask_mono

#             norm: Normalize = Normalize(self.sample_rate)
#             _: DaArray = norm.process(self.dask_mono)

#             # Verify _da_map_blocks was called with the processor function
#             mock_map_blocks.assert_called_once()
#             assert mock_map_blocks.call_args[0][0] == norm._processor_func
#             assert mock_map_blocks.call_args[0][1] is self.dask_mono

#     def test_create_operation_function(self) -> None:
#         """Test create_operation function correctly creates a Normalize instance."""
#         with mock.patch(
#             "wandas.core.lazy.time_series_operation.get_operation"
#         ) as mock_get_op:
#             mock_get_op.return_value = Normalize

#             op: AudioOperation = create_operation(
#                 "normalize", self.sample_rate, target_level=-15
#             )

#             mock_get_op.assert_called_with("normalize")
#             assert isinstance(op, Normalize)
#             assert op.target_level == -15
#             assert op.sampling_rate == self.sample_rate

#     def test_get_operation_function(self) -> None:
#         """Test get_operation returns the correct operation class."""
#         op_class: type[AudioOperation] = get_operation("normalize")
#         assert op_class == Normalize

#     def test_register_operation_function(self) -> None:
#         """Test register_operation adds an operation to the registry."""

#         # Create a mock operation class
#         class TestOperation(AudioOperation):
#             operation_name = "test_op"

#             def _create_processor(self) -> Callable[[NDArrayReal], NDArrayReal]:
#                 return lambda x: x

#         # Register the operation
#         register_operation("test_op", TestOperation)

#         # Verify it was added to the registry
#         assert get_operation("test_op") == TestOperation

#         # Clean up after test
#         if "test_op" in _OPERATION_REGISTRY:
#             del _OPERATION_REGISTRY["test_op"]

# def test_actual_normalization_results(self) -> None:
#     """Test the actual normalization results with real data processing."""
#     # Create data with varying amplitudes
#     data: NDArrayReal = np.array(
#         [
#             [0.8, 0.6, 0.4, 0.2],  # RMS ≈ 0.55
#             [0.2, 0.15, 0.1, 0.05],  # RMS ≈ 0.14
#         ]
#     )

#     # Create dask array and normalize
#     dask_data: DaArray = _da_from_array(data, chunks=(1, 2))
#     norm: Normalize = Normalize(
#         self.sample_rate, target_level=-20
#     )  # target_rms ≈ 0.1

#     # Process and compute result
#     result_dask: DaArray = norm.process(dask_data)
#     result: NDArrayReal = result_dask.compute()

#     # Channel 1: Scale by 0.1/0.55 ≈ 0.182
#     # Channel 2: Scale by 0.1/0.14 ≈ 0.714
#     expected: NDArrayReal = np.array(
#         [
#             data[0] * (0.1 / np.sqrt(np.mean(data[0] ** 2))),
#             data[1] * (0.1 / np.sqrt(np.mean(data[1] ** 2))),
#         ]
#     )

#     np.testing.assert_array_almost_equal(result, expected)


class TestHighPassFilter:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.cutoff: float = 500.0
        self.order: int = 4
        self.hpf: HighPassFilter = HighPassFilter(
            self.sample_rate, self.cutoff, self.order
        )

        # Create sample data with low and high frequency components
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)  # 1 second of audio

        # 50 Hz component (below cutoff) and 200 Hz component (above cutoff)
        self.low_freq: float = 50.0
        self.high_freq: float = 1000.0
        low_freq_signal = np.sin(2 * np.pi * self.low_freq * t)
        high_freq_signal = np.sin(2 * np.pi * self.high_freq * t)

        # Single channel signal with both components
        self.signal: NDArrayReal = np.array([low_freq_signal + high_freq_signal])

        # Create dask array
        self.dask_signal: DaArray = _da_from_array(self.signal, chunks=(1, 500))

    def test_initialization(self) -> None:
        """Test initialization with different parameters."""
        hpf = HighPassFilter(self.sample_rate, self.cutoff)
        assert hpf.sampling_rate == self.sample_rate
        assert hpf.cutoff == self.cutoff
        assert hpf.order == 4  # Default value

        custom_order = 6
        hpf = HighPassFilter(self.sample_rate, self.cutoff, order=custom_order)
        assert hpf.order == custom_order

    def test_filter_effect(self) -> None:
        """Test that the filter attenuates frequencies below cutoff."""
        processor = self.hpf._process_array
        result: NDArrayReal = processor(self.signal).compute()

        # Calculate FFT to check frequency content
        fft_original = np.abs(np.fft.rfft(self.signal[0]))
        fft_filtered = np.abs(np.fft.rfft(result[0]))

        freq_bins = np.fft.rfftfreq(len(self.signal[0]), 1 / self.sample_rate)

        # Find indices closest to our test frequencies
        low_idx = np.argmin(np.abs(freq_bins - self.low_freq))
        high_idx = np.argmin(np.abs(freq_bins - self.high_freq))

        # Low frequency should be attenuated, high frequency mostly preserved
        assert (
            fft_filtered[low_idx] < 0.1 * fft_original[low_idx]
        )  # At least 90% attenuation
        assert (
            fft_filtered[high_idx] > 0.9 * fft_original[high_idx]
        )  # At most 10% attenuation

    def test_invalid_cutoff_frequency(self) -> None:
        """Test that invalid cutoff frequencies raise ValueError."""
        # Cutoff too low
        with pytest.raises(ValueError):
            HighPassFilter(self.sample_rate, 0)

        # Cutoff too high (above Nyquist)
        with pytest.raises(ValueError):
            HighPassFilter(self.sample_rate, self.sample_rate / 2 + 1)


class TestLowPassFilter:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.cutoff: float = 500.0
        self.order: int = 4
        self.lpf: LowPassFilter = LowPassFilter(
            self.sample_rate, self.cutoff, self.order
        )

        # Create sample data with low and high frequency components
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)  # 1 second of audio

        # 50 Hz component (below cutoff) and 200 Hz component (above cutoff)
        self.low_freq: float = 50.0
        self.high_freq: float = 1000.0
        low_freq_signal = np.sin(2 * np.pi * self.low_freq * t)
        high_freq_signal = np.sin(2 * np.pi * self.high_freq * t)

        # Single channel signal with both components
        self.signal: NDArrayReal = np.array([low_freq_signal + high_freq_signal])

        # Create dask array
        self.dask_signal: DaArray = _da_from_array(self.signal, chunks=(1, 500))

    def test_initialization(self) -> None:
        """Test initialization with different parameters."""
        lpf = LowPassFilter(self.sample_rate, self.cutoff)
        assert lpf.sampling_rate == self.sample_rate
        assert lpf.cutoff == self.cutoff
        assert lpf.order == 4  # Default value

        custom_order = 6
        lpf = LowPassFilter(self.sample_rate, self.cutoff, order=custom_order)
        assert lpf.order == custom_order

    def test_filter_effect(self) -> None:
        """Test that the filter attenuates frequencies above cutoff."""
        processor = self.lpf._process_array
        result: NDArrayReal = processor(self.signal).compute()

        # Calculate FFT to check frequency content
        fft_original = np.abs(np.fft.rfft(self.signal[0]))
        fft_filtered = np.abs(np.fft.rfft(result[0]))

        freq_bins = np.fft.rfftfreq(len(self.signal[0]), 1 / self.sample_rate)

        # Find indices closest to our test frequencies
        low_idx = np.argmin(np.abs(freq_bins - self.low_freq))
        high_idx = np.argmin(np.abs(freq_bins - self.high_freq))

        # Low frequency should be preserved, high frequency attenuated
        assert (
            fft_filtered[low_idx] > 0.9 * fft_original[low_idx]
        )  # At most 10% attenuation
        assert (
            fft_filtered[high_idx] < 0.1 * fft_original[high_idx]
        )  # At least 90% attenuation

    def test_invalid_cutoff_frequency(self) -> None:
        """Test that invalid cutoff frequencies raise ValueError."""
        # Cutoff too low
        with pytest.raises(ValueError):
            LowPassFilter(self.sample_rate, 0)

        # Cutoff too high (above Nyquist)
        with pytest.raises(ValueError):
            LowPassFilter(self.sample_rate, self.sample_rate / 2 + 1)


class TestOperationRegistry:
    """Test registry-related functions."""

    def test_get_operation_normal(self) -> None:
        """Test get_operation returns a registered operation."""
        # Test for existing operations
        # assert get_operation("normalize") == Normalize
        assert get_operation("highpass_filter") == HighPassFilter
        assert get_operation("lowpass_filter") == LowPassFilter

    def test_get_operation_error(self) -> None:
        """Test get_operation raises ValueError for unknown operations."""
        with pytest.raises(ValueError, match="未知の操作タイプです"):
            get_operation("nonexistent_operation")

    def test_register_operation_normal(self) -> None:
        """Test registering a valid operation."""

        # Create a test operation class
        class TestOperation(AudioOperation):
            name = "test_register_op"

            def _create_processor(self) -> Callable[[NDArrayReal], NDArrayReal]:
                return lambda x: x

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
        # Create a normalize operation
        # norm_op = create_operation("normalize", 16000, target_level=-25)
        # assert isinstance(norm_op, Normalize)
        # assert norm_op.target_level == -25

        # Create a highpass filter operation
        hpf_op = create_operation("highpass_filter", 16000, cutoff=150.0, order=6)
        assert isinstance(hpf_op, HighPassFilter)
        assert hpf_op.cutoff == 150.0
        assert hpf_op.order == 6
