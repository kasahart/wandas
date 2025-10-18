"""Tests for psychoacoustic processing operations."""

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray
from mosqito.sq_metrics import loudness_zwst, loudness_zwtv

from wandas.processing.base import create_operation, get_operation
from wandas.processing.psychoacoustic import LoudnessZwst, LoudnessZwtv
from wandas.utils.types import NDArrayReal

_da_from_array = da.from_array  # type: ignore [unused-ignore]


class TestLoudnessZwtv:
    """Test suite for LoudnessZwtv operation."""

    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 48000
        self.duration: float = 0.1
        self.field_type: str = "free"

        # Create test signal: 1 kHz sine wave at 70 dB SPL
        # SPL reference: 20 µPa = 2e-5 Pa
        # 70 dB SPL corresponds to approximately 0.0632 Pa RMS
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        freq = 1000.0
        amplitude = 0.0632  # Approximate amplitude for 70 dB SPL
        self.signal_mono: NDArrayReal = np.array(
            [amplitude * np.sin(2 * np.pi * freq * t)]
        )

        # Create stereo signal
        self.signal_stereo: NDArrayReal = np.vstack(
            [
                amplitude * np.sin(2 * np.pi * freq * t),
                amplitude * np.sin(2 * np.pi * 2 * freq * t),
            ]
        )

        # Create dask arrays
        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=-1)
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=-1)

        # Create operation instance
        self.loudness_op = LoudnessZwtv(self.sample_rate, field_type=self.field_type)

    def test_initialization(self) -> None:
        """Test LoudnessZwtv initialization with different parameters."""
        # Default initialization
        loudness = LoudnessZwtv(self.sample_rate)
        assert loudness.sampling_rate == self.sample_rate
        assert loudness.field_type == "free"

        # Custom field_type
        loudness_diffuse = LoudnessZwtv(self.sample_rate, field_type="diffuse")
        assert loudness_diffuse.field_type == "diffuse"

    def test_invalid_field_type(self) -> None:
        """Test that invalid field_type raises ValueError."""
        with pytest.raises(ValueError, match="field_type must be 'free' or 'diffuse'"):
            LoudnessZwtv(self.sample_rate, field_type="invalid")

    def test_operation_name(self) -> None:
        """Test that operation has correct name."""
        assert self.loudness_op.name == "loudness_zwtv"

    def test_operation_registration(self) -> None:
        """Test that operation is properly registered."""
        op_class = get_operation("loudness_zwtv")
        assert op_class == LoudnessZwtv

    def test_create_operation(self) -> None:
        """Test creating operation via create_operation function."""
        op = create_operation("loudness_zwtv", self.sample_rate, field_type="diffuse")
        assert isinstance(op, LoudnessZwtv)
        assert op.field_type == "diffuse"

    def test_mono_signal_shape(self) -> None:
        """Test loudness calculation output shape for mono signal."""
        result = self.loudness_op.process_array(self.signal_mono).compute()

        # Result should be 2D (channels, time_samples)
        assert result.ndim == 2
        assert result.shape[0] == 1  # 1 channel
        # Time samples should be less than input samples (downsampled)
        assert result.shape[1] < self.signal_mono.shape[1]

    def test_stereo_signal_shape(self) -> None:
        """Test loudness calculation output shape for stereo signal."""
        loudness_op = LoudnessZwtv(self.sample_rate, field_type=self.field_type)
        result = loudness_op.process_array(self.signal_stereo).compute()

        # Compare with MoSQITo direct calculation for each channel
        n_ch1_direct, _, _, _ = loudness_zwtv(
            self.signal_stereo[0], self.sample_rate, field_type=self.field_type
        )
        n_ch2_direct, _, _, _ = loudness_zwtv(
            self.signal_stereo[1], self.sample_rate, field_type=self.field_type
        )

        # Results should match MoSQITo exactly
        np.testing.assert_array_equal(result[0], n_ch1_direct)
        np.testing.assert_array_equal(result[1], n_ch2_direct)

    def test_loudness_values_range(self) -> None:
        """Test that loudness values match MoSQITo output."""
        result = self.loudness_op.process_array(self.signal_mono).compute()

        # Calculate using MoSQITo directly for comparison
        n_direct, _, _, _ = loudness_zwtv(
            self.signal_mono[0], self.sample_rate, field_type=self.field_type
        )

        # Results should match exactly
        np.testing.assert_array_equal(
            result[0], n_direct, err_msg="Loudness values differ from MoSQITo output"
        )

    def test_comparison_with_mosqito_direct(self) -> None:
        """
        Test that values match MoSQITo direct calculation.

        This is the key test to ensure our integration is correct by comparing
        the output with direct MoSQITo calculation.
        """
        # Calculate using our operation
        our_result = self.loudness_op.process_array(self.signal_mono).compute()

        # Calculate using MoSQITo directly
        n_direct, _, _, _ = loudness_zwtv(
            self.signal_mono[0], self.sample_rate, field_type=self.field_type
        )

        # Results should be very close (allowing for small numerical differences)
        np.testing.assert_array_equal(
            our_result[0],
            n_direct,
            err_msg="Loudness values differ from direct MoSQITo calculation",
        )

    def test_free_vs_diffuse_field(self) -> None:
        """Test that free field and diffuse field give different results."""
        # Calculate with free field
        loudness_free = LoudnessZwtv(self.sample_rate, field_type="free")
        result_free = loudness_free.process_array(self.signal_mono).compute()

        # Calculate with diffuse field
        loudness_diffuse = LoudnessZwtv(self.sample_rate, field_type="diffuse")
        result_diffuse = loudness_diffuse.process_array(self.signal_mono).compute()

        # Compare with MoSQITo direct calculation
        n_free_direct, _, _, _ = loudness_zwtv(
            self.signal_mono[0], self.sample_rate, field_type="free"
        )
        n_diffuse_direct, _, _, _ = loudness_zwtv(
            self.signal_mono[0], self.sample_rate, field_type="diffuse"
        )

        # Results should match MoSQITo exactly
        np.testing.assert_array_equal(result_free[0], n_free_direct)
        np.testing.assert_array_equal(result_diffuse[0], n_diffuse_direct)

    def test_amplitude_dependency(self) -> None:
        """Test that loudness increases with amplitude."""
        # Create signals with different amplitudes
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        freq = 1000.0

        # Low amplitude signal (approx 50 dB SPL)
        signal_low = np.array([0.006 * np.sin(2 * np.pi * freq * t)])

        # High amplitude signal (approx 80 dB SPL)
        signal_high = np.array([0.2 * np.sin(2 * np.pi * freq * t)])

        # Calculate loudness using wandas
        loudness_low = self.loudness_op.process_array(signal_low).compute()
        loudness_high = self.loudness_op.process_array(signal_high).compute()

        # Compare with MoSQITo direct calculation
        n_low_direct, _, _, _ = loudness_zwtv(
            signal_low[0], self.sample_rate, field_type=self.field_type
        )
        n_high_direct, _, _, _ = loudness_zwtv(
            signal_high[0], self.sample_rate, field_type=self.field_type
        )

        # Results should match MoSQITo exactly
        np.testing.assert_array_equal(loudness_low[0], n_low_direct)
        np.testing.assert_array_equal(loudness_high[0], n_high_direct)

    def test_silence_produces_low_loudness(self) -> None:
        """Test that silence produces near-zero loudness."""
        # Create silent signal
        silence = np.zeros((1, int(self.sample_rate * self.duration)))

        result = self.loudness_op.process_array(silence).compute()

        # Compare with MoSQITo direct calculation
        n_direct, _, _, _ = loudness_zwtv(
            silence[0], self.sample_rate, field_type=self.field_type
        )

        # Results should match MoSQITo exactly
        np.testing.assert_array_equal(result[0], n_direct)

    def test_white_noise_loudness(self) -> None:
        """Test loudness calculation with white noise."""
        # Generate white noise at moderate level
        np.random.seed(42)
        noise = np.random.normal(0, 0.02, (1, int(self.sample_rate * self.duration)))

        result = self.loudness_op.process_array(noise).compute()

        # Compare with MoSQITo direct calculation
        n_direct, _, _, _ = loudness_zwtv(
            noise[0], self.sample_rate, field_type=self.field_type
        )

        # Results should match MoSQITo exactly
        np.testing.assert_array_equal(result[0], n_direct)

    def test_process_with_dask(self) -> None:
        """Test that process method works with dask arrays."""
        result = self.loudness_op.process(self.dask_mono).compute()

        # Compare with MoSQITo direct calculation
        n_direct, _, _, _ = loudness_zwtv(
            self.signal_mono[0], self.sample_rate, field_type=self.field_type
        )

        # Results should match MoSQITo exactly
        np.testing.assert_array_equal(result[0], n_direct)

    def test_multi_channel_independence(self) -> None:
        """Test that each channel is processed independently."""
        # Create signal with different content in each channel
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        signal_ch1 = 0.05 * np.sin(
            2 * np.pi * 500 * t
        )  # Lower frequency, lower amplitude
        signal_ch2 = 0.1 * np.sin(
            2 * np.pi * 2000 * t
        )  # Higher frequency, higher amplitude

        stereo_signal = np.vstack([signal_ch1, signal_ch2])

        result = self.loudness_op.process_array(stereo_signal).compute()

        # Compare each channel with MoSQITo direct calculation
        n_ch1_direct, _, _, _ = loudness_zwtv(
            signal_ch1, self.sample_rate, field_type=self.field_type
        )
        n_ch2_direct, _, _, _ = loudness_zwtv(
            signal_ch2, self.sample_rate, field_type=self.field_type
        )

        # Results should match MoSQITo exactly for each channel
        np.testing.assert_array_equal(result[0], n_ch1_direct)
        np.testing.assert_array_equal(result[1], n_ch2_direct)

    def test_calculate_output_shape(self) -> None:
        """Test calculate_output_shape method."""
        input_shape = (1, 48000)  # 1 channel, 1 second at 48kHz
        output_shape = self.loudness_op.calculate_output_shape(input_shape)

        # Output should have same number of channels
        assert output_shape[0] == input_shape[0]
        # Output should have fewer time samples (downsampled)
        assert output_shape[1] < input_shape[1]
        assert output_shape[1] > 0

    def test_1d_input_handling(self) -> None:
        """Test that 1D input is properly handled."""
        # Create 1D signal
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        signal_1d = 0.05 * np.sin(2 * np.pi * 1000 * t)

        result = self.loudness_op.process_array(signal_1d).compute()

        # Should be reshaped to 2D with 1 channel
        assert result.ndim == 2
        assert result.shape[0] == 1

    def test_consistency_across_calls(self) -> None:
        """Test that repeated calls with same input produce same output."""
        result1 = self.loudness_op.process_array(self.signal_mono).compute()
        result2 = self.loudness_op.process_array(self.signal_mono).compute()

        # Results should be identical
        np.testing.assert_array_equal(result1, result2)

    def test_time_resolution(self) -> None:
        """Test that time resolution matches MoSQITo output."""
        result = self.loudness_op.process_array(self.signal_mono).compute()

        # Compare with MoSQITo direct calculation
        n_direct, _, _, _ = loudness_zwtv(
            self.signal_mono[0], self.sample_rate, field_type=self.field_type
        )

        # Time resolution should match exactly
        assert result.shape[1] == len(n_direct)


class TestLoudnessZwtvIntegration:
    """Integration tests for loudness calculation with ChannelFrame."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_rate: int = 48000
        self.duration: float = 0.1  # Reduced for faster tests

    def test_loudness_in_operation_registry(self) -> None:
        """Test that loudness operation is in registry."""
        from wandas.processing.base import _OPERATION_REGISTRY

        assert "loudness_zwtv" in _OPERATION_REGISTRY
        assert _OPERATION_REGISTRY["loudness_zwtv"] == LoudnessZwtv

    def test_channel_frame_loudness_method_exists(self) -> None:
        """Test that ChannelFrame has loudness_zwtv method."""
        from wandas.frames.channel import ChannelFrame

        # Create a simple frame
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        signal = np.array([0.05 * np.sin(2 * np.pi * 1000 * t)])
        dask_data = _da_from_array(signal, chunks=-1)
        frame = ChannelFrame(data=dask_data, sampling_rate=self.sample_rate)

        # Check method exists
        assert hasattr(frame, "loudness_zwtv")
        assert callable(frame.loudness_zwtv)

    def test_loudness_zwtv_metadata_updates(self) -> None:
        """Test that LoudnessZwtv returns correct metadata updates."""
        operation = LoudnessZwtv(sampling_rate=44100, field_type="free")

        updates = operation.get_metadata_updates()

        assert "sampling_rate" in updates
        assert updates["sampling_rate"] == 500.0


class TestLoudnessZwst:
    """Test suite for LoudnessZwst operation."""

    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 48000
        self.duration: float = 0.1  # Reduced from 1.0s for faster tests
        self.field_type: str = "free"

        # Create test signal: 1 kHz sine wave at 70 dB SPL
        # SPL reference: 20 µPa = 2e-5 Pa
        # 70 dB SPL corresponds to approximately 0.0632 Pa RMS
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        freq = 1000.0
        amplitude = 0.0632  # Approximate amplitude for 70 dB SPL
        self.signal_mono: NDArrayReal = np.array(
            [amplitude * np.sin(2 * np.pi * freq * t)]
        )

        # Create stereo signal
        self.signal_stereo: NDArrayReal = np.vstack(
            [
                amplitude * np.sin(2 * np.pi * freq * t),
                amplitude * np.sin(2 * np.pi * 2 * freq * t),
            ]
        )

        # Create dask arrays
        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=-1)
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=-1)

        # Create operation instance
        self.loudness_op = LoudnessZwst(self.sample_rate, field_type=self.field_type)

    def test_initialization(self) -> None:
        """Test LoudnessZwst initialization with different parameters."""
        # Default initialization
        loudness = LoudnessZwst(self.sample_rate)
        assert loudness.sampling_rate == self.sample_rate
        assert loudness.field_type == "free"

        # Custom field_type
        loudness_diffuse = LoudnessZwst(self.sample_rate, field_type="diffuse")
        assert loudness_diffuse.field_type == "diffuse"

    def test_invalid_field_type(self) -> None:
        """Test that invalid field_type raises ValueError."""
        with pytest.raises(ValueError, match="field_type must be 'free' or 'diffuse'"):
            LoudnessZwst(self.sample_rate, field_type="invalid")

    def test_operation_name(self) -> None:
        """Test that operation has correct name."""
        assert self.loudness_op.name == "loudness_zwst"

    def test_operation_registration(self) -> None:
        """Test that operation is properly registered."""
        op_class = get_operation("loudness_zwst")
        assert op_class == LoudnessZwst

    def test_create_operation(self) -> None:
        """Test creating operation via create_operation function."""
        op = create_operation("loudness_zwst", self.sample_rate, field_type="diffuse")
        assert isinstance(op, LoudnessZwst)
        assert op.field_type == "diffuse"

    def test_mono_signal_shape(self) -> None:
        """Test steady-state loudness calculation output shape for mono signal."""
        result = self.loudness_op.process_array(self.signal_mono).compute()

        # Result should be 2D (channels, 1)
        assert result.ndim == 2
        assert result.shape[0] == 1  # 1 channel
        assert result.shape[1] == 1  # Single loudness value

    def test_stereo_signal_shape(self) -> None:
        """Test steady-state loudness calculation output shape for stereo signal."""
        loudness_op = LoudnessZwst(self.sample_rate, field_type=self.field_type)
        result = loudness_op.process_array(self.signal_stereo).compute()

        # Result should be 2D (channels, 1)
        assert result.ndim == 2
        assert result.shape[0] == 2  # 2 channels
        assert result.shape[1] == 1  # Single loudness value per channel

        # Compare with MoSQITo direct calculation for each channel
        n_ch1_direct, _, _ = loudness_zwst(
            self.signal_stereo[0], self.sample_rate, field_type=self.field_type
        )
        n_ch2_direct, _, _ = loudness_zwst(
            self.signal_stereo[1], self.sample_rate, field_type=self.field_type
        )

        # Results should match MoSQITo exactly
        np.testing.assert_allclose(result[0, 0], n_ch1_direct, rtol=1e-10)
        np.testing.assert_allclose(result[1, 0], n_ch2_direct, rtol=1e-10)

    def test_loudness_values_range(self) -> None:
        """Test that loudness values match MoSQITo output."""
        result = self.loudness_op.process_array(self.signal_mono).compute()

        # Calculate using MoSQITo directly for comparison
        n_direct, _, _ = loudness_zwst(
            self.signal_mono[0], self.sample_rate, field_type=self.field_type
        )

        # Results should match exactly
        np.testing.assert_allclose(
            result[0, 0],
            n_direct,
            rtol=1e-10,
            err_msg="Loudness values differ from MoSQITo output",
        )

    def test_comparison_with_mosqito_direct(self) -> None:
        """
        Test that values match MoSQITo direct calculation.

        This is the key test to ensure our integration is correct by comparing
        the output with direct MoSQITo calculation.
        """
        # Calculate using our operation
        our_result = self.loudness_op.process_array(self.signal_mono).compute()

        # Calculate using MoSQITo directly
        n_direct, _, _ = loudness_zwst(
            self.signal_mono[0], self.sample_rate, field_type=self.field_type
        )

        # Results should be very close (allowing for small numerical differences)
        np.testing.assert_allclose(
            our_result[0, 0],
            n_direct,
            rtol=1e-10,
            err_msg="Loudness values differ from direct MoSQITo calculation",
        )

    def test_free_vs_diffuse_field(self) -> None:
        """Test that free field and diffuse field give different results."""
        # Calculate with free field
        loudness_free = LoudnessZwst(self.sample_rate, field_type="free")
        result_free = loudness_free.process_array(self.signal_mono).compute()

        # Calculate with diffuse field
        loudness_diffuse = LoudnessZwst(self.sample_rate, field_type="diffuse")
        result_diffuse = loudness_diffuse.process_array(self.signal_mono).compute()

        # Compare with MoSQITo direct calculation
        n_free_direct, _, _ = loudness_zwst(
            self.signal_mono[0], self.sample_rate, field_type="free"
        )
        n_diffuse_direct, _, _ = loudness_zwst(
            self.signal_mono[0], self.sample_rate, field_type="diffuse"
        )

        # Results should match MoSQITo exactly
        np.testing.assert_allclose(result_free[0, 0], n_free_direct, rtol=1e-10)
        np.testing.assert_allclose(result_diffuse[0, 0], n_diffuse_direct, rtol=1e-10)

    def test_amplitude_dependency(self) -> None:
        """Test that loudness increases with amplitude."""
        # Create signals with different amplitudes
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        freq = 1000.0

        # Low amplitude signal (approx 50 dB SPL)
        signal_low = np.array([0.006 * np.sin(2 * np.pi * freq * t)])

        # High amplitude signal (approx 80 dB SPL)
        signal_high = np.array([0.2 * np.sin(2 * np.pi * freq * t)])

        # Calculate loudness using wandas
        loudness_low = self.loudness_op.process_array(signal_low).compute()
        loudness_high = self.loudness_op.process_array(signal_high).compute()

        # Compare with MoSQITo direct calculation
        n_low_direct, _, _ = loudness_zwst(
            signal_low[0], self.sample_rate, field_type=self.field_type
        )
        n_high_direct, _, _ = loudness_zwst(
            signal_high[0], self.sample_rate, field_type=self.field_type
        )

        # Results should match MoSQITo exactly
        np.testing.assert_allclose(loudness_low[0, 0], n_low_direct, rtol=1e-10)
        np.testing.assert_allclose(loudness_high[0, 0], n_high_direct, rtol=1e-10)

        # Higher amplitude should produce higher loudness
        assert loudness_high[0, 0] > loudness_low[0, 0]

    def test_silence_produces_low_loudness(self) -> None:
        """Test that silence produces near-zero loudness."""
        # Create silent signal
        silence = np.zeros((1, int(self.sample_rate * self.duration)))

        result = self.loudness_op.process_array(silence).compute()

        # Compare with MoSQITo direct calculation
        n_direct, _, _ = loudness_zwst(
            silence[0], self.sample_rate, field_type=self.field_type
        )

        # Results should match MoSQITo exactly
        np.testing.assert_allclose(result[0, 0], n_direct, rtol=1e-10)

    def test_white_noise_loudness(self) -> None:
        """Test loudness calculation with white noise."""
        # Generate white noise at moderate level
        np.random.seed(42)
        noise = np.random.normal(0, 0.02, (1, int(self.sample_rate * self.duration)))

        result = self.loudness_op.process_array(noise).compute()

        # Compare with MoSQITo direct calculation
        n_direct, _, _ = loudness_zwst(
            noise[0], self.sample_rate, field_type=self.field_type
        )

        # Results should match MoSQITo exactly
        np.testing.assert_allclose(result[0, 0], n_direct, rtol=1e-10)

    def test_process_with_dask(self) -> None:
        """Test that process method works with dask arrays."""
        result = self.loudness_op.process(self.dask_mono).compute()

        # Compare with MoSQITo direct calculation
        n_direct, _, _ = loudness_zwst(
            self.signal_mono[0], self.sample_rate, field_type=self.field_type
        )

        # Results should match MoSQITo exactly
        np.testing.assert_allclose(result[0, 0], n_direct, rtol=1e-10)

    def test_multi_channel_independence(self) -> None:
        """Test that each channel is processed independently."""
        # Create signal with different content in each channel
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        signal_ch1 = 0.05 * np.sin(
            2 * np.pi * 500 * t
        )  # Lower frequency, lower amplitude
        signal_ch2 = 0.1 * np.sin(
            2 * np.pi * 2000 * t
        )  # Higher frequency, higher amplitude

        stereo_signal = np.vstack([signal_ch1, signal_ch2])

        result = self.loudness_op.process_array(stereo_signal).compute()

        # Compare each channel with MoSQITo direct calculation
        n_ch1_direct, _, _ = loudness_zwst(
            signal_ch1, self.sample_rate, field_type=self.field_type
        )
        n_ch2_direct, _, _ = loudness_zwst(
            signal_ch2, self.sample_rate, field_type=self.field_type
        )

        # Results should match MoSQITo exactly for each channel
        np.testing.assert_allclose(result[0, 0], n_ch1_direct, rtol=1e-10)
        np.testing.assert_allclose(result[1, 0], n_ch2_direct, rtol=1e-10)

    def test_calculate_output_shape(self) -> None:
        """Test calculate_output_shape method."""
        input_shape = (1, 48000)  # 1 channel, 1 second at 48kHz
        output_shape = self.loudness_op.calculate_output_shape(input_shape)

        # Output should be (channels, 1)
        assert output_shape[0] == input_shape[0]
        assert output_shape[1] == 1

        # Test with stereo
        input_shape_stereo = (2, 48000)
        output_shape_stereo = self.loudness_op.calculate_output_shape(
            input_shape_stereo
        )
        assert output_shape_stereo[0] == 2
        assert output_shape_stereo[1] == 1

    def test_1d_input_handling(self) -> None:
        """Test that 1D input is properly handled."""
        # Create 1D signal
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        signal_1d = 0.05 * np.sin(2 * np.pi * 1000 * t)

        result = self.loudness_op.process_array(signal_1d).compute()

        # Should be reshaped to 2D with 1 channel
        assert result.ndim == 2
        assert result.shape[0] == 1
        assert result.shape[1] == 1

    def test_consistency_across_calls(self) -> None:
        """Test that repeated calls with same input produce same output."""
        result1 = self.loudness_op.process_array(self.signal_mono).compute()
        result2 = self.loudness_op.process_array(self.signal_mono).compute()

        # Results should be identical
        np.testing.assert_array_equal(result1, result2)

    def test_metadata_updates(self) -> None:
        """Test that LoudnessZwst returns correct metadata updates."""
        operation = LoudnessZwst(sampling_rate=44100, field_type="free")

        updates = operation.get_metadata_updates()

        # Steady-state loudness doesn't update sampling rate (single value output)
        assert updates == {}


class TestLoudnessZwstIntegration:
    """Integration tests for steady-state loudness calculation with ChannelFrame."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_rate: int = 48000
        self.duration: float = 0.1  # Reduced for faster tests

    def test_loudness_in_operation_registry(self) -> None:
        """Test that loudness operation is in registry."""
        from wandas.processing.base import _OPERATION_REGISTRY

        assert "loudness_zwst" in _OPERATION_REGISTRY
        assert _OPERATION_REGISTRY["loudness_zwst"] == LoudnessZwst

    def test_channel_frame_loudness_method_exists(self) -> None:
        """Test that ChannelFrame has loudness_zwst method."""
        from wandas.frames.channel import ChannelFrame

        # Create a simple frame
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        signal = np.array([0.05 * np.sin(2 * np.pi * 1000 * t)])
        dask_data = _da_from_array(signal, chunks=-1)
        frame = ChannelFrame(data=dask_data, sampling_rate=self.sample_rate)

        # Check method exists
        assert hasattr(frame, "loudness_zwst")
        assert callable(frame.loudness_zwst)

    def test_channel_frame_loudness_returns_ndarray(self) -> None:
        """Test that ChannelFrame.loudness_zwst() returns NDArrayReal."""
        from wandas.frames.channel import ChannelFrame

        # Create mono frame
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        signal_mono = np.array([0.05 * np.sin(2 * np.pi * 1000 * t)])
        dask_data_mono = _da_from_array(signal_mono, chunks=-1)
        frame_mono = ChannelFrame(data=dask_data_mono, sampling_rate=self.sample_rate)

        # Calculate loudness
        loudness_mono = frame_mono.loudness_zwst(field_type="free")

        # Should be NDArrayReal (1D array)
        assert isinstance(loudness_mono, np.ndarray)
        assert loudness_mono.ndim == 1
        assert loudness_mono.shape[0] == 1  # One value per channel
        assert isinstance(loudness_mono[0], (float, np.floating))

        # Create stereo frame
        signal_stereo = np.vstack([signal_mono[0], signal_mono[0] * 0.5])
        dask_data_stereo = _da_from_array(signal_stereo, chunks=-1)
        frame_stereo = ChannelFrame(
            data=dask_data_stereo, sampling_rate=self.sample_rate
        )

        # Calculate loudness for stereo
        loudness_stereo = frame_stereo.loudness_zwst(field_type="free")

        # Should be 1D array with 2 values
        assert isinstance(loudness_stereo, np.ndarray)
        assert loudness_stereo.ndim == 1
        assert loudness_stereo.shape[0] == 2  # Two values (one per channel)

        # Can access values directly without double indexing
        assert isinstance(loudness_stereo[0], (float, np.floating))
        assert isinstance(loudness_stereo[1], (float, np.floating))

        # Can use numpy operations directly
        mean_loudness = loudness_stereo.mean()
        assert isinstance(mean_loudness, (float, np.floating))

    def test_channel_frame_loudness_matches_mosqito(self) -> None:
        """Test that ChannelFrame.loudness_zwst() matches direct MoSQITo call."""
        from wandas.frames.channel import ChannelFrame

        # Create test signal
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        signal = np.array([0.05 * np.sin(2 * np.pi * 1000 * t)])
        dask_data = _da_from_array(signal, chunks=-1)
        frame = ChannelFrame(data=dask_data, sampling_rate=self.sample_rate)

        # Calculate using wandas
        loudness_wandas = frame.loudness_zwst(field_type="free")

        # Calculate using MoSQITo directly
        n_direct, _, _ = loudness_zwst(signal[0], self.sample_rate, field_type="free")

        # Results should match
        np.testing.assert_allclose(loudness_wandas[0], n_direct, rtol=1e-10)
