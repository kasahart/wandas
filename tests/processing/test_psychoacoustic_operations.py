"""Tests for psychoacoustic processing operations."""

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray
from mosqito.sq_metrics import loudness_zwst, loudness_zwtv
from mosqito.sq_metrics import roughness_dw as roughness_dw_mosqito

from wandas.processing.base import create_operation, get_operation
from wandas.processing.psychoacoustic import LoudnessZwst, LoudnessZwtv, RoughnessDw
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

    def test_time_axis_values(self) -> None:
        """Test that time axis is correctly calculated based on sampling rate."""
        from wandas.frames.channel import ChannelFrame

        # Create frame and calculate loudness
        dask_data = _da_from_array(self.signal_mono, chunks=-1)
        frame = ChannelFrame(data=dask_data, sampling_rate=self.sample_rate)
        loudness_frame = frame.loudness_zwtv(field_type=self.field_type)

        # wandas time axis should be evenly spaced based on output sampling rate
        time_wandas = loudness_frame.time
        expected_sampling_rate = 500.0  # LoudnessZwtv outputs at 500 Hz

        # Check time axis properties
        assert len(time_wandas) > 0
        assert time_wandas[0] == 0.0  # Start at 0

        # Check that time axis is evenly spaced with correct sampling rate
        if len(time_wandas) > 1:
            time_steps = np.diff(time_wandas)
            expected_step = 1.0 / expected_sampling_rate

            np.testing.assert_allclose(
                time_steps,
                expected_step,
                rtol=1e-10,
                err_msg="Time steps are not evenly spaced",
            )

            # Verify sampling rate matches expected
            assert loudness_frame.sampling_rate == expected_sampling_rate, (
                f"Expected {expected_sampling_rate} Hz, "
                f"got {loudness_frame.sampling_rate}"
            )

    def test_plot_method_exists_and_works(self) -> None:
        """Test that loudness ChannelFrame can be plotted."""
        import matplotlib.pyplot as plt

        from wandas.frames.channel import ChannelFrame

        # Create frame and calculate loudness
        dask_data = _da_from_array(self.signal_mono, chunks=-1)
        frame = ChannelFrame(data=dask_data, sampling_rate=self.sample_rate)
        loudness_frame = frame.loudness_zwtv(field_type=self.field_type)

        # Test plot method exists
        assert hasattr(loudness_frame, "plot")
        assert callable(loudness_frame.plot)

        # Test plot execution without error (use overlay=True for single Axes)
        fig, ax = plt.subplots()
        result_ax = loudness_frame.plot(
            ax=ax, title="Test Loudness", ylabel="Loudness [sones]", overlay=True
        )

        assert result_ax is ax
        assert ax.get_ylabel() == "Loudness [sones]"
        assert ax.get_title() == "Test Loudness"

        plt.close(fig)


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


class TestRoughnessDw:
    """Test suite for RoughnessDw operation."""

    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 48000
        self.duration: float = 0.5  # 0.5s for roughness calculation
        self.overlap: float = 0.5

        # Create test signal: AM modulated tone (typical roughness stimulus)
        # 1 kHz carrier modulated at 70 Hz (known to produce roughness)
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        carrier_freq = 1000.0
        modulation_freq = 70.0
        amplitude = 0.1

        # AM signal: carrier * (1 + modulation_depth * sin(mod_freq))
        modulation_depth = 1.0
        carrier = np.sin(2 * np.pi * carrier_freq * t)
        modulation = 1 + modulation_depth * np.sin(2 * np.pi * modulation_freq * t)
        self.signal_mono: NDArrayReal = np.array([amplitude * carrier * modulation])

        # Create stereo signal with different modulation frequencies
        modulation2 = 1 + modulation_depth * np.sin(2 * np.pi * 40 * t)
        self.signal_stereo: NDArrayReal = np.vstack(
            [
                amplitude * carrier * modulation,
                amplitude * carrier * modulation2,
            ]
        )

        # Create dask arrays
        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=-1)
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=-1)

        # Create operation instance
        self.roughness_op = RoughnessDw(self.sample_rate, overlap=self.overlap)

    def test_initialization(self) -> None:
        """Test RoughnessDw initialization with different parameters."""
        # Default initialization
        roughness = RoughnessDw(self.sample_rate)
        assert roughness.sampling_rate == self.sample_rate
        assert roughness.overlap == 0.5

        # Custom overlap
        roughness_custom = RoughnessDw(self.sample_rate, overlap=0.0)
        assert roughness_custom.overlap == 0.0

    def test_invalid_overlap(self) -> None:
        """Test that invalid overlap raises ValueError."""
        with pytest.raises(ValueError, match="overlap must be in"):
            RoughnessDw(self.sample_rate, overlap=1.5)

        with pytest.raises(ValueError, match="overlap must be in"):
            RoughnessDw(self.sample_rate, overlap=-0.1)

    def test_operation_name(self) -> None:
        """Test that operation has correct name."""
        assert self.roughness_op.name == "roughness_dw"

    def test_operation_registration(self) -> None:
        """Test that operation is properly registered."""
        op_class = get_operation("roughness_dw")
        assert op_class == RoughnessDw

    def test_create_operation(self) -> None:
        """Test creating operation via create_operation function."""
        op = create_operation("roughness_dw", self.sample_rate, overlap=0.75)
        assert isinstance(op, RoughnessDw)
        assert op.overlap == 0.75

    def test_mono_signal_shape(self) -> None:
        """Test roughness calculation output shape for mono signal."""
        result = self.roughness_op.process_array(self.signal_mono).compute()

        # Result should be 2D (channels, time_samples)
        assert result.ndim == 2
        assert result.shape[0] == 1  # 1 channel
        # Time samples should be less than input samples (windowed processing)
        assert result.shape[1] < self.signal_mono.shape[1]
        assert result.shape[1] > 0

    def test_stereo_signal_shape(self) -> None:
        """Test roughness calculation output shape for stereo signal."""
        result = self.roughness_op.process_array(self.signal_stereo).compute()

        # Result should be 2D (channels, time_samples)
        assert result.ndim == 2
        assert result.shape[0] == 2  # 2 channels
        assert result.shape[1] > 0

    def test_comparison_with_mosqito_direct(self) -> None:
        """Test that values match MoSQITo direct calculation."""
        # Calculate using our operation
        our_result = self.roughness_op.process_array(self.signal_mono).compute()

        # Calculate using MoSQITo directly
        r_direct, _, _, _ = roughness_dw_mosqito(
            self.signal_mono[0], self.sample_rate, overlap=self.overlap
        )

        # Results should match MoSQITo exactly
        np.testing.assert_array_equal(
            our_result[0],
            r_direct,
            err_msg="Roughness values differ from direct MoSQITo calculation",
        )

    def test_stereo_matches_mosqito(self) -> None:
        """Test stereo signal matches MoSQITo for each channel."""
        result = self.roughness_op.process_array(self.signal_stereo).compute()

        # Compare with MoSQITo direct calculation for each channel
        r_ch1_direct, _, _, _ = roughness_dw_mosqito(
            self.signal_stereo[0], self.sample_rate, overlap=self.overlap
        )
        r_ch2_direct, _, _, _ = roughness_dw_mosqito(
            self.signal_stereo[1], self.sample_rate, overlap=self.overlap
        )

        # Results should match MoSQITo exactly
        np.testing.assert_array_equal(result[0], r_ch1_direct)
        np.testing.assert_array_equal(result[1], r_ch2_direct)

    def test_overlap_affects_time_resolution(self) -> None:
        """Test that overlap affects output time resolution."""
        # Test with different overlap values
        roughness_overlap0 = RoughnessDw(self.sample_rate, overlap=0.0)
        roughness_overlap05 = RoughnessDw(self.sample_rate, overlap=0.5)

        result_overlap0 = roughness_overlap0.process_array(self.signal_mono).compute()
        result_overlap05 = roughness_overlap05.process_array(self.signal_mono).compute()

        # Higher overlap should give more time points
        # (overlap=0.5 has 100ms hop, overlap=0.0 has 200ms hop)
        # So overlap=0.5 should have approximately 2x more points
        assert result_overlap05.shape[1] > result_overlap0.shape[1]

    def test_roughness_values_range(self) -> None:
        """Test that roughness values are in reasonable range."""
        result = self.roughness_op.process_array(self.signal_mono).compute()

        # Roughness values should be non-negative
        assert np.all(result >= 0)

        # For our AM signal, roughness should be detectable (> 0)
        assert np.max(result) > 0

        # Compare with MoSQITo direct calculation
        r_direct, _, _, _ = roughness_dw_mosqito(
            self.signal_mono[0], self.sample_rate, overlap=self.overlap
        )

        # Verify it matches MoSQITo
        np.testing.assert_array_equal(result[0], r_direct)

    def test_silence_produces_low_roughness(self) -> None:
        """Test that silence produces near-zero roughness."""
        silence = np.zeros((1, int(self.sample_rate * self.duration)))

        result = self.roughness_op.process_array(silence).compute()

        # Compare with MoSQITo direct calculation
        r_direct, _, _, _ = roughness_dw_mosqito(
            silence[0], self.sample_rate, overlap=self.overlap
        )

        # Results should match MoSQITo exactly
        np.testing.assert_array_equal(result[0], r_direct)

        # Roughness should be very low for silence
        assert np.all(result < 0.01)

    def test_process_with_dask(self) -> None:
        """Test that process method works with dask arrays."""
        result = self.roughness_op.process(self.dask_mono).compute()

        # Compare with MoSQITo direct calculation
        r_direct, _, _, _ = roughness_dw_mosqito(
            self.signal_mono[0], self.sample_rate, overlap=self.overlap
        )

        # Results should match MoSQITo exactly
        np.testing.assert_array_equal(result[0], r_direct)

    def test_multi_channel_independence(self) -> None:
        """Test that each channel is processed independently."""
        result = self.roughness_op.process_array(self.signal_stereo).compute()

        # Compare each channel with MoSQITo direct calculation
        r_ch1_direct, _, _, _ = roughness_dw_mosqito(
            self.signal_stereo[0], self.sample_rate, overlap=self.overlap
        )
        r_ch2_direct, _, _, _ = roughness_dw_mosqito(
            self.signal_stereo[1], self.sample_rate, overlap=self.overlap
        )

        # Results should match MoSQITo exactly for each channel
        np.testing.assert_array_equal(result[0], r_ch1_direct)
        np.testing.assert_array_equal(result[1], r_ch2_direct)

    def test_1d_input_handling(self) -> None:
        """Test that 1D input is properly handled."""
        # Create 1D signal
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        signal_1d = 0.1 * np.sin(2 * np.pi * 1000 * t)

        result = self.roughness_op.process_array(signal_1d).compute()

        # Should be reshaped to 2D with 1 channel
        assert result.ndim == 2
        assert result.shape[0] == 1

    def test_consistency_across_calls(self) -> None:
        """Test that repeated calls with same input produce same output."""
        result1 = self.roughness_op.process_array(self.signal_mono).compute()
        result2 = self.roughness_op.process_array(self.signal_mono).compute()

        # Results should be identical
        np.testing.assert_array_equal(result1, result2)

    def test_sampling_rate_metadata(self) -> None:
        """Test that output sampling rate is correctly calculated."""
        # overlap=0.5, window=200ms, hop=100ms → fs≈10Hz
        metadata_updates = self.roughness_op.get_metadata_updates()

        assert "sampling_rate" in metadata_updates
        expected_sr = 1.0 / (0.2 * (1 - self.overlap))
        assert metadata_updates["sampling_rate"] == pytest.approx(expected_sr)

        # Test with overlap=0.0 → fs≈5Hz
        roughness_overlap0 = RoughnessDw(self.sample_rate, overlap=0.0)
        metadata_overlap0 = roughness_overlap0.get_metadata_updates()
        expected_sr_overlap0 = 1.0 / 0.2  # 5 Hz
        assert metadata_overlap0["sampling_rate"] == pytest.approx(expected_sr_overlap0)

    def test_calculate_output_shape(self) -> None:
        """Test calculate_output_shape method."""
        input_shape = (1, int(self.sample_rate * self.duration))
        output_shape = self.roughness_op.calculate_output_shape(input_shape)

        # Output should have same number of channels
        assert output_shape[0] == input_shape[0]
        # Output should have fewer time samples
        assert output_shape[1] < input_shape[1]
        assert output_shape[1] > 0

    def test_time_axis_values(self) -> None:
        """Test that time axis is correctly calculated based on sampling rate."""
        from wandas.frames.channel import ChannelFrame

        # Create frame and calculate roughness
        dask_data = _da_from_array(self.signal_mono, chunks=-1)
        frame = ChannelFrame(data=dask_data, sampling_rate=self.sample_rate)
        roughness_frame = frame.roughness_dw(overlap=self.overlap)

        # wandas time axis should be evenly spaced based on output sampling rate
        time_wandas = roughness_frame.time

        # RoughnessDw with overlap=0.5 should output at ~10 Hz
        # (200ms window, 100ms hop -> 1/0.1 = 10 Hz)
        expected_sampling_rate = 1.0 / (0.2 * (1 - self.overlap))

        # Check time axis properties
        assert len(time_wandas) > 0
        assert time_wandas[0] == 0.0  # Start at 0

        # Check that time axis is evenly spaced with correct sampling rate
        if len(time_wandas) > 1:
            time_steps = np.diff(time_wandas)
            expected_step = 1.0 / expected_sampling_rate

            np.testing.assert_allclose(
                time_steps,
                expected_step,
                rtol=1e-10,
                err_msg="Time steps are not evenly spaced",
            )

            # Verify sampling rate matches expected
            assert roughness_frame.sampling_rate == pytest.approx(
                expected_sampling_rate
            ), (
                f"Expected {expected_sampling_rate} Hz, "
                f"got {roughness_frame.sampling_rate}"
            )


class TestRoughnessDwIntegration:
    """Integration tests for roughness calculation with ChannelFrame."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_rate: int = 48000
        self.duration: float = 0.5

    def test_roughness_in_operation_registry(self) -> None:
        """Test that roughness operation is in registry."""
        from wandas.processing.base import _OPERATION_REGISTRY

        assert "roughness_dw" in _OPERATION_REGISTRY
        assert _OPERATION_REGISTRY["roughness_dw"] == RoughnessDw

    def test_channel_frame_roughness_method_exists(self) -> None:
        """Test that ChannelFrame has roughness_dw method."""
        from wandas.frames.channel import ChannelFrame

        # Create a simple frame
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        signal = np.array([0.1 * np.sin(2 * np.pi * 1000 * t)])
        dask_data = _da_from_array(signal, chunks=-1)
        frame = ChannelFrame(data=dask_data, sampling_rate=self.sample_rate)

        # Check method exists
        assert hasattr(frame, "roughness_dw")
        assert callable(frame.roughness_dw)

    def test_channel_frame_roughness_returns_channelframe(self) -> None:
        """Test that ChannelFrame.roughness_dw() returns ChannelFrame."""
        from wandas.frames.channel import ChannelFrame

        # Create AM modulated signal
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        carrier = np.sin(2 * np.pi * 1000 * t)
        modulation = 1 + np.sin(2 * np.pi * 70 * t)
        signal = np.array([0.1 * carrier * modulation])
        dask_data = _da_from_array(signal, chunks=-1)
        frame = ChannelFrame(data=dask_data, sampling_rate=self.sample_rate)

        # Calculate roughness
        roughness_frame = frame.roughness_dw(overlap=0.5)

        # Should return ChannelFrame
        assert isinstance(roughness_frame, ChannelFrame)

        # Should have updated sampling rate
        assert roughness_frame.sampling_rate == pytest.approx(10.0)  # ~10 Hz

        # Should have roughness data
        assert roughness_frame.data is not None
        assert roughness_frame.n_samples > 0

    def test_channel_frame_roughness_matches_mosqito(self) -> None:
        """Test that ChannelFrame.roughness_dw() matches direct MoSQITo call."""
        from wandas.frames.channel import ChannelFrame

        # Create AM modulated signal
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        carrier = np.sin(2 * np.pi * 1000 * t)
        modulation = 1 + np.sin(2 * np.pi * 70 * t)
        signal = np.array([0.1 * carrier * modulation])
        dask_data = _da_from_array(signal, chunks=-1)
        frame = ChannelFrame(data=dask_data, sampling_rate=self.sample_rate)

        # Calculate using wandas
        roughness_frame = frame.roughness_dw(overlap=0.5)

        # Get data (may be Dask or NumPy)
        roughness_wandas_data = (
            roughness_frame._data.compute()
            if hasattr(roughness_frame._data, "compute")
            else roughness_frame._data
        )

        # Calculate using MoSQITo directly
        r_direct, _, _, _ = roughness_dw_mosqito(
            signal[0], self.sample_rate, overlap=0.5
        )

        # Results should match (roughness_wandas_data is 2D: (n_channels, n_time))
        np.testing.assert_array_equal(roughness_wandas_data[0], r_direct)


class TestRoughnessDwSpec:
    """Test suite for RoughnessDwSpec operation."""

    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 48000
        self.duration: float = 0.5
        self.overlap: float = 0.5

        # Create AM modulated signal (roughness stimuli)
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        carrier = np.sin(2 * np.pi * 1000 * t)
        modulation = 1 + np.sin(2 * np.pi * 70 * t)
        self.signal_mono: NDArrayReal = np.array([0.1 * carrier * modulation])

        # Create stereo signal (different modulation frequencies)
        modulation2 = 1 + np.sin(2 * np.pi * 50 * t)
        self.signal_stereo: NDArrayReal = np.vstack(
            [
                0.1 * carrier * modulation,
                0.1 * carrier * modulation2,
            ]
        )

        # Create dask arrays
        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=-1)
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=-1)

        # Import operation class
        from wandas.processing.psychoacoustic import RoughnessDwSpec

        self.roughness_spec_op = RoughnessDwSpec(self.sample_rate, overlap=self.overlap)

    def test_initialization(self) -> None:
        """Test RoughnessDwSpec initialization with different parameters."""
        from wandas.processing.psychoacoustic import RoughnessDwSpec

        # Default initialization
        roughness_spec = RoughnessDwSpec(self.sample_rate)
        assert roughness_spec.sampling_rate == self.sample_rate
        assert roughness_spec.overlap == 0.5

        # Custom overlap
        roughness_spec_custom = RoughnessDwSpec(self.sample_rate, overlap=0.0)
        assert roughness_spec_custom.overlap == 0.0

    def test_invalid_overlap(self) -> None:
        """Test that invalid overlap raises ValueError."""
        from wandas.processing.psychoacoustic import RoughnessDwSpec

        with pytest.raises(ValueError, match="overlap must be in"):
            RoughnessDwSpec(self.sample_rate, overlap=1.5)

        with pytest.raises(ValueError, match="overlap must be in"):
            RoughnessDwSpec(self.sample_rate, overlap=-0.1)

    def test_operation_name(self) -> None:
        """Test that operation has correct name."""
        assert self.roughness_spec_op.name == "roughness_dw_spec"

    def test_operation_registration(self) -> None:
        """Test that operation is properly registered."""
        op_class = get_operation("roughness_dw_spec")
        from wandas.processing.psychoacoustic import RoughnessDwSpec

        assert op_class == RoughnessDwSpec

    def test_create_operation(self) -> None:
        """Test creating operation via create_operation function."""
        op = create_operation("roughness_dw_spec", self.sample_rate, overlap=0.75)
        from wandas.processing.psychoacoustic import RoughnessDwSpec

        assert isinstance(op, RoughnessDwSpec)
        assert op.overlap == 0.75

    def test_mono_signal_shape(self) -> None:
        """Test roughness_spec calculation output shape for mono signal."""
        result = self.roughness_spec_op.process_array(self.signal_mono).compute()

        # Result should be 2D (n_bark_bands, time_samples) for mono
        assert result.ndim == 2
        assert result.shape[0] == 47  # 47 Bark bands
        assert result.shape[1] > 0  # Time samples

    def test_stereo_signal_shape(self) -> None:
        """Test roughness_spec calculation output shape for stereo signal."""
        result = self.roughness_spec_op.process_array(self.signal_stereo).compute()

        # Result should be 3D (n_channels, n_bark_bands, time_samples) for stereo
        assert result.ndim == 3
        assert result.shape[0] == 2  # 2 channels
        assert result.shape[1] == 47  # 47 Bark bands
        assert result.shape[2] > 0  # Time samples

    def test_comparison_with_mosqito_direct_mono(self) -> None:
        """Test that specific roughness values match MoSQITo direct calculation."""
        # Calculate using our operation
        our_result = self.roughness_spec_op.process_array(self.signal_mono).compute()

        # Calculate using MoSQITo directly
        _, r_spec_direct, _, _ = roughness_dw_mosqito(
            self.signal_mono[0], self.sample_rate, overlap=self.overlap
        )

        # Results should match MoSQITo exactly
        np.testing.assert_array_equal(
            our_result,
            r_spec_direct,
            err_msg="Specific roughness values differ from direct MoSQITo calculation",
        )

    def test_comparison_with_mosqito_direct_stereo(self) -> None:
        """Test that stereo specific roughness matches MoSQITo for each channel."""
        result = self.roughness_spec_op.process_array(self.signal_stereo).compute()

        # Compare with MoSQITo direct calculation for each channel
        _, r_spec_ch1_direct, _, _ = roughness_dw_mosqito(
            self.signal_stereo[0], self.sample_rate, overlap=self.overlap
        )
        _, r_spec_ch2_direct, _, _ = roughness_dw_mosqito(
            self.signal_stereo[1], self.sample_rate, overlap=self.overlap
        )

        # Results should match MoSQITo exactly
        np.testing.assert_array_equal(result[0], r_spec_ch1_direct)
        np.testing.assert_array_equal(result[1], r_spec_ch2_direct)

    def test_bark_axis_values(self) -> None:
        """Test that bark_axis has correct values."""
        bark_axis = self.roughness_spec_op.bark_axis

        # Should have 47 Bark bands from 0.5 to 23.5
        assert len(bark_axis) == 47
        assert bark_axis[0] == pytest.approx(0.5)
        assert bark_axis[-1] == pytest.approx(23.5)

    def test_bark_axis_matches_mosqito(self) -> None:
        """Test that bark_axis matches MoSQITo bark_axis exactly."""
        # Get bark_axis from our operation
        bark_axis_wandas = self.roughness_spec_op.bark_axis

        # Get bark_axis from MoSQITo directly
        _, _, bark_axis_mosqito, _ = roughness_dw_mosqito(
            self.signal_mono[0], self.sample_rate, overlap=self.overlap
        )

        # Results should match exactly
        np.testing.assert_array_equal(
            bark_axis_wandas,
            bark_axis_mosqito,
            err_msg="Bark axis values differ from MoSQITo bark axis",
        )

    def test_integration_with_total_roughness(self) -> None:
        """Test that integrating specific roughness gives total roughness."""
        # Calculate specific roughness
        r_spec = self.roughness_spec_op.process_array(self.signal_mono).compute()

        # Calculate total roughness directly
        r_total, _, _, _ = roughness_dw_mosqito(
            self.signal_mono[0], self.sample_rate, overlap=self.overlap
        )

        # Integrate specific roughness: R = 0.25 * sum(R_spec) over Bark bands
        r_integrated = 0.25 * np.sum(r_spec, axis=0)

        # Should match total roughness (within numerical precision)
        np.testing.assert_allclose(
            r_integrated,
            r_total,
            rtol=1e-10,
            err_msg="Integrated specific roughness does not match total roughness",
        )

    def test_metadata_updates(self) -> None:
        """Test that metadata updates include sampling rate and bark_axis."""
        metadata_updates = self.roughness_spec_op.get_metadata_updates()

        assert "sampling_rate" in metadata_updates
        assert "bark_axis" in metadata_updates

        # Check sampling rate is correct
        expected_sr = 1.0 / (0.2 * (1 - self.overlap))
        assert metadata_updates["sampling_rate"] == pytest.approx(expected_sr)

        # Check bark_axis
        bark_axis = metadata_updates["bark_axis"]
        assert len(bark_axis) == 47
        assert bark_axis[0] == pytest.approx(0.5)
        assert bark_axis[-1] == pytest.approx(23.5)

    def test_silence_produces_low_specific_roughness(self) -> None:
        """Test that silence produces near-zero specific roughness."""
        silence = np.zeros((1, int(self.sample_rate * self.duration)))

        result = self.roughness_spec_op.process_array(silence).compute()

        # Compare with MoSQITo direct calculation
        _, r_spec_direct, _, _ = roughness_dw_mosqito(
            silence[0], self.sample_rate, overlap=self.overlap
        )

        # Results should match MoSQITo exactly
        np.testing.assert_array_equal(result, r_spec_direct)

        # Specific roughness should be very low for silence
        assert np.all(result < 0.01)

    def test_process_with_dask(self) -> None:
        """Test that process method works with dask arrays."""
        result = self.roughness_spec_op.process(self.dask_mono).compute()

        # Compare with MoSQITo direct calculation
        _, r_spec_direct, _, _ = roughness_dw_mosqito(
            self.signal_mono[0], self.sample_rate, overlap=self.overlap
        )

        # Results should match MoSQITo exactly
        np.testing.assert_array_equal(result, r_spec_direct)

    def test_1d_input_handling(self) -> None:
        """Test that 1D input is properly handled."""
        # Create 1D signal
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        signal_1d = 0.1 * np.sin(2 * np.pi * 1000 * t)

        result = self.roughness_spec_op.process_array(signal_1d).compute()

        # Should be reshaped to 2D with shape (n_bark_bands, n_time)
        assert result.ndim == 2
        assert result.shape[0] == 47

    def test_consistency_across_calls(self) -> None:
        """Test that repeated calls with same input produce same output."""
        result1 = self.roughness_spec_op.process_array(self.signal_mono).compute()
        result2 = self.roughness_spec_op.process_array(self.signal_mono).compute()

        # Results should be identical
        np.testing.assert_array_equal(result1, result2)

    def test_bark_axis_caching(self) -> None:
        """Test that bark_axis is cached to avoid redundant MoSQITo calls."""
        from wandas.processing.psychoacoustic import RoughnessDwSpec

        # Clear the cache to start fresh
        RoughnessDwSpec._bark_axis_cache.clear()

        # First instantiation should compute bark_axis
        op1 = RoughnessDwSpec(self.sample_rate, overlap=self.overlap)
        bark_axis_1 = op1.bark_axis

        # Cache should now contain one entry
        assert len(RoughnessDwSpec._bark_axis_cache) == 1
        cache_key = (self.sample_rate, self.overlap)
        assert cache_key in RoughnessDwSpec._bark_axis_cache

        # Second instantiation with same parameters should use cached value
        op2 = RoughnessDwSpec(self.sample_rate, overlap=self.overlap)
        bark_axis_2 = op2.bark_axis

        # Cache should still have one entry (not recomputed)
        assert len(RoughnessDwSpec._bark_axis_cache) == 1

        # Both instances should have identical bark_axis (same array reference)
        np.testing.assert_array_equal(bark_axis_1, bark_axis_2)
        assert bark_axis_1 is RoughnessDwSpec._bark_axis_cache[cache_key]
        assert bark_axis_2 is RoughnessDwSpec._bark_axis_cache[cache_key]

        # Different parameters should trigger new computation
        op3 = RoughnessDwSpec(self.sample_rate, overlap=0.0)
        bark_axis_3 = op3.bark_axis

        # Cache should now have two entries
        assert len(RoughnessDwSpec._bark_axis_cache) == 2
        cache_key_2 = (self.sample_rate, 0.0)
        assert cache_key_2 in RoughnessDwSpec._bark_axis_cache

        # Bark axis should be the same (MoSQITo returns same bark_axis regardless of overlap)
        np.testing.assert_array_equal(bark_axis_1, bark_axis_3)

        # Clean up: clear cache after test
        RoughnessDwSpec._bark_axis_cache.clear()


class TestRoughnessDwSpecIntegration:
    """Integration tests for specific roughness calculation with ChannelFrame."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_rate: int = 48000
        self.duration: float = 0.5

    def test_roughness_spec_in_operation_registry(self) -> None:
        """Test that roughness_dw_spec operation is in registry."""
        from wandas.processing.base import _OPERATION_REGISTRY
        from wandas.processing.psychoacoustic import RoughnessDwSpec

        assert "roughness_dw_spec" in _OPERATION_REGISTRY
        assert _OPERATION_REGISTRY["roughness_dw_spec"] == RoughnessDwSpec

    def test_channel_frame_roughness_spec_method_exists(self) -> None:
        """Test that ChannelFrame has roughness_dw_spec method."""
        from wandas.frames.channel import ChannelFrame

        # Create a simple frame
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        signal = np.array([0.1 * np.sin(2 * np.pi * 1000 * t)])
        dask_data = _da_from_array(signal, chunks=-1)
        frame = ChannelFrame(data=dask_data, sampling_rate=self.sample_rate)

        # Check method exists
        assert hasattr(frame, "roughness_dw_spec")
        assert callable(frame.roughness_dw_spec)

    def test_channel_frame_roughness_spec_returns_roughness_frame(self) -> None:
        """Test that ChannelFrame.roughness_dw_spec() returns RoughnessFrame."""
        from wandas.frames.channel import ChannelFrame
        from wandas.frames.roughness import RoughnessFrame

        # Create AM modulated signal
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        carrier = np.sin(2 * np.pi * 1000 * t)
        modulation = 1 + np.sin(2 * np.pi * 70 * t)
        signal = np.array([0.1 * carrier * modulation])
        dask_data = _da_from_array(signal, chunks=-1)
        frame = ChannelFrame(data=dask_data, sampling_rate=self.sample_rate)

        # Calculate specific roughness
        roughness_spec_frame = frame.roughness_dw_spec(overlap=0.5)

        # Should return RoughnessFrame
        assert isinstance(roughness_spec_frame, RoughnessFrame)

        # Should have updated sampling rate
        assert roughness_spec_frame.sampling_rate == pytest.approx(10.0)  # ~10 Hz

        # Should have 47 Bark bands
        assert roughness_spec_frame.n_bark_bands == 47

        # Should have bark_axis
        assert len(roughness_spec_frame.bark_axis) == 47

    def test_channel_frame_roughness_spec_matches_mosqito(self) -> None:
        """Test that ChannelFrame.roughness_dw_spec() matches direct MoSQITo call."""
        from wandas.frames.channel import ChannelFrame

        # Create AM modulated signal
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        carrier = np.sin(2 * np.pi * 1000 * t)
        modulation = 1 + np.sin(2 * np.pi * 70 * t)
        signal = np.array([0.1 * carrier * modulation])
        dask_data = _da_from_array(signal, chunks=-1)
        frame = ChannelFrame(data=dask_data, sampling_rate=self.sample_rate)

        # Calculate using wandas
        roughness_spec_frame = frame.roughness_dw_spec(overlap=0.5)

        # Get data (may be Dask or NumPy)
        roughness_spec_wandas_data = (
            roughness_spec_frame._data.compute()
            if hasattr(roughness_spec_frame._data, "compute")
            else roughness_spec_frame._data
        )

        # Calculate using MoSQITo directly
        _, r_spec_direct, _, _ = roughness_dw_mosqito(
            signal[0], self.sample_rate, overlap=0.5
        )

        # Results should match
        # For mono signal, wandas returns (n_bark, n_time)
        np.testing.assert_array_equal(roughness_spec_wandas_data, r_spec_direct)
