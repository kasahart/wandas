"""Tests for psychoacoustic processing operations."""

from unittest.mock import patch

import numpy as np
import pytest
from dask.array.core import Array as DaArray
from mosqito.sq_metrics import loudness_zwst, loudness_zwtv
from mosqito.sq_metrics import roughness_dw as roughness_dw_mosqito
from mosqito.sq_metrics import sharpness_din_st as sharpness_din_st_mosqito
from mosqito.sq_metrics import sharpness_din_tv as sharpness_din_tv_mosqito

import wandas.processing.psychoacoustic as psychoacoustic_module
from wandas.processing.base import create_operation, get_operation
from wandas.processing.psychoacoustic import (
    LoudnessZwst,
    LoudnessZwtv,
    RoughnessDw,
    RoughnessDwSpec,
    SharpnessDin,
    SharpnessDinSt,
)
from wandas.utils.dask_helpers import da_from_array
from wandas.utils.types import NDArrayReal

_SR: int = 48000


def _psychoacoustic_class(name: str) -> type:
    return getattr(psychoacoustic_module, name)


# -- Shared signal builders ------------------------------------------------


def _loudness_signal(
    duration: float = 0.1,
) -> tuple[NDArrayReal, NDArrayReal, DaArray, DaArray]:
    """1 kHz sine at ~70 dB SPL (0.0632 Pa) — mono + stereo + dask variants."""
    t = np.linspace(0, duration, int(_SR * duration))
    amp = 0.0632
    mono: NDArrayReal = np.array([amp * np.sin(2 * np.pi * 1000.0 * t)])
    stereo: NDArrayReal = np.vstack(
        [
            amp * np.sin(2 * np.pi * 1000.0 * t),
            amp * np.sin(2 * np.pi * 2000.0 * t),
        ]
    )
    return mono, stereo, da_from_array(mono, chunks=(1, -1)), da_from_array(stereo, chunks=(1, -1))


def _roughness_signal(
    duration: float = 0.5,
) -> tuple[NDArrayReal, NDArrayReal, DaArray, DaArray]:
    """AM-modulated 1 kHz carrier (70 Hz mod, depth 1.0) — roughness stimulus."""
    t = np.linspace(0, duration, int(_SR * duration))
    carrier = np.sin(2 * np.pi * 1000.0 * t)
    mod70 = 1 + np.sin(2 * np.pi * 70.0 * t)
    mod40 = 1 + np.sin(2 * np.pi * 40.0 * t)
    mono: NDArrayReal = np.array([0.1 * carrier * mod70])
    stereo: NDArrayReal = np.vstack([0.1 * carrier * mod70, 0.1 * carrier * mod40])
    return mono, stereo, da_from_array(mono, chunks=(1, -1)), da_from_array(stereo, chunks=(1, -1))


def _roughness_spec_signal(
    duration: float = 0.5,
) -> tuple[NDArrayReal, NDArrayReal, DaArray, DaArray]:
    """AM-modulated 1 kHz carrier (70/50 Hz mods) — specific roughness stimulus."""
    t = np.linspace(0, duration, int(_SR * duration))
    carrier = np.sin(2 * np.pi * 1000.0 * t)
    mod70 = 1 + np.sin(2 * np.pi * 70.0 * t)
    mod50 = 1 + np.sin(2 * np.pi * 50.0 * t)
    mono: NDArrayReal = np.array([0.1 * carrier * mod70])
    stereo: NDArrayReal = np.vstack([0.1 * carrier * mod70, 0.1 * carrier * mod50])
    return mono, stereo, da_from_array(mono, chunks=(1, -1)), da_from_array(stereo, chunks=(1, -1))


def _sharpness_signal(
    duration: float = 0.1,
) -> tuple[NDArrayReal, NDArrayReal, DaArray, DaArray]:
    """4 kHz sine at moderate level (0.05) — sharpness stimulus."""
    t = np.linspace(0, duration, int(_SR * duration))
    amp = 0.05
    mono: NDArrayReal = np.array([amp * np.sin(2 * np.pi * 4000.0 * t)])
    stereo: NDArrayReal = np.vstack(
        [
            amp * np.sin(2 * np.pi * 4000.0 * t),
            amp * np.sin(2 * np.pi * 8000.0 * t),
        ]
    )
    return mono, stereo, da_from_array(mono, chunks=(1, -1)), da_from_array(stereo, chunks=(1, -1))


class TestLoudnessZwtv:
    """Test suite for LoudnessZwtv operation."""

    def test_initialization(self) -> None:
        """Test LoudnessZwtv initialization with different parameters."""
        # Default initialization
        loudness = LoudnessZwtv(_SR)
        assert loudness.sampling_rate == _SR
        assert loudness.field_type == "free"

        # Custom field_type
        loudness_diffuse = LoudnessZwtv(_SR, field_type="diffuse")
        assert loudness_diffuse.field_type == "diffuse"

    def test_invalid_field_type(self) -> None:
        """Test that invalid field_type raises ValueError."""
        with pytest.raises(ValueError, match="field_type must be 'free' or 'diffuse'"):
            LoudnessZwtv(_SR, field_type="invalid")

    def test_operation_name(self) -> None:
        """Test that operation has correct name."""
        op = LoudnessZwtv(_SR, field_type="free")
        assert op.name == "loudness_zwtv"

    def test_operation_registration(self) -> None:
        """Test that operation is properly registered."""
        op_class = get_operation("loudness_zwtv")
        assert op_class is _psychoacoustic_class("LoudnessZwtv")

    def test_create_operation(self) -> None:
        """Test creating operation via create_operation function."""
        op = create_operation("loudness_zwtv", _SR, field_type="diffuse")
        assert type(op) is _psychoacoustic_class("LoudnessZwtv")
        assert getattr(op, "field_type") == "diffuse"

    def test_mono_signal_shape(self) -> None:
        """Test loudness calculation output shape for mono signal."""
        signal_mono, _, _, _ = _loudness_signal()
        op = LoudnessZwtv(_SR, field_type="free")
        result = op.process_array(signal_mono).compute()

        # Result should be 2D (channels, time_samples)
        assert result.ndim == 2
        assert result.shape[0] == 1  # 1 channel
        # Time samples should be less than input samples (downsampled)
        assert result.shape[1] < signal_mono.shape[1]

    def test_stereo_signal_shape(self) -> None:
        """Test loudness calculation output shape for stereo signal."""
        _, signal_stereo, _, _ = _loudness_signal()
        loudness_op = LoudnessZwtv(_SR, field_type="free")
        result = loudness_op.process_array(signal_stereo).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        n_ch1_direct, _, _, _ = loudness_zwtv(signal_stereo[0], _SR, field_type="free")
        n_ch2_direct, _, _, _ = loudness_zwtv(signal_stereo[1], _SR, field_type="free")

        np.testing.assert_array_equal(result[0], n_ch1_direct)
        np.testing.assert_array_equal(result[1], n_ch2_direct)

    def test_loudness_values_range(self) -> None:
        """Test that loudness values match MoSQITo output."""
        signal_mono, _, _, _ = _loudness_signal()
        op = LoudnessZwtv(_SR, field_type="free")
        result = op.process_array(signal_mono).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        n_direct, _, _, _ = loudness_zwtv(signal_mono[0], _SR, field_type="free")

        # Results should match exactly
        np.testing.assert_array_equal(result[0], n_direct, err_msg="Loudness values differ from MoSQITo output")

    def test_comparison_with_mosqito_direct(self) -> None:
        """
        Test that values match MoSQITo direct calculation.

        This is the key test to ensure our integration is correct by comparing
        the output with direct MoSQITo calculation.
        """
        signal_mono, _, _, _ = _loudness_signal()
        op = LoudnessZwtv(_SR, field_type="free")
        our_result = op.process_array(signal_mono).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        n_direct, _, _, _ = loudness_zwtv(signal_mono[0], _SR, field_type="free")
        np.testing.assert_array_equal(
            our_result[0],
            n_direct,
            err_msg="Loudness values differ from direct MoSQITo calculation",
        )

    def test_free_vs_diffuse_field(self) -> None:
        """Test that free field and diffuse field give different results."""
        signal_mono, _, _, _ = _loudness_signal()
        loudness_free = LoudnessZwtv(_SR, field_type="free")
        result_free = loudness_free.process_array(signal_mono).compute()

        # Calculate with diffuse field
        loudness_diffuse = LoudnessZwtv(_SR, field_type="diffuse")
        result_diffuse = loudness_diffuse.process_array(signal_mono).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        n_free_direct, _, _, _ = loudness_zwtv(signal_mono[0], _SR, field_type="free")
        n_diffuse_direct, _, _, _ = loudness_zwtv(signal_mono[0], _SR, field_type="diffuse")

        np.testing.assert_array_equal(result_free[0], n_free_direct)
        np.testing.assert_array_equal(result_diffuse[0], n_diffuse_direct)

    def test_amplitude_dependency(self) -> None:
        """Test that loudness increases with amplitude."""
        op = LoudnessZwtv(_SR, field_type="free")
        duration = 0.1
        # Create signals with different amplitudes
        t = np.linspace(0, duration, int(_SR * duration))
        freq = 1000.0

        # Low amplitude signal (approx 50 dB SPL)
        signal_low = np.array([0.006 * np.sin(2 * np.pi * freq * t)])

        # High amplitude signal (approx 80 dB SPL)
        signal_high = np.array([0.2 * np.sin(2 * np.pi * freq * t)])

        # Calculate loudness using wandas
        loudness_low = op.process_array(signal_low).compute()
        loudness_high = op.process_array(signal_high).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        n_low_direct, _, _, _ = loudness_zwtv(signal_low[0], _SR, field_type="free")
        n_high_direct, _, _, _ = loudness_zwtv(signal_high[0], _SR, field_type="free")

        np.testing.assert_array_equal(loudness_low[0], n_low_direct)
        np.testing.assert_array_equal(loudness_high[0], n_high_direct)

    def test_silence_produces_low_loudness(self) -> None:
        """Test that silence produces near-zero loudness."""
        op = LoudnessZwtv(_SR, field_type="free")
        duration = 0.1
        # Create silent signal
        silence = np.zeros((1, int(_SR * duration)))

        result = op.process_array(silence).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        n_direct, _, _, _ = loudness_zwtv(silence[0], _SR, field_type="free")

        np.testing.assert_array_equal(result[0], n_direct)

    def test_white_noise_loudness(self) -> None:
        """Test loudness calculation with white noise."""
        op = LoudnessZwtv(_SR, field_type="free")
        duration = 0.1
        # Generate white noise at moderate level
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.02, (1, int(_SR * duration)))

        result = op.process_array(noise).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        n_direct, _, _, _ = loudness_zwtv(noise[0], _SR, field_type="free")

        np.testing.assert_array_equal(result[0], n_direct)

    def test_process_with_dask(self) -> None:
        """Test that process method works with dask arrays."""
        signal_mono, _, _, dask_mono = _loudness_signal()
        op = LoudnessZwtv(_SR, field_type="free")
        result = op.process(dask_mono).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        n_direct, _, _, _ = loudness_zwtv(signal_mono[0], _SR, field_type="free")
        np.testing.assert_array_equal(result[0], n_direct)

    def test_multi_channel_independence(self) -> None:
        """Test that each channel is processed independently."""
        op = LoudnessZwtv(_SR, field_type="free")
        duration = 0.1
        # Create signal with different content in each channel
        t = np.linspace(0, duration, int(_SR * duration))
        signal_ch1 = 0.05 * np.sin(2 * np.pi * 500 * t)  # Lower frequency, lower amplitude
        signal_ch2 = 0.1 * np.sin(2 * np.pi * 2000 * t)  # Higher frequency, higher amplitude

        stereo_signal = np.vstack([signal_ch1, signal_ch2])

        result = op.process_array(stereo_signal).compute()

        # Compare each channel with MoSQITo direct calculation
        n_ch1_direct, _, _, _ = loudness_zwtv(signal_ch1, _SR, field_type="free")
        n_ch2_direct, _, _, _ = loudness_zwtv(signal_ch2, _SR, field_type="free")

        np.testing.assert_array_equal(result[0], n_ch1_direct)
        np.testing.assert_array_equal(result[1], n_ch2_direct)

    def test_calculate_output_shape(self) -> None:
        """Test calculate_output_shape method."""
        op = LoudnessZwtv(_SR, field_type="free")
        input_shape = (1, 48000)  # 1 channel, 1 second at 48kHz
        output_shape = op.calculate_output_shape(input_shape)

        # Output should have same number of channels
        assert output_shape[0] == input_shape[0]
        # Output should have fewer time samples (downsampled)
        assert output_shape[1] < input_shape[1]
        assert output_shape[1] > 0

    def test_1d_input_handling(self) -> None:
        """Test that 1D input is properly handled."""
        op = LoudnessZwtv(_SR, field_type="free")
        duration = 0.1
        # Create 1D signal
        t = np.linspace(0, duration, int(_SR * duration))
        signal_1d = 0.05 * np.sin(2 * np.pi * 1000 * t)

        result = op.process_array(signal_1d).compute()

        # Should be reshaped to 2D with 1 channel
        assert result.ndim == 2
        assert result.shape[0] == 1

    def test_consistency_across_calls(self) -> None:
        """Test that repeated calls with same input produce same output."""
        signal_mono, _, _, _ = _loudness_signal()
        op = LoudnessZwtv(_SR, field_type="free")
        result1 = op.process_array(signal_mono).compute()
        result2 = op.process_array(signal_mono).compute()

        # Results should be identical
        np.testing.assert_array_equal(result1, result2)

    def test_time_resolution(self) -> None:
        """Test that time resolution matches MoSQITo output."""
        signal_mono, _, _, _ = _loudness_signal()
        op = LoudnessZwtv(_SR, field_type="free")
        result = op.process_array(signal_mono).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        n_direct, _, _, _ = loudness_zwtv(signal_mono[0], _SR, field_type="free")

        # Time resolution should match exactly
        assert result.shape[1] == len(n_direct)

    def test_time_axis_values(self) -> None:
        """Test that time axis is correctly calculated based on sampling rate."""
        signal_mono, _, _, _ = _loudness_signal()
        from wandas.frames.channel import ChannelFrame

        # Create frame and calculate loudness
        dask_data = da_from_array(signal_mono, chunks=(1, -1))
        frame = ChannelFrame(data=dask_data, sampling_rate=_SR)
        loudness_frame = frame.loudness_zwtv(field_type="free")

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
                f"Expected {expected_sampling_rate} Hz, got {loudness_frame.sampling_rate}"
            )

    def test_plot_method_exists_and_works(self) -> None:
        """Test that loudness ChannelFrame can be plotted."""
        signal_mono, _, _, _ = _loudness_signal()
        import matplotlib.pyplot as plt

        from wandas.frames.channel import ChannelFrame

        # Create frame and calculate loudness
        dask_data = da_from_array(signal_mono, chunks=(1, -1))
        frame = ChannelFrame(data=dask_data, sampling_rate=_SR)
        loudness_frame = frame.loudness_zwtv(field_type="free")

        # Test plot method exists
        assert hasattr(loudness_frame, "plot")
        assert callable(loudness_frame.plot)

        # Test plot execution without error (use overlay=True for single Axes)
        fig, ax = plt.subplots()
        result_ax = loudness_frame.plot(ax=ax, title="Test Loudness", ylabel="Loudness [sones]", overlay=True)

        assert result_ax is ax
        assert ax.get_ylabel() == "Loudness [sones]"
        assert ax.get_title() == "Test Loudness"

        plt.close(fig)


class TestLoudnessZwtvIntegration:
    """Integration tests for loudness calculation with ChannelFrame."""

    def test_loudness_in_operation_registry(self) -> None:
        """Test that loudness operation is in registry."""
        from wandas.processing.base import _OPERATION_REGISTRY

        assert "loudness_zwtv" in _OPERATION_REGISTRY
        assert _OPERATION_REGISTRY["loudness_zwtv"] is _psychoacoustic_class("LoudnessZwtv")

    def test_channel_frame_loudness_method_exists(self) -> None:
        """Test that ChannelFrame has loudness_zwtv method."""
        duration = 0.1
        from wandas.frames.channel import ChannelFrame

        # Create a simple frame
        t = np.linspace(0, duration, int(_SR * duration))
        signal = np.array([0.05 * np.sin(2 * np.pi * 1000 * t)])
        dask_data = da_from_array(signal, chunks=(1, -1))
        frame = ChannelFrame(data=dask_data, sampling_rate=_SR)

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

    def test_initialization(self) -> None:
        """Test LoudnessZwst initialization with different parameters."""
        # Default initialization
        loudness = LoudnessZwst(_SR)
        assert loudness.sampling_rate == _SR
        assert loudness.field_type == "free"

        # Custom field_type
        loudness_diffuse = LoudnessZwst(_SR, field_type="diffuse")
        assert loudness_diffuse.field_type == "diffuse"

    def test_invalid_field_type(self) -> None:
        """Test that invalid field_type raises ValueError."""
        with pytest.raises(ValueError, match="field_type must be 'free' or 'diffuse'"):
            LoudnessZwst(_SR, field_type="invalid")

    def test_operation_name(self) -> None:
        """Test that operation has correct name."""
        op = LoudnessZwst(_SR, field_type="free")
        assert op.name == "loudness_zwst"

    def test_operation_registration(self) -> None:
        """Test that operation is properly registered."""
        op_class = get_operation("loudness_zwst")
        assert op_class is _psychoacoustic_class("LoudnessZwst")

    def test_create_operation(self) -> None:
        """Test creating operation via create_operation function."""
        op = create_operation("loudness_zwst", _SR, field_type="diffuse")
        assert type(op) is _psychoacoustic_class("LoudnessZwst")
        assert getattr(op, "field_type") == "diffuse"

    def test_mono_signal_shape(self) -> None:
        """Test steady-state loudness calculation output shape for mono signal."""
        signal_mono, _, _, _ = _loudness_signal()
        op = LoudnessZwst(_SR, field_type="free")
        result = op.process_array(signal_mono).compute()

        # Result should be 2D (channels, 1)
        assert result.ndim == 2
        assert result.shape[0] == 1  # 1 channel
        assert result.shape[1] == 1  # Single loudness value

    def test_stereo_signal_shape(self) -> None:
        """Test steady-state loudness calculation output shape for stereo signal."""
        _, signal_stereo, _, _ = _loudness_signal()
        loudness_op = LoudnessZwst(_SR, field_type="free")
        result = loudness_op.process_array(signal_stereo).compute()

        # Result should be 2D (channels, 1)
        assert result.ndim == 2
        assert result.shape[0] == 2  # 2 channels
        assert result.shape[1] == 1  # Single loudness value per channel

        # MoSQITo wrapper — exact match expected (same algorithm)
        n_ch1_direct, _, _ = loudness_zwst(signal_stereo[0], _SR, field_type="free")
        n_ch2_direct, _, _ = loudness_zwst(signal_stereo[1], _SR, field_type="free")

        np.testing.assert_allclose(result[0, 0], n_ch1_direct, rtol=1e-10)
        np.testing.assert_allclose(result[1, 0], n_ch2_direct, rtol=1e-10)

    def test_loudness_values_range(self) -> None:
        """Test that loudness values match MoSQITo output."""
        signal_mono, _, _, _ = _loudness_signal()
        op = LoudnessZwst(_SR, field_type="free")
        result = op.process_array(signal_mono).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        n_direct, _, _ = loudness_zwst(signal_mono[0], _SR, field_type="free")

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
        signal_mono, _, _, _ = _loudness_signal()
        op = LoudnessZwst(_SR, field_type="free")
        our_result = op.process_array(signal_mono).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        n_direct, _, _ = loudness_zwst(signal_mono[0], _SR, field_type="free")
        np.testing.assert_allclose(
            our_result[0, 0],
            n_direct,
            rtol=1e-10,
            err_msg="Loudness values differ from direct MoSQITo calculation",
        )

    def test_free_vs_diffuse_field(self) -> None:
        """Test that free field and diffuse field give different results."""
        signal_mono, _, _, _ = _loudness_signal()
        loudness_free = LoudnessZwst(_SR, field_type="free")
        result_free = loudness_free.process_array(signal_mono).compute()

        # Calculate with diffuse field
        loudness_diffuse = LoudnessZwst(_SR, field_type="diffuse")
        result_diffuse = loudness_diffuse.process_array(signal_mono).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        n_free_direct, _, _ = loudness_zwst(signal_mono[0], _SR, field_type="free")
        n_diffuse_direct, _, _ = loudness_zwst(signal_mono[0], _SR, field_type="diffuse")

        np.testing.assert_allclose(result_free[0, 0], n_free_direct, rtol=1e-10)
        np.testing.assert_allclose(result_diffuse[0, 0], n_diffuse_direct, rtol=1e-10)

    def test_amplitude_dependency(self) -> None:
        """Test that loudness increases with amplitude."""
        op = LoudnessZwst(_SR, field_type="free")
        duration = 0.1
        # Create signals with different amplitudes
        t = np.linspace(0, duration, int(_SR * duration))
        freq = 1000.0

        # Low amplitude signal (approx 50 dB SPL)
        signal_low = np.array([0.006 * np.sin(2 * np.pi * freq * t)])

        # High amplitude signal (approx 80 dB SPL)
        signal_high = np.array([0.2 * np.sin(2 * np.pi * freq * t)])

        # Calculate loudness using wandas
        loudness_low = op.process_array(signal_low).compute()
        loudness_high = op.process_array(signal_high).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        n_low_direct, _, _ = loudness_zwst(signal_low[0], _SR, field_type="free")
        n_high_direct, _, _ = loudness_zwst(signal_high[0], _SR, field_type="free")

        np.testing.assert_allclose(loudness_low[0, 0], n_low_direct, rtol=1e-10)
        np.testing.assert_allclose(loudness_high[0, 0], n_high_direct, rtol=1e-10)

        # Higher amplitude should produce higher loudness
        assert loudness_high[0, 0] > loudness_low[0, 0]

    def test_silence_produces_low_loudness(self) -> None:
        """Test that silence produces near-zero loudness."""
        op = LoudnessZwst(_SR, field_type="free")
        duration = 0.1
        # Create silent signal
        silence = np.zeros((1, int(_SR * duration)))

        result = op.process_array(silence).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        n_direct, _, _ = loudness_zwst(silence[0], _SR, field_type="free")

        np.testing.assert_allclose(result[0, 0], n_direct, rtol=1e-10)

    def test_white_noise_loudness(self) -> None:
        """Test loudness calculation with white noise."""
        op = LoudnessZwst(_SR, field_type="free")
        duration = 0.1
        # Generate white noise at moderate level
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.02, (1, int(_SR * duration)))

        result = op.process_array(noise).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        n_direct, _, _ = loudness_zwst(noise[0], _SR, field_type="free")

        np.testing.assert_allclose(result[0, 0], n_direct, rtol=1e-10)

    def test_process_with_dask(self) -> None:
        """Test that process method works with dask arrays."""
        signal_mono, _, _, dask_mono = _loudness_signal()
        op = LoudnessZwst(_SR, field_type="free")
        result = op.process(dask_mono).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        n_direct, _, _ = loudness_zwst(signal_mono[0], _SR, field_type="free")
        np.testing.assert_allclose(result[0, 0], n_direct, rtol=1e-10)

    def test_multi_channel_independence(self) -> None:
        """Test that each channel is processed independently."""
        op = LoudnessZwst(_SR, field_type="free")
        duration = 0.1
        # Create signal with different content in each channel
        t = np.linspace(0, duration, int(_SR * duration))
        signal_ch1 = 0.05 * np.sin(2 * np.pi * 500 * t)  # Lower frequency, lower amplitude
        signal_ch2 = 0.1 * np.sin(2 * np.pi * 2000 * t)  # Higher frequency, higher amplitude

        stereo_signal = np.vstack([signal_ch1, signal_ch2])

        result = op.process_array(stereo_signal).compute()

        # Compare each channel with MoSQITo direct calculation
        n_ch1_direct, _, _ = loudness_zwst(signal_ch1, _SR, field_type="free")
        n_ch2_direct, _, _ = loudness_zwst(signal_ch2, _SR, field_type="free")

        np.testing.assert_allclose(result[0, 0], n_ch1_direct, rtol=1e-10)
        np.testing.assert_allclose(result[1, 0], n_ch2_direct, rtol=1e-10)

    def test_calculate_output_shape(self) -> None:
        """Test calculate_output_shape method."""
        op = LoudnessZwst(_SR, field_type="free")
        input_shape = (1, 48000)  # 1 channel, 1 second at 48kHz
        output_shape = op.calculate_output_shape(input_shape)

        # Output should be (channels, 1)
        assert output_shape[0] == input_shape[0]
        assert output_shape[1] == 1

        # Test with stereo
        input_shape_stereo = (2, 48000)
        output_shape_stereo = op.calculate_output_shape(input_shape_stereo)
        assert output_shape_stereo[0] == 2
        assert output_shape_stereo[1] == 1

    def test_1d_input_handling(self) -> None:
        """Test that 1D input is properly handled."""
        op = LoudnessZwst(_SR, field_type="free")
        duration = 0.1
        # Create 1D signal
        t = np.linspace(0, duration, int(_SR * duration))
        signal_1d = 0.05 * np.sin(2 * np.pi * 1000 * t)

        result = op.process_array(signal_1d).compute()

        # Should be reshaped to 2D with 1 channel
        assert result.ndim == 2
        assert result.shape[0] == 1
        assert result.shape[1] == 1

    def test_consistency_across_calls(self) -> None:
        """Test that repeated calls with same input produce same output."""
        signal_mono, _, _, _ = _loudness_signal()
        op = LoudnessZwst(_SR, field_type="free")
        result1 = op.process_array(signal_mono).compute()
        result2 = op.process_array(signal_mono).compute()

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

    def test_loudness_in_operation_registry(self) -> None:
        """Test that loudness operation is in registry."""
        from wandas.processing.base import _OPERATION_REGISTRY

        assert "loudness_zwst" in _OPERATION_REGISTRY
        assert _OPERATION_REGISTRY["loudness_zwst"] is _psychoacoustic_class("LoudnessZwst")

    def test_channel_frame_loudness_method_exists(self) -> None:
        """Test that ChannelFrame has loudness_zwst method."""
        duration = 0.1
        from wandas.frames.channel import ChannelFrame

        # Create a simple frame
        t = np.linspace(0, duration, int(_SR * duration))
        signal = np.array([0.05 * np.sin(2 * np.pi * 1000 * t)])
        dask_data = da_from_array(signal, chunks=(1, -1))
        frame = ChannelFrame(data=dask_data, sampling_rate=_SR)

        # Check method exists
        assert hasattr(frame, "loudness_zwst")
        assert callable(frame.loudness_zwst)

    def test_channel_frame_loudness_returns_ndarray(self) -> None:
        """Test that ChannelFrame.loudness_zwst() returns NDArrayReal."""
        duration = 0.1
        from wandas.frames.channel import ChannelFrame

        # Create mono frame
        t = np.linspace(0, duration, int(_SR * duration))
        signal_mono = np.array([0.05 * np.sin(2 * np.pi * 1000 * t)])
        dask_data_mono = da_from_array(signal_mono, chunks=(1, -1))
        frame_mono = ChannelFrame(data=dask_data_mono, sampling_rate=_SR)

        # Calculate loudness
        loudness_mono = frame_mono.loudness_zwst(field_type="free")

        # Should be NDArrayReal (1D array)
        assert isinstance(loudness_mono, np.ndarray)
        assert loudness_mono.ndim == 1
        assert loudness_mono.shape[0] == 1  # One value per channel
        assert isinstance(loudness_mono[0], float | np.floating)

        # Create stereo frame
        signal_stereo = np.vstack([signal_mono[0], signal_mono[0] * 0.5])
        dask_data_stereo = da_from_array(signal_stereo, chunks=(1, -1))
        frame_stereo = ChannelFrame(data=dask_data_stereo, sampling_rate=_SR)

        # Calculate loudness for stereo
        loudness_stereo = frame_stereo.loudness_zwst(field_type="free")

        # Should be 1D array with 2 values
        assert isinstance(loudness_stereo, np.ndarray)
        assert loudness_stereo.ndim == 1
        assert loudness_stereo.shape[0] == 2  # Two values (one per channel)

        # Can access values directly without double indexing
        assert isinstance(loudness_stereo[0], float | np.floating)
        assert isinstance(loudness_stereo[1], float | np.floating)

        # Can use numpy operations directly
        mean_loudness = loudness_stereo.mean()
        assert isinstance(mean_loudness, float | np.floating)

    def test_channel_frame_loudness_matches_mosqito(self) -> None:
        """Test that ChannelFrame.loudness_zwst() matches direct MoSQITo call."""
        duration = 0.1
        from wandas.frames.channel import ChannelFrame

        # Create test signal
        t = np.linspace(0, duration, int(_SR * duration))
        signal = np.array([0.05 * np.sin(2 * np.pi * 1000 * t)])
        dask_data = da_from_array(signal, chunks=(1, -1))
        frame = ChannelFrame(data=dask_data, sampling_rate=_SR)

        # Calculate using wandas
        loudness_wandas = frame.loudness_zwst(field_type="free")

        # MoSQITo wrapper — exact match expected (same algorithm)
        n_direct, _, _ = loudness_zwst(signal[0], _SR, field_type="free")

        # Results should match
        np.testing.assert_allclose(loudness_wandas[0], n_direct, rtol=1e-10)


class TestRoughnessDw:
    """Test suite for RoughnessDw operation."""

    def test_initialization(self) -> None:
        """Test RoughnessDw initialization with different parameters."""
        # Default initialization
        roughness = RoughnessDw(_SR)
        assert roughness.sampling_rate == _SR
        assert roughness.overlap == 0.5

        # Custom overlap
        roughness_custom = RoughnessDw(_SR, overlap=0.0)
        assert roughness_custom.overlap == 0.0

    def test_invalid_overlap(self) -> None:
        """Test that invalid overlap raises ValueError."""
        with pytest.raises(ValueError, match="overlap must be in"):
            RoughnessDw(_SR, overlap=1.5)

        with pytest.raises(ValueError, match="overlap must be in"):
            RoughnessDw(_SR, overlap=-0.1)

    def test_operation_name(self) -> None:
        """Test that operation has correct name."""
        op = RoughnessDw(_SR, overlap=0.5)
        assert op.name == "roughness_dw"

    def test_operation_registration(self) -> None:
        """Test that operation is properly registered."""
        op_class = get_operation("roughness_dw")
        assert op_class is _psychoacoustic_class("RoughnessDw")

    def test_create_operation(self) -> None:
        """Test creating operation via create_operation function."""
        op = create_operation("roughness_dw", _SR, overlap=0.75)
        assert type(op) is _psychoacoustic_class("RoughnessDw")
        assert getattr(op, "overlap") == 0.75

    def test_mono_signal_shape(self) -> None:
        """Test roughness calculation output shape for mono signal."""
        signal_mono, _, _, _ = _roughness_signal()
        op = RoughnessDw(_SR, overlap=0.5)
        result = op.process_array(signal_mono).compute()

        # Result should be 2D (channels, time_samples)
        assert result.ndim == 2
        assert result.shape[0] == 1  # 1 channel
        # Time samples should be less than input samples (windowed processing)
        assert result.shape[1] < signal_mono.shape[1]
        assert result.shape[1] > 0

    def test_stereo_signal_shape(self) -> None:
        """Test roughness calculation output shape for stereo signal."""
        _, signal_stereo, _, _ = _roughness_signal()
        op = RoughnessDw(_SR, overlap=0.5)
        result = op.process_array(signal_stereo).compute()

        # Result should be 2D (channels, time_samples)
        assert result.ndim == 2
        assert result.shape[0] == 2  # 2 channels
        assert result.shape[1] > 0

    def test_comparison_with_mosqito_direct(self) -> None:
        """Test that values match MoSQITo direct calculation."""
        signal_mono, _, _, _ = _roughness_signal()
        op = RoughnessDw(_SR, overlap=0.5)
        overlap = 0.5
        our_result = op.process_array(signal_mono).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        r_direct, _, _, _ = roughness_dw_mosqito(signal_mono[0], _SR, overlap=overlap)

        np.testing.assert_array_equal(
            our_result[0],
            r_direct,
            err_msg="Roughness values differ from direct MoSQITo calculation",
        )

    def test_stereo_matches_mosqito(self) -> None:
        """Test stereo signal matches MoSQITo for each channel."""
        _, signal_stereo, _, _ = _roughness_signal()
        op = RoughnessDw(_SR, overlap=0.5)
        overlap = 0.5
        result = op.process_array(signal_stereo).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        r_ch1_direct, _, _, _ = roughness_dw_mosqito(signal_stereo[0], _SR, overlap=overlap)
        r_ch2_direct, _, _, _ = roughness_dw_mosqito(signal_stereo[1], _SR, overlap=overlap)

        np.testing.assert_array_equal(result[0], r_ch1_direct)
        np.testing.assert_array_equal(result[1], r_ch2_direct)

    def test_overlap_affects_time_resolution(self) -> None:
        """Test that overlap affects output time resolution."""
        signal_mono, _, _, _ = _roughness_signal()
        roughness_overlap0 = RoughnessDw(_SR, overlap=0.0)
        roughness_overlap05 = RoughnessDw(_SR, overlap=0.5)

        result_overlap0 = roughness_overlap0.process_array(signal_mono).compute()
        result_overlap05 = roughness_overlap05.process_array(signal_mono).compute()

        # Higher overlap should give more time points
        # (overlap=0.5 has 100ms hop, overlap=0.0 has 200ms hop)
        # So overlap=0.5 should have approximately 2x more points
        assert result_overlap05.shape[1] > result_overlap0.shape[1]

    def test_roughness_values_range(self) -> None:
        """Test that roughness values are in reasonable range."""
        signal_mono, _, _, _ = _roughness_signal()
        op = RoughnessDw(_SR, overlap=0.5)
        overlap = 0.5
        result = op.process_array(signal_mono).compute()

        # Roughness values should be non-negative
        assert np.all(result >= 0)

        # For our AM signal, roughness should be detectable (> 0)
        assert np.max(result) > 0

        # MoSQITo wrapper — exact match expected (same algorithm)
        r_direct, _, _, _ = roughness_dw_mosqito(signal_mono[0], _SR, overlap=overlap)

        # Verify it matches MoSQITo
        np.testing.assert_array_equal(result[0], r_direct)

    def test_silence_produces_low_roughness(self) -> None:
        """Test that silence produces near-zero roughness."""
        op = RoughnessDw(_SR, overlap=0.5)
        duration = 0.5
        overlap = 0.5
        silence = np.zeros((1, int(_SR * duration)))

        result = op.process_array(silence).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        r_direct, _, _, _ = roughness_dw_mosqito(silence[0], _SR, overlap=overlap)

        np.testing.assert_array_equal(result[0], r_direct)

        # Roughness should be very low for silence
        assert np.all(result < 0.01)

    def test_process_with_dask(self) -> None:
        """Test that process method works with dask arrays."""
        signal_mono, _, _, dask_mono = _roughness_signal()
        op = RoughnessDw(_SR, overlap=0.5)
        overlap = 0.5
        result = op.process(dask_mono).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        r_direct, _, _, _ = roughness_dw_mosqito(signal_mono[0], _SR, overlap=overlap)
        np.testing.assert_array_equal(result[0], r_direct)

    def test_multi_channel_independence(self) -> None:
        """Test that each channel is processed independently."""
        _, signal_stereo, _, _ = _roughness_signal()
        op = RoughnessDw(_SR, overlap=0.5)
        overlap = 0.5
        result = op.process_array(signal_stereo).compute()

        # Compare each channel with MoSQITo direct calculation
        r_ch1_direct, _, _, _ = roughness_dw_mosqito(signal_stereo[0], _SR, overlap=overlap)
        r_ch2_direct, _, _, _ = roughness_dw_mosqito(signal_stereo[1], _SR, overlap=overlap)

        np.testing.assert_array_equal(result[0], r_ch1_direct)
        np.testing.assert_array_equal(result[1], r_ch2_direct)

    def test_1d_input_handling(self) -> None:
        """Test that 1D input is properly handled."""
        op = RoughnessDw(_SR, overlap=0.5)
        duration = 0.5
        # Create 1D signal
        t = np.linspace(0, duration, int(_SR * duration))
        signal_1d = 0.1 * np.sin(2 * np.pi * 1000 * t)

        result = op.process_array(signal_1d).compute()

        # Should be reshaped to 2D with 1 channel
        assert result.ndim == 2
        assert result.shape[0] == 1

    def test_consistency_across_calls(self) -> None:
        """Test that repeated calls with same input produce same output."""
        signal_mono, _, _, _ = _roughness_signal()
        op = RoughnessDw(_SR, overlap=0.5)
        result1 = op.process_array(signal_mono).compute()
        result2 = op.process_array(signal_mono).compute()

        # Results should be identical
        np.testing.assert_array_equal(result1, result2)

    def test_sampling_rate_metadata(self) -> None:
        """Test that output sampling rate is correctly calculated."""
        op = RoughnessDw(_SR, overlap=0.5)
        overlap = 0.5
        # overlap=0.5, window=200ms, hop=100ms → fs≈10Hz
        metadata_updates = op.get_metadata_updates()

        assert "sampling_rate" in metadata_updates
        expected_sr = 1.0 / (0.2 * (1 - overlap))
        assert metadata_updates["sampling_rate"] == pytest.approx(expected_sr)

        # Test with overlap=0.0 → fs≈5Hz
        roughness_overlap0 = RoughnessDw(_SR, overlap=0.0)
        metadata_overlap0 = roughness_overlap0.get_metadata_updates()
        expected_sr_overlap0 = 1.0 / 0.2  # 5 Hz
        assert metadata_overlap0["sampling_rate"] == pytest.approx(expected_sr_overlap0)

    def test_calculate_output_shape(self) -> None:
        """Test calculate_output_shape method."""
        op = RoughnessDw(_SR, overlap=0.5)
        duration = 0.5
        input_shape = (1, int(_SR * duration))
        output_shape = op.calculate_output_shape(input_shape)

        # Output should have same number of channels
        assert output_shape[0] == input_shape[0]
        # Output should have fewer time samples
        assert output_shape[1] < input_shape[1]
        assert output_shape[1] > 0

    def test_time_axis_values(self) -> None:
        """Test that time axis is correctly calculated based on sampling rate."""
        signal_mono, _, _, _ = _roughness_signal()
        overlap = 0.5
        from wandas.frames.channel import ChannelFrame

        # Create frame and calculate roughness
        dask_data = da_from_array(signal_mono, chunks=(1, -1))
        frame = ChannelFrame(data=dask_data, sampling_rate=_SR)
        roughness_frame = frame.roughness_dw(overlap=overlap)

        # wandas time axis should be evenly spaced based on output sampling rate
        time_wandas = roughness_frame.time

        # RoughnessDw with overlap=0.5 should output at ~10 Hz
        # (200ms window, 100ms hop -> 1/0.1 = 10 Hz)
        expected_sampling_rate = 1.0 / (0.2 * (1 - overlap))

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
            assert roughness_frame.sampling_rate == pytest.approx(expected_sampling_rate), (
                f"Expected {expected_sampling_rate} Hz, got {roughness_frame.sampling_rate}"
            )


class TestRoughnessDwIntegration:
    """Integration tests for roughness calculation with ChannelFrame."""

    def test_roughness_in_operation_registry(self) -> None:
        """Test that roughness operation is in registry."""
        from wandas.processing.base import _OPERATION_REGISTRY

        assert "roughness_dw" in _OPERATION_REGISTRY
        assert _OPERATION_REGISTRY["roughness_dw"] is _psychoacoustic_class("RoughnessDw")

    def test_channel_frame_roughness_method_exists(self) -> None:
        """Test that ChannelFrame has roughness_dw method."""
        duration = 0.5
        from wandas.frames.channel import ChannelFrame

        # Create a simple frame
        t = np.linspace(0, duration, int(_SR * duration))
        signal = np.array([0.1 * np.sin(2 * np.pi * 1000 * t)])
        dask_data = da_from_array(signal, chunks=(1, -1))
        frame = ChannelFrame(data=dask_data, sampling_rate=_SR)

        # Check method exists
        assert hasattr(frame, "roughness_dw")
        assert callable(frame.roughness_dw)

    def test_channel_frame_roughness_returns_channelframe(self) -> None:
        """Test that ChannelFrame.roughness_dw() returns ChannelFrame."""
        duration = 0.5
        from wandas.frames.channel import ChannelFrame

        # Create AM modulated signal
        t = np.linspace(0, duration, int(_SR * duration))
        carrier = np.sin(2 * np.pi * 1000 * t)
        modulation = 1 + np.sin(2 * np.pi * 70 * t)
        signal = np.array([0.1 * carrier * modulation])
        dask_data = da_from_array(signal, chunks=(1, -1))
        frame = ChannelFrame(data=dask_data, sampling_rate=_SR)

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
        duration = 0.5
        from wandas.frames.channel import ChannelFrame

        # Create AM modulated signal
        t = np.linspace(0, duration, int(_SR * duration))
        carrier = np.sin(2 * np.pi * 1000 * t)
        modulation = 1 + np.sin(2 * np.pi * 70 * t)
        signal = np.array([0.1 * carrier * modulation])
        dask_data = da_from_array(signal, chunks=(1, -1))
        frame = ChannelFrame(data=dask_data, sampling_rate=_SR)

        # Calculate using wandas
        roughness_frame = frame.roughness_dw(overlap=0.5)

        # Get data (may be Dask or NumPy)
        roughness_wandas_data = (
            roughness_frame._data.compute() if hasattr(roughness_frame._data, "compute") else roughness_frame._data
        )

        # MoSQITo wrapper — exact match expected (same algorithm)
        r_direct, _, _, _ = roughness_dw_mosqito(signal[0], _SR, overlap=0.5)

        # Results should match (roughness_wandas_data is 2D: (n_channels, n_time))
        np.testing.assert_array_equal(roughness_wandas_data[0], r_direct)


class TestRoughnessDwSpec:
    """Test suite for RoughnessDwSpec operation."""

    def test_initialization(self) -> None:
        """Test RoughnessDwSpec initialization with different parameters."""
        from wandas.processing.psychoacoustic import RoughnessDwSpec

        # Default initialization
        roughness_spec = RoughnessDwSpec(_SR)
        assert roughness_spec.sampling_rate == _SR
        assert roughness_spec.overlap == 0.5

        # Custom overlap
        roughness_spec_custom = RoughnessDwSpec(_SR, overlap=0.0)
        assert roughness_spec_custom.overlap == 0.0

    def test_invalid_overlap(self) -> None:
        """Test that invalid overlap raises ValueError."""
        from wandas.processing.psychoacoustic import RoughnessDwSpec

        with pytest.raises(ValueError, match="overlap must be in"):
            RoughnessDwSpec(_SR, overlap=1.5)

        with pytest.raises(ValueError, match="overlap must be in"):
            RoughnessDwSpec(_SR, overlap=-0.1)

    def test_operation_name(self) -> None:
        """Test that operation has correct name."""
        op = RoughnessDwSpec(_SR, overlap=0.5)
        assert op.name == "roughness_dw_spec"

    def test_operation_registration(self) -> None:
        """Test that operation is properly registered."""
        op_class = get_operation("roughness_dw_spec")
        assert op_class is _psychoacoustic_class("RoughnessDwSpec")

    def test_create_operation(self) -> None:
        """Test creating operation via create_operation function."""
        op = create_operation("roughness_dw_spec", _SR, overlap=0.75)
        assert type(op) is _psychoacoustic_class("RoughnessDwSpec")
        assert getattr(op, "overlap") == 0.75

    def test_mono_signal_shape(self) -> None:
        """Test roughness_spec calculation output shape for mono signal."""
        signal_mono, _, _, _ = _roughness_spec_signal()
        op = RoughnessDwSpec(_SR, overlap=0.5)
        result = op.process_array(signal_mono).compute()

        # Result should be 2D (n_bark_bands, time_samples) for mono
        assert result.ndim == 2
        assert result.shape[0] == 47  # 47 Bark bands
        assert result.shape[1] > 0  # Time samples

    def test_stereo_signal_shape(self) -> None:
        """Test roughness_spec calculation output shape for stereo signal."""
        _, signal_stereo, _, _ = _roughness_spec_signal()
        op = RoughnessDwSpec(_SR, overlap=0.5)
        result = op.process_array(signal_stereo).compute()

        # Result should be 3D (n_channels, n_bark_bands, time_samples) for stereo
        assert result.ndim == 3
        assert result.shape[0] == 2  # 2 channels
        assert result.shape[1] == 47  # 47 Bark bands
        assert result.shape[2] > 0  # Time samples

    def test_comparison_with_mosqito_direct_mono(self) -> None:
        """Test that specific roughness values match MoSQITo direct calculation."""
        signal_mono, _, _, _ = _roughness_spec_signal()
        op = RoughnessDwSpec(_SR, overlap=0.5)
        overlap = 0.5
        our_result = op.process_array(signal_mono).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        _, r_spec_direct, _, _ = roughness_dw_mosqito(signal_mono[0], _SR, overlap=overlap)

        np.testing.assert_array_equal(
            our_result,
            r_spec_direct,
            err_msg="Specific roughness values differ from direct MoSQITo calculation",
        )

    def test_comparison_with_mosqito_direct_stereo(self) -> None:
        """Test that stereo specific roughness matches MoSQITo for each channel."""
        _, signal_stereo, _, _ = _roughness_spec_signal()
        op = RoughnessDwSpec(_SR, overlap=0.5)
        overlap = 0.5
        result = op.process_array(signal_stereo).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        _, r_spec_ch1_direct, _, _ = roughness_dw_mosqito(signal_stereo[0], _SR, overlap=overlap)
        _, r_spec_ch2_direct, _, _ = roughness_dw_mosqito(signal_stereo[1], _SR, overlap=overlap)

        np.testing.assert_array_equal(result[0], r_spec_ch1_direct)
        np.testing.assert_array_equal(result[1], r_spec_ch2_direct)

    def test_bark_axis_values(self) -> None:
        """Test that bark_axis has correct values."""
        op = RoughnessDwSpec(_SR, overlap=0.5)
        bark_axis = op.bark_axis

        # Should have 47 Bark bands from 0.5 to 23.5
        assert len(bark_axis) == 47
        assert bark_axis[0] == pytest.approx(0.5)
        assert bark_axis[-1] == pytest.approx(23.5)

    def test_bark_axis_matches_mosqito(self) -> None:
        """Test that bark_axis matches MoSQITo bark_axis exactly."""
        signal_mono, _, _, _ = _roughness_spec_signal()
        op = RoughnessDwSpec(_SR, overlap=0.5)
        overlap = 0.5
        bark_axis_wandas = op.bark_axis

        # MoSQITo wrapper — exact match expected (same bark axis)
        _, _, bark_axis_mosqito, _ = roughness_dw_mosqito(signal_mono[0], _SR, overlap=overlap)

        # Results should match exactly
        np.testing.assert_array_equal(
            bark_axis_wandas,
            bark_axis_mosqito,
            err_msg="Bark axis values differ from MoSQITo bark axis",
        )

    def test_integration_with_total_roughness(self) -> None:
        """Test that integrating specific roughness gives total roughness."""
        signal_mono, _, _, _ = _roughness_spec_signal()
        op = RoughnessDwSpec(_SR, overlap=0.5)
        overlap = 0.5
        # Calculate specific roughness
        r_spec = op.process_array(signal_mono).compute()

        # Calculate total roughness directly
        r_total, _, _, _ = roughness_dw_mosqito(signal_mono[0], _SR, overlap=overlap)

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
        op = RoughnessDwSpec(_SR, overlap=0.5)
        overlap = 0.5
        metadata_updates = op.get_metadata_updates()

        assert "sampling_rate" in metadata_updates
        assert "bark_axis" in metadata_updates

        # Check sampling rate is correct
        expected_sr = 1.0 / (0.2 * (1 - overlap))
        assert metadata_updates["sampling_rate"] == pytest.approx(expected_sr)

        # Check bark_axis
        bark_axis = metadata_updates["bark_axis"]
        assert len(bark_axis) == 47
        assert bark_axis[0] == pytest.approx(0.5)
        assert bark_axis[-1] == pytest.approx(23.5)

    def test_silence_produces_low_specific_roughness(self) -> None:
        """Test that silence produces near-zero specific roughness."""
        op = RoughnessDwSpec(_SR, overlap=0.5)
        duration = 0.5
        overlap = 0.5
        silence = np.zeros((1, int(_SR * duration)))

        result = op.process_array(silence).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        _, r_spec_direct, _, _ = roughness_dw_mosqito(silence[0], _SR, overlap=overlap)

        np.testing.assert_array_equal(result, r_spec_direct)

        # Specific roughness should be very low for silence
        assert np.all(result < 0.01)

    def test_process_with_dask(self) -> None:
        """Test that process method works with dask arrays."""
        signal_mono, _, dask_mono, _ = _roughness_spec_signal()
        op = RoughnessDwSpec(_SR, overlap=0.5)
        overlap = 0.5
        result = op.process(dask_mono).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        _, r_spec_direct, _, _ = roughness_dw_mosqito(signal_mono[0], _SR, overlap=overlap)
        np.testing.assert_array_equal(result, r_spec_direct)

    def test_1d_input_handling(self) -> None:
        """Test that 1D input is properly handled."""
        op = RoughnessDwSpec(_SR, overlap=0.5)
        duration = 0.5
        # Create 1D signal
        t = np.linspace(0, duration, int(_SR * duration))
        signal_1d = 0.1 * np.sin(2 * np.pi * 1000 * t)

        result = op.process_array(signal_1d).compute()

        # Should be reshaped to 2D with shape (n_bark_bands, n_time)
        assert result.ndim == 2
        assert result.shape[0] == 47

    def test_consistency_across_calls(self) -> None:
        """Test that repeated calls with same input produce same output."""
        signal_mono, _, _, _ = _roughness_spec_signal()
        op = RoughnessDwSpec(_SR, overlap=0.5)
        result1 = op.process_array(signal_mono).compute()
        result2 = op.process_array(signal_mono).compute()

        # Results should be identical
        np.testing.assert_array_equal(result1, result2)

    def test_bark_axis_none_guard(self) -> None:
        """Test that _bark_axis is populated from MoSQITo when it is None."""
        signal_mono, _, _, _ = _roughness_spec_signal()
        op = RoughnessDwSpec(_SR, overlap=0.5)
        # Force _bark_axis to None to exercise the guard in _process_array
        op._bark_axis = None  # ty: ignore[invalid-assignment]
        result = op._process_array(signal_mono)
        # After processing, _bark_axis should have been set from MoSQITo
        assert op._bark_axis is not None
        assert result.shape[0] == 47  # 47 Bark bands (mono → (n_bark, n_time))

    def test_bark_axis_caching(self) -> None:
        """Test that bark_axis is cached to avoid redundant MoSQITo calls."""
        overlap = 0.5

        # Clear the cache to start fresh
        RoughnessDwSpec._bark_axis_cache.clear()

        # First instantiation should compute bark_axis
        op1 = RoughnessDwSpec(_SR, overlap=overlap)
        bark_axis_1 = op1.bark_axis

        # Cache should now contain one entry
        assert len(RoughnessDwSpec._bark_axis_cache) == 1
        cache_key = (_SR, overlap)
        assert cache_key in RoughnessDwSpec._bark_axis_cache

        # Second instantiation with same parameters should use cached value
        op2 = RoughnessDwSpec(_SR, overlap=overlap)
        bark_axis_2 = op2.bark_axis

        # Cache should still have one entry (not recomputed)
        assert len(RoughnessDwSpec._bark_axis_cache) == 1

        # Both instances should have identical bark_axis (same array reference)
        np.testing.assert_array_equal(bark_axis_1, bark_axis_2)
        assert bark_axis_1 is RoughnessDwSpec._bark_axis_cache[cache_key]
        assert bark_axis_2 is RoughnessDwSpec._bark_axis_cache[cache_key]

        # Different parameters should trigger new computation
        op3 = RoughnessDwSpec(_SR, overlap=0.0)
        bark_axis_3 = op3.bark_axis

        # Cache should now have two entries
        assert len(RoughnessDwSpec._bark_axis_cache) == 2
        cache_key_2 = (_SR, 0.0)
        assert cache_key_2 in RoughnessDwSpec._bark_axis_cache

        # Bark axis should be the same (MoSQITo returns same bark_axis
        # regardless of overlap)
        np.testing.assert_array_equal(bark_axis_1, bark_axis_3)

        # Clean up: clear cache after test
        RoughnessDwSpec._bark_axis_cache.clear()


class TestRoughnessDwSpecIntegration:
    """Integration tests for specific roughness calculation with ChannelFrame."""

    def test_roughness_spec_in_operation_registry(self) -> None:
        """Test that roughness_dw_spec operation is in registry."""
        from wandas.processing.base import _OPERATION_REGISTRY

        assert "roughness_dw_spec" in _OPERATION_REGISTRY
        assert _OPERATION_REGISTRY["roughness_dw_spec"] is _psychoacoustic_class("RoughnessDwSpec")

    def test_channel_frame_roughness_spec_method_exists(self) -> None:
        """Test that ChannelFrame has roughness_dw_spec method."""
        duration = 0.5
        from wandas.frames.channel import ChannelFrame

        # Create a simple frame
        t = np.linspace(0, duration, int(_SR * duration))
        signal = np.array([0.1 * np.sin(2 * np.pi * 1000 * t)])
        dask_data = da_from_array(signal, chunks=(1, -1))
        frame = ChannelFrame(data=dask_data, sampling_rate=_SR)

        # Check method exists
        assert hasattr(frame, "roughness_dw_spec")
        assert callable(frame.roughness_dw_spec)

    def test_channel_frame_roughness_spec_returns_roughness_frame(self) -> None:
        """Test that ChannelFrame.roughness_dw_spec() returns RoughnessFrame."""
        duration = 0.5
        from wandas.frames.channel import ChannelFrame
        from wandas.frames.roughness import RoughnessFrame

        # Create AM modulated signal
        t = np.linspace(0, duration, int(_SR * duration))
        carrier = np.sin(2 * np.pi * 1000 * t)
        modulation = 1 + np.sin(2 * np.pi * 70 * t)
        signal = np.array([0.1 * carrier * modulation])
        dask_data = da_from_array(signal, chunks=(1, -1))
        frame = ChannelFrame(data=dask_data, sampling_rate=_SR)

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
        duration = 0.5
        from wandas.frames.channel import ChannelFrame

        # Create AM modulated signal
        t = np.linspace(0, duration, int(_SR * duration))
        carrier = np.sin(2 * np.pi * 1000 * t)
        modulation = 1 + np.sin(2 * np.pi * 70 * t)
        signal = np.array([0.1 * carrier * modulation])
        dask_data = da_from_array(signal, chunks=(1, -1))
        frame = ChannelFrame(data=dask_data, sampling_rate=_SR)

        # Calculate using wandas
        roughness_spec_frame = frame.roughness_dw_spec(overlap=0.5)

        # Get data (may be Dask or NumPy)
        roughness_spec_wandas_data = (
            roughness_spec_frame._data.compute()
            if hasattr(roughness_spec_frame._data, "compute")
            else roughness_spec_frame._data
        )

        # MoSQITo wrapper — exact match expected (same algorithm)
        _, r_spec_direct, _, _ = roughness_dw_mosqito(signal[0], _SR, overlap=0.5)

        # Results should match
        # For mono signal, wandas returns (n_bark, n_time)
        np.testing.assert_array_equal(roughness_spec_wandas_data, r_spec_direct)


class TestSharpnessDin:
    """Test suite for SharpnessDin operation."""

    def test_initialization(self) -> None:
        """Test SharpnessDin initialization."""
        # Default initialization
        sharpness = SharpnessDin(_SR)
        assert sharpness.sampling_rate == _SR

    def test_invalid_weighting(self) -> None:
        """Test that invalid weighting raises ValueError."""
        with pytest.raises(ValueError, match="Invalid weighting function"):
            SharpnessDin(_SR, weighting="invalid")

    def test_invalid_field_type(self) -> None:
        """Test that invalid field_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid field type"):
            SharpnessDin(_SR, field_type="invalid")

    def test_operation_name(self) -> None:
        """Test that operation has correct name."""
        op = SharpnessDin(_SR)
        assert op.name == "sharpness_din"

    def test_operation_registration(self) -> None:
        """Test that operation is properly registered."""
        op_class = get_operation("sharpness_din")
        assert op_class is _psychoacoustic_class("SharpnessDin")

    def test_create_operation(self) -> None:
        """Test creating operation via create_operation function."""
        op = create_operation("sharpness_din", _SR)
        assert type(op) is _psychoacoustic_class("SharpnessDin")

    def test_mono_signal_shape(self) -> None:
        """Test sharpness calculation output shape for mono signal."""
        signal_mono, _, _, _ = _sharpness_signal()
        op = SharpnessDin(_SR)
        result = op.process_array(signal_mono).compute()

        # Result should be 2D (channels, time_samples)
        assert result.ndim == 2
        assert result.shape[0] == 1  # 1 channel

    def test_stereo_signal_shape(self) -> None:
        """Test sharpness calculation output shape for stereo signal."""
        _, signal_stereo, _, _ = _sharpness_signal()
        op = SharpnessDin(_SR)
        result = op.process_array(signal_stereo).compute()

        # Result should be 2D (channels, time_samples)
        assert result.ndim == 2
        assert result.shape[0] == 2  # 2 channels
        assert result.shape[1] > 0

    def test_comparison_with_mosqito_direct(self) -> None:
        """Test that values match MoSQITo direct calculation."""
        signal_mono, _, _, _ = _sharpness_signal()
        op = SharpnessDin(_SR)
        our_result = op.process_array(signal_mono).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        s_direct, _ = sharpness_din_tv_mosqito(signal_mono[0], _SR)

        np.testing.assert_array_equal(
            our_result[0],
            s_direct,
            err_msg="Sharpness values differ from direct MoSQITo calculation",
        )

    def test_stereo_matches_mosqito(self) -> None:
        """Test stereo signal matches MoSQITo for each channel."""
        _, signal_stereo, _, _ = _sharpness_signal()
        op = SharpnessDin(_SR)
        result = op.process_array(signal_stereo).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        s_ch1_direct, _ = sharpness_din_tv_mosqito(signal_stereo[0], _SR)
        s_ch2_direct, _ = sharpness_din_tv_mosqito(signal_stereo[1], _SR)

        np.testing.assert_array_equal(result[0], s_ch1_direct)
        np.testing.assert_array_equal(result[1], s_ch2_direct)

    def test_sharpness_values_range(self) -> None:
        """Test that sharpness values are in reasonable range."""
        signal_mono, _, _, _ = _sharpness_signal()
        op = SharpnessDin(_SR)
        result = op.process_array(signal_mono).compute()

        # Sharpness values should be non-negative
        assert np.all(result >= 0)

        # For our high-frequency signal, sharpness should be detectable (> 0)
        assert np.max(result) > 0

        # MoSQITo wrapper — exact match expected (same algorithm)
        s_direct, _ = sharpness_din_tv_mosqito(signal_mono[0], _SR)

        # Verify it matches MoSQITo
        np.testing.assert_array_equal(result[0], s_direct)

    def test_silence_produces_low_sharpness(self) -> None:
        """Test that silence produces near-zero sharpness."""
        op = SharpnessDin(_SR)
        duration = 0.1
        silence = np.zeros((1, int(_SR * duration)))

        result = op.process_array(silence).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        s_direct, _ = sharpness_din_tv_mosqito(silence[0], _SR)

        np.testing.assert_array_equal(result[0], s_direct)

        # Sharpness should be very low for silence
        assert np.all(result < 0.01)

    def test_process_with_dask(self) -> None:
        """Test that process method works with dask arrays."""
        signal_mono, _, dask_mono, _ = _sharpness_signal()
        op = SharpnessDin(_SR)
        result = op.process(dask_mono).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        s_direct, _ = sharpness_din_tv_mosqito(signal_mono[0], _SR)
        np.testing.assert_array_equal(result[0], s_direct)

    def test_multi_channel_independence(self) -> None:
        """Test that each channel is processed independently."""
        _, signal_stereo, _, _ = _sharpness_signal()
        op = SharpnessDin(_SR)
        result = op.process_array(signal_stereo).compute()

        # Compare each channel with MoSQITo direct calculation
        s_ch1_direct, _ = sharpness_din_tv_mosqito(signal_stereo[0], _SR)
        s_ch2_direct, _ = sharpness_din_tv_mosqito(signal_stereo[1], _SR)

        np.testing.assert_array_equal(result[0], s_ch1_direct)
        np.testing.assert_array_equal(result[1], s_ch2_direct)

    def test_1d_input_handling(self) -> None:
        """Test that 1D input is properly handled."""
        op = SharpnessDin(_SR)
        duration = 0.1
        # Create 1D signal
        t = np.linspace(0, duration, int(_SR * duration))
        signal_1d = 0.05 * np.sin(2 * np.pi * 4000 * t)

        result = op.process_array(signal_1d).compute()

        # Should be reshaped to 2D with 1 channel
        assert result.ndim == 2
        assert result.shape[0] == 1

    def test_consistency_across_calls(self) -> None:
        """Test that repeated calls with same input produce same output."""
        signal_mono, _, _, _ = _sharpness_signal()
        op = SharpnessDin(_SR)
        result1 = op.process_array(signal_mono).compute()
        result2 = op.process_array(signal_mono).compute()

        # Results should be identical
        np.testing.assert_array_equal(result1, result2)

    def test_sampling_rate_metadata(self) -> None:
        """Test that output sampling rate is correctly calculated."""
        op = SharpnessDin(_SR)
        metadata_updates = op.get_metadata_updates()

        assert "sampling_rate" in metadata_updates
        assert metadata_updates["sampling_rate"] == 500.0

    def test_calculate_output_shape(self) -> None:
        """Test calculate_output_shape method."""
        op = SharpnessDin(_SR)
        input_shape = (1, 48000)  # 1 channel, 1 second at 48kHz
        output_shape = op.calculate_output_shape(input_shape)

        # Output should have same number of channels
        assert output_shape[0] == input_shape[0]
        # Output should have fewer time samples (downsampled)
        assert output_shape[1] < input_shape[1]
        assert output_shape[1] > 0

        # Test that empty input shape raises ValueError
        with pytest.raises(ValueError, match="Input shape must have at least one dimension"):
            op.calculate_output_shape(())

    def test_time_axis_values(self) -> None:
        """Test that time axis is correctly calculated based on sampling rate."""
        signal_mono, _, _, _ = _sharpness_signal()
        from wandas.frames.channel import ChannelFrame

        # Create frame and calculate sharpness
        dask_data = da_from_array(signal_mono, chunks=(1, -1))
        frame = ChannelFrame(data=dask_data, sampling_rate=_SR)
        sharpness_frame = frame.sharpness_din()

        # wandas time axis should be evenly spaced based on output sampling rate
        time_wandas = sharpness_frame.time
        expected_sampling_rate = 500.0  # SharpnessDin outputs at 500 Hz

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
            assert sharpness_frame.sampling_rate == expected_sampling_rate, (
                f"Expected {expected_sampling_rate} Hz, got {sharpness_frame.sampling_rate}"
            )

    def test_plot_method_exists_and_works(self) -> None:
        """Test that sharpness ChannelFrame can be plotted."""
        signal_mono, _, _, _ = _sharpness_signal()
        import matplotlib.pyplot as plt

        from wandas.frames.channel import ChannelFrame

        # Create frame and calculate sharpness
        dask_data = da_from_array(signal_mono, chunks=(1, -1))
        frame = ChannelFrame(data=dask_data, sampling_rate=_SR)
        sharpness_frame = frame.sharpness_din()

        # Test plot method exists
        assert hasattr(sharpness_frame, "plot")
        assert callable(sharpness_frame.plot)

        # Test plot execution without error (use overlay=True for single Axes)
        fig, ax = plt.subplots()
        result_ax = sharpness_frame.plot(ax=ax, title="Test Sharpness", ylabel="Sharpness [acum]")

        assert result_ax is ax
        assert ax.get_ylabel() == "Sharpness [acum]"
        assert ax.get_title() == "Test Sharpness"

        plt.close(fig)


class TestSharpnessDinIntegration:
    """Integration tests for sharpness calculation with ChannelFrame."""

    def test_sharpness_in_operation_registry(self) -> None:
        """Test that sharpness operation is in registry."""
        from wandas.processing.base import _OPERATION_REGISTRY

        assert "sharpness_din" in _OPERATION_REGISTRY
        assert _OPERATION_REGISTRY["sharpness_din"] is _psychoacoustic_class("SharpnessDin")

    def test_channel_frame_sharpness_method_exists(self) -> None:
        """Test that ChannelFrame has sharpness_din method."""
        duration = 0.1
        from wandas.frames.channel import ChannelFrame

        # Create a simple frame
        t = np.linspace(0, duration, int(_SR * duration))
        signal = np.array([0.05 * np.sin(2 * np.pi * 4000 * t)])
        dask_data = da_from_array(signal, chunks=(1, -1))
        frame = ChannelFrame(data=dask_data, sampling_rate=_SR)

        # Check method exists
        assert hasattr(frame, "sharpness_din")
        assert callable(frame.sharpness_din)

    def test_channel_frame_sharpness_returns_channel_frame(self) -> None:
        """Test that ChannelFrame.sharpness_din() returns ChannelFrame."""
        duration = 0.1
        from wandas.frames.channel import ChannelFrame

        # Create high-frequency signal
        t = np.linspace(0, duration, int(_SR * duration))
        signal = np.array([0.05 * np.sin(2 * np.pi * 4000 * t)])
        dask_data = da_from_array(signal, chunks=(1, -1))
        frame = ChannelFrame(data=dask_data, sampling_rate=_SR)

        # Calculate sharpness
        sharpness_frame = frame.sharpness_din()

        # Should return ChannelFrame
        assert isinstance(sharpness_frame, ChannelFrame)

        # Should have updated sampling rate
        assert sharpness_frame.sampling_rate == 500.0

        # Should have sharpness data
        assert sharpness_frame.data is not None
        assert sharpness_frame.n_samples > 0

    def test_channel_frame_sharpness_matches_mosqito(self) -> None:
        """Test that ChannelFrame.sharpness_din() matches direct MoSQITo call."""
        duration = 0.1
        from wandas.frames.channel import ChannelFrame

        # Create test signal
        t = np.linspace(0, duration, int(_SR * duration))
        signal = np.array([0.05 * np.sin(2 * np.pi * 4000 * t)])
        dask_data = da_from_array(signal, chunks=(1, -1))
        frame = ChannelFrame(data=dask_data, sampling_rate=_SR)

        # Calculate using wandas
        sharpness_frame = frame.sharpness_din()

        # Get data (may be Dask or NumPy)
        sharpness_wandas_data = (
            sharpness_frame._data.compute() if hasattr(sharpness_frame._data, "compute") else sharpness_frame._data
        )

        # MoSQITo wrapper — exact match expected (same algorithm)
        s_direct, _ = sharpness_din_tv_mosqito(signal[0], _SR)

        # Results should match (sharpness_wandas_data is 2D: (n_channels, n_time))
        np.testing.assert_array_equal(sharpness_wandas_data[0], s_direct)


class TestSharpnessDinSt:
    """Test suite for SharpnessDinSt operation."""

    def test_initialization(self) -> None:
        """Test SharpnessDinSt initialization."""
        # Default initialization
        sharpness = SharpnessDinSt(_SR)
        assert sharpness.sampling_rate == _SR

    def test_invalid_weighting(self) -> None:
        """Test that invalid weighting raises ValueError."""
        with pytest.raises(ValueError, match="Invalid weighting function"):
            SharpnessDinSt(_SR, weighting="invalid")

    def test_invalid_field_type(self) -> None:
        """Test that invalid field_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid field type"):
            SharpnessDinSt(_SR, field_type="invalid")

    def test_operation_name(self) -> None:
        """Test that operation has correct name."""
        op = SharpnessDinSt(_SR)
        assert op.name == "sharpness_din_st"

    def test_operation_registration(self) -> None:
        """Test that operation is properly registered."""
        op_class = get_operation("sharpness_din_st")
        assert op_class is _psychoacoustic_class("SharpnessDinSt")

    def test_create_operation(self) -> None:
        """Test creating operation via create_operation function."""
        op = create_operation("sharpness_din_st", _SR)
        assert type(op) is _psychoacoustic_class("SharpnessDinSt")

    def test_mono_signal_shape(self) -> None:
        """Test steady-state sharpness calculation output shape for mono signal."""
        signal_mono, _, _, _ = _sharpness_signal()
        op = SharpnessDinSt(_SR)
        result = op.process_array(signal_mono).compute()

        # Result should be 2D (channels, 1)
        assert result.ndim == 2
        assert result.shape[0] == 1  # 1 channel
        assert result.shape[1] == 1  # Single sharpness value

    def test_stereo_signal_shape(self) -> None:
        """Test steady-state sharpness calculation output shape for stereo signal."""
        _, signal_stereo, _, _ = _sharpness_signal()
        op = SharpnessDinSt(_SR)
        result = op.process_array(signal_stereo).compute()

        # Result should be 2D (channels, 1)
        assert result.ndim == 2
        assert result.shape[0] == 2  # 2 channels
        assert result.shape[1] == 1  # Single sharpness value per channel

    def test_comparison_with_mosqito_direct(self) -> None:
        """Test that values match MoSQITo direct calculation."""
        signal_mono, _, _, _ = _sharpness_signal()
        op = SharpnessDinSt(_SR)
        our_result = op.process_array(signal_mono).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        s_direct = sharpness_din_st_mosqito(signal_mono[0], _SR)

        # Results should match exactly
        np.testing.assert_allclose(
            our_result[0, 0],
            s_direct,
            rtol=1e-10,
            err_msg="Sharpness values differ from direct MoSQITo calculation",
        )

    def test_stereo_matches_mosqito(self) -> None:
        """Test stereo signal matches MoSQITo for each channel."""
        _, signal_stereo, _, _ = _sharpness_signal()
        op = SharpnessDinSt(_SR)
        result = op.process_array(signal_stereo).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        s_ch1_direct = sharpness_din_st_mosqito(signal_stereo[0], _SR)
        s_ch2_direct = sharpness_din_st_mosqito(signal_stereo[1], _SR)

        np.testing.assert_allclose(result[0, 0], s_ch1_direct, rtol=1e-10)
        np.testing.assert_allclose(result[1, 0], s_ch2_direct, rtol=1e-10)

    def test_sharpness_values_range(self) -> None:
        """Test that sharpness values are in reasonable range."""
        signal_mono, _, _, _ = _sharpness_signal()
        op = SharpnessDinSt(_SR)
        result = op.process_array(signal_mono).compute()

        # Sharpness values should be non-negative
        assert np.all(result >= 0)

        # For our high-frequency signal, sharpness should be detectable (> 0)
        assert np.max(result) > 0

        # MoSQITo wrapper — exact match expected (same algorithm)
        s_direct = sharpness_din_st_mosqito(signal_mono[0], _SR)

        # Verify it matches MoSQITo
        np.testing.assert_allclose(result[0, 0], s_direct, rtol=1e-10)

    def test_silence_produces_low_sharpness(self) -> None:
        """Test that silence produces near-zero sharpness."""
        op = SharpnessDinSt(_SR)
        duration = 0.1
        silence = np.zeros((1, int(_SR * duration)))

        result = op.process_array(silence).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        s_direct = sharpness_din_st_mosqito(silence[0], _SR)

        np.testing.assert_allclose(result[0, 0], s_direct, rtol=1e-10)

        # Sharpness should be very low for silence (or NaN for zero loudness)
        # MoSQITo returns NaN for silence due to division by zero in loudness
        if np.isnan(s_direct):
            assert np.isnan(result[0, 0])
        else:
            assert np.all(result < 0.01)

    def test_process_with_dask(self) -> None:
        """Test that process method works with dask arrays."""
        signal_mono, _, dask_mono, _ = _sharpness_signal()
        op = SharpnessDinSt(_SR)
        result = op.process(dask_mono).compute()

        # MoSQITo wrapper — exact match expected (same algorithm)
        s_direct = sharpness_din_st_mosqito(signal_mono[0], _SR)
        np.testing.assert_allclose(result[0, 0], s_direct, rtol=1e-10)

    def test_multi_channel_independence(self) -> None:
        """Test that each channel is processed independently."""
        _, signal_stereo, _, _ = _sharpness_signal()
        op = SharpnessDinSt(_SR)
        result = op.process_array(signal_stereo).compute()

        # Compare each channel with MoSQITo direct calculation
        s_ch1_direct = sharpness_din_st_mosqito(signal_stereo[0], _SR)
        s_ch2_direct = sharpness_din_st_mosqito(signal_stereo[1], _SR)

        np.testing.assert_allclose(result[0, 0], s_ch1_direct, rtol=1e-10)
        np.testing.assert_allclose(result[1, 0], s_ch2_direct, rtol=1e-10)

    def test_calculate_output_shape(self) -> None:
        """Test calculate_output_shape method."""
        op = SharpnessDinSt(_SR)
        input_shape = (1, 48000)  # 1 channel, 1 second at 48kHz
        output_shape = op.calculate_output_shape(input_shape)

        # Output should be (channels, 1)
        assert output_shape[0] == input_shape[0]
        assert output_shape[1] == 1

        # Test with stereo
        input_shape_stereo = (2, 48000)
        output_shape_stereo = op.calculate_output_shape(input_shape_stereo)
        assert output_shape_stereo[0] == 2
        assert output_shape_stereo[1] == 1

    def test_1d_input_handling(self) -> None:
        """Test that 1D input is properly handled."""
        op = SharpnessDinSt(_SR)
        duration = 0.1
        # Create 1D signal
        t = np.linspace(0, duration, int(_SR * duration))
        signal_1d = 0.05 * np.sin(2 * np.pi * 4000 * t)

        result = op.process_array(signal_1d).compute()

        # Should be reshaped to 2D with 1 channel
        assert result.ndim == 2
        assert result.shape[0] == 1
        assert result.shape[1] == 1

    def test_consistency_across_calls(self) -> None:
        """Test that repeated calls with same input produce same output."""
        signal_mono, _, _, _ = _sharpness_signal()
        op = SharpnessDinSt(_SR)
        result1 = op.process_array(signal_mono).compute()
        result2 = op.process_array(signal_mono).compute()

        # Results should be identical
        np.testing.assert_array_equal(result1, result2)

    def test_metadata_updates(self) -> None:
        """Test that SharpnessDinSt returns correct metadata updates."""
        operation = SharpnessDinSt(sampling_rate=44100, field_type="free")

        updates = operation.get_metadata_updates()

        # Steady-state sharpness doesn't update sampling rate (single value output)
        assert updates == {}


class TestSharpnessDinStIntegration:
    """Integration tests for steady-state sharpness calculation with ChannelFrame."""

    def test_sharpness_st_in_operation_registry(self) -> None:
        """Test that sharpness_din_st operation is in registry."""
        from wandas.processing.base import _OPERATION_REGISTRY

        assert "sharpness_din_st" in _OPERATION_REGISTRY
        assert _OPERATION_REGISTRY["sharpness_din_st"] is _psychoacoustic_class("SharpnessDinSt")

    def test_channel_frame_sharpness_st_method_exists(self) -> None:
        """Test that ChannelFrame has sharpness_din_st method."""
        duration = 0.1
        from wandas.frames.channel import ChannelFrame

        # Create a simple frame
        t = np.linspace(0, duration, int(_SR * duration))
        signal = np.array([0.05 * np.sin(2 * np.pi * 4000 * t)])
        dask_data = da_from_array(signal, chunks=(1, -1))
        frame = ChannelFrame(data=dask_data, sampling_rate=_SR)

        # Check method exists
        assert hasattr(frame, "sharpness_din_st")
        assert callable(frame.sharpness_din_st)

    def test_channel_frame_sharpness_st_returns_ndarray(self) -> None:
        """Test that ChannelFrame.sharpness_din_st() returns NDArrayReal."""
        duration = 0.1
        from wandas.frames.channel import ChannelFrame

        # Create mono frame
        t = np.linspace(0, duration, int(_SR * duration))
        signal_mono = np.array([0.05 * np.sin(2 * np.pi * 4000 * t)])
        dask_data_mono = da_from_array(signal_mono, chunks=(1, -1))
        frame_mono = ChannelFrame(data=dask_data_mono, sampling_rate=_SR)

        # Calculate sharpness
        sharpness_mono = frame_mono.sharpness_din_st(field_type="free")

        # Should be NDArrayReal (1D array)
        assert isinstance(sharpness_mono, np.ndarray)
        assert sharpness_mono.ndim == 1
        assert sharpness_mono.shape[0] == 1  # One value per channel
        assert isinstance(sharpness_mono[0], float | np.floating)

        # Create stereo frame
        signal_stereo = np.vstack([signal_mono[0], signal_mono[0] * 0.5])
        dask_data_stereo = da_from_array(signal_stereo, chunks=(1, -1))
        frame_stereo = ChannelFrame(data=dask_data_stereo, sampling_rate=_SR)

        # Calculate sharpness for stereo
        sharpness_stereo = frame_stereo.sharpness_din_st(field_type="free")

        # Should be 1D array with 2 values
        assert isinstance(sharpness_stereo, np.ndarray)
        assert sharpness_stereo.ndim == 1
        assert sharpness_stereo.shape[0] == 2  # Two values (one per channel)

        # Can access values directly without double indexing
        assert isinstance(sharpness_stereo[0], float | np.floating)
        assert isinstance(sharpness_stereo[1], float | np.floating)

        # Can use numpy operations directly
        mean_sharpness = sharpness_stereo.mean()
        assert isinstance(mean_sharpness, float | np.floating)

    def test_channel_frame_sharpness_st_matches_mosqito(self) -> None:
        """Test that ChannelFrame.sharpness_din_st() matches direct MoSQITo call."""
        duration = 0.1
        from wandas.frames.channel import ChannelFrame

        # Create test signal
        t = np.linspace(0, duration, int(_SR * duration))
        signal = np.array([0.05 * np.sin(2 * np.pi * 4000 * t)])
        dask_data = da_from_array(signal, chunks=(1, -1))
        frame = ChannelFrame(data=dask_data, sampling_rate=_SR)

        # Calculate using wandas
        sharpness_wandas = frame.sharpness_din_st(field_type="free")

        # MoSQITo wrapper — exact match expected (same algorithm)
        s_direct = sharpness_din_st_mosqito(signal[0], _SR)

        # Results should match
        np.testing.assert_allclose(sharpness_wandas[0], s_direct, rtol=1e-10)


# --- Coverage gap tests for psychoacoustic.py ---


class TestRoughnessBaseCoverageGaps:
    """Tests targeting uncovered lines in _RoughnessBase and RoughnessDwSpec."""

    def test_estimated_time_samples_hop_zero(self) -> None:
        """_estimated_time_samples returns 1 when hop_samples == 0 (line 350).

        overlap=1.0 makes (1 - overlap) == 0, so hop_samples rounds to 0.
        """
        # Use a unique sampling rate to avoid cache collision
        sr = 99991.0
        # Clear cache to avoid stale state
        RoughnessDwSpec._bark_axis_cache.pop((sr, 1.0), None)

        # Create a mock roughness operation using a real RoughnessDw (which uses _RoughnessBase)
        roughness_dw = RoughnessDw(sampling_rate=sr, overlap=1.0)
        result = roughness_dw._estimated_time_samples(n_samples=int(sr * 0.2))
        assert result == 1

    def test_roughness_dw_spec_init_mosqito_raises(self) -> None:
        """RoughnessDwSpec.__init__ raises RuntimeError when mosqito fails (lines 493-495)."""
        unique_sr = 99997.0
        RoughnessDwSpec._bark_axis_cache.pop((unique_sr, 0.5), None)

        with patch(
            "wandas.processing.psychoacoustic.roughness_dw_mosqito",
            side_effect=RuntimeError("mosqito internal error"),
        ):
            with pytest.raises(RuntimeError, match="Could not initialize RoughnessDwSpec"):
                RoughnessDwSpec(sampling_rate=unique_sr, overlap=0.5)

    def test_roughness_dw_spec_init_empty_bark_axis(self) -> None:
        """RoughnessDwSpec.__init__ raises RuntimeError when bark_axis is empty (lines 501-502)."""
        unique_sr = 99993.0
        RoughnessDwSpec._bark_axis_cache.pop((unique_sr, 0.5), None)

        # Return an empty array as bark_axis
        with patch(
            "wandas.processing.psychoacoustic.roughness_dw_mosqito",
            return_value=(None, None, np.array([]), None),
        ):
            with pytest.raises(RuntimeError, match="empty or None bark_axis"):
                RoughnessDwSpec(sampling_rate=unique_sr, overlap=0.5)

    def test_roughness_dw_spec_calculate_output_shape_1d_input(self) -> None:
        """calculate_output_shape handles 1D input shape (lines 520-521)."""
        sr = 48000.0
        # Ensure cached so we don't need mosqito
        if (sr, 0.5) not in RoughnessDwSpec._bark_axis_cache:
            RoughnessDwSpec(sampling_rate=sr, overlap=0.5)
        op = RoughnessDwSpec(sampling_rate=sr, overlap=0.5)
        n_samples = int(sr * 1.0)
        shape = op.calculate_output_shape((n_samples,))
        # For 1D input, n_channels=1, output is (n_bark_bands, estimated_time_samples)
        assert len(shape) == 2
        assert shape[0] == len(op.bark_axis)
