"""
Tests for psychoacoustic metrics operations.

This module tests the loudness calculation operations and compares results
with direct MoSQITo calculations.
"""

import numpy as np
import pytest
from dask.array import from_array as da_from_array
from dask.array.core import Array as DaArray

from wandas.processing.base import create_operation, get_operation
from wandas.processing.psychoacoustics import LoudnessZwst, LoudnessZwtv
from wandas.utils.types import NDArrayReal


class TestLoudnessZwtv:
    """Test time-varying loudness calculation."""

    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 48000

        # Create a test signal: 1 kHz tone at 70 dB SPL
        # SPL reference: 20 µPa (2e-5 Pa)
        # 70 dB SPL corresponds to approximately 0.0632 Pa RMS
        duration = 1.0  # 1 second
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        frequency = 1000.0  # Hz

        # Generate sine wave with RMS amplitude of 0.0632 Pa (70 dB SPL)
        # Peak amplitude = RMS * sqrt(2)
        rms_amplitude = 0.0632  # Pa
        peak_amplitude = rms_amplitude * np.sqrt(2)
        self.signal_mono: NDArrayReal = np.array(
            [peak_amplitude * np.sin(2 * np.pi * frequency * t)]
        )

        # Create stereo version
        self.signal_stereo: NDArrayReal = np.vstack(
            [self.signal_mono[0], self.signal_mono[0]]
        )

        self.dask_mono: DaArray = da_from_array(self.signal_mono, chunks=-1)
        self.dask_stereo: DaArray = da_from_array(self.signal_stereo, chunks=-1)

    def test_initialization(self) -> None:
        """Test initialization with different parameters."""
        # Test with free field
        loudness_op = LoudnessZwtv(self.sample_rate, field_type="free")
        assert loudness_op.sampling_rate == self.sample_rate
        assert loudness_op.field_type == "free"

        # Test with diffuse field
        loudness_op_diffuse = LoudnessZwtv(self.sample_rate, field_type="diffuse")
        assert loudness_op_diffuse.field_type == "diffuse"

    def test_invalid_field_type(self) -> None:
        """Test that invalid field_type raises ValueError."""
        with pytest.raises(ValueError, match="field_type must be"):
            LoudnessZwtv(self.sample_rate, field_type="invalid")  # type: ignore

    def test_invalid_sampling_rate(self) -> None:
        """Test that invalid sampling rate raises ValueError."""
        with pytest.raises(ValueError, match="Sampling rate must be positive"):
            LoudnessZwtv(-1000, field_type="free")

    def test_operation_registry(self) -> None:
        """Test that LoudnessZwtv is properly registered."""
        assert get_operation("loudness_zwtv") == LoudnessZwtv

        loudness_op = create_operation("loudness_zwtv", self.sample_rate)
        assert isinstance(loudness_op, LoudnessZwtv)
        assert loudness_op.sampling_rate == self.sample_rate

    def test_mono_signal_output_structure(self) -> None:
        """Test that output has correct structure for mono signal."""
        loudness_op = LoudnessZwtv(self.sample_rate, field_type="free")
        result = loudness_op.process(self.dask_mono)

        # Check that result is a dictionary
        assert isinstance(result, dict)

        # Check required keys
        assert "N" in result
        assert "N_spec" in result
        assert "bark_axis" in result
        assert "time_axis" in result
        assert "n_channels" in result

        # Check types
        assert isinstance(result["N"], np.ndarray)
        assert isinstance(result["N_spec"], np.ndarray)
        assert isinstance(result["bark_axis"], np.ndarray)
        assert isinstance(result["time_axis"], np.ndarray)
        assert result["n_channels"] == 1

        # Check dimensions
        assert result["N"].ndim == 1
        assert result["N_spec"].ndim == 2
        assert len(result["bark_axis"]) == result["N_spec"].shape[0]
        assert len(result["time_axis"]) == result["N_spec"].shape[1]
        assert len(result["time_axis"]) == len(result["N"])

    def test_stereo_signal_output_structure(self) -> None:
        """Test that output has correct structure for stereo signal."""
        loudness_op = LoudnessZwtv(self.sample_rate, field_type="free")
        result = loudness_op.process(self.dask_stereo)

        # Check that result is a dictionary
        assert isinstance(result, dict)
        assert result["n_channels"] == 2

        # Check dimensions
        assert result["N"].ndim == 1
        assert result["N_spec"].ndim == 2

    def test_comparison_with_mosqito_direct(self) -> None:
        """Test that results match direct MoSQITo calculation."""
        from mosqito.sq_metrics import loudness_zwtv

        # Calculate using our operation
        loudness_op = LoudnessZwtv(self.sample_rate, field_type="free")
        result_wandas = loudness_op.process(self.dask_mono)

        # Calculate using MoSQITo directly
        N_mosqito, N_spec_mosqito, bark_axis_mosqito, time_axis_mosqito = (
            loudness_zwtv(self.signal_mono[0], self.sample_rate, field_type="free")
        )

        # Compare results
        np.testing.assert_allclose(result_wandas["N"], N_mosqito, rtol=1e-10)
        np.testing.assert_allclose(result_wandas["N_spec"], N_spec_mosqito, rtol=1e-10)
        np.testing.assert_allclose(
            result_wandas["bark_axis"], bark_axis_mosqito, rtol=1e-10
        )
        np.testing.assert_allclose(
            result_wandas["time_axis"], time_axis_mosqito, rtol=1e-10
        )

    def test_loudness_values_range(self) -> None:
        """Test that loudness values are in a reasonable range."""
        loudness_op = LoudnessZwtv(self.sample_rate, field_type="free")
        result = loudness_op.process(self.dask_mono)

        # Loudness should be positive
        assert np.all(result["N"] >= 0)
        assert np.all(result["N_spec"] >= 0)

        # For 70 dB SPL tone, loudness should be in a reasonable range
        # (typically a few sones for a 1 kHz tone at 70 dB)
        mean_loudness = np.mean(result["N"])
        assert 0.1 < mean_loudness < 100  # Reasonable range for sones

    def test_bark_axis_range(self) -> None:
        """Test that Bark axis is in the expected range."""
        loudness_op = LoudnessZwtv(self.sample_rate, field_type="free")
        result = loudness_op.process(self.dask_mono)

        # Bark scale typically ranges from 0 to ~24 Bark
        assert np.min(result["bark_axis"]) >= 0
        assert np.max(result["bark_axis"]) <= 30  # Upper bound with margin

    def test_time_axis_duration(self) -> None:
        """Test that time axis corresponds to signal duration."""
        loudness_op = LoudnessZwtv(self.sample_rate, field_type="free")
        result = loudness_op.process(self.dask_mono)

        expected_duration = self.signal_mono.shape[1] / self.sample_rate
        actual_duration = np.max(result["time_axis"])

        # Time axis should span approximately the signal duration
        # Allow for some tolerance due to windowing/framing
        assert 0.9 * expected_duration <= actual_duration <= 1.1 * expected_duration

    def test_different_field_types(self) -> None:
        """Test that free and diffuse field give different results."""
        loudness_free = LoudnessZwtv(self.sample_rate, field_type="free")
        loudness_diffuse = LoudnessZwtv(self.sample_rate, field_type="diffuse")

        result_free = loudness_free.process(self.dask_mono)
        result_diffuse = loudness_diffuse.process(self.dask_mono)

        # Results should be different (but both valid)
        assert not np.allclose(result_free["N"], result_diffuse["N"])
        # Both should be positive
        assert np.all(result_free["N"] >= 0)
        assert np.all(result_diffuse["N"] >= 0)


class TestLoudnessZwst:
    """Test stationary loudness calculation."""

    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 48000

        # Create a test signal: 1 kHz tone at 70 dB SPL
        duration = 1.0  # 1 second
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        frequency = 1000.0  # Hz

        # Generate sine wave with RMS amplitude of 0.0632 Pa (70 dB SPL)
        rms_amplitude = 0.0632  # Pa
        peak_amplitude = rms_amplitude * np.sqrt(2)
        self.signal_mono: NDArrayReal = np.array(
            [peak_amplitude * np.sin(2 * np.pi * frequency * t)]
        )

        # Create stereo version
        self.signal_stereo: NDArrayReal = np.vstack(
            [self.signal_mono[0], self.signal_mono[0]]
        )

        self.dask_mono: DaArray = da_from_array(self.signal_mono, chunks=-1)
        self.dask_stereo: DaArray = da_from_array(self.signal_stereo, chunks=-1)

    def test_initialization(self) -> None:
        """Test initialization with different parameters."""
        # Test with free field
        loudness_op = LoudnessZwst(self.sample_rate, field_type="free")
        assert loudness_op.sampling_rate == self.sample_rate
        assert loudness_op.field_type == "free"

        # Test with diffuse field
        loudness_op_diffuse = LoudnessZwst(self.sample_rate, field_type="diffuse")
        assert loudness_op_diffuse.field_type == "diffuse"

    def test_invalid_field_type(self) -> None:
        """Test that invalid field_type raises ValueError."""
        with pytest.raises(ValueError, match="field_type must be"):
            LoudnessZwst(self.sample_rate, field_type="invalid")  # type: ignore

    def test_invalid_sampling_rate(self) -> None:
        """Test that invalid sampling rate raises ValueError."""
        with pytest.raises(ValueError, match="Sampling rate must be positive"):
            LoudnessZwst(-1000, field_type="free")

    def test_operation_registry(self) -> None:
        """Test that LoudnessZwst is properly registered."""
        assert get_operation("loudness_zwst") == LoudnessZwst

        loudness_op = create_operation("loudness_zwst", self.sample_rate)
        assert isinstance(loudness_op, LoudnessZwst)
        assert loudness_op.sampling_rate == self.sample_rate

    def test_mono_signal_output_structure(self) -> None:
        """Test that output has correct structure for mono signal."""
        loudness_op = LoudnessZwst(self.sample_rate, field_type="free")
        result = loudness_op.process(self.dask_mono)

        # Check that result is a dictionary
        assert isinstance(result, dict)

        # Check required keys
        assert "N" in result
        assert "N_spec" in result
        assert "bark_axis" in result
        assert "n_channels" in result

        # Check types
        assert isinstance(result["N"], (float, np.floating))
        assert isinstance(result["N_spec"], np.ndarray)
        assert isinstance(result["bark_axis"], np.ndarray)
        assert result["n_channels"] == 1

        # Check dimensions
        assert result["N_spec"].ndim == 1
        assert len(result["bark_axis"]) == len(result["N_spec"])

    def test_stereo_signal_output_structure(self) -> None:
        """Test that output has correct structure for stereo signal."""
        loudness_op = LoudnessZwst(self.sample_rate, field_type="free")
        result = loudness_op.process(self.dask_stereo)

        # Check that result is a dictionary
        assert isinstance(result, dict)
        assert result["n_channels"] == 2

        # Check dimensions
        assert isinstance(result["N"], (float, np.floating))
        assert result["N_spec"].ndim == 1

    def test_comparison_with_mosqito_direct(self) -> None:
        """Test that results match direct MoSQITo calculation."""
        from mosqito.sq_metrics import loudness_zwst

        # Calculate using our operation
        loudness_op = LoudnessZwst(self.sample_rate, field_type="free")
        result_wandas = loudness_op.process(self.dask_mono)

        # Calculate using MoSQITo directly
        N_mosqito, N_spec_mosqito, bark_axis_mosqito = loudness_zwst(
            self.signal_mono[0], self.sample_rate, field_type="free"
        )

        # Compare results - N should match exactly
        assert np.isclose(result_wandas["N"], N_mosqito, rtol=1e-10)
        np.testing.assert_allclose(result_wandas["N_spec"], N_spec_mosqito, rtol=1e-10)
        np.testing.assert_allclose(
            result_wandas["bark_axis"], bark_axis_mosqito, rtol=1e-10
        )

    def test_loudness_values_range(self) -> None:
        """Test that loudness values are in a reasonable range."""
        loudness_op = LoudnessZwst(self.sample_rate, field_type="free")
        result = loudness_op.process(self.dask_mono)

        # Loudness should be positive
        assert result["N"] >= 0
        assert np.all(result["N_spec"] >= 0)

        # For 70 dB SPL tone, loudness should be in a reasonable range
        assert 0.1 < result["N"] < 100  # Reasonable range for sones

    def test_bark_axis_range(self) -> None:
        """Test that Bark axis is in the expected range."""
        loudness_op = LoudnessZwst(self.sample_rate, field_type="free")
        result = loudness_op.process(self.dask_mono)

        # Bark scale typically ranges from 0 to ~24 Bark
        assert np.min(result["bark_axis"]) >= 0
        assert np.max(result["bark_axis"]) <= 30  # Upper bound with margin

    def test_different_field_types(self) -> None:
        """Test that free and diffuse field give different results."""
        loudness_free = LoudnessZwst(self.sample_rate, field_type="free")
        loudness_diffuse = LoudnessZwst(self.sample_rate, field_type="diffuse")

        result_free = loudness_free.process(self.dask_mono)
        result_diffuse = loudness_diffuse.process(self.dask_mono)

        # Results should be different (but both valid)
        assert not np.isclose(result_free["N"], result_diffuse["N"])
        # Both should be positive
        assert result_free["N"] >= 0
        assert result_diffuse["N"] >= 0

    def test_specific_loudness_peak_frequency(self) -> None:
        """Test that specific loudness peaks near the tone frequency."""
        loudness_op = LoudnessZwst(self.sample_rate, field_type="free")
        result = loudness_op.process(self.dask_mono)

        # For a 1 kHz tone, the peak in specific loudness should be around
        # 8-9 Bark (which corresponds to ~1 kHz)
        peak_bark_idx = np.argmax(result["N_spec"])
        peak_bark = result["bark_axis"][peak_bark_idx]

        # 1 kHz corresponds to approximately 8.5 Bark
        # Allow for some tolerance
        assert 7.0 < peak_bark < 10.0
