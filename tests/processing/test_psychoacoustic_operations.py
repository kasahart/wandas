"""Tests for psychoacoustic metrics operations."""

from typing import Any

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.processing.base import create_operation, get_operation
from wandas.processing.psychoacoustic import RoughnessDW
from wandas.utils.types import NDArrayReal

_da_from_array = da.from_array  # type: ignore [unused-ignore]


class TestRoughnessDWOperation:
    """Test suite for Roughness calculation using Daniel & Weber method."""

    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: float = 44100
        self.duration: float = 1.0
        self.n_samples: int = int(self.sample_rate * self.duration)

        # Create a test signal with amplitude modulation (to generate roughness)
        # Carrier frequency: 1000 Hz, Modulation frequency: 70 Hz (peak roughness)
        t = np.linspace(0, self.duration, self.n_samples, endpoint=False)
        carrier_freq = 1000.0
        mod_freq = 70.0  # 70 Hz modulation creates high roughness

        # Create modulated signal
        carrier = np.sin(2 * np.pi * carrier_freq * t)
        modulator = 0.5 * (1 + np.sin(2 * np.pi * mod_freq * t))
        self.signal_mono: NDArrayReal = np.array([carrier * modulator])

        # Create stereo signal (different modulation frequencies)
        mod_freq2 = 100.0
        modulator2 = 0.5 * (1 + np.sin(2 * np.pi * mod_freq2 * t))
        self.signal_stereo: NDArrayReal = np.array(
            [carrier * modulator, carrier * modulator2]
        )

        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=-1)
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=-1)

    def test_initialization_time_method(self) -> None:
        """Test RoughnessDW initialization with time method."""
        roughness_op = RoughnessDW(self.sample_rate, method="time", overlap=0.0)
        assert roughness_op.sampling_rate == self.sample_rate
        assert roughness_op.method == "time"
        assert roughness_op.overlap == 0.0

    def test_initialization_freq_method(self) -> None:
        """Test RoughnessDW initialization with freq method."""
        roughness_op = RoughnessDW(self.sample_rate, method="freq", overlap=0.0)
        assert roughness_op.sampling_rate == self.sample_rate
        assert roughness_op.method == "freq"
        assert roughness_op.overlap == 0.0

    def test_initialization_with_overlap(self) -> None:
        """Test RoughnessDW initialization with custom overlap."""
        roughness_op = RoughnessDW(self.sample_rate, method="time", overlap=0.5)
        assert roughness_op.overlap == 0.5

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Invalid method"):
            RoughnessDW(self.sample_rate, method="invalid")  # type: ignore

    def test_invalid_overlap_raises_error(self) -> None:
        """Test that invalid overlap raises ValueError."""
        with pytest.raises(ValueError, match="Invalid overlap"):
            RoughnessDW(self.sample_rate, method="time", overlap=1.5)

        with pytest.raises(ValueError, match="Invalid overlap"):
            RoughnessDW(self.sample_rate, method="time", overlap=-0.1)

    def test_operation_registration(self) -> None:
        """Test that RoughnessDW operation is registered."""
        op_class = get_operation("roughness_dw")
        assert op_class == RoughnessDW

    def test_create_operation(self) -> None:
        """Test creating operation using create_operation."""
        operation = create_operation(
            "roughness_dw", self.sample_rate, method="time", overlap=0.0
        )
        assert isinstance(operation, RoughnessDW)
        assert operation.sampling_rate == self.sample_rate

    def test_output_shape_calculation(self) -> None:
        """Test that output shape calculation is correct."""
        roughness_op = RoughnessDW(self.sample_rate, method="time")

        # Mono signal: (1, n_samples) -> (1, 1)
        input_shape = (1, self.n_samples)
        output_shape = roughness_op.calculate_output_shape(input_shape)
        assert output_shape == (1, 1)

        # Stereo signal: (2, n_samples) -> (2, 1)
        input_shape = (2, self.n_samples)
        output_shape = roughness_op.calculate_output_shape(input_shape)
        assert output_shape == (2, 1)

    def test_roughness_time_method_mono(self) -> None:
        """Test roughness calculation with time method for mono signal."""
        roughness_op = RoughnessDW(self.sample_rate, method="time", overlap=0.0)
        result = roughness_op._process_array(self.signal_mono)

        # Check shape
        assert result.shape == (1, 1)

        # Check that roughness value is non-negative
        assert result[0, 0] >= 0

        # Check that roughness value is in reasonable range (typically 0-10 asper)
        assert result[0, 0] < 100  # Very rough signal should not exceed this

    def test_roughness_time_method_stereo(self) -> None:
        """Test roughness calculation with time method for stereo signal."""
        roughness_op = RoughnessDW(self.sample_rate, method="time", overlap=0.0)
        result = roughness_op._process_array(self.signal_stereo)

        # Check shape
        assert result.shape == (2, 1)

        # Check that roughness values are non-negative
        assert np.all(result >= 0)

        # Check that roughness values are different for different channels
        # (since they have different modulation frequencies)
        assert result[0, 0] != result[1, 0]

    def test_roughness_freq_method_mono(self) -> None:
        """Test roughness calculation with freq method for mono signal."""
        roughness_op = RoughnessDW(self.sample_rate, method="freq", overlap=0.0)
        result = roughness_op._process_array(self.signal_mono)

        # Check shape
        assert result.shape == (1, 1)

        # Check that roughness value is non-negative
        assert result[0, 0] >= 0

        # Check that roughness value is in reasonable range
        assert result[0, 0] < 100

    def test_roughness_freq_method_stereo(self) -> None:
        """Test roughness calculation with freq method for stereo signal."""
        roughness_op = RoughnessDW(self.sample_rate, method="freq", overlap=0.0)
        result = roughness_op._process_array(self.signal_stereo)

        # Check shape
        assert result.shape == (2, 1)

        # Check that roughness values are non-negative
        assert np.all(result >= 0)

    def test_roughness_with_overlap(self) -> None:
        """Test roughness calculation with different overlap values."""
        roughness_op_no_overlap = RoughnessDW(
            self.sample_rate, method="time", overlap=0.0
        )
        result_no_overlap = roughness_op_no_overlap._process_array(self.signal_mono)

        roughness_op_overlap = RoughnessDW(
            self.sample_rate, method="time", overlap=0.5
        )
        result_overlap = roughness_op_overlap._process_array(self.signal_mono)

        # Both should return valid roughness values
        assert result_no_overlap[0, 0] >= 0
        assert result_overlap[0, 0] >= 0

    def test_roughness_pure_tone(self) -> None:
        """Test roughness of pure tone (should be close to zero).

        A pure tone without modulation should have very low roughness.
        """
        # Create pure tone (no modulation)
        t = np.linspace(0, self.duration, self.n_samples, endpoint=False)
        pure_tone = np.sin(2 * np.pi * 1000 * t)
        pure_tone_signal = np.array([pure_tone])

        roughness_op = RoughnessDW(self.sample_rate, method="time", overlap=0.0)
        result = roughness_op._process_array(pure_tone_signal)

        # Pure tone should have very low roughness (close to 0)
        assert result[0, 0] < 0.5  # Threshold for "low" roughness

    def test_comparison_with_direct_mosqito_time(self) -> None:
        """Test that wandas results match direct MoSQITo calculations (time method).

        This test compares the roughness values calculated by wandas with
        those calculated directly using MoSQITo's API.
        """
        from mosqito.sq_metrics import roughness_dw_time

        # Calculate using wandas
        roughness_op = RoughnessDW(self.sample_rate, method="time", overlap=0.0)
        wandas_result = roughness_op._process_array(self.signal_mono)

        # Calculate directly using MoSQITo
        mosqito_result = roughness_dw_time(
            self.signal_mono[0, :],
            fs=self.sample_rate,
            overlap=0.0,
        )

        # Extract roughness value from MoSQITo result
        # roughness_dw_time returns (R, R_specific, bark_axis, time_axis)
        if isinstance(mosqito_result, tuple):
            mosqito_roughness = float(np.mean(mosqito_result[0]))
        else:
            mosqito_roughness = float(mosqito_result)

        # Compare results (allow small numerical differences)
        np.testing.assert_allclose(
            wandas_result[0, 0],
            mosqito_roughness,
            rtol=0.01,  # 1% relative tolerance
            atol=0.01,  # 0.01 asper absolute tolerance
        )

    def test_comparison_with_direct_mosqito_freq(self) -> None:
        """Test that wandas results match direct MoSQITo calculations (freq method).

        This test compares the roughness values calculated by wandas with
        those calculated directly using MoSQITo's API.
        """
        from mosqito.sq_metrics import roughness_dw_freq

        # Calculate using wandas
        roughness_op = RoughnessDW(self.sample_rate, method="freq", overlap=0.0)
        wandas_result = roughness_op._process_array(self.signal_mono)

        # Calculate directly using MoSQITo
        signal = self.signal_mono[0, :]
        spectrum = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), d=1.0 / self.sample_rate)
        magnitude = np.abs(spectrum)

        mosqito_result = roughness_dw_freq(magnitude, freqs)

        # Extract roughness value from MoSQITo result
        # roughness_dw_freq returns (R, R_specific, bark_axis)
        if isinstance(mosqito_result, tuple):
            mosqito_roughness = float(mosqito_result[0])
        else:
            mosqito_roughness = float(mosqito_result)

        # Compare results (allow small numerical differences)
        np.testing.assert_allclose(
            wandas_result[0, 0],
            mosqito_roughness,
            rtol=0.01,  # 1% relative tolerance
            atol=0.01,  # 0.01 asper absolute tolerance
        )

    def test_process_with_dask_array(self) -> None:
        """Test processing with Dask array."""
        roughness_op = RoughnessDW(self.sample_rate, method="time", overlap=0.0)

        # Process using Dask array
        result_dask = roughness_op.process(self.dask_mono)

        # Compute result
        result = result_dask.compute()

        # Check shape
        assert result.shape == (1, 1)

        # Check that result is valid
        assert result[0, 0] >= 0
