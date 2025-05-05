from unittest import mock

import dask.array as da
import numpy as np
import pytest
import scipy.signal as signal
from dask.array.core import Array as DaArray

from wandas.processing.base import create_operation, get_operation

# インポートパスを修正 (filter → filters)
from wandas.processing.filters import (
    AWeighting,
    HighPassFilter,
    LowPassFilter,
)
from wandas.utils.types import NDArrayReal

_da_from_array = da.from_array  # type: ignore [unused-ignore]


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
        # process_arrayの代わりにprocessメソッドを使用
        result: NDArrayReal = self.hpf.process(self.dask_signal).compute()

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
        # process_arrayの代わりにprocessメソッドを使用
        result: NDArrayReal = self.lpf.process(self.dask_signal).compute()

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


class TestAWeightingOperation:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 300000
        self.a_weight = AWeighting(self.sample_rate)

        # Different frequency components
        # (A-weighting affects different frequencies differently)
        self.low_freq: float = 100.0  # heavily attenuated by A-weighting
        self.mid_freq: float = 1000.0  # slight boost around 1-2kHz
        self.high_freq: float = 10000.0  # some attenuation at higher frequencies

        # Single channel signal with all components
        self.signal: NDArrayReal = signal.unit_impulse(self.sample_rate).reshape(1, -1)
        # Create dask array
        self.dask_signal: DaArray = _da_from_array(self.signal, chunks=-1)

    def test_initialization(self) -> None:
        """Test initialization with parameters."""
        a_weight = AWeighting(self.sample_rate)
        assert a_weight.sampling_rate == self.sample_rate

    def test_filter_effect(self) -> None:
        """Test that A-weighting affects frequencies as expected."""
        # process_arrayの代わりにprocessメソッドを使用
        result: NDArrayReal = self.a_weight.process(self.dask_signal).compute()

        # Check shape preservation
        assert result.shape == self.signal.shape

        # Calculate FFT to check frequency content
        fft_original = np.abs(np.fft.rfft(self.signal[0]))
        fft_filtered = np.abs(np.fft.rfft(result[0]))

        freq_bins = np.fft.rfftfreq(len(self.signal[0]), 1 / self.sample_rate)

        # Find indices closest to our test frequencies
        low_idx = np.argmin(np.abs(freq_bins - self.low_freq))
        mid_idx = np.argmin(np.abs(freq_bins - self.mid_freq))
        high_idx = np.argmin(np.abs(freq_bins - self.high_freq))

        # Low frequency should be heavily attenuated by A-weighting
        assert int(20 * np.log10(fft_filtered[low_idx] / fft_original[low_idx])) == -19
        # Mid frequency might be slightly boosted or preserved
        # A-weighting typically has less effect around 1kHz
        assert int(20 * np.log10(fft_filtered[mid_idx] / fft_original[mid_idx])) == 0

        # High frequency should be somewhat attenuated 小数点1桁まで確認。
        assert (
            int(20 * np.log10(fft_filtered[high_idx] / fft_original[high_idx]) * 10)
            == -2.5 * 10
        )

    def test_process(self) -> None:
        """Test the process method with Dask array."""
        # Process using the high-level process method
        result = self.a_weight.process(self.dask_signal)

        # Check that the result is a Dask array
        assert isinstance(result, DaArray)

        # Compute and check shape
        computed_result = result.compute()
        assert computed_result.shape == self.signal.shape

        with mock.patch.object(
            DaArray, "compute", return_value=self.signal
        ) as mock_compute:
            # Just creating the object shouldn't call compute
            # Verify compute hasn't been called

            result = self.a_weight.process(self.dask_signal)
            mock_compute.assert_not_called()
            # Now call compute
            computed_result = result.compute()
            # Verify compute was called once
            mock_compute.assert_called_once()

    def test_operation_registry(self) -> None:
        """Test that AWeighting is properly registered in the operation registry."""
        # Verify AWeighting can be accessed through the registry
        assert get_operation("a_weighting") == AWeighting

        # Create operation through the factory function
        a_weight_op = create_operation("a_weighting", self.sample_rate)

        # Verify the operation was created correctly
        assert isinstance(a_weight_op, AWeighting)
        assert a_weight_op.sampling_rate == self.sample_rate
