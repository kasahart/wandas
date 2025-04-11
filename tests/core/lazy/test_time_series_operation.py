from typing import Callable
from unittest import mock

import dask.array as da
import librosa
import numpy as np
import pytest
from dask.array.core import Array as DaArray
from scipy import fft, signal

from wandas.core.lazy.time_series_operation import (
    _OPERATION_REGISTRY,
    FFT,
    ISTFT,
    STFT,
    AudioOperation,
    AWeighting,
    HighPassFilter,
    HpssHarmonic,
    HpssPercussive,
    LowPassFilter,
    RmsTrend,
    Trim,
    create_operation,
    get_operation,
    register_operation,
)
from wandas.utils.types import NDArrayComplex, NDArrayReal

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
        processor = self.hpf.process_array
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
        processor = self.lpf.process_array
        result: NDArrayReal = processor(self.signal).compute()

        # Calculate FFT to check frequency content
        fft_original = np.abs(np.fft.rfft(self.signal[0]))
        fft_filtered = np.abs(np.fft.rfft(result[0]))

        freq_bins = fft.rfftfreq(len(self.signal[0]), 1 / self.sample_rate)

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
        processor = self.a_weight.process_array
        result: NDArrayReal = processor(self.signal).compute()

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


class TestSTFTOperation:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.n_fft: int = 1024
        self.hop_length: int = 256
        self.win_length: int = 1024
        self.window: str = "hann"
        self.boundary: str = "zeros"

        # Create a test signal (1 second sine wave at 440 Hz)
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        self.signal_mono: NDArrayReal = np.array([np.sin(2 * np.pi * 1000 * t)]) * 4
        self.signal_stereo: NDArrayReal = np.array(
            [np.sin(2 * np.pi * 1000 * t), np.sin(2 * np.pi * 2000 * t)]
        )

        # Create dask arrays
        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=-1)
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=-1)

        # Initialize STFT
        self.stft = STFT(
            sampling_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            boundary=self.boundary,
        )

        # Initialize ISTFT
        self.istft = ISTFT(
            sampling_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            boundary=self.boundary,
        )

    def test_stft_initialization(self) -> None:
        """Test STFT initialization with different parameters."""
        # Default initialization
        stft = STFT(self.sample_rate)
        assert stft.sampling_rate == self.sample_rate
        assert stft.n_fft == 2048
        assert stft.hop_length == 512  # 2048 // 4
        assert stft.win_length == 2048
        assert stft.window == "hann"
        assert stft.boundary == "zeros"  # デフォルト値を確認

        # Custom initialization
        custom_stft = STFT(
            sampling_rate=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            win_length=512,
            window="hamming",
            boundary="reflect",
        )
        assert custom_stft.n_fft == 1024
        assert custom_stft.hop_length == 256
        assert custom_stft.win_length == 512
        assert custom_stft.window == "hamming"
        assert custom_stft.boundary == "reflect"  # カスタム値を確認

    def test_istft_initialization(self) -> None:
        """Test ISTFT initialization with different parameters."""
        # Default initialization
        istft = ISTFT(self.sample_rate)
        assert istft.sampling_rate == self.sample_rate
        assert istft.n_fft == 2048
        assert istft.hop_length == 512  # 2048 // 4
        assert istft.win_length == 2048
        assert istft.window == "hann"
        assert istft.boundary == "zeros"  # デフォルト値を確認
        assert istft.length is None

        # Custom initialization
        custom_istft = ISTFT(
            sampling_rate=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            win_length=512,
            window="hamming",
            boundary=None,
            length=16000,
        )
        assert custom_istft.n_fft == 1024
        assert custom_istft.hop_length == 256
        assert custom_istft.win_length == 512
        assert custom_istft.window == "hamming"
        assert custom_istft.boundary is None
        assert custom_istft.length == 16000

    def test_stft_shape_mono(self) -> None:
        """Test STFT output shape for mono signal."""
        # Process the mono signal
        stft_result = self.stft.process_array(self.signal_mono).compute()

        # Check the shape of the result
        assert stft_result.ndim == 3, (
            "Output should be 3D (channels, frequencies, time)"
        )

        # Expected shape: (channels, frequencies, time frames)
        expected_n_channels = 1
        expected_n_freqs = self.n_fft // 2 + 1
        expected_n_frames = int(np.ceil(len(self.signal_mono[0]) / self.hop_length))
        if self.boundary != "none":
            expected_n_frames += 1  # パディングがある場合はフレーム数が増える

        expected_shape = (expected_n_channels, expected_n_freqs, expected_n_frames)
        assert stft_result.shape == expected_shape, (
            f"Expected {expected_shape}, got {stft_result.shape}"
        )

    def test_stft_shape_stereo(self) -> None:
        """Test STFT output shape for stereo signal."""
        # Process the stereo signal
        stft_result = self.stft.process_array(self.signal_stereo).compute()

        assert stft_result.ndim == 3, (
            "Output should be 3D (channels, frequencies, time)"
        )

        # Expected shape: (channels, frequencies, time frames)
        expected_n_channels = 2
        expected_n_freqs = self.n_fft // 2 + 1
        expected_n_frames = int(np.ceil(len(self.signal_stereo[0]) / self.hop_length))
        if self.boundary != "none":
            expected_n_frames += 1  # パディングがある場合はフレーム数が増える

        expected_shape = (expected_n_channels, expected_n_freqs, expected_n_frames)
        assert stft_result.shape == expected_shape, (
            f"Expected {expected_shape}, got {stft_result.shape}"
        )

    def test_stft_content(self) -> None:
        """Test STFT content correctness."""
        # Process the mono signal
        stft_result = self.stft.process_array(self.signal_mono).compute()

        assert stft_result.ndim == 3, (
            "Output should be 3D (channels, frequencies, time)"
        )

        # Calculate the expected STFT using scipy directly for comparison
        from scipy import signal as ss

        _, _, expected_stft = ss.stft(
            self.signal_mono[0],
            fs=self.sample_rate,
            window=self.window,
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length,
            nfft=self.n_fft,
            boundary=self.boundary,
            padded=True,
        )
        expected_stft[..., 1:-1, :] *= 2.0
        # Add channel dimension for comparison
        expected_stft = expected_stft.reshape(1, *expected_stft.shape)

        # Compare the results - should be very close but not exactly
        # the same due to floating point
        np.testing.assert_allclose(stft_result, expected_stft, rtol=1e-10, atol=1e-10)

        np.testing.assert_allclose(np.abs(stft_result).max(), 4, rtol=0.1)

    def test_istft_shape(self) -> None:
        """Test ISTFT output shape."""
        # First get some STFT data
        stft_data = self.stft.process_array(self.signal_mono)

        # Process with ISTFT
        istft_result = self.istft.process_array(stft_data).compute()

        # Check the shape
        assert istft_result.ndim == 2, "Output should be 2D (channels, time)"

        # One channel
        assert istft_result.shape[0] == 1

        # Length should be approximately the original signal length
        # The exact length depends on STFT/ISTFT parameters
        expected_length = len(self.signal_mono[0])
        assert abs(istft_result.shape[1] - expected_length) < self.win_length

    def test_roundtrip_reconstruction(self) -> None:
        """Test signal reconstruction quality through STFT->ISTFT roundtrip."""
        # Process with STFT then ISTFT
        stft_data = self.stft.process_array(self.signal_mono)
        istft_data = self.istft.process_array(stft_data).compute()

        # Compare with original signal (trim or pad to the same length)
        orig_length = self.signal_mono.shape[1]
        reconstructed_trimmed = istft_data[:, :orig_length]
        np.testing.assert_allclose(
            reconstructed_trimmed[..., 16:-16],
            self.signal_mono[..., 16:-16],
            rtol=1e-6,
            atol=1e-5,
        )

    def test_1d_input_handling(self) -> None:
        """Test that 1D input is properly reshaped to (1, samples)."""
        # Create a 1D array
        signal_1d = np.sin(
            2 * np.pi * 440 * np.linspace(0, 1, self.sample_rate, endpoint=False)
        )

        # Process 1D array
        stft_result = self.stft.process_array(signal_1d).compute()

        # Should be reshaped to 3D: (1, freqs, time)
        assert stft_result.ndim == 3
        assert stft_result.shape[0] == 1  # Single channel

    def test_istft_2d_input_handling(self) -> None:
        """
        Test that 2D input (single channel spectrogram) is
        properly reshaped to (1, freqs, frames).
        """
        # Create STFT data and remove channel dimension
        stft_data = self.stft.process_array(self.signal_mono).compute()
        stft_2d = stft_data[0]  # Remove channel dimension to get 2D

        # Process with ISTFT
        istft_result = self.istft.process_array(stft_2d).compute()  # Added compute()

        # Should be reshaped to 2D: (1, time)
        assert istft_result.ndim == 2
        assert istft_result.shape[0] == 1  # Single channel

    def test_boundary_parameter(self) -> None:
        """
        Test that different boundary parameters affect the output shape and content.
        """
        # Test with different boundary settings
        boundaries = ["even", "odd", "constant", "zeros", None]

        for boundary in boundaries:
            # Create STFT with the specific boundary
            stft = STFT(
                sampling_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                boundary=boundary,
            )

            # Process signal
            stft_result = stft.process_array(self.signal_mono).compute()

            # Check that we got a result with reasonable dimensions
            assert stft_result.ndim == 3
            assert stft_result.shape[0] == 1  # Single channel
            assert (
                stft_result.shape[1] == self.n_fft // 2 + 1
            )  # Number of frequency bins

            # Number of frames may differ based on boundary condition
            frame_count = stft_result.shape[2]
            min_frames = int(np.ceil(len(self.signal_mono[0]) / self.hop_length))
            # If boundary is not None, frame count is typically more than minimum
            if boundary is not None:
                assert frame_count >= min_frames

            # For ISTFT as well
            istft = ISTFT(
                sampling_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                boundary=boundary,
                length=len(self.signal_mono[0]),  # Set length to match original signal
            )

            # Roundtrip test
            istft_result = istft.process_array(stft_result).compute()
            assert istft_result.shape[0] == 1  # Single channel

            assert istft_result.ndim == 2  # Should be 2D (channels, time)
            assert istft_result.shape[1] == len(self.signal_mono[0])

    def test_stft_operation_registry(self) -> None:
        """Test that STFT is properly registered in the operation registry."""
        # Verify that STFT and ISTFT can be accessed through the registry
        assert get_operation("stft") == STFT
        assert get_operation("istft") == ISTFT

        # Create operation through the factory function
        stft_op = create_operation(
            "stft", self.sample_rate, n_fft=512, hop_length=128, boundary="reflect"
        )
        istft_op = create_operation(
            "istft", self.sample_rate, n_fft=512, hop_length=128, boundary="reflect"
        )

        # Verify the operations were created with correct parameters
        assert isinstance(stft_op, STFT)
        assert stft_op.n_fft == 512
        assert stft_op.hop_length == 128
        assert stft_op.boundary == "reflect"

        assert isinstance(istft_op, ISTFT)
        assert istft_op.n_fft == 512
        assert istft_op.hop_length == 128
        assert istft_op.boundary == "reflect"


class TestRmsTrend:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.frame_length: int = 1024
        self.hop_length: int = 256

        # 正弦波信号を作成（振幅1.0）
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        sine_wave = np.sin(2 * np.pi * 440 * t)  # 440Hzの正弦波

        # 正弦波（振幅1.0）のRMS値は1/√2 = 0.7071...
        self.expected_rms = 1.0 / np.sqrt(2)

        # シングルチャンネルとマルチチャンネル信号を準備
        self.signal_mono: NDArrayReal = np.array([sine_wave])
        self.signal_stereo: NDArrayReal = np.array(
            [sine_wave, sine_wave * 0.5]
        )  # 第2チャンネルは振幅0.5

        # Dask配列を作成
        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=(1, 1000))
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=(1, 1000))

        # RmsTrend操作を初期化
        self.rms_op = RmsTrend(
            sampling_rate=self.sample_rate,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            Aw=False,
        )

        # A-weightingを適用するRmsTrend操作
        self.rms_aw_op = RmsTrend(
            sampling_rate=self.sample_rate,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            Aw=True,
        )

    def test_initialization(self) -> None:
        """パラメータ初期化のテスト"""
        # デフォルト値でのイニシャライズ
        rms_default = RmsTrend(self.sample_rate)
        assert rms_default.sampling_rate == self.sample_rate
        assert rms_default.frame_length == 2048  # デフォルト値
        assert rms_default.hop_length == 512  # デフォルト値
        assert rms_default.Aw is False  # デフォルト値

        # カスタム値でのイニシャライズ
        custom_frame_length = 4096
        custom_hop_length = 1024
        rms_custom = RmsTrend(
            self.sample_rate,
            frame_length=custom_frame_length,
            hop_length=custom_hop_length,
            Aw=True,
        )
        assert rms_custom.sampling_rate == self.sample_rate
        assert rms_custom.frame_length == custom_frame_length
        assert rms_custom.hop_length == custom_hop_length
        assert rms_custom.Aw is True

    def test_rms_calculation(self) -> None:
        """RMS計算が正しく行われるかテスト"""
        # RMS処理を実行
        result = self.rms_op.process_array(self.signal_mono).compute()

        # 形状をチェック
        expected_frames = librosa.feature.rms(
            y=self.signal_mono,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )[..., 0, :].shape[1]
        assert result.shape == (1, expected_frames)

        # 正弦波のRMS値は振幅の1/√2に近いはず
        # フレーム分割とウィンドウ適用による誤差があるため厳密な一致ではなく近似値を確認
        np.testing.assert_allclose(np.mean(result), self.expected_rms, rtol=0.1)

        # ステレオ信号でもテスト
        result_stereo = self.rms_op.process_array(self.signal_stereo).compute()
        assert result_stereo.shape == (2, expected_frames)

        # 第2チャンネル（振幅0.5）のRMS値は第1チャンネルの約半分のはず
        ratio = np.mean(result_stereo[1]) / np.mean(result_stereo[0])
        np.testing.assert_allclose(ratio, 0.5, rtol=0.1)

    def test_a_weighting_effect(self) -> None:
        """A-weightingフィルタの効果をテスト"""
        # 通常のRMS計算
        result_normal = self.rms_op.process_array(self.signal_mono).compute()

        # A-weightingを適用したRMS計算
        result_aw = self.rms_aw_op.process_array(self.signal_mono).compute()

        # A-weightingを適用した場合と適用しない場合で結果が異なることを確認
        # 440Hzの信号に対しては大きな変化はないが、少なくとも同一ではないはず
        assert not np.allclose(result_normal, result_aw, rtol=1e-5)

    def test_delayed_execution(self) -> None:
        """Daskの遅延実行が正しく行われるかテスト"""
        # モックを使ってcompute()が呼ばれないことを検証
        with mock.patch.object(DaArray, "compute") as mock_compute:
            # process()メソッド呼び出し時にはcomputeは実行されない
            result = self.rms_op.process(self.dask_mono)
            mock_compute.assert_not_called()

            # process_array()メソッド呼び出し時にもcomputeは実行されない
            _ = self.rms_op.process_array(self.dask_mono)
            mock_compute.assert_not_called()

            # compute()を明示的に呼び出すと実行される
            _ = result.compute()
            mock_compute.assert_called()

    def test_operation_registry(self) -> None:
        """操作レジストリからRmsTrendが取得できるかテスト"""
        # レジストリから操作を取得
        assert get_operation("rms_trend") == RmsTrend

        # create_operationを使って操作を作成
        rms_op = create_operation(
            "rms_trend", self.sample_rate, frame_length=2048, hop_length=512, Aw=True
        )

        # 作成された操作が期待通りであることを確認
        assert isinstance(rms_op, RmsTrend)
        assert rms_op.sampling_rate == self.sample_rate
        assert rms_op.frame_length == 2048
        assert rms_op.hop_length == 512
        assert rms_op.Aw is True


class TestTrim:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000  # 16 kHz
        self.start_time: float = 1.0  # Start trimming at 1 second
        self.end_time: float = 2.0  # End trimming at 2 seconds
        self.trim_op: Trim = Trim(self.sample_rate, self.start_time, self.end_time)

        # Create a test signal (3 seconds sine wave at 440 Hz)
        t: np.ndarray = np.linspace(
            0, 3, self.sample_rate * 3, endpoint=False
        )  # 3 seconds
        self.signal: NDArrayReal = np.sin(2 * np.pi * 440 * t).reshape(
            1, -1
        )  # Single channel

        # Create a Dask array
        self.dask_signal: DaArray = _da_from_array(self.signal, chunks=-1)

    def test_initialization(self) -> None:
        """Test initialization of the Trim operation."""
        assert self.trim_op.start == self.start_time
        assert self.trim_op.end == self.end_time
        assert self.trim_op.start_sample == int(self.start_time * self.sample_rate)
        assert self.trim_op.end_sample == int(self.end_time * self.sample_rate)

    def test_trim_effect(self) -> None:
        """Test that the Trim operation correctly trims the signal."""
        result = self.trim_op.process_array(self.signal)

        # Compute the result
        computed_result: NDArrayReal = result.compute()

        # Check the shape of the trimmed signal
        expected_length: int = int((self.end_time - self.start_time) * self.sample_rate)
        assert computed_result.shape == (1, expected_length)

        # Check the content of the trimmed signal
        start_idx: int = self.trim_op.start_sample
        end_idx: int = self.trim_op.end_sample
        np.testing.assert_array_equal(
            computed_result, self.signal[..., start_idx:end_idx]
        )

    def test_dask_delayed_execution(self) -> None:
        """Test that the Trim operation uses Dask's delayed execution."""
        # Use mock to verify compute() isn't called during processing
        with mock.patch.object(DaArray, "compute") as mock_compute:
            # Just processing shouldn't trigger computation
            result: DaArray = self.trim_op.process(self.dask_signal)

            # Verify compute hasn't been called
            mock_compute.assert_not_called()

            # The result should be a Dask array
            assert isinstance(result, DaArray)

            # Now explicitly compute the result
            _: NDArrayReal = result.compute()

            # Verify compute was called
            mock_compute.assert_called_once()


class TestHpssHarmonic:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.hpss_harmonic = HpssHarmonic(self.sample_rate, kernel_size=31, power=2)

        # Create a test signal (1 second sine wave at 440 Hz)
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        self.signal: NDArrayReal = np.array([np.sin(2 * np.pi * 440 * t)])

        # Create a Dask array
        self.dask_signal: DaArray = _da_from_array(self.signal, chunks=(1, 1000))

    def test_initialization(self) -> None:
        """Test initialization of the HpssHarmonic operation."""
        assert self.hpss_harmonic.sampling_rate == self.sample_rate
        assert self.hpss_harmonic.kwargs["kernel_size"] == 31
        assert self.hpss_harmonic.kwargs["power"] == 2

    def test_harmonic_extraction(self) -> None:
        """Test that the HpssHarmonic operation extracts harmonic components."""
        with mock.patch(
            "librosa.effects.harmonic", return_value=self.signal
        ) as mock_harmonic:
            result = self.hpss_harmonic.process_array(self.signal).compute()

            # Verify that librosa.effects.harmonic was called with the correct arguments
            mock_harmonic.assert_called_once_with(self.signal, kernel_size=31, power=2)

            # Check that the result matches the mocked harmonic output
            np.testing.assert_array_equal(result, self.signal)

    def test_delayed_execution(self) -> None:
        """Test that HPSS Harmonic operation is executed lazily."""
        with mock.patch("dask.array.core.Array.compute") as mock_compute:
            # Apply the harmonic operation lazily
            result = self.hpss_harmonic.process(self.dask_signal)

            # Ensure compute is not called during the operation setup
            mock_compute.assert_not_called()

            # Trigger computation explicitly
            _ = result.compute()

            # Verify compute is called once during explicit computation
            mock_compute.assert_called_once()


class TestHpssPercussive:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.hpss_percussive = HpssPercussive(self.sample_rate, kernel_size=31, power=2)

        # Create a test signal (1 second sine wave at 440 Hz)
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        self.signal: NDArrayReal = np.array([np.sin(2 * np.pi * 440 * t)])

        # Create a Dask array
        self.dask_signal: DaArray = _da_from_array(self.signal, chunks=(1, 1000))

    def test_initialization(self) -> None:
        """Test initialization of the HpssPercussive operation."""
        assert self.hpss_percussive.sampling_rate == self.sample_rate
        assert self.hpss_percussive.kwargs["kernel_size"] == 31
        assert self.hpss_percussive.kwargs["power"] == 2

    def test_percussive_extraction(self) -> None:
        """Test that the HpssPercussive operation extracts percussive components."""
        with mock.patch(
            "librosa.effects.percussive", return_value=self.signal
        ) as mock_percussive:
            result = self.hpss_percussive.process_array(self.signal).compute()

            # Verify that librosa.effects.percussive was
            # called with the correct arguments
            mock_percussive.assert_called_once_with(
                self.signal, kernel_size=31, power=2
            )

            # Check that the result matches the mocked percussive output
            np.testing.assert_array_equal(result, self.signal)

    def test_delayed_execution(self) -> None:
        """Test that HPSS Percussive operation is executed lazily."""
        with mock.patch("dask.array.core.Array.compute") as mock_compute:
            # Apply the percussive operation lazily
            result = self.hpss_percussive.process(self.dask_signal)

            # Ensure compute is not called during the operation setup
            mock_compute.assert_not_called()

            # Trigger computation explicitly
            _ = result.compute()

            # Verify compute is called once during explicit computation
            mock_compute.assert_called_once()


class TestFFTOperation:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.n_fft: int = 1024
        self.window: str = "hann"
        self.fft = FFT(self.sample_rate, n_fft=self.n_fft, window=self.window)

        # Create a test signal (1 second sine wave at 500 Hz)
        self.freq: float = 500
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        self.signal_mono: NDArrayReal = (
            np.array([np.sin(2 * np.pi * self.freq * t)]) * 4
        )

        # Create a stereo signal
        self.signal_stereo: NDArrayReal = np.array(
            [
                np.sin(2 * np.pi * self.freq * t),  # 500 Hz
                np.sin(2 * np.pi * self.freq * 2 * t),  # 1000 Hz
            ]
        )

        # Create dask arrays
        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=-1)
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=-1)

    def test_initialization(self) -> None:
        """Test FFT initialization with different parameters."""
        # Default initialization
        fft = FFT(self.sample_rate)
        assert fft.sampling_rate == self.sample_rate
        assert fft.n_fft is None
        assert fft.window == "hann"

        # Custom initialization
        custom_fft = FFT(self.sample_rate, n_fft=2048, window="hamming")
        assert custom_fft.n_fft == 2048
        assert custom_fft.window == "hamming"

    def test_fft_shape(self) -> None:
        """Test FFT output shape."""
        # Process mono signal
        fft_result = self.fft.process_array(self.signal_mono).compute()

        # Check shape
        expected_freqs = self.n_fft // 2 + 1
        assert fft_result.shape == (1, expected_freqs)

        # Process stereo signal
        fft_result_stereo = self.fft.process_array(self.signal_stereo).compute()
        assert fft_result_stereo.shape == (2, expected_freqs)

    def test_fft_content(self) -> None:
        """Test FFT content correctness."""
        fft_result = self.fft.process_array(self.signal_mono).compute()

        # Calculate frequency bins
        freq_bins = np.fft.rfftfreq(self.n_fft, 1.0 / self.sample_rate)

        # Find index of the bin closest to our test frequency (440 Hz)
        target_idx = np.argmin(np.abs(freq_bins - self.freq))

        # Get magnitude spectrum
        magnitude = np.abs(fft_result[0])

        # The peak should be at the bin closest to 440 Hz
        peak_idx = np.argmax(magnitude)
        assert abs(peak_idx - target_idx) <= 1  # Allow for slight bin difference

        # Check that other frequencies have much lower magnitude
        # Exclude a small region around the peak
        mask = np.ones_like(magnitude, dtype=bool)
        region = 5  # Number of bins around peak to exclude
        lower = max(0, peak_idx - region)
        upper = min(len(magnitude), peak_idx + region + 1)
        mask[lower:upper] = False

        # The peak should be significantly higher than other frequencies
        assert np.max(magnitude[mask]) < 0.1 * magnitude[peak_idx]

    def test_amplitude_scaling(self) -> None:
        """Test that FFT amplitude scaling is correct."""
        # Create a cosine wave with amplitude 1.0
        # For a real cosine wave, the amplitude should be 0.5 in the FFT
        fft_inst = FFT(self.sample_rate, n_fft=None, window=self.window)
        amp = 2.0
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        cos_wave = amp * np.cos(2 * np.pi * self.freq * t)

        # Apply window to match what the FFT class does internally
        from scipy.signal import get_window

        win = get_window(self.window, len(cos_wave))
        scaled_cos = cos_wave * win
        scaling_factor = np.sum(win)

        # Get FFT result using our class
        fft_result = fft_inst.process_array(np.array([cos_wave])).compute()

        # Calculate expected FFT using numpy directly
        expected_fft: NDArrayComplex = fft.rfft(scaled_cos)
        expected_fft[1:-1] *= 2.0  # Double to account for single-sided spectrum
        expected_fft /= scaling_factor  # Apply scaling factor

        # Compare results
        np.testing.assert_allclose(fft_result[0], expected_fft, rtol=1e-10)

        # Find the actual frequency bin where the peak is located

        # Find the peak in the results
        peak_idx = np.argmax(np.abs(fft_result[0]))
        peak_mag = np.abs(fft_result[0, peak_idx])
        expected_mag = amp

        # Allow some tolerance due to window effects and frequency binning
        np.testing.assert_allclose(peak_mag, expected_mag, rtol=0.1)

    def test_delayed_execution(self) -> None:
        """Test that FFT operation uses dask's delayed execution."""
        with mock.patch.object(DaArray, "compute") as mock_compute:
            # Process should not trigger computation
            result = self.fft.process(self.dask_mono)
            mock_compute.assert_not_called()

            # The result should be a Dask array
            assert isinstance(result, DaArray)

            # Now explicitly compute
            _ = result.compute()
            mock_compute.assert_called_once()

    def test_window_function_effect(self) -> None:
        """Test different window functions have different effects."""
        # Test with rectangular window (no windowing)
        rect_fft = FFT(self.sample_rate, n_fft=None, window="boxcar")
        rect_result = rect_fft.process_array(self.signal_mono).compute()

        # Test with Hann window
        hann_fft = FFT(self.sample_rate, n_fft=None, window="hann")
        hann_result = hann_fft.process_array(self.signal_mono).compute()

        # The results should be different due to different window functions
        assert not np.allclose(rect_result, hann_result)

        rect_mag = np.abs(rect_result[0])
        hann_mag = np.abs(hann_result[0])

        np.testing.assert_allclose(rect_mag.max(), 4, rtol=0.1)
        np.testing.assert_allclose(hann_mag.max(), 4, rtol=0.1)

    def test_operation_registry(self) -> None:
        """Test that FFT is properly registered in the operation registry."""
        # Verify FFT can be accessed through the registry
        assert get_operation("fft") == FFT

        # Create operation through the factory function
        fft_op = create_operation("fft", self.sample_rate, n_fft=512, window="hamming")

        # Verify the operation was created correctly
        assert isinstance(fft_op, FFT)
        assert fft_op.sampling_rate == self.sample_rate
        assert fft_op.n_fft == 512
        assert fft_op.window == "hamming"
