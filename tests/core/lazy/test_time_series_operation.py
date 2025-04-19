from typing import Callable
from unittest import mock

import dask.array as da
import librosa
import numpy as np
import pytest
from dask.array.core import Array as DaArray
from mosqito.sound_level_meter import noct_spectrum, noct_synthesis
from mosqito.sound_level_meter.noct_spectrum._center_freq import _center_freq
from scipy import fft, signal

from wandas.core.lazy.time_series_operation import (
    _OPERATION_REGISTRY,
    ABS,
    FFT,
    IFFT,
    ISTFT,
    STFT,
    AudioOperation,
    AWeighting,
    ChannelDifference,
    HighPassFilter,
    HpssHarmonic,
    HpssPercussive,
    LowPassFilter,
    Mean,
    NOctSpectrum,
    NOctSynthesis,
    Power,
    ReSampling,
    RmsTrend,
    Sum,
    Trim,
    Welch,
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
        class TestOperation(AudioOperation[NDArrayReal, NDArrayReal]):
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
        self.boundary: str | None = "zeros"

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
        )

        # Initialize ISTFT
        self.istft = ISTFT(
            sampling_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
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

        # Custom initialization
        custom_stft = STFT(
            sampling_rate=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            win_length=512,
            window="hamming",
        )
        assert custom_stft.n_fft == 1024
        assert custom_stft.hop_length == 256
        assert custom_stft.win_length == 512
        assert custom_stft.window == "hamming"

    def test_istft_initialization(self) -> None:
        """Test ISTFT initialization with different parameters."""
        # Default initialization
        istft = ISTFT(self.sample_rate)
        assert istft.sampling_rate == self.sample_rate
        assert istft.n_fft == 2048
        assert istft.hop_length == 512  # 2048 // 4
        assert istft.win_length == 2048
        assert istft.window == "hann"
        assert istft.length is None

        # Custom initialization
        custom_istft = ISTFT(
            sampling_rate=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            win_length=512,
            window="hamming",
            length=16000,
        )
        assert custom_istft.n_fft == 1024
        assert custom_istft.hop_length == 256
        assert custom_istft.win_length == 512
        assert custom_istft.window == "hamming"
        assert custom_istft.length == 16000

    def test_stft_shape_mono(self) -> None:
        """Test STFT output shape for mono signal."""
        from scipy.signal import ShortTimeFFT as ScipySTFT
        from scipy.signal import get_window

        # Process the mono signal
        stft_result = self.stft.process_array(self.signal_mono).compute()

        # Check the shape of the result
        assert stft_result.ndim == 3, (
            "Output should be 3D (channels, frequencies, time)"
        )

        # Expected shape: (channels, frequencies, time frames)
        sft = ScipySTFT(
            win=get_window(self.window, self.win_length),
            hop=self.hop_length,
            fs=self.sample_rate,
            mfft=self.n_fft,
            scale_to="magnitude",
        )
        expected_n_channels = 1
        expected_n_freqs = sft.f.shape[0]
        expected_n_frames = sft.t(self.signal_mono.shape[-1]).shape[0]

        expected_shape = (expected_n_channels, expected_n_freqs, expected_n_frames)
        assert stft_result.shape == expected_shape, (
            f"Expected {expected_shape}, got {stft_result.shape}"
        )

    def test_stft_shape_stereo(self) -> None:
        """Test STFT output shape for stereo signal."""
        from scipy.signal import ShortTimeFFT as ScipySTFT
        from scipy.signal import get_window

        # Process the stereo signal
        stft_result = self.stft.process_array(self.signal_stereo).compute()

        assert stft_result.ndim == 3, (
            "Output should be 3D (channels, frequencies, time)"
        )

        # Expected shape: (channels, frequencies, time frames)
        sft = ScipySTFT(
            win=get_window(self.window, self.win_length),
            hop=self.hop_length,
            fs=self.sample_rate,
            mfft=self.n_fft,
            scale_to="magnitude",
        )
        expected_n_channels = 2
        expected_n_freqs = sft.f.shape[0]
        expected_n_frames = sft.t(self.signal_mono.shape[-1]).shape[0]

        expected_shape = (expected_n_channels, expected_n_freqs, expected_n_frames)
        assert stft_result.shape == expected_shape, (
            f"Expected {expected_shape}, got {stft_result.shape}"
        )

    def test_stft_content(self) -> None:
        """Test STFT content correctness."""
        # Process the mono signal using the class under test
        stft_result = self.stft.process(self.signal_mono).compute()

        assert stft_result.ndim == 3, (
            "Output should be 3D (channels, frequencies, time)"
        )

        # Calculate the expected STFT using scipy.signal.ShortTimeFFT directly
        from scipy.signal import ShortTimeFFT as ScipySTFT
        from scipy.signal import get_window

        # Ensure parameters match the self.stft instance
        sft = ScipySTFT(
            win=get_window(self.window, self.win_length),
            hop=self.hop_length,
            fs=self.sample_rate,
            mfft=self.n_fft,
            scale_to="magnitude",
        )
        # Calculate STFT on the raw signal data (first channel)
        expected_stft_raw = sft.stft(self.signal_mono[0])
        expected_stft_raw[..., 1:-1, :] *= 2.0
        # Reshape scipy's output (freqs, time) to match class
        # output (channels, freqs, time)
        expected_stft = expected_stft_raw.reshape(1, *expected_stft_raw.shape)

        # Check the peak magnitude
        #  (should be close to the original amplitude 4 due to scale_to='magnitude')
        np.testing.assert_allclose(np.abs(stft_result).max(), 4, rtol=1e-5)
        # Compare the results from the class with the directly calculated scipy result
        np.testing.assert_allclose(stft_result, expected_stft, rtol=1e-5, atol=1e-5)

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
        expected_length = len(self.signal_mono[0])
        assert abs(istft_result.shape[1] - expected_length) < self.win_length

    def test_roundtrip_reconstruction(self) -> None:
        """Test signal reconstruction quality through STFT->ISTFT roundtrip."""
        # Process with STFT then ISTFT
        stft_data = self.stft.process_array(self.signal_mono)
        istft_data = self.istft.process_array(stft_data).compute()

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
        signal_1d = np.sin(
            2 * np.pi * 440 * np.linspace(0, 1, self.sample_rate, endpoint=False)
        )

        stft_result = self.stft.process_array(signal_1d).compute()

        assert stft_result.ndim == 3
        assert stft_result.shape[0] == 1

    def test_istft_2d_input_handling(self) -> None:
        """
        Test that 2D input (single channel spectrogram) is
        properly reshaped to (1, freqs, frames).
        """
        stft_data = self.stft.process_array(self.signal_mono).compute()
        stft_2d = stft_data[0]

        istft_result = self.istft.process_array(stft_2d).compute()

        assert istft_result.ndim == 2
        assert istft_result.shape[0] == 1

    def test_stft_operation_registry(self) -> None:
        """Test that STFT is properly registered in the operation registry."""
        assert get_operation("stft") == STFT
        assert get_operation("istft") == ISTFT

        stft_op = create_operation("stft", self.sample_rate, n_fft=512, hop_length=128)
        istft_op = create_operation(
            "istft", self.sample_rate, n_fft=512, hop_length=128
        )

        assert isinstance(stft_op, STFT)
        assert stft_op.n_fft == 512
        assert stft_op.hop_length == 128

        assert isinstance(istft_op, ISTFT)
        assert istft_op.n_fft == 512
        assert istft_op.hop_length == 128


class TestRmsTrend:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.frame_length: int = 1024
        self.hop_length: int = 256

        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        sine_wave = np.sin(2 * np.pi * 440 * t)

        self.expected_rms = 1.0 / np.sqrt(2)

        self.signal_mono: NDArrayReal = np.array([sine_wave])
        self.signal_stereo: NDArrayReal = np.array([sine_wave, sine_wave * 0.5])

        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=(1, 1000))
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=(1, 1000))

        self.rms_op = RmsTrend(
            sampling_rate=self.sample_rate,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            Aw=False,
        )

        self.rms_aw_op = RmsTrend(
            sampling_rate=self.sample_rate,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            Aw=True,
        )

    def test_initialization(self) -> None:
        """パラメータ初期化のテスト"""
        rms_default = RmsTrend(self.sample_rate)
        assert rms_default.sampling_rate == self.sample_rate
        assert rms_default.frame_length == 2048
        assert rms_default.hop_length == 512
        assert rms_default.Aw is False

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
        # process_arrayの代わりにprocessメソッドを使用
        result = self.rms_op.process(self.dask_mono).compute()

        expected_frames = librosa.feature.rms(
            y=self.signal_mono,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )[..., 0, :].shape[1]
        assert result.shape == (1, expected_frames)

        np.testing.assert_allclose(np.mean(result), self.expected_rms, rtol=0.1)

        result_stereo = self.rms_op.process(self.dask_stereo).compute()
        assert result_stereo.shape == (2, expected_frames)

        # 第2チャンネル（振幅0.5）のRMS値は第1チャンネルの約半分のはず
        ratio = np.mean(result_stereo[1]) / np.mean(result_stereo[0])
        np.testing.assert_allclose(ratio, 0.5, rtol=0.1)

    def test_a_weighting_effect(self) -> None:
        """A-weightingフィルタの効果をテスト"""
        # 通常のRMS計算
        result_normal = self.rms_op.process(self.dask_mono).compute()

        # A-weightingを適用したRMS計算
        result_aw = self.rms_aw_op.process(self.dask_mono).compute()

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

        assert isinstance(rms_op, RmsTrend)
        assert rms_op.sampling_rate == self.sample_rate
        assert rms_op.frame_length == 2048
        assert rms_op.hop_length == 512
        assert rms_op.Aw is True


class TestTrim:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.start_time: float = 1.0
        self.end_time: float = 2.0
        self.trim_op: Trim = Trim(self.sample_rate, self.start_time, self.end_time)

        t: NDArrayReal = np.linspace(0, 3, self.sample_rate * 3, endpoint=False)
        self.signal: NDArrayReal = np.sin(2 * np.pi * 440 * t).reshape(1, -1)

        self.dask_signal: DaArray = _da_from_array(self.signal, chunks=-1)

    def test_initialization(self) -> None:
        """Test initialization of the Trim operation."""
        assert self.trim_op.start == self.start_time
        assert self.trim_op.end == self.end_time
        assert self.trim_op.start_sample == int(self.start_time * self.sample_rate)
        assert self.trim_op.end_sample == int(self.end_time * self.sample_rate)

    def test_trim_effect(self) -> None:
        """Test that the Trim operation correctly trims the signal."""
        # process_arrayの代わりにprocessメソッドを使用
        result = self.trim_op.process(self.dask_signal)

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

            mock_compute.assert_not_called()

            assert isinstance(result, DaArray)

            _: NDArrayReal = result.compute()

            mock_compute.assert_called_once()


class TestHpssHarmonic:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.hpss_harmonic = HpssHarmonic(self.sample_rate, kernel_size=31, power=2)

        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        self.signal: NDArrayReal = np.array([np.sin(2 * np.pi * 440 * t)])

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

            mock_harmonic.assert_called_once_with(self.signal, kernel_size=31, power=2)

            np.testing.assert_array_equal(result, self.signal)

    def test_delayed_execution(self) -> None:
        """Test that HPSS Harmonic operation is executed lazily."""
        with mock.patch("dask.array.core.Array.compute") as mock_compute:
            result = self.hpss_harmonic.process(self.dask_signal)

            mock_compute.assert_not_called()

            _ = result.compute()

            mock_compute.assert_called_once()


class TestHpssPercussive:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.hpss_percussive = HpssPercussive(self.sample_rate, kernel_size=31, power=2)

        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        self.signal: NDArrayReal = np.array([np.sin(2 * np.pi * 440 * t)])

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

            mock_percussive.assert_called_once_with(
                self.signal, kernel_size=31, power=2
            )

            np.testing.assert_array_equal(result, self.signal)

    def test_delayed_execution(self) -> None:
        """Test that HPSS Percussive operation is executed lazily."""
        with mock.patch("dask.array.core.Array.compute") as mock_compute:
            result = self.hpss_percussive.process(self.dask_signal)

            mock_compute.assert_not_called()

            _ = result.compute()

            mock_compute.assert_called_once()


class TestFFTOperation:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.n_fft: int = 1024
        self.window: str = "hann"
        self.fft = FFT(self.sample_rate, n_fft=self.n_fft, window=self.window)

        self.freq: float = 500
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        self.signal_mono: NDArrayReal = (
            np.array([np.sin(2 * np.pi * self.freq * t)]) * 4
        )

        self.signal_stereo: NDArrayReal = np.array(
            [
                np.sin(2 * np.pi * self.freq * t),
                np.sin(2 * np.pi * self.freq * 2 * t),
            ]
        )

        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=-1)
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=-1)

    def test_initialization(self) -> None:
        """Test FFT initialization with different parameters."""
        fft = FFT(self.sample_rate)
        assert fft.sampling_rate == self.sample_rate
        assert fft.n_fft is None
        assert fft.window == "hann"

        custom_fft = FFT(self.sample_rate, n_fft=2048, window="hamming")
        assert custom_fft.n_fft == 2048
        assert custom_fft.window == "hamming"

    def test_fft_shape(self) -> None:
        """Test FFT output shape."""
        fft_result = self.fft.process_array(self.signal_mono).compute()

        expected_freqs = self.n_fft // 2 + 1
        assert fft_result.shape == (1, expected_freqs)

        fft_result_stereo = self.fft.process_array(self.signal_stereo).compute()
        assert fft_result_stereo.shape == (2, expected_freqs)

    def test_fft_content(self) -> None:
        """Test FFT content correctness."""
        fft_result = self.fft.process_array(self.signal_mono).compute()

        freq_bins = np.fft.rfftfreq(self.n_fft, 1.0 / self.sample_rate)

        target_idx = np.argmin(np.abs(freq_bins - self.freq))

        magnitude = np.abs(fft_result[0])

        peak_idx = np.argmax(magnitude)
        assert abs(peak_idx - target_idx) <= 1

        mask = np.ones_like(magnitude, dtype=bool)
        region = 5
        lower: int = int(max(0, peak_idx - region))
        upper: int = int(min(len(magnitude), peak_idx + region + 1))
        mask[lower:upper] = False

        assert np.max(magnitude[mask]) < 0.1 * magnitude[peak_idx]

    def test_amplitude_scaling(self) -> None:
        """Test that FFT amplitude scaling is correct."""
        fft_inst = FFT(self.sample_rate, n_fft=None, window=self.window)
        amp = 2.0
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        cos_wave = amp * np.cos(2 * np.pi * self.freq * t)

        from scipy.signal import get_window

        win = get_window(self.window, len(cos_wave))
        scaled_cos = cos_wave * win
        scaling_factor = np.sum(win)

        fft_result = fft_inst.process_array(np.array([cos_wave])).compute()

        expected_fft: NDArrayComplex = np.fft.rfft(scaled_cos)
        expected_fft[1:-1] *= 2.0
        expected_fft /= scaling_factor

        np.testing.assert_allclose(fft_result[0], expected_fft, rtol=1e-10)

        peak_idx = np.argmax(np.abs(fft_result[0]))
        peak_mag = np.abs(fft_result[0, peak_idx])
        expected_mag = amp

        np.testing.assert_allclose(peak_mag, expected_mag, rtol=0.1)

    def test_delayed_execution(self) -> None:
        """Test that FFT operation uses dask's delayed execution."""
        with mock.patch.object(DaArray, "compute") as mock_compute:
            result = self.fft.process(self.dask_mono)
            mock_compute.assert_not_called()

            assert isinstance(result, DaArray)

            _ = result.compute()
            mock_compute.assert_called_once()

    def test_window_function_effect(self) -> None:
        """Test different window functions have different effects."""
        rect_fft = FFT(self.sample_rate, n_fft=None, window="boxcar")
        rect_result = rect_fft.process_array(self.signal_mono).compute()

        hann_fft = FFT(self.sample_rate, n_fft=None, window="hann")
        hann_result = hann_fft.process_array(self.signal_mono).compute()

        assert not np.allclose(rect_result, hann_result)

        rect_mag = np.abs(rect_result[0])
        hann_mag = np.abs(hann_result[0])

        np.testing.assert_allclose(rect_mag.max(), 4, rtol=0.1)
        np.testing.assert_allclose(hann_mag.max(), 4, rtol=0.1)

    def test_operation_registry(self) -> None:
        """Test that FFT is properly registered in the operation registry."""
        assert get_operation("fft") == FFT

        fft_op = create_operation("fft", self.sample_rate, n_fft=512, window="hamming")

        assert isinstance(fft_op, FFT)
        assert fft_op.sampling_rate == self.sample_rate
        assert fft_op.n_fft == 512
        assert fft_op.window == "hamming"


class TestIFFTOperation:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.n_fft: int = 1024
        self.window: str = "hann"

        # Create frequency domain signal
        freq_bins = np.fft.rfftfreq(self.n_fft, 1.0 / self.sample_rate)
        f0 = 500.0  # Target frequency
        target_idx = np.argmin(np.abs(freq_bins - f0))

        # Create complex spectrum with single peak at f0
        spectrum = np.zeros(self.n_fft // 2 + 1, dtype=complex)
        spectrum[target_idx] = 1.0

        # For stereo test
        spectrum2 = np.zeros(self.n_fft // 2 + 1, dtype=complex)
        spectrum2[target_idx // 2] = 1.0

        self.signal_mono: NDArrayComplex = np.array([spectrum])
        self.signal_stereo: NDArrayComplex = np.array([spectrum, spectrum2])

        # Create dask arrays
        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=-1)
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=-1)

        # Initialize IFFT
        self.ifft = IFFT(self.sample_rate, n_fft=self.n_fft, window=self.window)

    def test_initialization(self) -> None:
        """Test IFFT initialization with different parameters."""
        ifft = IFFT(self.sample_rate)
        assert ifft.sampling_rate == self.sample_rate
        assert ifft.n_fft is None
        assert ifft.window == "hann"

        custom_ifft = IFFT(self.sample_rate, n_fft=2048, window="hamming")
        assert custom_ifft.n_fft == 2048
        assert custom_ifft.window == "hamming"

    def test_ifft_shape(self) -> None:
        """Test IFFT output shape."""
        ifft_result = self.ifft.process_array(self.signal_mono).compute()

        # Expected time domain signal length
        expected_length = self.n_fft
        assert ifft_result.shape == (1, expected_length)

        ifft_result_stereo = self.ifft.process_array(self.signal_stereo).compute()
        assert ifft_result_stereo.shape == (2, expected_length)

    def test_ifft_content(self) -> None:
        """Test that IFFT properly transforms frequency domain to time domain."""
        # Process the mono signal
        ifft_result = self.ifft.process_array(self.signal_mono).compute()

        # Check that the result is real
        assert np.isrealobj(ifft_result)

        # For a single frequency component, we expect a sinusoidal time signal
        # Find the peak frequency in the time domain by FFT
        fft_of_result = np.fft.rfft(ifft_result[0])
        peak_idx = np.argmax(np.abs(fft_of_result))
        freq_bins = np.fft.rfftfreq(len(ifft_result[0]), 1.0 / self.sample_rate)
        detected_freq = freq_bins[peak_idx]

        # Check frequency matches our input
        np.testing.assert_allclose(detected_freq, 500.0, rtol=1e-1)

    def test_delayed_execution(self) -> None:
        """Test that IFFT operation uses dask's delayed execution."""
        with mock.patch.object(DaArray, "compute") as mock_compute:
            result = self.ifft.process(self.dask_mono)
            mock_compute.assert_not_called()

            assert isinstance(result, DaArray)

            _ = result.compute()
            mock_compute.assert_called_once()

    def test_1d_input_handling(self) -> None:
        """Test that 1D input is properly reshaped."""
        signal_1d = np.zeros((1, self.n_fft // 2 + 1), dtype=complex)
        signal_1d[0, 5] = 1.0  # Add a frequency component
        dask_signal_1d: DaArray = _da_from_array(signal_1d, chunks=-1)

        ifft_result = self.ifft.process(dask_signal_1d).compute()

        assert ifft_result.ndim == 2
        assert ifft_result.shape[0] == 1

    def test_operation_registry(self) -> None:
        """Test that IFFT is properly registered in the operation registry."""
        assert get_operation("ifft") == IFFT

        ifft_op = create_operation(
            "ifft", self.sample_rate, n_fft=512, window="hamming"
        )

        assert isinstance(ifft_op, IFFT)
        assert ifft_op.sampling_rate == self.sample_rate
        assert ifft_op.n_fft == 512
        assert ifft_op.window == "hamming"


class TestReSamplingOperation:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.orig_sr: int = 16000
        self.target_sr: int = 8000
        self.resampling = ReSampling(self.orig_sr, self.target_sr)

        # Create a simple sine wave
        t = np.linspace(0, 1, self.orig_sr, endpoint=False)
        self.signal_mono: NDArrayReal = np.array([np.sin(2 * np.pi * 440 * t)])
        self.signal_stereo: NDArrayReal = np.array(
            [np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 880 * t)]
        )

        # Create dask arrays
        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=(1, 1000))
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=(2, 1000))

    def test_initialization(self) -> None:
        """Test initialization with parameters."""
        resampling = ReSampling(self.orig_sr, self.target_sr)
        assert resampling.sampling_rate == self.orig_sr
        assert resampling.target_sr == self.target_sr

    def test_resampling_effect(self) -> None:
        """Test that resampling changes the signal length appropriately."""
        result = self.resampling.process_array(self.signal_mono).compute()

        # Check the shape - should be half the original length
        expected_length = int(
            self.signal_mono.shape[1] * (self.target_sr / self.orig_sr)
        )
        assert result.shape == (1, expected_length)

        # Test with stereo signal
        result_stereo = self.resampling.process_array(self.signal_stereo).compute()
        assert result_stereo.shape == (2, expected_length)

    def test_resampling_output_sample_rate(self) -> None:
        """Test that the output signal contains the expected frequency content."""
        # Testing with a specific frequency
        freq = 440.0
        duration = 1.0
        t_orig = np.linspace(0, duration, int(self.orig_sr * duration), endpoint=False)
        signal = np.array([np.sin(2 * np.pi * freq * t_orig)])

        # Resample
        resampled = self.resampling.process_array(signal).compute()

        # Generate expected signal directly at target sample rate
        t_target = np.linspace(
            0, duration, int(self.target_sr * duration), endpoint=False
        )
        expected = np.sin(2 * np.pi * freq * t_target)

        # Compare first channel waveforms - they should match closely
        np.testing.assert_allclose(resampled[0][:100], expected[:100], atol=0.1)

    def test_delayed_execution(self) -> None:
        """Test that resampling operation is executed lazily."""
        with mock.patch.object(DaArray, "compute") as mock_compute:
            result = self.resampling.process(self.dask_mono)
            mock_compute.assert_not_called()

            _ = result.compute()
            mock_compute.assert_called_once()

    def test_output_shape_calculation(self) -> None:
        """Test that the output shape calculation is correct."""
        input_shape = (2, 16000)  # 2 channels, 1 second at 16kHz
        expected_output_shape = (2, 8000)  # 2 channels, 1 second at 8kHz

        output_shape = self.resampling.calculate_output_shape(input_shape)
        assert output_shape == expected_output_shape

    def test_operation_registry(self) -> None:
        """Test that ReSampling is properly registered in the operation registry."""
        assert get_operation("resampling") == ReSampling

        resampling_op = create_operation("resampling", self.orig_sr, target_sr=44100)

        assert isinstance(resampling_op, ReSampling)
        assert resampling_op.sampling_rate == self.orig_sr
        assert resampling_op.target_sr == 44100


class TestABSOperation:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.abs_op = ABS(self.sample_rate)

        # Create a test signal with positive and negative values
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        sine_wave = np.sin(2 * np.pi * 440 * t)

        self.signal_mono: NDArrayReal = np.array([sine_wave])
        self.signal_stereo: NDArrayReal = np.array([sine_wave, -0.5 * sine_wave])

        # Create dask arrays
        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=(1, 1000))
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=(2, 1000))

    def test_initialization(self) -> None:
        """Test initialization."""
        abs_op = ABS(self.sample_rate)
        assert abs_op.sampling_rate == self.sample_rate

    def test_abs_effect(self) -> None:
        """Test that the absolute value operation works correctly."""
        result = self.abs_op.process(self.dask_mono).compute()

        # Check shape preservation
        assert result.shape == self.signal_mono.shape

        # Check that all values are non-negative
        assert np.all(result >= 0)

        # Check that the operation correctly takes absolute values
        np.testing.assert_allclose(result, np.abs(self.signal_mono))

        # Test with stereo signal
        result_stereo = self.abs_op.process(self.dask_stereo).compute()
        assert result_stereo.shape == self.signal_stereo.shape
        assert np.all(result_stereo >= 0)
        np.testing.assert_allclose(result_stereo, np.abs(self.signal_stereo))

    def test_delayed_execution(self) -> None:
        """Test that ABS operation is executed lazily."""
        with mock.patch.object(DaArray, "compute") as mock_compute:
            result = self.abs_op.process(self.dask_mono)
            mock_compute.assert_not_called()

            _ = result.compute()
            mock_compute.assert_called_once()

    def test_operation_registry(self) -> None:
        """Test that ABS is properly registered in the operation registry."""
        assert get_operation("abs") == ABS

        abs_op = create_operation("abs", self.sample_rate)

        assert isinstance(abs_op, ABS)
        assert abs_op.sampling_rate == self.sample_rate


class TestPowerOperation:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.exponent: float = 2.0
        self.power_op = Power(self.sample_rate, self.exponent)

        # Create a test signal
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        sine_wave = 0.5 * np.sin(2 * np.pi * 440 * t)

        self.signal_mono: NDArrayReal = np.array([sine_wave])
        self.signal_stereo: NDArrayReal = np.array([sine_wave, 0.25 * sine_wave])

        # Create dask arrays
        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=(1, 1000))
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=(2, 1000))

    def test_initialization(self) -> None:
        """Test initialization with parameters."""
        power_op = Power(self.sample_rate, self.exponent)
        assert power_op.sampling_rate == self.sample_rate
        assert power_op.exp == self.exponent

    def test_power_effect(self) -> None:
        """Test that the power operation works correctly."""
        result = self.power_op.process(self.dask_mono).compute()

        # Check shape preservation
        assert result.shape == self.signal_mono.shape

        # Check that the operation correctly computes powers
        np.testing.assert_allclose(result, np.power(self.signal_mono, self.exponent))

        # Test with stereo signal
        result_stereo = self.power_op.process(self.dask_stereo).compute()
        assert result_stereo.shape == self.signal_stereo.shape
        np.testing.assert_allclose(
            result_stereo, np.power(self.signal_stereo, self.exponent)
        )

    def test_different_exponents(self) -> None:
        """Test with different exponent values."""
        # Square root
        sqrt_op = Power(self.sample_rate, 0.5)
        sqrt_result = sqrt_op.process(self.dask_mono).compute()
        np.testing.assert_allclose(sqrt_result, np.power(self.signal_mono, 0.5))

        # Cube
        cube_op = Power(self.sample_rate, 3.0)
        cube_result = cube_op.process(self.dask_stereo).compute()
        np.testing.assert_allclose(cube_result, np.power(self.signal_stereo, 3.0))

    def test_delayed_execution(self) -> None:
        """Test that Power operation is executed lazily."""
        with mock.patch.object(DaArray, "compute") as mock_compute:
            result = self.power_op.process(self.dask_mono)
            mock_compute.assert_not_called()

            _ = result.compute()
            mock_compute.assert_called_once()

    def test_operation_registry(self) -> None:
        """Test that Power is properly registered in the operation registry."""
        assert get_operation("power") == Power

        power_op = create_operation("power", self.sample_rate, exponent=3.0)

        assert isinstance(power_op, Power)
        assert power_op.sampling_rate == self.sample_rate
        assert power_op.exp == 3.0


class TestWelchOperation:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.n_fft: int = 1024
        self.hop_length: int = 256
        self.win_length: int = 1024
        self.window: str = "hann"
        self.average: str = "mean"
        self.detrend: str = "constant"

        self.welch = Welch(
            sampling_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            average=self.average,
            detrend=self.detrend,
        )

        # Create a test signal with a known frequency
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        self.freq = 1000.0
        self.signal_mono: NDArrayReal = np.array([np.sin(2 * np.pi * self.freq * t)])
        self.signal_stereo: NDArrayReal = np.array(
            [np.sin(2 * np.pi * self.freq * t), np.sin(2 * np.pi * 2000 * t)]
        )

        # Create dask arrays
        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=(1, 1000))
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=(2, 1000))

    def test_initialization(self) -> None:
        """Test initialization with different parameters."""
        # Default initialization
        welch = Welch(self.sample_rate)
        assert welch.sampling_rate == self.sample_rate
        assert welch.n_fft == 2048
        assert welch.win_length == 2048
        assert welch.hop_length == 512  # 2048 // 4
        assert welch.window == "hann"
        assert welch.average == "mean"
        assert welch.detrend == "constant"

        # Custom initialization
        custom_welch = Welch(
            sampling_rate=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            win_length=512,
            window="hamming",
            average="median",
            detrend="linear",
        )
        assert custom_welch.n_fft == 1024
        assert custom_welch.win_length == 512
        assert custom_welch.hop_length == 256
        assert custom_welch.window == "hamming"
        assert custom_welch.average == "median"
        assert custom_welch.detrend == "linear"

    def test_welch_shape(self) -> None:
        """Test Welch output shape."""
        result = self.welch.process_array(self.signal_mono).compute()

        # Expected frequency bins
        expected_bins = self.n_fft // 2 + 1
        assert result.shape == (1, expected_bins)

        # Test with stereo signal
        result_stereo = self.welch.process_array(self.signal_stereo).compute()
        assert result_stereo.shape == (2, expected_bins)

    def test_welch_content(self) -> None:
        """Test that Welch correctly identifies frequency content."""
        result = self.welch.process_array(self.signal_mono).compute()

        # Get frequency bins
        freq_bins = np.fft.rfftfreq(self.n_fft, 1.0 / self.sample_rate)

        # Find the peak frequency
        peak_idx = np.argmax(result[0])
        detected_freq = freq_bins[peak_idx]

        # Check that the detected frequency is close to the actual frequency
        np.testing.assert_allclose(detected_freq, self.freq, rtol=0.05)

        # For stereo signal, check second channel
        result_stereo = self.welch.process_array(self.signal_stereo).compute()
        peak_idx_ch2 = np.argmax(result_stereo[1])
        detected_freq_ch2 = freq_bins[peak_idx_ch2]

        # Second channel should show peak at 2000 Hz
        np.testing.assert_allclose(detected_freq_ch2, 2000.0, rtol=0.05)

    def test_delayed_execution(self) -> None:
        """Test that Welch operation uses dask's delayed execution."""
        with mock.patch.object(DaArray, "compute") as mock_compute:
            result = self.welch.process(self.dask_mono)
            mock_compute.assert_not_called()

            assert isinstance(result, DaArray)

            _ = result.compute()
            mock_compute.assert_called_once()

    def test_operation_registry(self) -> None:
        """Test that Welch is properly registered in the operation registry."""
        assert get_operation("welch") == Welch

        welch_op = create_operation(
            "welch",
            self.sample_rate,
            n_fft=512,
            win_length=512,
            hop_length=128,
            window="hamming",
        )

        assert isinstance(welch_op, Welch)
        assert welch_op.sampling_rate == self.sample_rate
        assert welch_op.n_fft == 512
        assert welch_op.win_length == 512
        assert welch_op.hop_length == 128
        assert welch_op.window == "hamming"


class TestNOctSpectrumOperation:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 51200
        self.fmin: float = 24.0
        self.fmax: float = 12600
        self.n: int = 3
        self.G: int = 10
        self.fr: int = 1000

        self.noct_spectrum = NOctSpectrum(
            sampling_rate=self.sample_rate,
            fmin=self.fmin,
            fmax=self.fmax,
            n=self.n,
            G=self.G,
            fr=self.fr,
        )

        # Create a test signal with pink noise
        np.random.seed(42)  # For reproducibility
        white_noise = np.random.randn(self.sample_rate)

        # Simple approximation of pink noise by filtering white noise
        k = np.fft.rfftfreq(len(white_noise))[1:]
        X = np.fft.rfft(white_noise)  # noqa: N806
        S = 1.0 / np.sqrt(k)  # noqa: N806
        X[1:] *= S
        pink_noise = np.fft.irfft(X, len(white_noise))
        pink_noise /= np.abs(pink_noise).max()  # Normalize

        self.signal_mono: NDArrayReal = np.array([pink_noise])
        self.signal_stereo: NDArrayReal = np.array(
            [pink_noise, white_noise / np.abs(white_noise).max()]
        )

        # Create dask arrays
        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=(1, -1))
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=(2, -1))

    def test_initialization(self) -> None:
        """Test initialization with parameters."""
        noct = NOctSpectrum(
            self.sample_rate,
            fmin=self.fmin,
            fmax=self.fmax,
            n=self.n,
            G=self.G,
            fr=self.fr,
        )
        assert noct.sampling_rate == self.sample_rate
        assert noct.fmin == self.fmin
        assert noct.fmax == self.fmax
        assert noct.n == self.n
        assert noct.G == self.G
        assert noct.fr == self.fr

    def test_noct_spectrum_shape(self) -> None:
        """外部ライブラリのnoct_spectrum関数が正確にラップされているかテスト"""
        # 実際にnoct_spectrum関数を実行
        result = self.noct_spectrum.process(self.dask_mono).compute()

        # 外部ライブラリを直接呼び出した場合の結果を取得
        expected_spectrum, expected_freqs = noct_spectrum(
            sig=self.signal_mono.T,
            fs=self.sample_rate,
            fmin=self.fmin,
            fmax=self.fmax,
            n=self.n,
            G=self.G,
            fr=self.fr,
        )

        # 形状が一致することを確認
        assert result.shape == (1, expected_spectrum.shape[0])

        # 結果が一致することを確認
        np.testing.assert_allclose(result[0], expected_spectrum, rtol=1e-6)

        # 周波数帯域数が適切かチェック
        _, center_freqs = _center_freq(
            fmin=self.fmin, fmax=self.fmax, n=self.n, G=self.G, fr=self.fr
        )
        assert result.shape[1] == len(center_freqs)

    def test_noct_spectrum_stereo(self) -> None:
        """ステレオ信号でnoct_spectrum関数が正確にラップされているかテスト"""
        # ステレオ信号用のテスト
        result = self.noct_spectrum.process(self.dask_stereo).compute()

        # 外部ライブラリを直接呼び出した場合の結果を取得（第1チャンネル）
        expected_spectrum_ch1, expected_freqs = noct_spectrum(
            sig=self.signal_stereo[0:1].T,  # 第1チャンネルのみ
            fs=self.sample_rate,
            fmin=self.fmin,
            fmax=self.fmax,
            n=self.n,
            G=self.G,
            fr=self.fr,
        )

        # 外部ライブラリを直接呼び出した場合の結果を取得（第2チャンネル）
        expected_spectrum_ch2, _ = noct_spectrum(
            sig=self.signal_stereo[1:2].T,  # 第2チャンネルのみ
            fs=self.sample_rate,
            fmin=self.fmin,
            fmax=self.fmax,
            n=self.n,
            G=self.G,
            fr=self.fr,
        )

        # 形状が正しいことを確認（チャンネル数 x 周波数帯域数）
        assert result.shape == (2, expected_spectrum_ch1.shape[0])

        # 第1チャンネルの結果が一致することを確認
        np.testing.assert_allclose(result[0], expected_spectrum_ch1, rtol=1e-6)

        # 第2チャンネルの結果が一致することを確認
        np.testing.assert_allclose(result[1], expected_spectrum_ch2, rtol=1e-6)

        # 白色ノイズと有色ノイズのスペクトルが異なることを確認
        # （第1チャンネルはピンクノイズ、第2チャンネルは白色ノイズ）
        assert not np.allclose(result[0], result[1])

    def test_delayed_execution(self) -> None:
        """Test that NOctSpectrum operation uses dask's delayed execution."""
        with mock.patch.object(DaArray, "compute") as mock_compute:
            result = self.noct_spectrum.process(self.dask_mono)
            mock_compute.assert_not_called()

            _ = result.compute()
            mock_compute.assert_called_once()

    def test_operation_registry(self) -> None:
        """Test that NOctSpectrum is properly registered in the operation registry."""
        assert get_operation("noct_spectrum") == NOctSpectrum

        noct_op = create_operation(
            "noct_spectrum",
            self.sample_rate,
            fmin=100.0,
            fmax=5000.0,
            n=1,
            G=20,
            fr=1000,
        )

        assert isinstance(noct_op, NOctSpectrum)
        assert noct_op.sampling_rate == self.sample_rate
        assert noct_op.fmin == 100.0
        assert noct_op.fmax == 5000.0
        assert noct_op.n == 1
        assert noct_op.G == 20
        assert noct_op.fr == 1000


class TestNOctSynthesisOperation:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 48000
        self.fmin: float = 24.0
        self.fmax: float = 12600
        self.n: int = 3
        self.G: int = 10
        self.fr: int = 1000

        self.noct_synthesis = NOctSynthesis(
            sampling_rate=self.sample_rate,
            fmin=self.fmin,
            fmax=self.fmax,
            n=self.n,
            G=self.G,
            fr=self.fr,
        )

        # Create a test signal with pink noise
        np.random.seed(42)  # For reproducibility
        white_noise = np.random.randn(self.sample_rate)

        # Simple approximation of pink noise by filtering white noise
        k = np.fft.rfftfreq(len(white_noise))[1:]
        X = np.fft.rfft(white_noise)  # noqa: N806
        S = 1.0 / np.sqrt(k)  # noqa: N806
        X[1:] *= S
        pink_noise = np.fft.irfft(X, len(white_noise))
        pink_noise /= np.abs(pink_noise).max()  # Normalize

        self.signal_mono: NDArrayReal = np.array([pink_noise])
        self.signal_stereo: NDArrayReal = np.array(
            [pink_noise, white_noise / np.abs(white_noise).max()]
        )

        # Create dask arrays
        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=(1, 1000))
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=(2, 1000))

    def test_initialization(self) -> None:
        """Test initialization with parameters."""
        noct = NOctSynthesis(
            self.sample_rate,
            fmin=self.fmin,
            fmax=self.fmax,
            n=self.n,
            G=self.G,
            fr=self.fr,
        )
        assert noct.sampling_rate == self.sample_rate
        assert noct.fmin == self.fmin
        assert noct.fmax == self.fmax
        assert noct.n == self.n
        assert noct.G == self.G
        assert noct.fr == self.fr

    def test_noct_synthesis_integration(self) -> None:
        """外部ライブラリのnoct_synthesis関数が正確にラップされているかテスト"""
        # スペクトル計算に必要なデータを用意
        # まずfftでスペクトルを計算
        fft = FFT(self.sample_rate, n_fft=None, window="hann")
        spectrum = fft.process(self.dask_mono).compute()

        # NOctSynthesisによる合成
        result = self.noct_synthesis.process(spectrum).compute()

        # 外部ライブラリを直接呼び出した場合の結果
        # Note: NOctSynthesisのprocess_array内の処理を再現
        n = spectrum.shape[-1]
        if n % 2 == 0:
            n = n * 2 - 1
        else:
            n = (n - 1) * 2
        freqs = np.fft.rfftfreq(n, d=1 / self.sample_rate)

        expected_signal, expected_freqs = noct_synthesis(
            spectrum=np.abs(spectrum).T,
            freqs=freqs,
            fmin=self.fmin,
            fmax=self.fmax,
            n=self.n,
            G=self.G,
            fr=self.fr,
        )
        expected_signal = expected_signal.T

        # 形状が一致することを確認
        assert result.shape == expected_signal.shape

        # 結果が一致することを確認
        np.testing.assert_allclose(result[0], expected_signal[0])

    def test_noct_synthesis_stereo(self) -> None:
        """ステレオ信号でnoct_synthesis関数が正確にラップされているかテスト"""
        # FFTでスペクトル計算するための準備
        fft = FFT(self.sample_rate, n_fft=None, window="hann")

        # ステレオ信号のスペクトルを計算
        stereo_spectrum = fft.process(self.dask_stereo).compute()

        # NOctSynthesisによる合成
        result = self.noct_synthesis.process(stereo_spectrum).compute()

        # 外部ライブラリを直接呼び出した場合の結果
        n = stereo_spectrum.shape[-1]
        if n % 2 == 0:
            n = n * 2 - 1
        else:
            n = (n - 1) * 2
        freqs = np.fft.rfftfreq(n, d=1 / self.sample_rate)

        # 第1チャンネル
        expected_signal_ch1, expected_freqs = noct_synthesis(
            spectrum=np.abs(stereo_spectrum[0:1]).T,
            freqs=freqs,
            fmin=self.fmin,
            fmax=self.fmax,
            n=self.n,
            G=self.G,
            fr=self.fr,
        )
        expected_signal_ch1 = expected_signal_ch1.T

        # 第2チャンネル
        expected_signal_ch2, _ = noct_synthesis(
            spectrum=np.abs(stereo_spectrum[1:2]).T,
            freqs=freqs,
            fmin=self.fmin,
            fmax=self.fmax,
            n=self.n,
            G=self.G,
            fr=self.fr,
        )
        expected_signal_ch2 = expected_signal_ch2.T

        # 結果の形状を確認
        assert result.shape == (2, expected_signal_ch1.shape[1])

        # 各チャンネルの結果が外部ライブラリの結果と一致するか確認
        np.testing.assert_allclose(
            result[0], expected_signal_ch1[0], rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            result[1], expected_signal_ch2[0], rtol=1e-5, atol=1e-5
        )

        # ピンクノイズと白色ノイズのチャンネルで異なる結果が出ることを確認
        # 完全に異なるスペクトルから合成したので、結果も異なるはず
        assert not np.allclose(result[0], result[1])

    def test_delayed_execution(self) -> None:
        """Test that NOctSynthesis operation uses dask's delayed execution."""
        with mock.patch.object(DaArray, "compute") as mock_compute:
            with mock.patch(
                "wandas.core.lazy.time_series_operation.noct_synthesis"
            ) as mock_noct:
                # Create a dummy result for the mock
                dummy_signal = np.zeros((1, self.sample_rate))
                dummy_result = (dummy_signal, np.zeros(27))
                mock_noct.return_value = dummy_result

                result = self.noct_synthesis.process(self.dask_mono)
                mock_compute.assert_not_called()

                _ = result.compute()
                mock_compute.assert_called_once()

    def test_operation_registry(self) -> None:
        """Test that NOctSynthesis is properly registered in the operation registry."""
        assert get_operation("noct_synthesis") == NOctSynthesis

        noct_op = create_operation(
            "noct_synthesis",
            self.sample_rate,
            fmin=100.0,
            fmax=5000.0,
            n=1,
            G=20,
            fr=1000,
        )

        assert isinstance(noct_op, NOctSynthesis)
        assert noct_op.sampling_rate == self.sample_rate
        assert noct_op.fmin == 100.0
        assert noct_op.fmax == 5000.0
        assert noct_op.n == 1
        assert noct_op.G == 20
        assert noct_op.fr == 1000


class TestSumOperation:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.sum_op = Sum(self.sample_rate)

        # Create a test signal
        self.signal_mono: NDArrayReal = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        self.signal_stereo: NDArrayReal = np.array(
            [[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]]
        )

        # Create dask arrays
        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=(1, -1))
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=(2, -1))

    def test_initialization(self) -> None:
        """Test initialization."""
        sum_op = Sum(self.sample_rate)
        assert sum_op.sampling_rate == self.sample_rate

    def test_sum_effect(self) -> None:
        """Test that the Sum operation works correctly."""
        result = self.sum_op.process(self.dask_mono).compute()

        # Expected shape: (1, 5)
        assert result.shape == (1, 5)

        # Check the sum value
        assert result[0, 0] == 1.0

        # Test with stereo signal
        result_stereo = self.sum_op.process(self.dask_stereo).compute()
        assert result_stereo.shape == (1, 5)
        assert result_stereo[0, 0] == 6.0
        assert result_stereo[0, 1] == 6.0

    def test_delayed_execution(self) -> None:
        """Test that Sum operation is executed lazily."""
        with mock.patch.object(DaArray, "compute") as mock_compute:
            result = self.sum_op.process(self.dask_mono)
            mock_compute.assert_not_called()
            assert result.shape == (1, 5)
            _ = result.compute()
            mock_compute.assert_called_once()

    def test_operation_registry(self) -> None:
        """Test that Sum is properly registered in the operation registry."""
        assert get_operation("sum") == Sum

        sum_op = create_operation("sum", self.sample_rate)

        assert isinstance(sum_op, Sum)
        assert sum_op.sampling_rate == self.sample_rate


class TestMeanOperation:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.mean_op = Mean(self.sample_rate)

        # Create a test signal
        self.signal_mono: NDArrayReal = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        self.signal_stereo: NDArrayReal = np.array(
            [[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]]
        )

        # Create dask arrays
        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=(1, -1))
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=(2, -1))

    def test_initialization(self) -> None:
        """Test initialization."""
        mean_op = Mean(self.sample_rate)
        assert mean_op.sampling_rate == self.sample_rate

    def test_mean_effect(self) -> None:
        """Test that the Mean operation works correctly."""
        result = self.mean_op.process(self.dask_mono).compute()

        # Expected shape: (1, 5)
        assert result.shape == (1, 5)

        # Check the mean value
        assert result[0, 0] == 1.0

        # Test with stereo signal
        result_stereo = self.mean_op.process(self.dask_stereo).compute()
        assert result_stereo.shape == (1, 5)
        assert result_stereo[0, 0] == 3.0
        assert result_stereo[0, 1] == 3.0

    def test_delayed_execution(self) -> None:
        """Test that Mean operation is executed lazily."""
        with mock.patch.object(DaArray, "compute") as mock_compute:
            result = self.mean_op.process(self.dask_mono)
            mock_compute.assert_not_called()
            assert result.shape == (1, 5)
            _ = result.compute()
            mock_compute.assert_called_once()

    def test_operation_registry(self) -> None:
        """Test that Mean is properly registered in the operation registry."""
        assert get_operation("mean") == Mean

        mean_op = create_operation("mean", self.sample_rate)

        assert isinstance(mean_op, Mean)
        assert mean_op.sampling_rate == self.sample_rate


class TestChannelDifferenceOperation:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.ch_diff = ChannelDifference(self.sample_rate)

        # Create test signals
        self.signal_mono: NDArrayReal = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        self.signal_stereo: NDArrayReal = np.array(
            [[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]]
        )
        self.signal_multi: NDArrayReal = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [5.0, 4.0, 3.0, 2.0, 1.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
            ]
        )

        # Create dask arrays
        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=(1, -1))
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=(2, -1))
        self.dask_multi: DaArray = _da_from_array(self.signal_multi, chunks=(3, -1))

    def test_initialization(self) -> None:
        """Test initialization."""
        ch_diff = ChannelDifference(self.sample_rate)
        assert ch_diff.sampling_rate == self.sample_rate

    def test_channel_difference_effect(self) -> None:
        """Test that the channel difference operation works correctly."""
        # For mono channel, should return zeros
        result_mono = self.ch_diff.process(self.dask_mono).compute()
        assert result_mono.shape == (1, 5)
        np.testing.assert_array_equal(result_mono, np.zeros_like(self.signal_mono))

        # For stereo, should compute [ch0 - ch0, ch0 - ch1]
        result_stereo = self.ch_diff.process(self.dask_stereo).compute()
        expected_stereo = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [5.0 - 1.0, 4.0 - 2.0, 3.0 - 3.0, 2.0 - 4.0, 1.0 - 5.0],
            ]
        )
        assert result_stereo.shape == (2, 5)
        np.testing.assert_array_equal(result_stereo, expected_stereo)

        # For multi-channel, should return ch0 - ch1
        result_multi = self.ch_diff.process(self.dask_multi).compute()
        expected_multi = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [5.0 - 1.0, 4.0 - 2.0, 3.0 - 3.0, 2.0 - 4.0, 1.0 - 5.0],
                [0.0 - 1.0, 1.0 - 2.0, 2.0 - 3.0, 3.0 - 4.0, 4.0 - 5.0],
            ]
        )
        assert result_multi.shape == (3, 5)
        np.testing.assert_array_equal(result_multi, expected_multi)

    def test_delayed_execution(self) -> None:
        """Test that ChannelDifference operation is executed lazily."""
        with mock.patch.object(DaArray, "compute") as mock_compute:
            result = self.ch_diff.process(self.dask_stereo)
            mock_compute.assert_not_called()
            assert result.shape == (2, 5)
            _ = result.compute()
            mock_compute.assert_called_once()

    def test_operation_registry(self) -> None:
        """
        Test that ChannelDifference is properly registered in the operation registry.
        """
        assert get_operation("channel_difference") == ChannelDifference

        ch_diff_op = create_operation("channel_difference", self.sample_rate)

        assert isinstance(ch_diff_op, ChannelDifference)
        assert ch_diff_op.sampling_rate == self.sample_rate
