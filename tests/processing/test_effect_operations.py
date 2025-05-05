import dask.array as da
import numpy as np
from dask.array.core import Array as DaArray

from wandas.processing.base import create_operation, get_operation
from wandas.processing.effects import (
    AddWithSNR,
    HpssHarmonic,
    HpssPercussive,
)
from wandas.utils.types import NDArrayReal

_da_from_array = da.from_array  # type: ignore [unused-ignore]


class TestHpssHarmonic:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.hpss_h = HpssHarmonic(self.sample_rate)

        # Create sample signal: 1 second mixed harmonic/percussive content
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        # Harmonic content (sine waves)
        harmonic = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
        # Percussive content (short impulses)
        percussive = np.zeros_like(t)
        impulse_locations = np.arange(0, self.sample_rate, self.sample_rate // 8)
        percussive[impulse_locations] = 1.0

        self.mixed_signal: NDArrayReal = np.array([harmonic + percussive])
        self.dask_signal: DaArray = _da_from_array(self.mixed_signal, chunks=-1)

    def test_initialization(self) -> None:
        """Test initialization with different parameters."""
        hpss = HpssHarmonic(self.sample_rate)
        assert hpss.sampling_rate == self.sample_rate

        hpss_custom = HpssHarmonic(self.sample_rate, margin=2.0)
        assert hpss_custom.kwargs.get("margin") == 2.0

    def test_shape_preservation(self) -> None:
        """Test output shape matches input shape."""
        result = self.hpss_h.process(self.dask_signal).compute()
        assert result.shape == self.mixed_signal.shape

    def test_harmonic_extraction(self) -> None:
        """Test that harmonic content is preserved and percussive content is reduced."""
        result = self.hpss_h.process(self.dask_signal).compute()

        # Compute energy in frequency bands
        n_fft = 2048

        # Original signal
        orig_spec = np.abs(np.fft.rfft(self.mixed_signal[0], n_fft))

        # Processed signal
        result_spec = np.abs(np.fft.rfft(result[0], n_fft))

        # Check pitch continuity by comparing spectral flux
        from librosa.feature import spectral_contrast

        orig_contrast = spectral_contrast(y=self.mixed_signal[0], sr=self.sample_rate)
        result_contrast = spectral_contrast(y=result[0], sr=self.sample_rate)

        # Harmonic extraction should increase spectral contrast
        assert np.mean(result_contrast) > np.mean(orig_contrast)

        # For harmonic content, energy should be concentrated
        # around fundamental frequencies and their harmonics,
        # resulting in a more peaky spectrum
        orig_flatness = np.exp(np.mean(np.log(orig_spec + 1e-10))) / np.mean(orig_spec)
        result_flatness = np.exp(np.mean(np.log(result_spec + 1e-10))) / np.mean(
            result_spec
        )

        # Lower flatness indicates more harmonic (less noise-like) content
        assert result_flatness < orig_flatness

    def test_operation_registry(self) -> None:
        """Test that HpssHarmonic is properly registered in the operation registry."""
        assert get_operation("hpss_harmonic") == HpssHarmonic

        hpss_op = create_operation("hpss_harmonic", 16000, margin=3.0)

        assert isinstance(hpss_op, HpssHarmonic)
        assert hpss_op.sampling_rate == 16000
        assert hpss_op.kwargs.get("margin") == 3.0


class TestHpssPercussive:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.hpss_p = HpssPercussive(self.sample_rate)

        # Create sample signal: 1 second mixed harmonic/percussive content
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        # Harmonic content (sine waves)
        harmonic = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
        # Percussive content (short impulses)
        percussive = np.zeros_like(t)
        impulse_locations = np.arange(0, self.sample_rate, self.sample_rate // 8)
        percussive[impulse_locations] = 1.0

        self.mixed_signal: NDArrayReal = np.array([harmonic + percussive])
        self.dask_signal: DaArray = _da_from_array(self.mixed_signal, chunks=-1)

    def test_initialization(self) -> None:
        """Test initialization with different parameters."""
        hpss = HpssPercussive(self.sample_rate)
        assert hpss.sampling_rate == self.sample_rate

        hpss_custom = HpssPercussive(self.sample_rate, margin=2.0)
        assert hpss_custom.kwargs.get("margin") == 2.0

    def test_shape_preservation(self) -> None:
        """Test output shape matches input shape."""
        result = self.hpss_p.process(self.dask_signal).compute()
        assert result.shape == self.mixed_signal.shape

    def test_percussive_extraction(self) -> None:
        """Test that percussive content is preserved and harmonic content is reduced."""
        result = self.hpss_p.process(self.dask_signal).compute()

        # Check temporal characteristics - percussive content has sharper onsets
        from librosa.onset import onset_strength

        # Original onset strength
        orig_onset = onset_strength(y=self.mixed_signal[0], sr=self.sample_rate)

        # Processed onset strength
        result_onset = onset_strength(y=result[0], sr=self.sample_rate)

        # Percussive extraction should enhance onsets
        assert np.max(result_onset) > np.max(orig_onset)

        # Check spectral characteristics - percussive content is
        # more noise-like across frequency
        n_fft = 2048

        # Original signal
        orig_spec = np.abs(np.fft.rfft(self.mixed_signal[0], n_fft))

        # Processed signal
        result_spec = np.abs(np.fft.rfft(result[0], n_fft))

        # For percussive content, energy should be more spread across frequencies,
        # resulting in a flatter spectrum
        orig_flatness = np.exp(np.mean(np.log(orig_spec + 1e-10))) / np.mean(orig_spec)
        result_flatness = np.exp(np.mean(np.log(result_spec + 1e-10))) / np.mean(
            result_spec
        )

        # Higher flatness indicates more noise-like (less harmonic) content
        assert result_flatness > orig_flatness

    def test_operation_registry(self) -> None:
        """Test that HpssPercussive is properly registered in the operation registry."""
        assert get_operation("hpss_percussive") == HpssPercussive

        hpss_op = create_operation("hpss_percussive", 16000, margin=3.0)

        assert isinstance(hpss_op, HpssPercussive)
        assert hpss_op.sampling_rate == 16000
        assert hpss_op.kwargs.get("margin") == 3.0


class TestAddWithSNR:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.snr: float = 10.0  # dB

        # Create clean signal (sine wave)
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        self.clean_signal: NDArrayReal = np.array([np.sin(2 * np.pi * 440 * t)])

        # Create noise signal (white noise)
        np.random.seed(42)  # For reproducibility
        self.noise_signal: NDArrayReal = np.array([np.random.randn(self.sample_rate)])

        # Convert to dask arrays
        self.dask_clean: DaArray = _da_from_array(self.clean_signal, chunks=-1)
        self.dask_noise: DaArray = _da_from_array(self.noise_signal, chunks=-1)

        # Initialize the operation
        self.add_with_snr = AddWithSNR(self.sample_rate, self.dask_noise, self.snr)

    def test_initialization(self) -> None:
        """Test initialization with different parameters."""
        op = AddWithSNR(self.sample_rate, self.dask_noise, self.snr)
        assert op.sampling_rate == self.sample_rate
        assert op.snr == self.snr

    def test_shape_preservation(self) -> None:
        """Test output shape matches clean signal shape."""
        result = self.add_with_snr.process(self.dask_clean).compute()
        assert result.shape == self.clean_signal.shape

    def test_snr_adjustment(self) -> None:
        """Test that the noise is added with the correct SNR."""
        from wandas.utils import util

        result = self.add_with_snr.process(self.dask_clean).compute()

        # Calculate actual SNR
        clean_power = util.calculate_rms(self.clean_signal) ** 2
        # Extract noise from result
        noise_component = result - self.clean_signal
        noise_power = util.calculate_rms(noise_component) ** 2

        actual_snr = 10 * np.log10(clean_power / noise_power)

        # Check if the actual SNR is close to the target SNR
        np.testing.assert_allclose(actual_snr, self.snr, rtol=0.1)

    def test_stereo_signal(self) -> None:
        """Test with stereo signals."""
        # Create stereo clean signal
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        stereo_clean: NDArrayReal = np.array(
            [np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 880 * t)]
        )

        # Create stereo noise
        np.random.seed(42)
        stereo_noise: NDArrayReal = np.array(
            [np.random.randn(self.sample_rate), np.random.randn(self.sample_rate)]
        )

        # Convert to dask arrays
        dask_stereo_clean = _da_from_array(stereo_clean, chunks=-1)
        dask_stereo_noise = _da_from_array(stereo_noise, chunks=-1)

        # Create operation
        add_with_snr = AddWithSNR(self.sample_rate, dask_stereo_noise, self.snr)

        # Process
        result = add_with_snr.process(dask_stereo_clean).compute()

        # Check shape
        assert result.shape == stereo_clean.shape

    def test_operation_registry(self) -> None:
        """Test that AddWithSNR is properly registered in the operation registry."""
        assert get_operation("add_with_snr") == AddWithSNR
