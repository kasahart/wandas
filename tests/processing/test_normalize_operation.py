import dask.array as da
import numpy as np
from dask.array.core import Array as DaArray

from wandas.processing.base import create_operation, get_operation
from wandas.processing.effects import Normalize
from wandas.utils.types import NDArrayReal

_da_from_array = da.from_array  # type: ignore [unused-ignore]


class TestNormalize:
    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 16000
        self.target_level: float = -20.0
        self.normalize_op = Normalize(
            self.sample_rate, target_level=self.target_level, channel_wise=True
        )

        # Create test signal with known RMS
        # Sine wave with amplitude A has RMS = A / sqrt(2)
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        # Amplitude 1.0 -> RMS = 0.707
        self.signal_mono: NDArrayReal = np.array([np.sin(2 * np.pi * 440 * t)])
        # Two channels with different amplitudes
        self.signal_stereo: NDArrayReal = np.array(
            [
                np.sin(2 * np.pi * 440 * t),  # RMS ~0.707
                np.sin(2 * np.pi * 880 * t) * 0.5,  # RMS ~0.354
            ]
        )

        self.dask_mono: DaArray = _da_from_array(self.signal_mono, chunks=-1)
        self.dask_stereo: DaArray = _da_from_array(self.signal_stereo, chunks=-1)

    def test_initialization(self) -> None:
        """Test initialization with different parameters."""
        norm_op = Normalize(self.sample_rate, target_level=-10.0, channel_wise=False)
        assert norm_op.sampling_rate == self.sample_rate
        assert norm_op.target_level == -10.0
        assert norm_op.channel_wise is False

    def test_shape_preservation(self) -> None:
        """Test that output shape matches input shape."""
        result = self.normalize_op.process(self.dask_mono).compute()
        assert result.shape == self.signal_mono.shape

        result_stereo = self.normalize_op.process(self.dask_stereo).compute()
        assert result_stereo.shape == self.signal_stereo.shape

    def test_normalize_to_target_level_mono(self) -> None:
        """Test that normalization reaches the target RMS level for mono signal."""
        result = self.normalize_op.process(self.dask_mono).compute()

        # Calculate RMS of result
        result_rms = np.sqrt(np.mean(np.square(result)))

        # Target RMS in linear scale: 10^(-20/20) = 0.1
        target_rms = 10 ** (self.target_level / 20)

        # Verify the result RMS matches target (within tolerance)
        np.testing.assert_allclose(result_rms, target_rms, rtol=1e-5)

    def test_normalize_channel_wise(self) -> None:
        """Test channel-wise normalization."""
        norm_op = Normalize(
            self.sample_rate, target_level=-20.0, channel_wise=True
        )
        result = norm_op.process(self.dask_stereo).compute()

        # Calculate RMS per channel
        for ch in range(result.shape[0]):
            channel_rms = np.sqrt(np.mean(np.square(result[ch])))
            target_rms = 10 ** (-20.0 / 20)

            # Each channel should reach the target RMS independently
            np.testing.assert_allclose(channel_rms, target_rms, rtol=1e-5)

    def test_normalize_global(self) -> None:
        """Test global normalization (same scaling for all channels)."""
        norm_op = Normalize(
            self.sample_rate, target_level=-20.0, channel_wise=False
        )
        result = norm_op.process(self.dask_stereo).compute()

        # Calculate overall RMS (across all channels)
        overall_rms = np.sqrt(np.mean(np.square(result)))
        target_rms = 10 ** (-20.0 / 20)

        # Overall RMS should match target
        np.testing.assert_allclose(overall_rms, target_rms, rtol=1e-5)

        # Verify that the ratio between channels is preserved
        # Original ratio: channel 0 has 2x amplitude of channel 1
        original_ratio = np.max(np.abs(self.signal_stereo[0])) / np.max(
            np.abs(self.signal_stereo[1])
        )
        result_ratio = np.max(np.abs(result[0])) / np.max(np.abs(result[1]))

        np.testing.assert_allclose(result_ratio, original_ratio, rtol=1e-5)

    def test_different_target_levels(self) -> None:
        """Test normalization with different target levels."""
        target_levels = [-30.0, -20.0, -10.0, -3.0]

        for target_level in target_levels:
            norm_op = Normalize(
                self.sample_rate, target_level=target_level, channel_wise=True
            )
            result = norm_op.process(self.dask_mono).compute()

            # Calculate RMS of result
            result_rms = np.sqrt(np.mean(np.square(result)))

            # Target RMS in linear scale
            target_rms = 10 ** (target_level / 20)

            # Verify the result RMS matches target
            np.testing.assert_allclose(result_rms, target_rms, rtol=1e-5)

    def test_zero_signal_handling(self) -> None:
        """Test that zero signal doesn't cause division by zero."""
        zero_signal: NDArrayReal = np.zeros((1, self.sample_rate))
        zero_dask: DaArray = _da_from_array(zero_signal, chunks=-1)

        # Should not raise an error
        result = self.normalize_op.process(zero_dask).compute()

        # Result should also be zero (gain is 1.0 for zero signal)
        np.testing.assert_allclose(result, zero_signal)

    def test_very_quiet_signal_handling(self) -> None:
        """Test handling of very quiet signals (near zero)."""
        # Very quiet signal (below threshold)
        quiet_signal: NDArrayReal = np.array([np.ones(self.sample_rate) * 1e-12])
        quiet_dask: DaArray = _da_from_array(quiet_signal, chunks=-1)

        # Should not raise an error or produce NaN/Inf
        result = self.normalize_op.process(quiet_dask).compute()

        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_preserve_phase_and_shape(self) -> None:
        """Test that normalization preserves waveform shape (only scales amplitude)."""
        result = self.normalize_op.process(self.dask_mono).compute()

        # Normalize both original and result to have same scale
        original_normalized = (
            self.signal_mono / np.max(np.abs(self.signal_mono))
        )
        result_normalized = result / np.max(np.abs(result))

        # Shapes should match (allowing for numerical precision)
        np.testing.assert_allclose(
            original_normalized, result_normalized, rtol=1e-10
        )

    def test_multi_channel_independence(self) -> None:
        """Test that channel-wise normalization treats channels independently."""
        # Create signal with very different channel amplitudes
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)
        signal_varied: NDArrayReal = np.array(
            [
                np.sin(2 * np.pi * 440 * t) * 0.1,  # Very quiet
                np.sin(2 * np.pi * 440 * t) * 1.0,  # Normal
                np.sin(2 * np.pi * 440 * t) * 10.0,  # Very loud
            ]
        )
        dask_varied: DaArray = _da_from_array(signal_varied, chunks=-1)

        norm_op = Normalize(
            self.sample_rate, target_level=-20.0, channel_wise=True
        )
        result = norm_op.process(dask_varied).compute()

        # All channels should reach the same target RMS
        target_rms = 10 ** (-20.0 / 20)
        for ch in range(result.shape[0]):
            channel_rms = np.sqrt(np.mean(np.square(result[ch])))
            np.testing.assert_allclose(channel_rms, target_rms, rtol=1e-5)

    def test_operation_registry(self) -> None:
        """Test that Normalize is properly registered in the operation registry."""
        assert get_operation("normalize") == Normalize

        norm_op = create_operation(
            "normalize", self.sample_rate, target_level=-15.0, channel_wise=False
        )
        assert isinstance(norm_op, Normalize)
        assert norm_op.sampling_rate == self.sample_rate
        assert norm_op.target_level == -15.0
        assert norm_op.channel_wise is False

    def test_positive_target_level(self) -> None:
        """Test normalization with positive target level (amplification)."""
        # Positive dB means amplification
        norm_op = Normalize(
            self.sample_rate, target_level=3.0, channel_wise=True
        )
        result = norm_op.process(self.dask_mono).compute()

        # Calculate RMS of result
        result_rms = np.sqrt(np.mean(np.square(result)))

        # Target RMS in linear scale: 10^(3/20) ≈ 1.413
        target_rms = 10 ** (3.0 / 20)

        # Verify the result RMS matches target
        np.testing.assert_allclose(result_rms, target_rms, rtol=1e-5)

        # Result should be louder than original
        original_rms = np.sqrt(np.mean(np.square(self.signal_mono)))
        assert result_rms > original_rms

    def test_negative_target_level(self) -> None:
        """Test normalization with negative target level (attenuation)."""
        # Negative dB means attenuation
        norm_op = Normalize(
            self.sample_rate, target_level=-40.0, channel_wise=True
        )
        result = norm_op.process(self.dask_mono).compute()

        # Calculate RMS of result
        result_rms = np.sqrt(np.mean(np.square(result)))

        # Target RMS in linear scale: 10^(-40/20) = 0.01
        target_rms = 10 ** (-40.0 / 20)

        # Verify the result RMS matches target
        np.testing.assert_allclose(result_rms, target_rms, rtol=1e-5)

        # Result should be quieter than original
        original_rms = np.sqrt(np.mean(np.square(self.signal_mono)))
        assert result_rms < original_rms
