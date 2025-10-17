"""Integration tests for roughness calculation on ChannelFrame."""

import numpy as np
import pytest

import wandas as wd
from wandas.utils.types import NDArrayReal


class TestChannelFrameRoughness:
    """Integration tests for roughness calculation method on ChannelFrame."""

    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: float = 44100
        self.duration: float = 1.0

        # Create test signal with amplitude modulation
        # Carrier: 1000 Hz, Modulation: 70 Hz (peak roughness frequency)
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        carrier = np.sin(2 * np.pi * 1000 * t)
        modulator = 0.5 * (1 + np.sin(2 * np.pi * 70 * t))

        self.modulated_signal = carrier * modulator

        # Create pure tone (low roughness reference)
        self.pure_tone = np.sin(2 * np.pi * 1000 * t)

    def test_roughness_method_exists(self) -> None:
        """Test that roughness method is available on ChannelFrame."""
        cf = wd.ChannelFrame(
            data=self.modulated_signal,
            sampling_rate=self.sample_rate,
        )
        assert hasattr(cf, "roughness")
        assert callable(cf.roughness)

    def test_roughness_time_method_mono(self) -> None:
        """Test roughness calculation with time method on mono signal."""
        cf = wd.ChannelFrame(
            data=self.modulated_signal,
            sampling_rate=self.sample_rate,
        )

        roughness = cf.roughness(method="time")

        # Check that result is a scalar for mono signal
        assert isinstance(roughness, (np.ndarray, float, np.number))
        if isinstance(roughness, np.ndarray):
            assert roughness.shape == ()  # Scalar numpy array

        # Check that roughness is non-negative
        float_roughness = float(roughness)
        assert float_roughness >= 0

        # Check that roughness is in reasonable range
        assert float_roughness < 100

    def test_roughness_time_method_stereo(self) -> None:
        """Test roughness calculation with time method on stereo signal."""
        # Create stereo signal with different modulation frequencies
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        carrier = np.sin(2 * np.pi * 1000 * t)

        mod1 = 0.5 * (1 + np.sin(2 * np.pi * 70 * t))
        mod2 = 0.5 * (1 + np.sin(2 * np.pi * 100 * t))

        stereo_signal = np.array([carrier * mod1, carrier * mod2])

        cf = wd.ChannelFrame(
            data=stereo_signal,
            sampling_rate=self.sample_rate,
        )

        roughness = cf.roughness(method="time")

        # Check that result is array with one value per channel
        assert isinstance(roughness, np.ndarray)
        assert roughness.shape == (2,)

        # Check that both values are non-negative
        assert np.all(roughness >= 0)

        # Check that values are different (different modulation frequencies)
        assert roughness[0] != roughness[1]

    def test_roughness_freq_method_mono(self) -> None:
        """Test roughness calculation with freq method on mono signal."""
        cf = wd.ChannelFrame(
            data=self.modulated_signal,
            sampling_rate=self.sample_rate,
        )

        roughness = cf.roughness(method="freq")

        # Check that result is valid
        assert isinstance(roughness, (np.ndarray, float, np.number))
        float_roughness = float(roughness)
        assert float_roughness >= 0
        assert float_roughness < 100

    def test_roughness_with_overlap(self) -> None:
        """Test roughness calculation with different overlap values."""
        cf = wd.ChannelFrame(
            data=self.modulated_signal,
            sampling_rate=self.sample_rate,
        )

        roughness_no_overlap = cf.roughness(method="time", overlap=0.0)
        roughness_with_overlap = cf.roughness(method="time", overlap=0.5)

        # Both should produce valid results
        assert float(roughness_no_overlap) >= 0
        assert float(roughness_with_overlap) >= 0

    def test_roughness_pure_tone_vs_modulated(self) -> None:
        """Test that modulated signal has higher roughness than pure tone.

        This test validates that the roughness metric correctly identifies
        roughness: a modulated signal should have higher roughness than
        a pure tone.
        """
        cf_pure = wd.ChannelFrame(
            data=self.pure_tone,
            sampling_rate=self.sample_rate,
        )

        cf_modulated = wd.ChannelFrame(
            data=self.modulated_signal,
            sampling_rate=self.sample_rate,
        )

        roughness_pure = cf_pure.roughness(method="time")
        roughness_modulated = cf_modulated.roughness(method="time")

        # Pure tone should have lower roughness than modulated signal
        assert float(roughness_pure) < float(roughness_modulated)

        # Pure tone should have very low roughness
        assert float(roughness_pure) < 0.5

        # Modulated signal should have noticeable roughness
        assert float(roughness_modulated) > 0.1

    def test_roughness_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        cf = wd.ChannelFrame(
            data=self.modulated_signal,
            sampling_rate=self.sample_rate,
        )

        with pytest.raises(ValueError, match="Invalid method"):
            cf.roughness(method="invalid")  # type: ignore

    def test_roughness_invalid_overlap_raises_error(self) -> None:
        """Test that invalid overlap raises ValueError."""
        cf = wd.ChannelFrame(
            data=self.modulated_signal,
            sampling_rate=self.sample_rate,
        )

        with pytest.raises(ValueError, match="Invalid overlap"):
            cf.roughness(method="time", overlap=1.5)

        with pytest.raises(ValueError, match="Invalid overlap"):
            cf.roughness(method="time", overlap=-0.1)

    def test_roughness_comparison_with_mosqito(self) -> None:
        """Test that ChannelFrame.roughness matches direct MoSQITo calculation.

        This integration test ensures that the wandas implementation
        produces results consistent with direct MoSQITo usage.
        """
        from mosqito.sq_metrics import roughness_dw_time

        cf = wd.ChannelFrame(
            data=self.modulated_signal,
            sampling_rate=self.sample_rate,
        )

        # Calculate using wandas
        wandas_roughness = cf.roughness(method="time", overlap=0.0)

        # Calculate directly using MoSQITo
        mosqito_result = roughness_dw_time(
            self.modulated_signal,
            fs=self.sample_rate,
            overlap=0.0,
        )

        # Extract roughness value
        if isinstance(mosqito_result, tuple):
            mosqito_roughness = float(np.mean(mosqito_result[0]))
        else:
            mosqito_roughness = float(mosqito_result)

        # Compare results
        np.testing.assert_allclose(
            float(wandas_roughness),
            mosqito_roughness,
            rtol=0.01,  # 1% relative tolerance
            atol=0.01,  # 0.01 asper absolute tolerance
        )

    def test_roughness_with_generated_signal(self) -> None:
        """Test roughness calculation with wandas generated signal."""
        # Generate signal using wandas
        signal = wd.generate_sin(
            freqs=[1000],
            duration=1.0,
            sampling_rate=self.sample_rate,
        )

        # Apply amplitude modulation manually
        t = np.linspace(0, 1.0, signal.n_samples)
        modulator = 0.5 * (1 + np.sin(2 * np.pi * 70 * t))

        # Modulate the signal
        import dask.array as da

        data_array = signal._data.compute()
        modulated_data = data_array * modulator

        modulated_signal = wd.ChannelFrame(
            data=modulated_data,
            sampling_rate=self.sample_rate,
        )

        roughness = modulated_signal.roughness(method="time")

        # Check that roughness is calculated
        assert float(roughness) > 0

    def test_roughness_different_sampling_rates(self) -> None:
        """Test roughness calculation with different sampling rates."""
        # Test with 44100 Hz
        cf_44100 = wd.ChannelFrame(
            data=self.modulated_signal,
            sampling_rate=44100,
        )
        roughness_44100 = cf_44100.roughness(method="time")

        # Create signal with 48000 Hz
        t_48k = np.linspace(0, self.duration, int(48000 * self.duration))
        carrier_48k = np.sin(2 * np.pi * 1000 * t_48k)
        modulator_48k = 0.5 * (1 + np.sin(2 * np.pi * 70 * t_48k))
        signal_48k = carrier_48k * modulator_48k

        cf_48000 = wd.ChannelFrame(
            data=signal_48k,
            sampling_rate=48000,
        )
        roughness_48000 = cf_48000.roughness(method="time")

        # Both should produce valid roughness values
        assert float(roughness_44100) >= 0
        assert float(roughness_48000) >= 0

        # Values should be similar (within reasonable tolerance)
        # The exact values may differ slightly due to sampling rate effects
        np.testing.assert_allclose(
            float(roughness_44100),
            float(roughness_48000),
            rtol=0.2,  # 20% relative tolerance for different sampling rates
        )
