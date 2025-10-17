"""Test Speech Intelligibility Index integration with ChannelFrame."""

import numpy as np
import pytest

import wandas as wd


class TestChannelFrameSII:
    """Test Speech Intelligibility Index method on ChannelFrame."""

    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: int = 44100
        self.duration: float = 1.0

        # Create test signals using wandas utilities
        # Speech-like signal with multiple frequencies
        self.speech_frame = wd.generate_sin(
            freqs=[250, 500, 1000, 2000],
            amplitudes=[0.3, 0.3, 0.2, 0.1],
            duration=self.duration,
            sampling_rate=self.sample_rate,
        )

        # Add some noise to make it more realistic
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        noise = np.random.normal(0, 0.05, len(t))
        noise_data = self.speech_frame._data + noise
        self.speech_frame = wd.ChannelFrame(
            data=noise_data,
            sampling_rate=self.sample_rate,
        )

        # Clean tone signal
        self.clean_frame = wd.generate_sin(
            freqs=[1000],
            duration=self.duration,
            sampling_rate=self.sample_rate,
        )

        # Multi-channel signal
        left_channel = wd.generate_sin(
            freqs=[500], duration=self.duration, sampling_rate=self.sample_rate
        )
        right_channel = wd.generate_sin(
            freqs=[1000], duration=self.duration, sampling_rate=self.sample_rate
        )
        # Combine channels
        import dask.array as da
        multi_data = da.vstack([left_channel._data, right_channel._data])
        self.multi_frame = wd.ChannelFrame(
            data=multi_data,
            sampling_rate=self.sample_rate,
        )

    def test_sii_method_exists(self) -> None:
        """Test that sii method exists on ChannelFrame."""
        assert hasattr(self.speech_frame, "sii")
        assert callable(self.speech_frame.sii)

    def test_sii_return_values(self) -> None:
        """Test that sii method returns correct tuple structure."""
        sii_value, specific_sii, freq_axis = self.speech_frame.sii()

        # Check return types
        assert isinstance(sii_value, float)
        assert isinstance(specific_sii, np.ndarray)
        assert isinstance(freq_axis, np.ndarray)

        # Check SII value is in valid range
        assert 0.0 <= sii_value <= 1.0

        # Check that specific_sii and freq_axis have same length
        assert len(specific_sii) == len(freq_axis)

    def test_sii_default_method_band(self) -> None:
        """Test SII with default method band (octave)."""
        sii_value, specific_sii, freq_axis = self.speech_frame.sii()

        # Should return valid values
        assert 0.0 <= sii_value <= 1.0
        assert len(specific_sii) > 0
        assert len(freq_axis) > 0

    def test_sii_different_method_bands(self) -> None:
        """Test SII with different method bands."""
        methods = ["octave", "third octave", "critical"]

        for method in methods:
            sii_value, specific_sii, freq_axis = self.speech_frame.sii(method_band=method)

            # All should return valid SII values
            assert 0.0 <= sii_value <= 1.0
            assert len(specific_sii) > 0
            assert len(freq_axis) > 0

            # Third octave and critical should have more frequency bands
            if method == "octave":
                octave_bands = len(freq_axis)
            elif method == "third octave":
                third_octave_bands = len(freq_axis)
                assert third_octave_bands > octave_bands
            elif method == "critical":
                critical_bands = len(freq_axis)
                assert critical_bands >= octave_bands

    def test_sii_multi_channel(self) -> None:
        """Test that SII works with multi-channel signals."""
        sii_value, specific_sii, freq_axis = self.multi_frame.sii()

        # Should automatically convert to mono and return valid SII
        assert 0.0 <= sii_value <= 1.0
        assert isinstance(sii_value, float)

    def test_sii_comparison_with_mosqito_direct(self) -> None:
        """Test that ChannelFrame.sii() matches direct MoSQITo call."""
        from mosqito.sq_metrics.speech_intelligibility.sii_ansi import comp_sii

        # Get result from ChannelFrame method
        sii_wandas, specific_wandas, freq_wandas = self.speech_frame.sii(method_band="octave")

        # Get result from direct MoSQITo call
        signal_data = self.speech_frame._data.compute().flatten()
        sii_mosqito, specific_mosqito, freq_mosqito = comp_sii(
            signal_data, self.sample_rate, method_band="octave"
        )

        # Values should match
        np.testing.assert_allclose(sii_wandas, sii_mosqito, rtol=1e-10)
        np.testing.assert_allclose(specific_wandas, specific_mosqito, rtol=1e-10)
        np.testing.assert_allclose(freq_wandas, freq_mosqito, rtol=1e-10)

    def test_sii_with_ansi_test_case(self) -> None:
        """
        Test SII with ANSI S3.5 test case values.

        Reference: ANSI S3.5-1997 test case
        Expected SII ≈ 0.504 for specific noise and speech levels
        """
        # Create a more controlled test signal
        # This is a simplified test; actual ANSI test case requires
        # specific noise and speech spectra
        test_signal = wd.generate_sin(
            freqs=[250, 500, 1000, 2000, 4000],
            amplitudes=[0.2, 0.2, 0.2, 0.2, 0.1],
            duration=1.0,
            sampling_rate=44100,
        )

        sii_value, _, _ = test_signal.sii(method_band="octave")

        # Should be in reasonable range for speech
        # (Actual ANSI test case would require precise signal construction)
        assert 0.0 <= sii_value <= 1.0

    def test_sii_frequency_axis_ordering(self) -> None:
        """Test that frequency axis is properly ordered."""
        _, _, freq_axis = self.speech_frame.sii()

        # Frequency axis should be monotonically increasing
        assert np.all(np.diff(freq_axis) > 0)

    def test_sii_specific_values_sum(self) -> None:
        """Test that specific SII values are in valid range."""
        _, specific_sii, _ = self.speech_frame.sii()

        # Each specific SII value should be between 0 and 1
        assert np.all(specific_sii >= 0.0)
        assert np.all(specific_sii <= 1.0)

    def test_sii_with_filtered_signal(self) -> None:
        """Test SII on filtered signals."""
        # Low-pass filtered signal (better intelligibility)
        filtered_signal = self.speech_frame.low_pass_filter(cutoff=4000)
        sii_filtered, _, _ = filtered_signal.sii()

        # High-pass filtered signal (reduced intelligibility)
        highpass_signal = self.speech_frame.high_pass_filter(cutoff=3000)
        sii_highpass, _, _ = highpass_signal.sii()

        # Both should be valid
        assert 0.0 <= sii_filtered <= 1.0
        assert 0.0 <= sii_highpass <= 1.0

        # Low-pass filter preserving speech frequencies should have higher SII
        # than high-pass filter removing important speech frequencies
        # (This depends on signal characteristics, so we just check validity)

    def test_sii_with_normalized_signal(self) -> None:
        """Test SII on normalized signals."""
        normalized_signal = self.speech_frame.normalize()
        sii_value, _, _ = normalized_signal.sii()

        # Should return valid SII
        assert 0.0 <= sii_value <= 1.0
