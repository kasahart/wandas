"""Tests for improved error messages in temporal operations."""

import pytest

from wandas.processing.temporal import ReSampling


class TestReSamplingErrors:
    """Test error messages for ReSampling operation."""

    def test_negative_target_sr_error_message(self) -> None:
        """Test that negative target_sr raises informative error."""
        with pytest.raises(ValueError) as exc_info:
            ReSampling(sampling_rate=16000, target_sr=-8000)

        error_msg = str(exc_info.value)
        assert "Invalid target sampling rate" in error_msg
        assert "Given: -8000" in error_msg
        assert "Expected: Positive number > 0 Hz" in error_msg
        assert "Solution" in error_msg
        assert "Common rates" in error_msg
        assert "Background" in error_msg

    def test_zero_target_sr_error_message(self) -> None:
        """Test that zero target_sr raises informative error."""
        with pytest.raises(ValueError) as exc_info:
            ReSampling(sampling_rate=16000, target_sr=0)

        error_msg = str(exc_info.value)
        assert "Invalid target sampling rate" in error_msg
        assert "Given: 0" in error_msg
        assert "Positive number > 0 Hz" in error_msg

    def test_same_target_sr_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that target_sr equal to sampling_rate logs a warning."""
        import logging
        
        with caplog.at_level(logging.WARNING):
            ReSampling(sampling_rate=16000, target_sr=16000)
        
        assert len(caplog.records) == 1
        assert "Target sampling rate (16000" in caplog.text
        assert "equals current sampling rate" in caplog.text
        assert "No resampling will be performed" in caplog.text

    def test_valid_target_sr_no_error(self) -> None:
        """Test that valid target_sr does not raise error."""
        resampler = ReSampling(sampling_rate=16000, target_sr=8000)
        assert resampler.sampling_rate == 16000
        assert resampler.target_sr == 8000

    def test_upsample_valid(self) -> None:
        """Test that upsampling (higher target_sr) is valid."""
        resampler = ReSampling(sampling_rate=16000, target_sr=44100)
        assert resampler.target_sr == 44100

    def test_downsample_valid(self) -> None:
        """Test that downsampling (lower target_sr) is valid."""
        resampler = ReSampling(sampling_rate=44100, target_sr=16000)
        assert resampler.target_sr == 16000
