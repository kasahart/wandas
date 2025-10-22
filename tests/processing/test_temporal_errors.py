"""Tests for temporal operation error messages."""

import pytest

from wandas.processing.temporal import ReSampling


class TestReSamplingErrors:
    """Test error messages in ReSampling."""

    def test_target_sr_zero(self) -> None:
        """Test error message when target sampling rate is zero."""
        with pytest.raises(ValueError) as exc_info:
            ReSampling(sampling_rate=16000, target_sr=0)

        error_msg = str(exc_info.value)
        assert "Invalid target sampling rate" in error_msg
        assert "Given: 0" in error_msg
        assert "positive number" in error_msg
        assert "Solution" in error_msg
        assert "Background" in error_msg

    def test_target_sr_negative(self) -> None:
        """Test error message when target sampling rate is negative."""
        with pytest.raises(ValueError) as exc_info:
            ReSampling(sampling_rate=16000, target_sr=-8000)

        error_msg = str(exc_info.value)
        assert "Invalid target sampling rate" in error_msg
        assert "Given: -8000" in error_msg
        assert "positive number" in error_msg
        assert "Solution" in error_msg
        assert "Background" in error_msg

    def test_extreme_upsampling_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test warning message for extreme upsampling (>10x)."""
        import logging

        with caplog.at_level(logging.WARNING):
            ReSampling(sampling_rate=8000, target_sr=100000)

        assert any("Extreme upsampling detected" in record.message for record in caplog.records)
        warning_msg = next(
            record.message
            for record in caplog.records
            if "Extreme upsampling detected" in record.message
        )
        assert "Original rate: 8000" in warning_msg
        assert "Target rate: 100000" in warning_msg
        assert "Ratio: 12.5x" in warning_msg
        assert "memory usage" in warning_msg

    def test_extreme_downsampling_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test warning message for extreme downsampling (<0.1x)."""
        import logging

        with caplog.at_level(logging.WARNING):
            ReSampling(sampling_rate=48000, target_sr=4000)

        assert any(
            "Extreme downsampling detected" in record.message for record in caplog.records
        )
        warning_msg = next(
            record.message
            for record in caplog.records
            if "Extreme downsampling detected" in record.message
        )
        assert "Original rate: 48000" in warning_msg
        assert "Target rate: 4000" in warning_msg
        assert "Ratio: 0.1x" in warning_msg
        assert "aliasing" in warning_msg

    def test_valid_resampling_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that normal resampling doesn't produce warnings."""
        import logging

        with caplog.at_level(logging.WARNING):
            # 2x upsampling - should be fine
            ReSampling(sampling_rate=16000, target_sr=32000)
            # 2x downsampling - should be fine
            ReSampling(sampling_rate=48000, target_sr=24000)

        # Should not have any extreme resampling warnings
        assert not any(
            "Extreme" in record.message and "sampling detected" in record.message
            for record in caplog.records
        )

    def test_valid_common_sampling_rates(self) -> None:
        """Test that common sampling rate conversions are accepted."""
        # Common conversions
        conversions = [
            (44100, 48000),  # CD to DAT
            (48000, 44100),  # DAT to CD
            (16000, 8000),  # Downsampling for telephony
            (8000, 16000),  # Upsampling from telephony
        ]

        for orig_sr, target_sr in conversions:
            rs = ReSampling(sampling_rate=orig_sr, target_sr=target_sr)
            assert rs.sampling_rate == orig_sr
            assert rs.target_sr == target_sr
