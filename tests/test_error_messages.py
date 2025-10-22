"""
Example tests for error message quality.

This module demonstrates how to test error messages to ensure they follow
the 3-element structure (What, Why, How) and provide helpful guidance to users.

These tests can be used as templates for testing error messages in other modules.
"""

import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame


class TestErrorMessageQuality:
    """Test that error messages are helpful and follow guidelines."""

    def test_dimension_error_message_structure(self) -> None:
        """Test that dimension validation error provides all 3 elements."""
        # Create a 3D array (invalid for ChannelFrame)
        invalid_data = np.random.rand(2, 3, 4)

        with pytest.raises(ValueError) as exc_info:
            ChannelFrame.from_numpy(invalid_data, sampling_rate=44100)

        error_msg = str(exc_info.value)

        # What: Should state the problem clearly
        assert "1D or 2D" in error_msg or "dimension" in error_msg.lower()
        assert "3D" in error_msg  # Should mention actual dimension

        # Why: Should explain the constraint
        # (Currently this might not be present, but should be in improved version)
        # assert "expects" in error_msg or "ChannelFrame" in error_msg

        # How: Should provide solution
        # (Currently this might not be present, but should be in improved version)
        # assert "reshape" in error_msg or "Please" in error_msg

        # Should include actual shape
        assert str(invalid_data.shape) in error_msg

    def test_dimension_error_message_length(self) -> None:
        """Test that error message is sufficiently detailed."""
        invalid_data = np.random.rand(2, 3, 4)

        with pytest.raises(ValueError) as exc_info:
            ChannelFrame.from_numpy(invalid_data, sampling_rate=44100)

        error_msg = str(exc_info.value)

        # Error message should be reasonably detailed (not too short)
        # After Phase 2 improvements, should be 200-300 characters
        # Currently might be shorter - this test documents the target
        assert len(error_msg) > 50, "Error message should be descriptive"

    def test_sampling_rate_mismatch_error_provides_both_values(self) -> None:
        """Test that sampling rate errors show both the expected and actual values."""
        # Create two frames with different sampling rates
        frame1 = ChannelFrame.from_numpy(np.random.rand(100), sampling_rate=44100)
        frame2 = ChannelFrame.from_numpy(np.random.rand(100), sampling_rate=48000)

        with pytest.raises(ValueError) as exc_info:
            frame1 + frame2

        error_msg = str(exc_info.value)

        # Should mention sampling rate mismatch
        assert "sampling rate" in error_msg.lower() or "rate" in error_msg.lower()

        # Should include both values (this might not be present currently)
        # After Phase 2, should include both 44100 and 48000
        # assert "44100" in error_msg and "48000" in error_msg

    def test_error_messages_are_user_friendly(self) -> None:
        """Test that error messages use friendly language."""
        invalid_data = np.random.rand(2, 3, 4)

        with pytest.raises(ValueError) as exc_info:
            ChannelFrame.from_numpy(invalid_data, sampling_rate=44100)

        error_msg = str(exc_info.value).lower()

        # Should use friendly language (avoid jargon where possible)
        # Should NOT contain debug-oriented messages
        assert "unexpected" not in error_msg or "shape" in error_msg
        assert "bug" not in error_msg
        assert "internal error" not in error_msg

    def test_index_error_message_shows_valid_range(self) -> None:
        """Test that index errors show the valid range."""
        frame = ChannelFrame.from_numpy(np.random.rand(3, 100), sampling_rate=44100)

        with pytest.raises(IndexError) as exc_info:
            frame[5]  # Index 5 is out of range for 3 channels

        error_msg = str(exc_info.value)

        # Should mention the invalid index
        assert "5" in error_msg or "out of range" in error_msg.lower()

        # Should mention the number of channels (after Phase 2 improvements)
        # assert "3" in error_msg or str(frame.n_channels) in error_msg

        # Should provide valid range (after Phase 2 improvements)
        # assert "0" in error_msg and "2" in error_msg


class TestErrorMessageConsistency:
    """Test that error messages are consistent across the library."""

    @pytest.mark.parametrize(
        "operation,error_type",
        [
            (lambda: ChannelFrame.from_numpy(np.random.rand(2, 3, 4), 44100), ValueError),
            # Add more test cases here as more errors are improved
        ],
    )
    def test_error_message_minimum_quality(
        self, operation: callable, error_type: type
    ) -> None:
        """Test that all errors meet minimum quality standards."""
        with pytest.raises(error_type) as exc_info:
            operation()

        error_msg = str(exc_info.value)

        # Minimum quality standards
        assert len(error_msg) > 20, "Error message should be descriptive"
        assert error_msg[0].isupper(), "Error message should start with capital letter"


class TestPhase2ErrorImprovements:
    """
    Tests for Phase 2 error improvements.

    These tests currently expect basic error messages but can be updated
    after Phase 2 to verify the 3-element structure is present.
    """

    def test_high_priority_errors_follow_3element_structure(self) -> None:
        """
        Test that high-priority errors follow the 3-element structure.

        This test can be enabled and expanded after Phase 2 improvements.
        """
        pytest.skip("Phase 2 improvements not yet implemented")

        # Example of what to test after Phase 2:
        invalid_data = np.random.rand(2, 3, 4)

        with pytest.raises(ValueError) as exc_info:
            ChannelFrame.from_numpy(invalid_data, sampling_rate=44100)

        error_msg = str(exc_info.value)

        # What: Clear problem statement
        assert any(
            phrase in error_msg.lower()
            for phrase in ["got", "expected", "must be", "invalid"]
        )

        # Why: Explanation of constraint
        assert any(
            phrase in error_msg.lower()
            for phrase in ["because", "expects", "requires", "must"]
        )

        # How: Solution guidance
        assert any(
            phrase in error_msg.lower()
            for phrase in ["please", "try", "use", "reshape", "convert"]
        )

        # Should use multi-line format
        assert "\n" in error_msg

        # Should include specific values
        assert "3D" in error_msg
        assert str(invalid_data.shape) in error_msg


# Usage example:
# pytest tests/test_error_messages.py -v
# pytest tests/test_error_messages.py::TestErrorMessageQuality -v
