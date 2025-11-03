"""
Test channel label updates for audio processing operations.

This module tests that channel labels are properly updated when
operations are applied, enabling better tracking and visualization
of processing pipelines.
"""

import dask.array as da
import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame

_da_from_array = da.from_array  # type: ignore [unused-ignore]


class TestChannelLabelUpdates:
    """Test channel label updates for unary operations."""

    def setup_method(self) -> None:
        """Set up test fixtures for each test."""
        self.sample_rate: float = 16000
        self.data: np.ndarray = np.random.random((2, 16000))  # 2 channels, 1 second
        self.dask_data = _da_from_array(self.data, chunks=(1, 4000))

    def test_normalize_updates_channel_labels(self) -> None:
        """Test that normalize operation updates channel labels."""
        frame = ChannelFrame(
            data=self.dask_data,
            sampling_rate=self.sample_rate,
            label="test_audio",
            channel_metadata=[
                {"label": "ch0", "unit": "", "extra": {}},
                {"label": "ch1", "unit": "", "extra": {}},
            ],
        )

        result = frame.normalize()

        # Verify labels are updated
        assert result.labels == ["normalize(ch0)", "normalize(ch1)"]
        # Verify original frame is unchanged
        assert frame.labels == ["ch0", "ch1"]

    def test_low_pass_filter_updates_labels(self) -> None:
        """Test that low_pass_filter updates channel labels."""
        frame = ChannelFrame(
            data=self.dask_data,
            sampling_rate=self.sample_rate,
            label="test_audio",
            channel_metadata=[
                {"label": "acc_x", "unit": "m/s^2", "extra": {}},
                {"label": "acc_y", "unit": "m/s^2", "extra": {}},
            ],
        )

        result = frame.low_pass_filter(cutoff=1000)

        assert result.labels == ["lowpass_filter(acc_x)", "lowpass_filter(acc_y)"]

    def test_high_pass_filter_updates_labels(self) -> None:
        """Test that high_pass_filter updates channel labels."""
        frame = ChannelFrame(
            data=self.dask_data,
            sampling_rate=self.sample_rate,
            channel_metadata=[
                {"label": "signal_a", "unit": "V", "extra": {}},
                {"label": "signal_b", "unit": "V", "extra": {}},
            ],
        )

        result = frame.high_pass_filter(cutoff=100)

        assert result.labels == ["highpass_filter(signal_a)", "highpass_filter(signal_b)"]

    def test_band_pass_filter_updates_labels(self) -> None:
        """Test that band_pass_filter updates channel labels."""
        frame = ChannelFrame(
            data=self.dask_data,
            sampling_rate=self.sample_rate,
            channel_metadata=[
                {"label": "mic1", "unit": "Pa", "extra": {}},
                {"label": "mic2", "unit": "Pa", "extra": {}},
            ],
        )

        result = frame.band_pass_filter(low_cutoff=200, high_cutoff=5000)

        assert result.labels == ["bandpass_filter(mic1)", "bandpass_filter(mic2)"]

    def test_a_weighting_updates_labels(self) -> None:
        """Test that a_weighting updates channel labels."""
        frame = ChannelFrame(
            data=self.dask_data,
            sampling_rate=self.sample_rate,
            channel_metadata=[
                {"label": "audio_left", "unit": "Pa", "extra": {}},
                {"label": "audio_right", "unit": "Pa", "extra": {}},
            ],
        )

        result = frame.a_weighting()

        assert result.labels == ["a_weighting(audio_left)", "a_weighting(audio_right)"]

    def test_abs_updates_labels(self) -> None:
        """Test that abs operation updates channel labels."""
        frame = ChannelFrame(
            data=self.dask_data,
            sampling_rate=self.sample_rate,
            channel_metadata=[
                {"label": "ch0", "unit": "", "extra": {}},
                {"label": "ch1", "unit": "", "extra": {}},
            ],
        )

        result = frame.abs()

        assert result.labels == ["abs(ch0)", "abs(ch1)"]

    def test_power_updates_labels(self) -> None:
        """Test that power operation updates channel labels."""
        frame = ChannelFrame(
            data=self.dask_data,
            sampling_rate=self.sample_rate,
            channel_metadata=[
                {"label": "signal", "unit": "V", "extra": {}},
                {"label": "reference", "unit": "V", "extra": {}},
            ],
        )

        result = frame.power(exponent=2.0)

        assert result.labels == ["power(signal)", "power(reference)"]


class TestChainedOperationLabels:
    """Test label updates for chained operations."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_rate: float = 16000
        self.data: np.ndarray = np.random.random((1, 16000))  # 1 channel
        self.dask_data = _da_from_array(self.data, chunks=(1, 4000))

    def test_chained_operations_nest_labels(self) -> None:
        """Test that chained operations properly nest labels."""
        frame = ChannelFrame(
            data=self.dask_data,
            sampling_rate=self.sample_rate,
            channel_metadata=[{"label": "ch0", "unit": "", "extra": {}}],
        )

        result = frame.normalize().low_pass_filter(cutoff=1000)

        assert result.labels == ["lowpass_filter(normalize(ch0))"]

    def test_triple_chained_operations(self) -> None:
        """Test three operations chained together."""
        frame = ChannelFrame(
            data=self.dask_data,
            sampling_rate=self.sample_rate,
            channel_metadata=[{"label": "raw", "unit": "", "extra": {}}],
        )

        result = (
            frame.normalize()
            .high_pass_filter(cutoff=100)
            .low_pass_filter(cutoff=1000)
        )

        assert result.labels == ["lowpass_filter(highpass_filter(normalize(raw)))"]

    def test_chained_operations_preserve_metadata(self) -> None:
        """Test that chained operations preserve non-label metadata."""
        frame = ChannelFrame(
            data=self.dask_data,
            sampling_rate=self.sample_rate,
            channel_metadata=[
                {"label": "ch0", "unit": "Pa", "extra": {"sensor_id": 123}}
            ],
        )

        result = frame.normalize().low_pass_filter(cutoff=1000)

        # Label should be updated
        assert result.labels == ["lowpass_filter(normalize(ch0))"]
        # But other metadata should be preserved
        assert result.channels[0].unit == "Pa"
        assert result.channels[0].extra == {"sensor_id": 123}


class TestBinaryOperationLabelCompatibility:
    """Test that updated labels work with binary operations."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_rate: float = 16000
        self.data: np.ndarray = np.random.random((1, 16000))
        self.dask_data = _da_from_array(self.data, chunks=(1, 4000))

    def test_binary_op_with_processed_frame(self) -> None:
        """Test binary operation with a frame that has updated labels."""
        frame1 = ChannelFrame(
            data=self.dask_data,
            sampling_rate=self.sample_rate,
            channel_metadata=[{"label": "ch0", "unit": "", "extra": {}}],
        ).normalize()

        frame2 = ChannelFrame(
            data=self.dask_data,
            sampling_rate=self.sample_rate,
            channel_metadata=[{"label": "ch0", "unit": "", "extra": {}}],
        )

        result = frame1 + frame2

        # Binary operation should include the processed label
        assert "normalize(ch0)" in result.labels[0]
        assert "ch0" in result.labels[0]

    def test_add_two_processed_frames(self) -> None:
        """Test adding two frames with different processing."""
        frame1 = ChannelFrame(
            data=self.dask_data,
            sampling_rate=self.sample_rate,
            channel_metadata=[{"label": "ch0", "unit": "", "extra": {}}],
        ).normalize()

        frame2 = ChannelFrame(
            data=self.dask_data,
            sampling_rate=self.sample_rate,
            channel_metadata=[{"label": "ch0", "unit": "", "extra": {}}],
        ).low_pass_filter(cutoff=1000)

        result = frame1 + frame2

        # Both processed labels should appear
        assert "normalize(ch0)" in result.labels[0]
        assert "lowpass_filter(ch0)" in result.labels[0]


class TestEdgeCases:
    """Test edge cases for label updates."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_rate: float = 16000
        self.data: np.ndarray = np.random.random((1, 16000))
        self.dask_data = _da_from_array(self.data, chunks=(1, 4000))

    def test_operation_on_single_channel(self) -> None:
        """Test operations on single-channel frame."""
        frame = ChannelFrame(
            data=self.dask_data,
            sampling_rate=self.sample_rate,
            channel_metadata=[{"label": "mono", "unit": "", "extra": {}}],
        )

        result = frame.normalize()

        assert len(result.labels) == 1
        assert result.labels[0] == "normalize(mono)"

    def test_operation_with_special_characters_in_label(self) -> None:
        """Test that special characters in labels are handled correctly."""
        frame = ChannelFrame(
            data=self.dask_data,
            sampling_rate=self.sample_rate,
            channel_metadata=[
                {"label": "sensor-1_output", "unit": "", "extra": {}}
            ],
        )

        result = frame.normalize()

        assert result.labels[0] == "normalize(sensor-1_output)"

    def test_label_update_preserves_channel_metadata_structure(self) -> None:
        """Test that label updates preserve the ChannelMetadata structure."""
        from wandas.core.metadata import ChannelMetadata

        frame = ChannelFrame(
            data=self.dask_data,
            sampling_rate=self.sample_rate,
            channel_metadata=[
                ChannelMetadata(
                    label="ch0", unit="Pa", ref=2e-5, extra={"calibration": 1.0}
                )
            ],
        )

        result = frame.normalize()

        # Check that the result has ChannelMetadata objects
        assert isinstance(result.channels[0], ChannelMetadata)
        # Check that all fields are preserved except label
        assert result.channels[0].label == "normalize(ch0)"
        assert result.channels[0].unit == "Pa"
        assert result.channels[0].ref == 2e-5
        assert result.channels[0].extra == {"calibration": 1.0}


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.sample_rate: float = 16000
        self.data: np.ndarray = np.random.random((2, 16000))
        self.dask_data = _da_from_array(self.data, chunks=(1, 4000))

    def test_default_channel_labels_still_work(self) -> None:
        """Test that default channel labels (ch0, ch1, ...) still work."""
        frame = ChannelFrame(
            data=self.dask_data,
            sampling_rate=self.sample_rate,
        )

        # Default labels should be ch0, ch1
        assert frame.labels == ["ch0", "ch1"]

        # After operation, they should be updated
        result = frame.normalize()
        assert result.labels == ["normalize(ch0)", "normalize(ch1)"]

    def test_operation_history_still_tracked(self) -> None:
        """Test that operation history is still tracked correctly."""
        frame = ChannelFrame(
            data=self.dask_data,
            sampling_rate=self.sample_rate,
        )

        result = frame.normalize().low_pass_filter(cutoff=1000)

        # Operation history should have two entries
        assert len(result.operation_history) == 2
        assert result.operation_history[0]["operation"] == "normalize"
        assert result.operation_history[1]["operation"] == "lowpass_filter"

    def test_previous_reference_maintained(self) -> None:
        """Test that the previous frame reference is maintained."""
        frame = ChannelFrame(
            data=self.dask_data,
            sampling_rate=self.sample_rate,
        )

        result = frame.normalize()

        assert result.previous is frame
        assert result.previous.labels == ["ch0", "ch1"]
