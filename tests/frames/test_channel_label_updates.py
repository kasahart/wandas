"""
Test channel label updates for audio processing operations.

This module tests that channel labels are properly updated when
operations are applied, enabling better tracking and visualization
of processing pipelines.
"""

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.frames.channel import ChannelFrame

_da_from_array = da.from_array

# --- Module-level deterministic test data (signal content irrelevant for label tests) ---
_SAMPLE_RATE: float = 16000
_rng = np.random.default_rng(42)
_DATA_2CH: np.ndarray = _rng.random((2, 16000))  # 2 channels, 1 second
_DATA_1CH: np.ndarray = _rng.random((1, 16000))  # 1 channel, 1 second
_DASK_2CH = _da_from_array(_DATA_2CH, chunks=(1, 4000))
_DASK_1CH = _da_from_array(_DATA_1CH, chunks=(1, 4000))


class TestChannelLabelUpdates:
    """Test channel label updates for unary operations."""

    def test_normalize_updates_channel_labels(self) -> None:
        """Test that normalize operation updates channel labels."""
        frame = ChannelFrame(
            data=_DASK_2CH,
            sampling_rate=_SAMPLE_RATE,
            label="test_audio",
            channel_metadata=[
                {"label": "ch0", "unit": "", "extra": {}},
                {"label": "ch1", "unit": "", "extra": {}},
            ],
        )

        result = frame.normalize()

        # Verify labels are updated (using display name "norm")
        assert result.labels == ["norm(ch0)", "norm(ch1)"]
        # Pillar 1: immutability — new instance, original unchanged
        assert result is not frame
        assert isinstance(result._data, DaArray)  # Dask laziness preserved
        assert frame.labels == ["ch0", "ch1"]
        assert frame.sampling_rate == _SAMPLE_RATE

    def test_low_pass_filter_updates_labels(self) -> None:
        """Test that low_pass_filter updates channel labels."""
        frame = ChannelFrame(
            data=_DASK_2CH,
            sampling_rate=_SAMPLE_RATE,
            label="test_audio",
            channel_metadata=[
                {"label": "acc_x", "unit": "m/s^2", "extra": {}},
                {"label": "acc_y", "unit": "m/s^2", "extra": {}},
            ],
        )

        result = frame.low_pass_filter(cutoff=1000)

        assert result.labels == ["lpf(acc_x)", "lpf(acc_y)"]
        assert isinstance(result._data, DaArray)  # Dask laziness preserved

    def test_high_pass_filter_updates_labels(self) -> None:
        """Test that high_pass_filter updates channel labels."""
        frame = ChannelFrame(
            data=_DASK_2CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[
                {"label": "signal_a", "unit": "V", "extra": {}},
                {"label": "signal_b", "unit": "V", "extra": {}},
            ],
        )

        result = frame.high_pass_filter(cutoff=100)

        assert result.labels == [
            "hpf(signal_a)",
            "hpf(signal_b)",
        ]
        assert isinstance(result._data, DaArray)  # Dask laziness preserved

    def test_band_pass_filter_updates_labels(self) -> None:
        """Test that band_pass_filter updates channel labels."""
        frame = ChannelFrame(
            data=_DASK_2CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[
                {"label": "mic1", "unit": "Pa", "extra": {}},
                {"label": "mic2", "unit": "Pa", "extra": {}},
            ],
        )

        result = frame.band_pass_filter(low_cutoff=200, high_cutoff=5000)

        assert result.labels == ["bpf(mic1)", "bpf(mic2)"]

    def test_a_weighting_updates_labels(self) -> None:
        """Test that a_weighting updates channel labels."""
        frame = ChannelFrame(
            data=_DASK_2CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[
                {"label": "audio_left", "unit": "Pa", "extra": {}},
                {"label": "audio_right", "unit": "Pa", "extra": {}},
            ],
        )

        result = frame.a_weighting()

        assert result.labels == ["Aw(audio_left)", "Aw(audio_right)"]

    def test_sound_level_updates_labels(self) -> None:
        """Test that sound_level operation updates channel labels."""
        frame = ChannelFrame(
            data=_DASK_2CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[
                {"label": "audio_left", "unit": "Pa", "extra": {}},
                {"label": "audio_right", "unit": "Pa", "extra": {}},
            ],
        )

        result = frame.sound_level(freq_weighting="A", time_weighting="Fast", dB=True)

        assert result.labels == ["LAF(audio_left)", "LAF(audio_right)"]

    def test_sound_level_linear_output_updates_labels(self) -> None:
        """Test that linear sound_level output uses RMS-aware labels."""
        frame = ChannelFrame(
            data=_DASK_2CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[
                {"label": "audio_left", "unit": "Pa", "extra": {}},
                {"label": "audio_right", "unit": "Pa", "extra": {}},
            ],
        )

        result = frame.sound_level(freq_weighting="A", time_weighting="Fast", dB=False)

        assert result.labels == ["AFRMS(audio_left)", "AFRMS(audio_right)"]

    def test_abs_updates_labels(self) -> None:
        """Test that abs operation updates channel labels."""
        frame = ChannelFrame(
            data=_DASK_2CH,
            sampling_rate=_SAMPLE_RATE,
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
            data=_DASK_2CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[
                {"label": "signal", "unit": "V", "extra": {}},
                {"label": "reference", "unit": "V", "extra": {}},
            ],
        )

        result = frame.power(exponent=2.0)

        assert result.labels == ["pow(signal)", "pow(reference)"]


class TestChainedOperationLabels:
    """Test label updates for chained operations."""

    def test_chained_operations_nest_labels(self) -> None:
        """Test that chained operations properly nest labels."""
        frame = ChannelFrame(
            data=_DASK_1CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[{"label": "ch0", "unit": "", "extra": {}}],
        )

        result = frame.normalize().low_pass_filter(cutoff=1000)

        assert result.labels == ["lpf(norm(ch0))"]
        assert isinstance(result._data, DaArray)  # Dask laziness preserved

    def test_triple_chained_operations(self) -> None:
        """Test three operations chained together."""
        frame = ChannelFrame(
            data=_DASK_1CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[{"label": "raw", "unit": "", "extra": {}}],
        )

        result = frame.normalize().high_pass_filter(cutoff=100).low_pass_filter(cutoff=1000)

        assert result.labels == ["lpf(hpf(norm(raw)))"]

    def test_chained_operations_preserve_metadata(self) -> None:
        """Test that chained operations preserve non-label metadata."""
        frame = ChannelFrame(
            data=_DASK_1CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[{"label": "ch0", "unit": "Pa", "extra": {"sensor_id": 123}}],
        )

        result = frame.normalize().low_pass_filter(cutoff=1000)

        # Label should be updated (using display names)
        assert result.labels == ["lpf(norm(ch0))"]
        # But other metadata should be preserved
        assert result.channels[0].unit == "Pa"
        assert result.channels[0].extra == {"sensor_id": 123}


class TestBinaryOperationLabelCompatibility:
    """Test that updated labels work with binary operations."""

    def test_binary_op_with_processed_frame(self) -> None:
        """Test binary operation with a frame that has updated labels."""
        frame1 = ChannelFrame(
            data=_DASK_1CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[{"label": "ch0", "unit": "", "extra": {}}],
        ).normalize()

        frame2 = ChannelFrame(
            data=_DASK_1CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[{"label": "ch0", "unit": "", "extra": {}}],
        )

        result = frame1 + frame2

        # Binary operation should include the processed label (display name "norm")
        assert "norm(ch0)" in result.labels[0]
        assert "ch0" in result.labels[0]

    def test_add_two_processed_frames(self) -> None:
        """Test adding two frames with different processing."""
        frame1 = ChannelFrame(
            data=_DASK_1CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[{"label": "ch0", "unit": "", "extra": {}}],
        ).normalize()

        frame2 = ChannelFrame(
            data=_DASK_1CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[{"label": "ch0", "unit": "", "extra": {}}],
        ).low_pass_filter(cutoff=1000)

        result = frame1 + frame2

        # Both processed labels should appear (using display names)
        assert "norm(ch0)" in result.labels[0]
        assert "lpf(ch0)" in result.labels[0]


class TestEdgeCases:
    """Test edge cases for label updates."""

    def test_operation_on_single_channel(self) -> None:
        """Test operations on single-channel frame."""
        frame = ChannelFrame(
            data=_DASK_1CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[{"label": "mono", "unit": "", "extra": {}}],
        )

        result = frame.normalize()

        assert len(result.labels) == 1
        assert result.labels[0] == "norm(mono)"

    def test_operation_with_special_characters_in_label(self) -> None:
        """Test that special characters in labels are handled correctly."""
        frame = ChannelFrame(
            data=_DASK_1CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[{"label": "sensor-1_output", "unit": "", "extra": {}}],
        )

        result = frame.normalize()

        assert result.labels[0] == "norm(sensor-1_output)"

    def test_label_update_preserves_channel_metadata_structure(self) -> None:
        """Test that label updates preserve the ChannelMetadata structure."""
        from wandas.core.metadata import ChannelMetadata

        frame = ChannelFrame(
            data=_DASK_1CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[ChannelMetadata(label="ch0", unit="Pa", ref=2e-5, extra={"calibration": 1.0})],
        )

        result = frame.normalize()

        # Check that the result has ChannelMetadata objects
        assert isinstance(result.channels[0], ChannelMetadata)
        # Check that all fields are preserved except label (which uses display name)
        assert result.channels[0].label == "norm(ch0)"
        assert result.channels[0].unit == "Pa"
        assert result.channels[0].ref == 2e-5
        assert result.channels[0].extra == {"calibration": 1.0}


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_default_channel_labels_still_work(self) -> None:
        """Test that default channel labels (ch0, ch1, ...) still work."""
        frame = ChannelFrame(
            data=_DASK_2CH,
            sampling_rate=_SAMPLE_RATE,
        )

        # Default labels should be ch0, ch1
        assert frame.labels == ["ch0", "ch1"]

        # After operation, they should be updated (using display name)
        result = frame.normalize()
        assert result.labels == ["norm(ch0)", "norm(ch1)"]

    def test_operation_history_still_tracked(self) -> None:
        """Test that operation history is still tracked correctly."""
        frame = ChannelFrame(
            data=_DASK_2CH,
            sampling_rate=_SAMPLE_RATE,
        )

        result = frame.normalize().low_pass_filter(cutoff=1000)

        # Operation history should have two entries
        assert len(result.operation_history) == 2
        assert result.operation_history[0]["operation"] == "normalize"
        assert result.operation_history[1]["operation"] == "lowpass_filter"

    def test_previous_reference_maintained(self) -> None:
        """Test that the previous frame reference is maintained."""
        frame = ChannelFrame(
            data=_DASK_2CH,
            sampling_rate=_SAMPLE_RATE,
        )

        result = frame.normalize()

        assert result.previous is frame
        assert result.previous.labels == ["ch0", "ch1"]


class TestAddChannelWithLabelPrefix:
    """Test add_channel with label prefix for ChannelFrame input."""

    def test_add_channel_frame_with_label_prefix(self) -> None:
        """Test that label parameter adds prefix to ChannelFrame channel labels."""
        frame1 = ChannelFrame(
            data=_DASK_2CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[
                {"label": "left", "unit": "", "extra": {}},
                {"label": "right", "unit": "", "extra": {}},
            ],
        )

        # Create another frame to add (same structure)
        frame2 = ChannelFrame(
            data=_DASK_2CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[
                {"label": "left", "unit": "", "extra": {}},
                {"label": "right", "unit": "", "extra": {}},
            ],
        )

        # Add with label prefix
        result = frame1.add_channel(frame2, label="ref")

        # Existing channels should remain unchanged
        assert len(result.labels) == 4
        assert result.labels[0] == "left"
        assert result.labels[1] == "right"
        # New channels should have prefix applied
        assert result.labels[2] == "ref_left"
        assert result.labels[3] == "ref_right"

    def test_add_channel_frame_without_label_prefix(self) -> None:
        """Test that without label parameter, original labels are used."""
        frame1 = ChannelFrame(
            data=_DASK_2CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[
                {"label": "left", "unit": "", "extra": {}},
                {"label": "right", "unit": "", "extra": {}},
            ],
        )

        frame2 = ChannelFrame(
            data=_DASK_2CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[
                {"label": "vocals", "unit": "", "extra": {}},
                {"label": "instrumental", "unit": "", "extra": {}},
            ],
        )

        result = frame1.add_channel(frame2)  # No label parameter

        assert result.labels == ["left", "right", "vocals", "instrumental"]

    def test_add_channel_frame_with_label_prefix_and_suffix_on_dup(self) -> None:
        """Test label prefix with suffix_on_dup for handling duplicates."""
        frame1 = ChannelFrame(
            data=_DASK_2CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[{"label": "ch0", "unit": "", "extra": {}}],
        )

        frame2 = ChannelFrame(
            data=_DASK_2CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[{"label": "ch0", "unit": "", "extra": {}}],
        )

        # With label prefix, should not conflict
        result = frame1.add_channel(frame2, label="ref")
        assert result.labels == ["ch0", "ref_ch0"]

        # Without label, would conflict and use suffix_on_dup
        result2 = frame1.add_channel(frame2, suffix_on_dup="_dup")
        assert result2.labels == ["ch0", "ch0_dup"]


class TestRenameChannels:
    """Test rename_channels method."""

    def test_rename_channels_by_index(self) -> None:
        """Test renaming channels using index keys."""
        frame = ChannelFrame(
            data=_DASK_2CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[
                {"label": "ch0", "unit": "", "extra": {}},
                {"label": "ch1", "unit": "", "extra": {}},
            ],
        )

        result = frame.rename_channels({0: "left", 1: "right"})

        assert result.labels == ["left", "right"]

    def test_rename_channels_by_label(self) -> None:
        """Test renaming channels using label keys."""
        frame = ChannelFrame(
            data=_DASK_2CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[
                {"label": "ch0", "unit": "", "extra": {}},
                {"label": "ch1", "unit": "", "extra": {}},
            ],
        )

        result = frame.rename_channels({"ch0": "left", "ch1": "right"})

        assert result.labels == ["left", "right"]

    def test_rename_channels_partial(self) -> None:
        """Test renaming only some channels."""
        frame = ChannelFrame(
            data=_DASK_2CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[
                {"label": "ch0", "unit": "", "extra": {}},
                {"label": "ch1", "unit": "", "extra": {}},
            ],
        )

        result = frame.rename_channels({0: "left"})  # Only rename first channel

        assert result.labels == ["left", "ch1"]

    def test_rename_channels_inplace(self) -> None:
        """Test renaming channels with inplace=True."""
        frame = ChannelFrame(
            data=_DASK_2CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[
                {"label": "ch0", "unit": "", "extra": {}},
                {"label": "ch1", "unit": "", "extra": {}},
            ],
        )

        result = frame.rename_channels({0: "left", 1: "right"}, inplace=True)

        assert result is frame
        assert frame.labels == ["left", "right"]

    def test_rename_channels_nonexistent_index_error(self) -> None:
        """Test error when renaming non-existent index."""
        frame = ChannelFrame(
            data=_DASK_2CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[{"label": "ch0", "unit": "", "extra": {}}],
        )

        with pytest.raises(KeyError):
            frame.rename_channels({5: "invalid"})

    def test_rename_channels_nonexistent_label_error(self) -> None:
        """Test error when renaming non-existent label."""
        frame = ChannelFrame(
            data=_DASK_2CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[{"label": "ch0", "unit": "", "extra": {}}],
        )

        with pytest.raises(KeyError):
            frame.rename_channels({"nonexistent": "new_label"})

    def test_rename_channels_duplicate_error(self) -> None:
        """Test error when rename would create duplicate labels."""
        frame = ChannelFrame(
            data=_DASK_2CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[{"label": "ch0", "unit": "", "extra": {}}, {"label": "ch1", "unit": "", "extra": {}}],
        )

        with pytest.raises(ValueError):
            frame.rename_channels({"ch0": "ch1"})  # Would create duplicate

    def test_rename_channels_preserves_metadata(self) -> None:
        """Test that rename_channels preserves channel metadata."""
        from wandas.core.metadata import ChannelMetadata

        frame = ChannelFrame(
            data=_DASK_2CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[
                ChannelMetadata(label="ch0", unit="Pa", ref=2e-5, extra={"calibration": 1.0}),
                ChannelMetadata(label="ch1", unit="V", ref=1.0, extra={"gain": 10}),
            ],
        )

        result = frame.rename_channels({0: "left", 1: "right"})

        assert result.labels == ["left", "right"]
        # Metadata should be preserved
        assert result.channels[0].unit == "Pa"
        assert result.channels[0].ref == 2e-5
        assert result.channels[0].extra == {"calibration": 1.0}
        assert result.channels[1].unit == "V"
        assert result.channels[1].extra == {"gain": 10}

    def test_rename_channels_noop_succeeds(self) -> None:
        """Test that renaming a channel to its current label (no-op) succeeds."""
        frame = ChannelFrame(
            data=_DASK_2CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[
                {"label": "ch0", "unit": "", "extra": {}},
                {"label": "ch1", "unit": "", "extra": {}},
            ],
        )

        result = frame.rename_channels({0: "ch0"})

        assert result.labels == ["ch0", "ch1"]

    def test_rename_channels_swap_succeeds(self) -> None:
        """Test that swapping two channel labels succeeds without duplicate error."""
        frame = ChannelFrame(
            data=_DASK_2CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[
                {"label": "ch0", "unit": "", "extra": {}},
                {"label": "ch1", "unit": "", "extra": {}},
            ],
        )

        result = frame.rename_channels({0: "ch1", 1: "ch0"})

        assert result.labels == ["ch1", "ch0"]

    def test_rename_channels_mixed_key_duplicate_index_error(self) -> None:
        """Test error when both an int index and a string label target the same channel."""
        frame = ChannelFrame(
            data=_DASK_2CH,
            sampling_rate=_SAMPLE_RATE,
            channel_metadata=[
                {"label": "ch0", "unit": "", "extra": {}},
                {"label": "ch1", "unit": "", "extra": {}},
            ],
        )

        # Index 0 and label "ch0" both refer to channel 0 — ambiguous mapping
        with pytest.raises(ValueError):
            frame.rename_channels({0: "a", "ch0": "b"})
