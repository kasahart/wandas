import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.frames.channel import ChannelFrame
from wandas.frames.segments import ChannelFrameSegments

_SR = 1000
_HITS = (0.5, 1.5, 3.0)


def _hammer_frame() -> ChannelFrame:
    """Force pulses at known times on ch0 and a decaying response on ch1."""
    data = np.zeros((2, 4 * _SR))
    for hit in _HITS:
        start = int(hit * _SR)
        data[0, start : start + 5] = 20.0
        data[1, start : start + 100] = np.exp(-np.arange(100) / 20.0)
    return ChannelFrame.from_numpy(data, sampling_rate=_SR, ch_labels=["force", "response"])


class TestSegment:
    def test_segments_cut_all_channels_with_pre_trigger(self) -> None:
        cf = _hammer_frame()
        segments = cf.segment(_HITS, duration=0.2, pre=0.01)
        assert isinstance(segments, ChannelFrameSegments)
        assert len(segments) == len(_HITS)
        assert segments.pre == 0.01
        assert segments.trigger_channel is None
        np.testing.assert_array_equal(segments.event_ids, [0, 1, 2])
        np.testing.assert_allclose(segments.trigger_times, _HITS)
        for seg, hit in zip(segments, _HITS, strict=True):
            assert isinstance(seg, ChannelFrame)
            assert seg.labels == ["force", "response"]
            assert seg.n_samples == 200
            np.testing.assert_allclose(seg.source_time_offset, [hit - 0.01, hit - 0.01])
            start = int(round((hit - 0.01) * _SR))
            np.testing.assert_array_equal(seg.data, cf.data[:, start : start + 200])

    def test_segments_are_lazy_with_one_lineage_entry(self) -> None:
        cf = _hammer_frame()
        seg = cf.segment(0.5, duration=0.1)[0]
        assert isinstance(seg._data, DaArray)
        assert len(seg.operation_history) == len(cf.operation_history) + 1

    def test_segments_preserve_calibration_and_metadata(self) -> None:
        data = _hammer_frame().data
        cf = ChannelFrame.from_numpy(data, sampling_rate=_SR, metadata={"test": "A"}).with_calibration({0: 0.1})
        seg = cf.segment(0.5, duration=0.1)[0]
        assert seg.channels[0].calibration == cf.channels[0].calibration
        assert seg.metadata == {"test": "A"}

    def test_events_outside_signal_are_skipped_with_warning(self) -> None:
        cf = _hammer_frame()
        with pytest.warns(UserWarning, match="Skipped 2 segment"):
            segments = cf.segment([0.005, 1.5, 3.95], duration=0.1, pre=0.01)
        assert len(segments) == 1
        np.testing.assert_array_equal(segments.event_ids, [1])
        np.testing.assert_allclose(segments[0].source_time_offset, [1.49, 1.49])

    def test_all_events_outside_signal_give_empty_segments(self) -> None:
        with pytest.warns(UserWarning, match="Skipped 1 segment"):
            segments = _hammer_frame().segment(3.99, duration=0.1)
        assert len(segments) == 0
        with pytest.raises(ValueError, match="no segments"):
            segments.transfer_function(input="force")

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"times": [[0.5]], "duration": 0.1}, "1-D"),
            ({"times": 0.5, "duration": 0.0001}, "shorter than one sample"),
            ({"times": 0.5, "duration": 0.1, "pre": -0.1}, "non-negative"),
            ({"times": [np.nan, 1.5], "duration": 0.1}, "finite"),
            ({"times": [np.inf], "duration": 0.1}, "finite"),
            ({"times": 0.5, "duration": float("nan")}, "finite"),
            ({"times": 0.5, "duration": 0.1, "pre": float("nan")}, "finite"),
            ({"times": 0.5, "duration": 0.1, "pre": 0.1}, "shorter than the segment duration"),
            ({"times": 0.5, "duration": 0.1, "pre": 0.2}, "shorter than the segment duration"),
        ],
    )
    def test_invalid_arguments_raise(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            _hammer_frame().segment(**kwargs)


class TestTrigger:
    def test_trigger_cuts_every_hit(self) -> None:
        cf = _hammer_frame()
        hits = cf.trigger(10.0, duration=0.2, pre=0.01, channel="force")
        assert isinstance(hits, ChannelFrameSegments)
        np.testing.assert_allclose(hits.trigger_times, _HITS)
        assert hits.trigger_channel == "force"
        expected = cf.segment(_HITS, duration=0.2, pre=0.01)
        for hit, ref in zip(hits, expected, strict=True):
            np.testing.assert_array_equal(hit.data, ref.data)
            assert hit.data[0, 10] == 20.0
            assert hit.data[0, 9] == 0.0

    def test_trigger_channel_can_be_selected_by_index(self) -> None:
        hits = _hammer_frame().trigger(0.5, duration=0.2, channel=1)
        np.testing.assert_allclose(hits.trigger_times, _HITS)
        assert hits.trigger_channel == "response"

    def test_negative_trigger_channel_index_counts_from_the_end(self) -> None:
        hits = _hammer_frame().trigger(0.5, duration=0.2, channel=-1)
        np.testing.assert_allclose(hits.trigger_times, _HITS)
        assert hits.trigger_channel == "response"

    def test_threshold_uses_calibrated_physical_unit(self) -> None:
        cf = _hammer_frame().with_calibration({"force": 0.1})
        assert len(cf.trigger(10.0, duration=0.2)) == 0
        np.testing.assert_allclose(cf.trigger(1.0, duration=0.2).trigger_times, _HITS)

    def test_falling_slope_detects_end_of_pulse(self) -> None:
        hits = _hammer_frame().trigger(10.0, duration=0.2, slope="falling")
        np.testing.assert_allclose(hits.trigger_times, np.array(_HITS) + 5 / _SR)

    def test_default_holdoff_waits_for_the_end_of_the_segment(self) -> None:
        cf = _hammer_frame()
        hits = cf.trigger(10.0, duration=0.6, pre=0.1)
        np.testing.assert_allclose(hits.trigger_times, _HITS)
        # 1.5 s falls within 1.1 s of the hit at 0.5 s, so it is not a new trigger.
        with pytest.warns(UserWarning, match="Skipped 1 segment"):
            long_hits = cf.trigger(10.0, duration=1.2, pre=0.1)
        np.testing.assert_allclose(long_hits.trigger_times, [0.5])

    def test_explicit_holdoff_is_used(self) -> None:
        hits = _hammer_frame().trigger(10.0, duration=0.2, holdoff=1.2)
        np.testing.assert_allclose(hits.trigger_times, [0.5, 3.0])
        np.testing.assert_array_equal(hits.event_ids, [0, 1])

    def test_default_holdoff_is_counted_in_segment_samples(self) -> None:
        data = np.zeros((1, 100))
        data[0, [10, 13]] = 1.0
        cf = ChannelFrame.from_numpy(data, sampling_rate=1000)
        # 3.5 samples round to 4 and 0.5 to 0, so the segment at 10 covers 10-13.
        hits = cf.trigger(0.5, duration=0.0035, pre=0.0005)
        np.testing.assert_allclose(hits.trigger_times, [0.01])

    def test_chains_from_a_processed_frame(self) -> None:
        hits = _hammer_frame().high_pass_filter(cutoff=5).trigger(5.0, duration=0.2, pre=0.01)
        assert len(hits) == len(_HITS)

    def test_detection_does_not_change_frame(self) -> None:
        cf = _hammer_frame()
        history = list(cf.operation_history)
        cf.trigger(10.0, duration=0.2)
        assert cf.operation_history == history

    @pytest.mark.parametrize("holdoff", [-0.0004, float("nan"), float("inf")])
    def test_invalid_holdoff_raises(self, holdoff: float) -> None:
        with pytest.raises(ValueError, match="finite and non-negative"):
            _hammer_frame().trigger(10.0, duration=0.2, holdoff=holdoff)

    def test_unknown_label_raises(self) -> None:
        with pytest.raises(KeyError):
            _hammer_frame().trigger(10.0, duration=0.2, channel="missing")

    @pytest.mark.parametrize(("duration", "pre"), [(float("nan"), 0.0), (0.1, float("nan")), (-1.0, 0.0)])
    def test_invalid_window_is_reported_as_segment_error(self, duration: float, pre: float) -> None:
        with pytest.raises(ValueError, match="Segment duration"):
            _hammer_frame().trigger(10.0, duration=duration, pre=pre)
