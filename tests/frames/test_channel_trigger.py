import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.frames.channel import ChannelFrame
from wandas.utils.frame_dataset import ChannelFrameDataset

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


class TestFindTriggers:
    def test_returns_trigger_times_in_seconds(self) -> None:
        times = _hammer_frame().find_triggers(10.0)
        assert isinstance(times, np.ndarray)
        np.testing.assert_allclose(times, _HITS)

    def test_trigger_channel_can_be_selected_by_label(self) -> None:
        cf = _hammer_frame()
        np.testing.assert_allclose(cf.find_triggers(0.5, channel="response"), _HITS)

    def test_threshold_uses_calibrated_physical_unit(self) -> None:
        cf = _hammer_frame().with_calibration({"force": 0.1})
        assert cf.find_triggers(10.0).size == 0
        np.testing.assert_allclose(cf.find_triggers(1.0), _HITS)

    def test_falling_slope_detects_end_of_pulse(self) -> None:
        times = _hammer_frame().find_triggers(10.0, slope="falling")
        np.testing.assert_allclose(times, np.array(_HITS) + 5 / _SR)

    def test_holdoff_in_seconds_suppresses_close_triggers(self) -> None:
        times = _hammer_frame().find_triggers(10.0, holdoff=1.2)
        np.testing.assert_allclose(times, [0.5, 3.0])

    def test_detection_does_not_change_frame(self) -> None:
        cf = _hammer_frame()
        history = list(cf.operation_history)
        cf.find_triggers(10.0)
        assert cf.operation_history == history

    @pytest.mark.parametrize("holdoff", [-0.0004, float("nan"), float("inf")])
    def test_invalid_holdoff_raises(self, holdoff: float) -> None:
        with pytest.raises(ValueError, match="finite and non-negative"):
            _hammer_frame().find_triggers(10.0, holdoff=holdoff)

    def test_unknown_label_raises(self) -> None:
        with pytest.raises(KeyError):
            _hammer_frame().find_triggers(10.0, channel="missing")


class TestSegment:
    def test_segments_cut_all_channels_with_pre_trigger(self) -> None:
        cf = _hammer_frame()
        segments = cf.segment(_HITS, duration=0.2, pre=0.01)
        assert isinstance(segments, ChannelFrameDataset)
        assert len(segments) == len(_HITS)
        for index, hit in enumerate(_HITS):
            seg = segments[index]
            assert isinstance(seg, ChannelFrame)
            assert seg.labels == ["force", "response"]
            assert seg.n_samples == 200
            np.testing.assert_allclose(seg.source_time_offset, [hit - 0.01, hit - 0.01])
            start = int(round((hit - 0.01) * _SR))
            np.testing.assert_array_equal(seg.data, cf.data[:, start : start + 200])

    def test_segments_are_lazy_with_one_lineage_entry(self) -> None:
        cf = _hammer_frame()
        seg = cf.segment(0.5, duration=0.1)[0]
        assert seg is not None
        assert isinstance(seg._data, DaArray)
        assert len(seg.operation_history) == len(cf.operation_history) + 1

    def test_segments_preserve_calibration(self) -> None:
        cf = _hammer_frame().with_calibration({"force": 0.1})
        seg = cf.segment(0.5, duration=0.1)[0]
        assert seg is not None
        assert seg.channels[0].calibration == cf.channels[0].calibration

    def test_events_outside_signal_are_skipped_with_warning(self) -> None:
        cf = _hammer_frame()
        with pytest.warns(UserWarning, match="Skipped 2 segment"):
            segments = cf.segment([0.005, 1.5, 3.95], duration=0.1, pre=0.01)
        assert len(segments) == 1
        seg = segments[0]
        assert seg is not None
        np.testing.assert_allclose(seg.source_time_offset, [1.49, 1.49])
        assert seg.metadata["event"] == 1

    def test_items_carry_event_metadata_for_selection(self) -> None:
        cf = _hammer_frame()
        segments = cf.segment(_HITS, duration=0.2, pre=0.01)
        seg = segments.select(event=2)[0]
        assert seg is not None
        assert seg.metadata["trigger_time"] == 3.0
        np.testing.assert_allclose(seg.source_time_offset, [2.99, 2.99])

    def test_dataset_methods_apply_to_every_event(self) -> None:
        segments = _hammer_frame().segment(_HITS, duration=0.2, pre=0.01)
        spectrograms = segments.stft(n_fft=64, hop_length=32)
        assert len(spectrograms) == len(_HITS)
        assert all(spectrograms[i] is not None for i in range(len(_HITS)))

    def test_find_triggers_and_segment_compose(self) -> None:
        cf = _hammer_frame()
        segments = cf.segment(cf.find_triggers(10.0, channel="force"), duration=0.2, pre=0.01)
        for seg in segments:
            assert seg is not None
            assert seg.data[0, 10] == 20.0
            assert seg.data[0, 9] == 0.0

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
        ],
    )
    def test_invalid_arguments_raise(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            _hammer_frame().segment(**kwargs)
