import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from dask.array.core import Array as DaArray
from matplotlib.axes import Axes

from wandas.frames.channel import ChannelFrame
from wandas.frames.pairwise import CoherenceFrame, TransferFunctionFrame
from wandas.frames.segments import ChannelFrameSegments, FrameSegments
from wandas.frames.spectral import SpectralFrame
from wandas.pipeline import RecipeExecutionError, RecipeExtractionError, RecipePlan
from wandas.processing.semantic import thaw_params

_SR = 1000
_LENGTH = 200
_N_HITS = 5


def _hits(n_channels: int = 3) -> tuple[ChannelFrameSegments, np.ndarray]:
    """Random segments placed back to back, and their values as (hit, channel, sample)."""
    rng = np.random.default_rng(0)
    values = rng.standard_normal((_N_HITS, n_channels, _LENGTH))
    data = np.concatenate(list(values), axis=-1)
    labels = ["force", "accel", "mic"][:n_channels]
    cf = ChannelFrame.from_numpy(data, sampling_rate=_SR, ch_labels=labels)
    times = np.arange(_N_HITS) * _LENGTH / _SR
    return cf.segment(times, duration=_LENGTH / _SR), values


def _with_trigger_channel(hits: ChannelFrameSegments, index: int = 0) -> ChannelFrameSegments:
    return ChannelFrameSegments(list(hits), hits.event_ids, hits.trigger_times, pre=hits.pre, trigger_channel=index)


def _spectra(values: np.ndarray, n_fft: int = _LENGTH, window: np.ndarray | None = None) -> np.ndarray:
    weights = np.ones(values.shape[-1]) if window is None else window
    return np.fft.rfft(values * weights, n=n_fft, axis=-1)


def _expected_h1(values: np.ndarray, n_fft: int = _LENGTH, window: np.ndarray | None = None) -> np.ndarray:
    spectra = _spectra(values, n_fft=n_fft, window=window)
    x, y = spectra[:, 0], spectra[:, 1:]
    return (np.conj(x)[:, None] * y).sum(axis=0) / (np.abs(x) ** 2).sum(axis=0)


def _expected_coherence(values: np.ndarray) -> np.ndarray:
    spectra = _spectra(values)
    x, y = spectra[:, 0], spectra[:, 1:]
    cross = (np.conj(x)[:, None] * y).sum(axis=0)
    return np.abs(cross) ** 2 / ((np.abs(x) ** 2).sum(axis=0) * (np.abs(y) ** 2).sum(axis=0))


class TestCollection:
    def test_integer_index_returns_one_frame(self) -> None:
        hits, values = _hits()
        assert isinstance(hits[1], ChannelFrame)
        np.testing.assert_allclose(hits[1].data, values[1])
        np.testing.assert_allclose(hits[-1].data, values[-1])

    def test_selection_keeps_event_ids_and_type(self) -> None:
        hits, _ = _hits()
        for subset in (hits[1:3], hits[[1, 2]], hits[np.array([False, True, True, False, False])]):
            assert isinstance(subset, ChannelFrameSegments)
            np.testing.assert_array_equal(subset.event_ids, [1, 2])
            np.testing.assert_allclose(subset.trigger_times, [0.2, 0.4])
            assert subset.pre == hits.pre

    def test_drop_uses_event_ids_not_positions(self) -> None:
        hits, values = _hits()
        kept = hits.drop(0).drop([2, 4])
        np.testing.assert_array_equal(kept.event_ids, [1, 3])
        np.testing.assert_allclose(kept[1].data, values[3])

    @pytest.mark.parametrize(
        ("ids", "error", "match"), [(7, KeyError, "Unknown segment event IDs"), ([1, 1], ValueError, "repeat")]
    )
    def test_drop_rejects_unknown_or_repeated_ids(self, ids: object, error: type[Exception], match: str) -> None:
        hits, _ = _hits()
        with pytest.raises(error, match=match):
            hits.drop(ids)  # ty: ignore[invalid-argument-type]

    @pytest.mark.parametrize(
        ("key", "error", "match"),
        [
            ([0, 0], ValueError, "repeat"),
            ([5], IndexError, "out of range"),
            (np.array([True, False]), ValueError, "mask length"),
            ("force", TypeError, "indexed by"),
        ],
    )
    def test_invalid_selection_raises(self, key: object, error: type[Exception], match: str) -> None:
        hits, _ = _hits()
        with pytest.raises(error, match=match):
            hits[key]  # ty: ignore[invalid-argument-type]

    def test_event_ids_and_times_are_read_only(self) -> None:
        hits, _ = _hits()
        with pytest.raises(ValueError):
            hits.event_ids[0] = 9
        with pytest.raises(ValueError):
            hits.trigger_times[0] = 9.0

    def test_event_ids_and_times_are_copied_from_the_caller(self) -> None:
        hits, _ = _hits()
        ids = np.array([5, 6], dtype=np.int64)
        times = np.array([0.1, 0.2])
        segments = FrameSegments([hits[0], hits[1]], ids, times)
        ids[0] = 6
        times[0] = 9.0
        np.testing.assert_array_equal(segments.event_ids, [5, 6])
        np.testing.assert_allclose(segments.trigger_times, [0.1, 0.2])

    def test_properties_and_repr(self) -> None:
        hits, _ = _hits()
        assert hits.labels == ["force", "accel", "mic"]
        assert hits.sampling_rate == _SR
        assert hits.n_samples == _LENGTH
        assert hits.duration == pytest.approx(0.2)
        assert "n_segments=5" in repr(hits)

    def test_to_numpy_stacks_segments_and_keeps_channel_axis(self) -> None:
        hits, values = _hits(n_channels=1)
        array = hits.to_numpy()
        assert array.shape == (_N_HITS, 1, _LENGTH)
        np.testing.assert_allclose(array, values)

    def test_mismatched_segments_are_rejected(self) -> None:
        hits, _ = _hits()
        with pytest.raises(ValueError, match="differs from the first"):
            FrameSegments([hits[0], hits[0][:, :10]], [0, 1], [0.0, 1.0])
        renamed = hits[1].rename_channels({"mic": "other"})
        with pytest.raises(ValueError, match="other channels"):
            ChannelFrameSegments([hits[0], renamed], [0, 1], [0.0, 1.0])
        with pytest.raises(ValueError, match="unique"):
            FrameSegments([hits[0], hits[1]], [3, 3], [0.0, 1.0])


class TestValidation:
    def test_counts_types_and_pre_are_checked(self) -> None:
        hits, _ = _hits()
        spectrum = hits[0].fft()
        with pytest.raises(ValueError, match="one event ID and one trigger time"):
            FrameSegments([hits[0]], [0, 1], [0.0])
        with pytest.raises(TypeError, match="must be Frames"):
            FrameSegments([np.zeros(3)], [0], [0.0])  # ty: ignore[invalid-argument-type]
        with pytest.raises(TypeError, match="another Frame type"):
            FrameSegments([hits[0], spectrum], [0, 1], [0.0, 1.0])
        with pytest.raises(TypeError, match="need ChannelFrames"):
            ChannelFrameSegments([spectrum], [0], [0.0])  # ty: ignore[invalid-argument-type]
        with pytest.raises(ValueError, match="finite and non-negative"):
            ChannelFrameSegments([hits[0]], [0], [0.0], pre=-1.0)
        with pytest.raises(ValueError, match="out of range"):
            ChannelFrameSegments([hits[0]], [0], [0.0], trigger_channel=3)

    def test_empty_segments(self) -> None:
        empty = ChannelFrameSegments([], [], [])
        assert len(empty) == 0
        assert repr(empty) == "ChannelFrameSegments(n_segments=0)"
        assert empty.trigger_channel is None
        with pytest.raises(ValueError, match="no segments to convert"):
            empty.to_numpy()
        with pytest.raises(ValueError, match="no segments"):
            empty.average()
        kept = ChannelFrameSegments([], [], [], pre=0.01).apply(lambda hit: hit)
        assert isinstance(kept, ChannelFrameSegments)
        assert kept.pre == 0.01

    def test_single_channel_has_no_output(self) -> None:
        hits, _ = _hits(n_channels=1)
        with pytest.raises(ValueError, match="no output channel"):
            hits.transfer_function(input=0)


class TestApply:
    def test_channel_frame_results_stay_channel_segments(self) -> None:
        hits = _with_trigger_channel(_hits()[0])
        filtered = hits.drop(1).apply(lambda hit: hit.high_pass_filter(cutoff=20))
        assert isinstance(filtered, ChannelFrameSegments)
        np.testing.assert_array_equal(filtered.event_ids, [0, 2, 3, 4])
        assert filtered.trigger_channel == filtered.labels[0]
        np.testing.assert_allclose(filtered[0].data, hits[0].high_pass_filter(cutoff=20).data)

    def test_other_frame_results_become_frame_segments(self) -> None:
        hits, _ = _hits()
        spectra = hits.apply(lambda hit: hit.fft())
        assert type(spectra) is FrameSegments
        assert isinstance(spectra[0], SpectralFrame)
        np.testing.assert_array_equal(spectra.event_ids, hits.event_ids)
        assert spectra.to_numpy().shape == (_N_HITS, 3, _LENGTH // 2 + 1)

    def test_generic_segments_select_and_describe(self) -> None:
        hits, _ = _hits()
        spectra = hits.apply(lambda hit: hit.fft())
        subset = spectra.drop(0)[:2]
        assert type(subset) is FrameSegments
        np.testing.assert_array_equal(subset.event_ids, [1, 2])
        assert repr(subset) == f"FrameSegments(n_segments=2, frame=SpectralFrame, shape=(3, {_LENGTH // 2 + 1}))"

    def test_changing_the_channel_count_forgets_the_trigger_channel(self) -> None:
        hits = _with_trigger_channel(_hits()[0])
        mono = hits.apply(lambda hit: hit[[1]])
        assert isinstance(mono, ChannelFrameSegments)
        assert mono.trigger_channel is None

    def test_reordered_channels_keep_the_trigger_channel(self) -> None:
        hits = _with_trigger_channel(_hits()[0])
        reordered = hits.apply(lambda hit: hit[[2, 1, 0]])
        assert isinstance(reordered, ChannelFrameSegments)
        assert reordered.trigger_channel == "force"
        renamed = hits.apply(lambda hit: hit.rename_channels({"force": "F"}))
        assert isinstance(renamed, ChannelFrameSegments)
        assert renamed.trigger_channel == "F"

    def test_moving_the_segment_start_updates_pre(self) -> None:
        cf = ChannelFrame.from_numpy(np.zeros((1, 1000)), sampling_rate=_SR)
        hits = cf.segment([0.2, 0.5], duration=0.1, pre=0.02)
        trimmed = hits.apply(lambda hit: hit[:, 10:])
        assert isinstance(trimmed, ChannelFrameSegments)
        assert trimmed.pre == pytest.approx(0.01)
        with pytest.raises(ValueError, match="cut the event out"):
            hits.apply(lambda hit: hit[:, 30:])
        starts = iter([5, 10])
        with pytest.raises(ValueError, match="different amounts"):
            hits.apply(lambda hit: hit[:, next(starts) :][:, :50])

    def test_cutting_the_segment_end_before_the_event_is_rejected(self) -> None:
        cf = ChannelFrame.from_numpy(np.zeros((1, 1000)), sampling_rate=_SR)
        hits = cf.segment([0.2, 0.5], duration=0.1, pre=0.02)
        shortened = hits.apply(lambda hit: hit[:, :21])
        assert isinstance(shortened, ChannelFrameSegments)
        assert shortened.pre == pytest.approx(0.02)
        for end in (10, 20):
            with pytest.raises(ValueError, match="cut the event out"):
                hits.apply(lambda hit, end=end: hit[:, :end])

    def test_results_of_different_shapes_are_rejected(self) -> None:
        hits, _ = _hits()
        lengths = iter(range(10, 20))
        with pytest.raises(ValueError, match="differs from the first"):
            hits.apply(lambda hit: hit[:, : next(lengths)])


class TestEnsembleAnalysis:
    def test_transfer_function_is_h1_over_segments(self) -> None:
        hits, values = _hits()
        frf = _with_trigger_channel(hits).transfer_function()
        assert isinstance(frf, TransferFunctionFrame)
        assert frf.shape == (2, _LENGTH // 2 + 1)
        np.testing.assert_allclose(frf.data, _expected_h1(values), rtol=1e-10, atol=1e-12)

    def test_transfer_function_channels_and_window(self) -> None:
        hits, values = _hits()
        frf = hits.transfer_function(input="force", output="mic", n_fft=256, window="hann")
        expected = _expected_h1(values[:, [0, 2]], n_fft=256, window=np.hanning(_LENGTH + 1)[:-1])
        assert frf.shape == (129,)
        np.testing.assert_allclose(frf.data, expected[0], rtol=1e-10, atol=1e-12)

    def test_transfer_function_follows_the_given_input(self) -> None:
        hits, values = _hits()
        frf = hits.transfer_function(input=1, output=[0])
        np.testing.assert_allclose(frf.data, _expected_h1(values[:, [1, 0]])[0], rtol=1e-10, atol=1e-12)

    def test_negative_channel_indices_are_resolved_before_pairing(self) -> None:
        hits, values = _hits()
        frf = hits.transfer_function(input=-3)
        assert frf.shape == (2, _LENGTH // 2 + 1)
        np.testing.assert_allclose(frf.data, _expected_h1(values), rtol=1e-10, atol=1e-12)
        with pytest.raises(ValueError, match="differ from the input"):
            hits.transfer_function(input=0, output=-3)

    def test_coherence_from_averaged_spectra(self) -> None:
        hits, values = _hits(n_channels=2)
        coherence = hits.coherence(input="force")
        assert isinstance(coherence, CoherenceFrame)
        np.testing.assert_allclose(coherence.data, _expected_coherence(values)[0], rtol=1e-8, atol=1e-12)
        assert np.all(coherence.data < 1.0)

    def test_coherence_needs_two_segments(self) -> None:
        hits, _ = _hits()
        with pytest.raises(ValueError, match="at least two segments"):
            hits[[0]].coherence(input="force")

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({}, "Give the input channel"),
            ({"input": "force", "output": "force"}, "differ from the input"),
            ({"input": "force", "output": ["mic", "mic"]}, "distinct"),
            ({"input": "force", "n_fft": 100}, "not be shorter than a segment"),
        ],
    )
    def test_invalid_analysis_arguments_raise(self, kwargs: dict, match: str) -> None:
        hits, _ = _hits()
        with pytest.raises(ValueError, match=match):
            hits.transfer_function(**kwargs)

    def test_analysis_is_lazy_and_records_segment_lineage(self) -> None:
        hits, _ = _hits()
        frf = hits.drop(3).transfer_function(input="force")
        assert isinstance(frf._data, DaArray)
        operations = [record["operation"] for record in frf.operation_history]
        assert operations[-3:] == [
            "wandas.audio.segment",
            "wandas.segments.drop",
            "wandas.segments.transfer_function",
        ]
        drop = next(record for record in frf.operation_history if record["operation"] == "wandas.segments.drop")
        assert drop["params"] == {"event_ids": [3]}

    def test_average_is_the_sample_mean_of_the_segments(self) -> None:
        hits, values = _hits()
        average = hits.average()
        assert isinstance(average, ChannelFrame)
        assert average.labels == hits.labels
        np.testing.assert_allclose(average.data, values.mean(axis=0))
        assert average.operation_history[-1]["operation"] == "wandas.segments.average"


def _impacts(seed: int = 0, n_hits: int = 6) -> ChannelFrame:
    """A force channel with half-sine hits and a response channel ringing after each."""
    rng = np.random.default_rng(seed)
    n = n_hits * _LENGTH + _LENGTH
    force = np.zeros(n)
    width = 5
    for hit in range(n_hits):
        start = _LENGTH // 2 + hit * _LENGTH
        force[start : start + width] += (8 + rng.random()) * np.sin(np.pi * np.arange(width) / width)
    t = np.arange(_LENGTH) / _SR
    impulse = np.exp(-t / 0.02) * np.sin(2 * np.pi * 120 * t)
    response = np.convolve(force, impulse)[:n] + 0.01 * rng.standard_normal(n)
    return ChannelFrame.from_numpy(
        np.stack([force, response]), sampling_rate=_SR, ch_labels=["force", "accel"], ch_units=["N", "m/s^2"]
    )


class TestRecipe:
    def _round_trip(self, frame: object) -> RecipePlan:
        plan = RecipePlan.from_frame(frame)
        return RecipePlan.from_dict(plan.to_dict())

    def test_trigger_drop_and_transfer_function_replay(self) -> None:
        cf = _impacts()
        hits = cf.high_pass_filter(cutoff=5).trigger(4.0, duration=0.15, pre=0.01, channel=0)
        frf = hits.drop([2]).transfer_function()
        plan = self._round_trip(frf)
        operations = [node.operation for node in plan.nodes]
        assert operations == [
            "wandas.audio.highpass_filter",
            "wandas.audio.trigger",
            "wandas.segments.drop",
            "wandas.segments.transfer_function",
        ]
        replayed = plan.apply(cf)
        assert isinstance(replayed, TransferFunctionFrame)
        np.testing.assert_allclose(replayed.data, frf.data)

    def test_replay_detects_events_on_another_recording(self) -> None:
        plan = RecipePlan.from_frame(_impacts(0).trigger(4.0, duration=0.15, pre=0.01).average())
        other = _impacts(1, n_hits=4)
        expected = other.trigger(4.0, duration=0.15, pre=0.01).average()
        np.testing.assert_allclose(plan.apply(other).data, expected.data)

    def test_segment_times_and_coherence_replay(self) -> None:
        cf = _impacts()
        times = np.array([0.1, 0.3, 0.5, 0.7])
        coherence = cf.segment(times, duration=0.15, pre=0.0).coherence(input="force")
        plan = self._round_trip(coherence)
        node = plan.nodes[0]
        assert node.operation == "wandas.audio.segment"
        np.testing.assert_allclose(plan.apply(cf).data, coherence.data)

    def test_selection_in_event_order_is_recorded_as_drop(self) -> None:
        hits, _ = _hits()
        mask = np.array([True, False, True, True, False])
        average = hits[mask].average()
        drop = next(node for node in RecipePlan.from_frame(average).nodes if node.operation == "wandas.segments.drop")
        assert thaw_params(drop.params) == {"event_ids": [1, 4]}

    def test_reordered_selection_is_not_a_recipe(self) -> None:
        hits, _ = _hits()
        with pytest.raises(RecipeExtractionError, match="another order"):
            RecipePlan.from_frame(hits[[2, 0]].average())

    def test_apply_is_not_a_recipe(self) -> None:
        hits, _ = _hits()
        with pytest.raises(RecipeExtractionError, match="Python function"):
            filtered = hits.apply(lambda hit: hit.high_pass_filter(cutoff=20))
            assert isinstance(filtered, ChannelFrameSegments)
            RecipePlan.from_frame(filtered.average())

    def test_segments_built_from_frames_are_not_a_recipe(self) -> None:
        hits = _with_trigger_channel(_hits()[0])
        with pytest.raises(RecipeExtractionError, match="built directly from Frames"):
            RecipePlan.from_frame(hits.average())

    def test_each_segment_keeps_its_own_time_slice_lineage(self) -> None:
        cf = _impacts()
        hits = cf.trigger(4.0, duration=0.15, pre=0.01)
        assert hits.lineage.operation is not None
        assert hits.lineage.operation.operation_id == "wandas.audio.trigger"
        plan = RecipePlan.from_frame(hits[1])
        assert [node.operation for node in plan.nodes] == ["wandas.frame.index"]

    def test_recipe_ending_with_segments_is_rejected_at_run_time(self) -> None:
        cf = _impacts()
        data = RecipePlan.from_frame(cf.trigger(4.0, duration=0.15, pre=0.01).average()).to_dict()
        data["nodes"] = data["nodes"][:1]
        data["output"] = data["nodes"][0]["id"]
        plan = RecipePlan.from_dict(data)
        with pytest.raises(RecipeExecutionError, match="not a Frame"):
            plan.apply(cf)

    def test_registry_describes_segment_steps(self) -> None:
        from wandas.pipeline import default_recipe_registry

        described = {item.operation_id: item for item in default_recipe_registry().describe()}
        assert described["wandas.audio.trigger"].receiver == (ChannelFrame,)
        assert described["wandas.audio.trigger"].returns is ChannelFrameSegments
        assert described["wandas.segments.drop"].receiver == (FrameSegments,)
        assert described["wandas.segments.drop"].returns == "receiver"
        assert described["wandas.segments.average"].receiver == (ChannelFrameSegments,)
        assert described["wandas.segments.average"].returns is ChannelFrame
        assert described["wandas.segments.transfer_function"].returns is TransferFunctionFrame
        assert described["wandas.segments.coherence"].returns is CoherenceFrame


class TestPlot:
    def test_plot_overlays_segments_per_channel(self) -> None:
        hits, _ = _hits()
        result = hits.drop(0).plot()
        assert not isinstance(result, Axes)
        axes = list(result)
        assert len(axes) == 3
        lines = axes[0].get_lines()
        assert [line.get_label() for line in lines] == ["#1", "#2", "#3", "#4"]
        assert axes[0].get_ylabel() == "force"
        assert axes[-1].get_xlabel() == "Time from event [s]"
        plt.close("all")

    def test_plot_one_channel_on_given_axes(self) -> None:
        cf = ChannelFrame.from_numpy(np.zeros((1, 1000)), sampling_rate=_SR, ch_units="N")
        hits = cf.segment([0.2, 0.5], duration=0.1, pre=0.02)
        _, ax = plt.subplots()
        result = hits.plot(channel=0, ax=ax)
        assert result is ax
        assert ax.get_ylabel().endswith("[N]")
        np.testing.assert_allclose(np.asarray(ax.get_lines()[0].get_xdata())[0], -0.02)
        plt.close("all")

    def test_axes_need_one_channel(self) -> None:
        hits, _ = _hits()
        _, ax = plt.subplots()
        with pytest.raises(ValueError, match="one channel"):
            hits.plot(ax=ax)
        plt.close("all")
