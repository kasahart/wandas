"""Selecting time-axis Frames by seconds: trim and SpectrogramFrame.at_time."""

import json

import dask.array as da
import numpy as np
import pytest

import wandas as wd
from tests.frame_helpers import channel_first_values
from wandas.frames.cepstrogram import CepstrogramFrame
from wandas.frames.channel import ChannelFrame
from wandas.frames.spectral import SpectralFrame
from wandas.frames.spectrogram import SpectrogramFrame
from wandas.pipeline import RecipePlan

_SR = 8000
_HOP = 128
_STEP = _HOP / _SR


def _ramp(n_samples: int = 100, sampling_rate: int = 100, offset: float | list[float] = 0.0) -> ChannelFrame:
    data = np.tile(np.arange(n_samples, dtype=np.float64), (np.size(offset), 1))
    return ChannelFrame(da.from_array(data, chunks=-1), sampling_rate=sampling_rate, source_time_offset=offset)


def _spectrogram() -> SpectrogramFrame:
    return wd.generate_sin(freqs=[440], duration=2.0, sampling_rate=_SR).stft(n_fft=512, hop_length=_HOP)


def _local_centers(frame: SpectrogramFrame | CepstrogramFrame) -> np.ndarray:
    return frame.frame_center_times[0] - frame.source_time_offset[0]


def _replay(frame: object, source: object) -> object:
    plan = RecipePlan.from_frame(frame, input_names=("signal",))
    loaded = RecipePlan.from_dict(json.loads(json.dumps(plan.to_dict(), allow_nan=False)))
    return loaded.apply({"signal": source})


class TestChannelFrameTrim:
    @pytest.mark.parametrize(
        ("start", "end"),
        [
            pytest.param("100ms", "0.5s", id="ms-and-s"),
            pytest.param(".1s", "500ms", id="leading-dot"),
            pytest.param("0.1 s", 0.5, id="space-and-number"),
        ],
    )
    def test_unit_strings_select_seconds(self, start: str, end: str | float) -> None:
        trimmed = _ramp().trim(start, end)

        np.testing.assert_array_equal(channel_first_values(trimmed), np.arange(10, 50).reshape(1, -1))

    def test_minutes_and_hours(self) -> None:
        frame = _ramp(n_samples=4000, sampling_rate=1)

        trimmed = frame.trim("1min30s", "1h")

        assert trimmed.n_samples == 3600 - 90
        np.testing.assert_array_equal(trimmed.source_time_offset, [90.0])

    @pytest.mark.parametrize("value", ["90", "1:30", "1m", "1s1s", "s", "-1s", "1.2.3s"])
    def test_invalid_time_strings_raise_type_error(self, value: str) -> None:
        with pytest.raises(TypeError, match="Invalid start time string"):
            _ramp().trim(value)

    @pytest.mark.parametrize("value", [True, None, [1.0]])
    def test_non_time_start_raises_type_error(self, value: object) -> None:
        with pytest.raises(TypeError, match="start must be a number of seconds"):
            _ramp().trim(value)  # ty: ignore[invalid-argument-type]

    def test_non_finite_time_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            _ramp().trim(0, float("inf"))

    def test_times_round_to_the_nearest_sample(self) -> None:
        # 0.29 * 100 is 28.999... in floating point; truncation would start one sample early.
        trimmed = _ramp().trim(0.29, 0.5)

        np.testing.assert_array_equal(channel_first_values(trimmed), np.arange(29, 50).reshape(1, -1))

    def test_adjacent_trims_neither_overlap_nor_leave_a_gap(self) -> None:
        frame = _ramp()

        parts = [frame.trim(0.0, 0.33), frame.trim(0.33, 0.67), frame.trim(0.67)]

        np.testing.assert_array_equal(np.concatenate([channel_first_values(part)[0] for part in parts]), np.arange(100))

    def test_local_reference_counts_from_this_frame(self) -> None:
        trimmed = _ramp(offset=10.0).trim(0.2).trim(0.1, 0.3)

        np.testing.assert_array_equal(channel_first_values(trimmed), np.arange(30, 50).reshape(1, -1))
        np.testing.assert_allclose(trimmed.source_time_offset, [10.3])

    def test_source_reference_uses_the_source_timeline(self) -> None:
        frame = _ramp(offset=10.0).trim(0.2)

        trimmed = frame.trim(10.3, 10.5, reference="source")

        np.testing.assert_array_equal(channel_first_values(trimmed), np.arange(30, 50).reshape(1, -1))

    def test_source_reference_before_the_frame_is_clipped(self) -> None:
        trimmed = _ramp(offset=10.0).trim(5.0, 10.2, reference="source")

        np.testing.assert_array_equal(channel_first_values(trimmed), np.arange(0, 20).reshape(1, -1))

    def test_source_reference_needs_one_shared_offset(self) -> None:
        with pytest.raises(ValueError, match="Channels start at different source times"):
            _ramp(offset=[0.0, 1.0]).trim(1.2, reference="source")

    def test_unknown_reference_raises(self) -> None:
        with pytest.raises(ValueError, match="reference must be 'local' or 'source'"):
            _ramp().trim(0.1, reference="absolute")  # ty: ignore[invalid-argument-type]

    def test_recipe_records_the_requested_strings(self) -> None:
        frame = _ramp(n_samples=200, sampling_rate=100)
        trimmed = frame.trim("500ms", "1s", reference="local")

        assert trimmed.operation_history[-1] == {
            "operation": "wandas.frame.time_slice",
            "version": 2,
            "params": {"start": "500ms", "end": "1s", "reference": "local"},
        }
        replayed = _replay(trimmed, _ramp(n_samples=400, sampling_rate=200))
        assert isinstance(replayed, ChannelFrame)
        np.testing.assert_array_equal(channel_first_values(replayed), np.arange(100, 200).reshape(1, -1))


class TestSpectrogramTrim:
    def test_keeps_frames_whose_centers_are_in_the_range(self) -> None:
        spectrogram = _spectrogram()

        trimmed = spectrogram.trim(0.5, 1.0)

        centers = _local_centers(spectrogram)
        expected = np.flatnonzero((centers >= 0.5) & (centers < 1.0))
        assert type(trimmed) is SpectrogramFrame
        np.testing.assert_allclose(trimmed.frame_center_times[0], spectrogram.frame_center_times[0][expected])
        np.testing.assert_array_equal(trimmed.data, spectrogram.data[..., expected])

    def test_center_on_the_end_is_excluded_and_on_the_start_included(self) -> None:
        spectrogram = _spectrogram()
        centers = _local_centers(spectrogram)

        trimmed = spectrogram.trim(float(centers[10]), float(centers[20]))

        np.testing.assert_allclose(trimmed.frame_center_times[0], spectrogram.frame_center_times[0][10:20])

    def test_local_reference_counts_from_the_source_offset(self) -> None:
        spectrogram = _spectrogram()
        first = spectrogram.trim(0.5, 1.0)
        offset = float(first.source_time_offset[0])

        trimmed = first.trim(0.1, 0.2)

        np.testing.assert_allclose(
            trimmed.frame_center_times[0], spectrogram.trim(offset + 0.1, offset + 0.2).frame_center_times[0]
        )

    def test_source_reference(self) -> None:
        spectrogram = _spectrogram()
        first = spectrogram.trim(0.5, 1.5)

        trimmed = first.trim(0.75, 1.0, reference="source")

        np.testing.assert_allclose(trimmed.frame_center_times[0], spectrogram.trim(0.75, 1.0).frame_center_times[0])

    def test_range_past_the_frames_is_clipped(self) -> None:
        spectrogram = _spectrogram()

        trimmed = spectrogram.trim(1.9, 100)

        assert trimmed.n_frames == int(np.count_nonzero(_local_centers(spectrogram) >= 1.9 - 1e-9))

    @pytest.mark.parametrize(("start", "end"), [(5.0, 6.0), (0.5005, 0.501)])
    def test_range_without_a_frame_center_raises(self, start: float, end: float) -> None:
        with pytest.raises(ValueError, match="Time range selects no frame"):
            _spectrogram().trim(start, end)

    def test_unknown_frame_centers_raise(self) -> None:
        spectrogram = _spectrogram()
        unknown = SpectrogramFrame(
            spectrogram._data, _SR, n_fft=512, hop_length=_HOP, win_length=512, frame_time_origin=None
        )

        with pytest.raises(ValueError, match="Physical frame center times are unknown"):
            unknown.trim(0.5, 1.0)

    def test_recipe_replays_seconds_on_another_hop(self) -> None:
        trimmed = _spectrogram().trim(0.5, 1.0)
        runtime = wd.generate_sin(freqs=[440], duration=2.0, sampling_rate=16_000)

        replayed = _replay(trimmed, runtime)

        assert isinstance(replayed, SpectrogramFrame)
        centers = replayed.frame_center_times[0]
        assert centers[0] >= 0.5 - 1e-9
        assert centers[-1] < 1.0
        assert replayed.n_frames == 2 * trimmed.n_frames

    def test_trim_stays_lazy(self) -> None:
        trimmed = _spectrogram().trim(0.5, 1.0)

        assert trimmed.operation_history[-1]["operation"] == "wandas.spectrogram.trim"
        assert hasattr(trimmed._data, "dask")


class TestCepstrogramTrim:
    def test_keeps_frames_whose_centers_are_in_the_range(self) -> None:
        cepstrogram = _spectrogram().cepstrum()

        trimmed = cepstrogram.trim("500ms", "1s")

        centers = _local_centers(cepstrogram)
        expected = np.flatnonzero((centers >= 0.5) & (centers < 1.0))
        assert type(trimmed) is CepstrogramFrame
        np.testing.assert_allclose(trimmed.frame_center_times[0], cepstrogram.frame_center_times[0][expected])
        np.testing.assert_array_equal(trimmed.data, cepstrogram.data[..., expected])
        assert trimmed.operation_history[-1]["operation"] == "wandas.cepstrogram.trim"


class TestSpectrogramAtTime:
    def test_picks_the_nearest_frame_center(self) -> None:
        spectrogram = _spectrogram()
        centers = _local_centers(spectrogram)
        index = int(np.argmin(np.abs(centers - 0.51)))

        spectrum = spectrogram.at_time(0.51)

        assert type(spectrum) is SpectralFrame
        np.testing.assert_array_equal(spectrum.data, spectrogram.get_frame_at(index).data)
        assert spectrum.label.endswith(f"(Frame {index}, Center {centers[index]:.3f}s)")

    def test_tie_goes_to_the_earlier_frame(self) -> None:
        spectrogram = _spectrogram()
        centers = _local_centers(spectrogram)
        midpoint = float(centers[10] + centers[11]) / 2

        spectrum = spectrogram.at_time(midpoint)

        np.testing.assert_array_equal(spectrum.data, spectrogram.get_frame_at(10).data)

    def test_unit_string_and_source_reference(self) -> None:
        spectrogram = _spectrogram()
        trimmed = spectrogram.trim(0.5, 1.5)

        offset = float(trimmed.source_time_offset[0])

        local = trimmed.at_time("250ms")
        source = trimmed.at_time(offset + 0.25, reference="source")

        np.testing.assert_array_equal(local.data, source.data)
        np.testing.assert_array_equal(source.data, spectrogram.at_time(offset + 0.25).data)

    def test_half_a_hop_beyond_the_last_center_is_accepted(self) -> None:
        spectrogram = _spectrogram()
        last = float(_local_centers(spectrogram)[-1])

        spectrum = spectrogram.at_time(last + 0.4 * _STEP)

        np.testing.assert_array_equal(spectrum.data, spectrogram.get_frame_at(spectrogram.n_frames - 1).data)

    @pytest.mark.parametrize("time", [-1.0, 3.0])
    def test_time_outside_the_spectrogram_raises(self, time: float) -> None:
        with pytest.raises(ValueError, match="Time is outside the spectrogram"):
            _spectrogram().at_time(time)

    def test_tolerance(self) -> None:
        spectrogram = _spectrogram()
        centers = _local_centers(spectrogram)
        time = float(centers[10]) + 0.004

        assert spectrogram.at_time(time, tolerance="5ms").n_channels == 1
        with pytest.raises(ValueError, match="No frame center within the tolerance"):
            spectrogram.at_time(time, tolerance=0.001)
        with pytest.raises(ValueError, match="tolerance must be non-negative"):
            spectrogram.at_time(time, tolerance=-1)

    def test_recipe_replays_the_time(self) -> None:
        spectrum = _spectrogram().at_time(0.51)
        runtime = wd.generate_sin(freqs=[880], duration=2.0, sampling_rate=_SR)

        replayed = _replay(spectrum, runtime)

        assert isinstance(replayed, SpectralFrame)
        assert spectrum.operation_history[-1] == {
            "operation": "wandas.spectrogram.at_time",
            "version": 1,
            "params": {"time": 0.51},
        }
        np.testing.assert_array_equal(replayed.data, runtime.stft(n_fft=512, hop_length=_HOP).at_time(0.51).data)
