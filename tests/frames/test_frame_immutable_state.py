import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import dask.array as da
import numpy as np
import pytest

from tests.builtin_frame_cases import BUILTIN_FRAME_CASES
from wandas.core.base_frame import BaseFrame
from wandas.core.metadata import ChannelCalibration
from wandas.frames.channel import ChannelFrame


def _frame() -> ChannelFrame:
    return ChannelFrame.from_numpy(
        np.arange(16.0).reshape(2, 8),
        sampling_rate=8,
        label="source",
        metadata={"nested": {"items": [1]}},
        ch_labels=["left", "right"],
    )


@dataclass
class _DeepCopyProbe:
    items: list[int]
    calls: list[int]

    def __deepcopy__(self, memo: dict[int, Any]) -> "_DeepCopyProbe":
        self.calls[0] += 1
        result = _DeepCopyProbe(copy.deepcopy(self.items, memo), self.calls)
        memo[id(self)] = result
        return result


@pytest.mark.parametrize(
    "frame_factory",
    [pytest.param(case.factory, id=case.id) for case in BUILTIN_FRAME_CASES],
)
def test_annotation_updates_preserve_every_frame_family(
    frame_factory: Callable[[], BaseFrame[Any]],
) -> None:
    frame = frame_factory()
    state = frame._get_additional_init_kwargs()
    updated = (
        frame.with_label("annotated")
        .with_metadata({"new": {"items": [2]}})
        .with_channel_extra(0, {"sensor": {"serial": "42"}})
    )

    assert type(updated) is type(frame)
    assert updated is not frame
    assert updated._data is frame._data
    assert isinstance(updated._data, da.Array)
    assert updated._channel_ids == frame._channel_ids
    assert updated.labels == frame.labels
    assert updated.lineage is frame.lineage
    assert updated.operation_history == frame.operation_history
    np.testing.assert_array_equal(updated.source_time_offset, frame.source_time_offset)
    assert updated.metadata == {**frame.metadata, "new": {"items": [2]}}
    assert updated.channels[0].extra == {
        **frame.channels[0].extra,
        "sensor": {"serial": "42"},
    }
    for name, expected in state.items():
        actual = updated._get_additional_init_kwargs()[name]
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual == expected


def test_with_metadata_deep_copies_caller_value_once_at_reconstruction_boundary() -> None:
    frame = _frame()
    calls = [0]
    nested = _DeepCopyProbe([2], calls)
    metadata = {"new": nested}

    updated = frame.with_metadata(metadata)

    assert calls == [1]
    nested.items.append(99)
    snapshot = updated.metadata
    assert snapshot["new"].items == [2]
    assert frame.metadata == {"nested": {"items": [1]}}


def test_with_channel_extra_deep_copies_caller_value_once_at_reconstruction_boundary() -> None:
    frame = _frame()
    calls = [0]
    nested = _DeepCopyProbe([3], calls)
    extra = {"sensor": nested}

    updated = frame.with_channel_extra("left", extra)

    assert calls == [1]
    nested.items.append(99)
    snapshot = updated.channels[0].extra
    assert snapshot["sensor"].items == [3]
    assert frame.channels[0].extra == {}


@pytest.mark.parametrize(
    "frame_factory",
    [pytest.param(case.factory, id=case.id) for case in BUILTIN_FRAME_CASES],
)
def test_sampling_rate_assignment_is_read_only_for_every_frame_family(
    frame_factory: Callable[[], BaseFrame[Any]],
) -> None:
    frame = frame_factory()

    with pytest.raises(AttributeError):
        setattr(frame, "sampling_rate", frame.sampling_rate)
    with pytest.raises(AttributeError):
        setattr(frame, "sampling_rate", frame.sampling_rate * 2)


def test_channel_frame_stores_validated_binary64_sampling_rate() -> None:
    frame = ChannelFrame.from_numpy(np.arange(8.0), sampling_rate=cast(Any, np.longdouble("48000.25")))

    assert type(frame._xr.attrs["sampling_rate"]) is float
    assert type(frame.sampling_rate) is float
    assert np.isfinite(frame.sampling_rate)
    assert frame.sampling_rate > 0
    assert frame.sampling_rate == float(np.longdouble("48000.25"))


def test_channel_frame_rejects_sampling_rate_that_overflows_binary64() -> None:
    with pytest.raises(ValueError, match=r"Invalid sampling_rate"):
        ChannelFrame.from_numpy(np.arange(8.0), sampling_rate=10**400)


@pytest.mark.skipif(
    bool(np.finfo(np.longdouble).max <= np.finfo(np.float64).max),
    reason="platform longdouble has no wider upper range than binary64",
)
def test_channel_frame_rejects_wide_finite_sampling_rate_that_normalizes_to_infinity() -> None:
    sampling_rate = np.longdouble(np.finfo(np.float64).max) * np.longdouble(2)
    assert np.isfinite(sampling_rate)

    with pytest.raises(ValueError, match=r"Invalid sampling_rate"):
        ChannelFrame.from_numpy(np.arange(8.0), sampling_rate=cast(Any, sampling_rate))


@pytest.mark.skipif(
    bool(np.finfo(np.longdouble).smallest_subnormal >= np.finfo(np.float64).smallest_subnormal),
    reason="platform longdouble has no wider lower range than binary64",
)
def test_channel_frame_rejects_positive_sampling_rate_that_normalizes_to_zero() -> None:
    sampling_rate = np.longdouble(np.nextafter(np.float64(0), np.float64(1))) / np.longdouble(2)
    assert sampling_rate > 0

    with pytest.raises(ValueError, match=r"Invalid sampling_rate"):
        ChannelFrame.from_numpy(np.arange(8.0), sampling_rate=cast(Any, sampling_rate))


def test_public_mutable_paths_are_read_only_and_nested_snapshots_are_detached() -> None:
    frame = _frame().with_channel_extra(0, {"nested": {"items": [1]}})
    metadata = frame.metadata
    extra = frame.channels[0].extra
    offsets = frame.source_time_offset
    metadata["nested"]["items"].append(2)
    extra["nested"]["items"].append(2)
    offsets[0] = 9

    assert frame.metadata == {"nested": {"items": [1]}}
    assert frame.channels[0].extra == {"nested": {"items": [1]}}
    np.testing.assert_array_equal(frame.source_time_offset, np.zeros(2))
    for attribute, value in [
        ("metadata", {}),
        ("label", "changed"),
        ("sampling_rate", 16),
        ("source_time_offset", [1, 2]),
    ]:
        with pytest.raises(AttributeError):
            setattr(frame, attribute, value)
    for attribute, value in [("label", "changed"), ("extra", {}), ("unit", "Pa"), ("ref", 1.0)]:
        with pytest.raises(AttributeError):
            setattr(frame.channels[0], attribute, value)


def test_channel_name_and_stable_id_lookup_use_distinct_paths() -> None:
    frame = ChannelFrame(
        da.arange(16.0, chunks=16).reshape(2, 8),
        sampling_rate=8,
        channel_ids=["left-id", "collision"],
        channel_metadata=[{"label": "collision"}, {"label": "right"}],
    )

    assert frame.channels["collision"].id == "left-id"
    assert frame.channels.by_id("collision").label == "right"
    updated = frame.with_channel_extra("collision", {"selected": "name"})
    assert updated.channels[0].extra == {"selected": "name"}
    assert updated.channels[1].extra == {}


def test_empty_and_duplicate_stable_ids_are_rejected() -> None:
    data = da.arange(16.0, chunks=16).reshape(2, 8)
    with pytest.raises(ValueError, match="non-empty"):
        ChannelFrame(data, sampling_rate=8, channel_ids=["", "c1"])
    with pytest.raises(ValueError, match="unique"):
        ChannelFrame(data, sampling_rate=8, channel_ids=["same", "same"])
    with pytest.raises(TypeError, match="strings"):
        ChannelFrame(data, sampling_rate=8, channel_ids=cast(Any, [0, "c1"]))


def test_calibration_partial_updates_preserve_factor_and_reset_ref_for_unit() -> None:
    calibration = ChannelCalibration(factor=2, unit="Pa", ref=0.5)

    changed_unit = calibration.with_unit("V")
    custom_ref = changed_unit.with_ref(0.25)

    assert changed_unit == ChannelCalibration(factor=2, unit="V")
    assert custom_ref == ChannelCalibration(factor=2, unit="V", ref=0.25)
    assert calibration.with_factor(3) == ChannelCalibration(factor=3, unit="Pa", ref=0.5)


@pytest.mark.parametrize(
    ("invoke", "error", "message"),
    [
        (lambda frame: frame.with_label(cast(Any, 1)), TypeError, "Frame label"),
        (lambda frame: frame.with_metadata(cast(Any, [])), TypeError, "Frame metadata"),
        (lambda frame: frame.with_channel_extra(cast(Any, True), {}), TypeError, "Channel selector"),
        (lambda frame: frame.with_source_time_offset(cast(Any, "1.0")), TypeError, "finite numeric"),
        (lambda frame: frame.rename_channels({0: cast(Any, 1)}), TypeError, "Channel label"),
    ],
)
def test_immutable_updates_reject_invalid_inputs(
    invoke: Callable[[ChannelFrame], Any],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        invoke(_frame())


def test_immutable_update_error_branches_are_explicit() -> None:
    frame = _frame()
    with pytest.raises(TypeError, match="Frame metadata"):
        frame.with_metadata(cast(Any, None))
    with pytest.raises(IndexError, match="out of range"):
        frame.with_channel_extra(2, {})
    with pytest.raises(KeyError, match="not found"):
        frame.with_channel_extra("missing", {})
    with pytest.raises(TypeError, match="replace"):
        frame.with_metadata({}, replace=cast(Any, 1))
    with pytest.raises(TypeError, match="channel_extra"):
        frame._with_annotations(channel_extra=cast(Any, []))
    with pytest.raises(ValueError, match="Duplicate channel selector"):
        frame._with_annotations(channel_extra={0: {"a": 1}, -2: {"b": 2}})

    duplicated = ChannelFrame.from_numpy(
        np.arange(16.0).reshape(2, 8),
        sampling_rate=8,
        ch_labels=["same", "same"],
    )
    with pytest.raises(ValueError, match="ambiguous"):
        duplicated.with_channel_extra("same", {})


def test_zero_dimensional_source_offset_preserves_scalar_recipe_intent() -> None:
    result = _frame().with_source_time_offset(np.array(0.25))
    assert result.operation_history[-1]["params"] == {"value": 0.25}


def test_rename_channels_is_lazy_and_available_on_derived_frames() -> None:
    frame = _frame().fft(n_fft=8)
    renamed = frame.rename_channels({0: "renamed"})

    assert type(renamed) is type(frame)
    assert renamed._data is frame._data
    assert renamed.labels == ["renamed", "right"]
    np.testing.assert_array_equal(renamed.freqs, frame.freqs)
