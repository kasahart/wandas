import warnings
from collections.abc import Callable
from typing import Any, cast

import dask.array as da
import numpy as np
import pytest

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


def test_with_annotations_is_atomic_lazy_and_lineage_neutral() -> None:
    frame = _frame()
    updates = {"new": {"items": [2]}}
    extras = {frame.channels[0].id: {"sensor": {"serials": [3]}}}
    result = frame.with_annotations(label="updated", metadata=updates, channel_extra=extras)

    assert result is not frame
    assert isinstance(result._data, da.Array)
    assert result._data is frame._data
    assert result.label == "updated"
    assert result.metadata == {"nested": {"items": [1]}, "new": {"items": [2]}}
    assert result.channels[0].extra == {"sensor": {"serials": [3]}}
    assert result.lineage is frame.lineage
    assert result.operation_history == frame.operation_history
    assert result._channel_ids == frame._channel_ids
    updates["new"]["items"].append(99)
    extras[frame.channels[0].id]["sensor"]["serials"].append(99)
    assert result.metadata["new"]["items"] == [2]
    assert result.channels[0].extra["sensor"]["serials"] == [3]


def test_annotation_merge_replace_and_channel_selector_contract() -> None:
    frame = _frame().with_channel_extra(0, {"old": 1})
    assert frame.with_metadata({"replacement": 2}, replace=True).metadata == {"replacement": 2}
    assert frame.with_channel_extra("left", {"new": 2}).channels[0].extra == {"old": 1, "new": 2}
    assert frame.with_channel_extra("c0", {"new": 2}, replace=True).channels[0].extra == {"new": 2}
    with pytest.raises(KeyError, match="Channel selector not found"):
        frame.with_channel_extra("missing", {})
    with pytest.raises(ValueError, match="Duplicate channel selector"):
        frame.with_annotations(channel_extra={0: {"a": 1}, "c0": {"b": 2}})


def test_with_label_none_resets_to_default_while_annotations_none_is_omitted() -> None:
    frame = _frame()
    assert frame.with_label(None).label == "unnamed_frame"
    assert frame.with_annotations(label=None).label == "source"


def test_metadata_access_wraps_plain_xarray_attrs_for_nested_warning() -> None:
    frame = _frame()
    frame._xr.attrs["metadata"] = {"nested": {"items": [1]}}
    with pytest.warns(DeprecationWarning, match="with_metadata"):
        frame.metadata["nested"]["items"].append(2)
    assert frame.metadata == {"nested": {"items": [1, 2]}}


def test_derived_sampling_rate_warning_points_to_source_channel_frame() -> None:
    spectral = _frame().fft(n_fft=8)
    with pytest.warns(DeprecationWarning, match="source ChannelFrame"):
        spectral.sampling_rate = 16
    assert spectral.sampling_rate == 16


def test_deprecated_list_extend_self_preserves_builtin_snapshot_semantics() -> None:
    frame = _frame().with_metadata({"tags": ["a", "b"]})
    tags = frame.metadata["tags"]
    with pytest.warns(DeprecationWarning, match="with_metadata"):
        tags.extend(tags)
    assert tags == ["a", "b", "a", "b"]


def test_deprecated_container_inplace_operators_warn_and_preserve_values() -> None:
    frame = _frame().with_metadata({"tags": ["a"], "details": {"left": 1}})
    tags = frame.metadata["tags"]
    details = frame.metadata["details"]
    with pytest.warns(DeprecationWarning, match="with_metadata"):
        tags += ["b"]
    with pytest.warns(DeprecationWarning, match="with_metadata"):
        tags *= 2
    with pytest.warns(DeprecationWarning, match="with_metadata"):
        details |= {"right": 2}
    assert tags == ["a", "b", "a", "b"]
    assert details == {"left": 1, "right": 2}


@pytest.mark.parametrize(
    ("mutate", "expected", "returned"),
    [
        (
            lambda value: value.__setitem__("added", {"items": [3]}),
            {"left": 1, "right": 2, "added": {"items": [3]}},
            None,
        ),
        (lambda value: value.__delitem__("left"), {"right": 2}, None),
        (lambda value: value.clear(), {}, None),
        (lambda value: value.pop("left"), {"right": 2}, 1),
        (lambda value: value.pop("missing", "fallback"), {"left": 1, "right": 2}, "fallback"),
        (lambda value: value.popitem(), {"left": 1}, ("right", 2)),
        (lambda value: value.setdefault("added", [3]), {"left": 1, "right": 2, "added": [3]}, [3]),
    ],
)
def test_deprecated_metadata_mapping_mutations_warn_and_preserve_builtin_semantics(
    mutate: Callable[[dict[str, Any]], Any],
    expected: dict[str, Any],
    returned: Any,
) -> None:
    mapping = _frame().with_metadata({"mapping": {"left": 1, "right": 2}}).metadata["mapping"]

    with pytest.warns(DeprecationWarning, match="with_metadata") as caught:
        actual = mutate(mapping)

    assert mapping == expected
    assert actual == returned
    assert caught[0].filename == __file__


def test_deprecated_metadata_mapping_setdefault_existing_value_does_not_warn() -> None:
    mapping = _frame().with_metadata({"mapping": {"left": {"items": [1]}}}).metadata["mapping"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        existing = mapping.setdefault("left", {"items": [2]})

    assert existing == {"items": [1]}
    assert mapping == {"left": {"items": [1]}}
    assert caught == []


def test_deprecated_metadata_mapping_assignment_copies_caller_owned_value() -> None:
    mapping = _frame().with_metadata({"mapping": {}}).metadata["mapping"]
    caller_value = {"items": [1]}

    with pytest.warns(DeprecationWarning, match="with_metadata"):
        mapping["added"] = caller_value
    caller_value["items"].append(2)

    assert mapping == {"added": {"items": [1]}}


@pytest.mark.parametrize(
    ("mutate", "expected", "returned"),
    [
        (lambda value: value.__setitem__(0, 9), [9, 1, 2], None),
        (lambda value: value.__setitem__(slice(1, None), [8, 7]), [3, 8, 7], None),
        (lambda value: value.__delitem__(0), [1, 2], None),
        (lambda value: value.__delitem__(slice(0, 2)), [2], None),
        (lambda value: value.append(4), [3, 1, 2, 4], None),
        (lambda value: value.insert(1, 4), [3, 4, 1, 2], None),
        (lambda value: value.pop(), [3, 1], 2),
        (lambda value: value.remove(1), [3, 2], None),
        (lambda value: value.clear(), [], None),
        (lambda value: value.reverse(), [2, 1, 3], None),
        (lambda value: value.sort(reverse=True), [3, 2, 1], None),
    ],
)
def test_deprecated_metadata_list_mutations_warn_and_preserve_builtin_semantics(
    mutate: Callable[[list[Any]], Any],
    expected: list[Any],
    returned: Any,
) -> None:
    values = _frame().with_metadata({"values": [3, 1, 2]}).metadata["values"]

    with pytest.warns(DeprecationWarning, match="with_metadata") as caught:
        actual = mutate(values)

    assert values == expected
    assert actual == returned
    assert caught[0].filename == __file__


def test_deprecated_metadata_list_slice_assignment_copies_caller_owned_values() -> None:
    values = _frame().with_metadata({"values": [0]}).metadata["values"]
    caller_values = [{"items": [1]}]

    with pytest.warns(DeprecationWarning, match="with_metadata"):
        values[:] = caller_values
    caller_values[0]["items"].append(2)

    assert values == [{"items": [1]}]


@pytest.mark.parametrize(
    ("invoke", "error", "message"),
    [
        (lambda frame: frame.with_label(cast(Any, 1)), TypeError, "Label must be a string or None"),
        (lambda frame: frame.with_metadata(cast(Any, [])), TypeError, "Metadata updates must be a mapping"),
        (
            lambda frame: frame.with_annotations(channel_extra=cast(Any, [])),
            TypeError,
            "channel_extra must map channel selectors",
        ),
        (
            lambda frame: frame.with_annotations(channel_extra={0: cast(Any, [])}),
            TypeError,
            "Channel extra updates must be mappings",
        ),
        (
            lambda frame: frame.with_channel_extra(cast(Any, True), {}),
            TypeError,
            "Channel selector must be a stable ID",
        ),
        (lambda frame: frame.with_channel_extra(2, {}), IndexError, "Channel index out of range"),
        (
            lambda frame: frame.rename_channels({0: cast(Any, 1)}),
            TypeError,
            "Channel labels must be strings",
        ),
    ],
)
def test_immutable_annotation_invalid_inputs_raise_explicit_errors(
    invoke: Callable[[ChannelFrame], Any],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        invoke(_frame())


@pytest.mark.parametrize(
    "invoke",
    [
        lambda frame: frame.with_channel_extra("duplicate", {}),
        lambda frame: frame.rename_channels({"duplicate": "renamed"}),
    ],
)
def test_channel_label_selector_when_ambiguous_raises_error(invoke: Callable[[ChannelFrame], Any]) -> None:
    frame = ChannelFrame.from_numpy(
        np.arange(16.0).reshape(2, 8),
        sampling_rate=8,
        ch_labels=["duplicate", "duplicate"],
    )

    with pytest.raises(ValueError, match="Channel label is ambiguous"):
        invoke(frame)


def test_rename_channels_is_available_on_derived_frame() -> None:
    spectral = _frame().fft(n_fft=8)
    renamed = spectral.rename_channels({0: "renamed"})
    assert type(renamed) is type(spectral)
    assert renamed.labels == ["renamed", "right"]
    np.testing.assert_array_equal(renamed.freqs, spectral.freqs)


def test_calibration_typed_replacements_are_public() -> None:
    calibration = ChannelCalibration(factor=2, unit="Pa")
    assert calibration.with_unit("V").unit == "V"
    assert calibration.with_ref(0.5).ref == 0.5


@pytest.mark.parametrize(
    ("mutate", "migration"),
    [
        (lambda frame: setattr(frame, "label", "x"), "with_label"),
        (lambda frame: frame.metadata["nested"]["items"].append(2), "with_metadata"),
        (lambda frame: setattr(frame.channels[0], "label", "x"), "rename_channels"),
        (lambda frame: setattr(frame.channels[0], "unit", "Pa"), "with_calibration"),
        (lambda frame: setattr(frame.channels[0], "ref", 2.0), "with_calibration"),
        (lambda frame: frame.channels[0].extra.update({"x": [1]}), "with_channel_extra"),
        (lambda frame: setattr(frame, "sampling_rate", 16), "resampling"),
        (lambda frame: setattr(frame, "source_time_offset", [1, 2]), "with_source_time_offset"),
    ],
)
def test_direct_mutation_warns_with_exact_migration(mutate: Callable[[ChannelFrame], None], migration: str) -> None:
    frame = _frame()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mutate(frame)
    assert len(caught) == 1
    assert caught[0].category is DeprecationWarning
    assert migration in str(caught[0].message)
    assert caught[0].filename == __file__
