import warnings
from collections.abc import Callable

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
