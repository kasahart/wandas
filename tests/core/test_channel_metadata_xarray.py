from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pytest

from wandas.core.channel_metadata import ChannelMetadataIndexer, ChannelMetadataView
from wandas.core.metadata import ChannelCalibration, ChannelMetadata
from wandas.frames.channel import ChannelFrame
from wandas.utils.dask_helpers import da_from_array


def _frame() -> ChannelFrame:
    return ChannelFrame(
        data=da_from_array(np.array([[1.0, 2.0], [3.0, 4.0]]), chunks=(1, -1)),
        sampling_rate=48_000,
        channel_metadata=[
            ChannelMetadata(label="left", unit="Pa", extra={"sensitivity": 50.0}),
            {"label": "right", "unit": "V", "ref": 0.5, "extra": {"sensitivity": 48.5}},
        ],
    )


def test_channel_metadata_storage_schema_is_xarray_backed() -> None:
    frame = _frame()

    assert list(frame._xr.dims) == ["channel", "time"]
    assert frame._xr.coords["channel"].values.tolist() == ["c0", "c1"]
    assert frame._xr.coords["channel_label"].values.tolist() == ["left", "right"]
    assert frame._xr.coords["channel_unit"].values.tolist() == ["Pa", "V"]
    assert frame._xr.coords["channel_ref"].values.tolist() == [2e-5, 0.5]
    assert frame._xr.coords["channel_calibration_factor"].values.tolist() == [1.0, 1.0]
    assert frame._xr.attrs["channel_extra"] == {
        "c0": {"sensitivity": 50.0},
        "c1": {"sensitivity": 48.5},
    }
    assert (
        not {
            "channel_ids",
            "channel_label",
            "channel_unit",
            "channel_ref",
            "channel_calibration_factor",
        }
        & frame._xr.attrs.keys()
    )


def test_channel_indexer_reads_and_writes_xarray_storage() -> None:
    frame = _frame()

    assert frame.channels[0].id == "c0"
    assert frame.channels[0].label == "left"
    assert frame.channels[0].unit == "Pa"
    assert frame.channels[0].ref == 2e-5
    assert frame.channels[0].extra == {"sensitivity": 50.0}

    frame.channels[0].label = "front-left"
    frame.channels[0].unit = "Hz"
    frame.channels[0].ref = 0.25
    frame.channels[0]["calibrated"] = True

    assert frame._xr.coords["channel"].values.tolist() == ["c0", "c1"]
    assert frame._xr.coords["channel_label"].values.tolist()[0] == "front-left"
    assert frame._xr.coords["channel_unit"].values.tolist()[0] == "Hz"
    assert float(frame._xr.coords["channel_ref"].values[0]) == 0.25
    assert frame._xr.attrs["channel_extra"]["c0"] == {"sensitivity": 50.0, "calibrated": True}


def test_channel_indexer_is_sequence_not_list_backed() -> None:
    frame = _frame()

    assert isinstance(frame.channels, Sequence)
    assert not isinstance(frame.channels, list)
    assert repr(frame.channels) == repr(frame.channels.to_list())


def test_channel_metadata_view_setters_validate_like_value_object() -> None:
    frame = _frame()

    with pytest.raises(TypeError, match="label must be a string"):
        setattr(frame.channels[0], "label", 123)
    with pytest.raises(TypeError, match="Invalid channel calibration unit"):
        setattr(frame.channels[0], "unit", 123)
    with pytest.raises(TypeError, match="Invalid channel calibration reference"):
        setattr(frame.channels[0], "ref", True)
    with pytest.raises(TypeError, match="Invalid channel calibration reference"):
        setattr(frame.channels[0], "ref", "1.0")
    with pytest.raises(AttributeError, match="use frame.with_calibration"):
        frame.channels[0].calibration = ChannelCalibration(2.0)

    assert frame.channels[0].label == "left"
    assert frame.channels[0].unit == "Pa"
    assert frame.channels[0].ref == 2e-5


def test_internal_channel_calibration_update_rejects_untyped_values() -> None:
    frame = _frame()

    with pytest.raises(TypeError, match="calibration must be a ChannelCalibration"):
        frame._set_channel_calibration(0, cast(Any, "bad"))


def test_channel_selection_reorder_keeps_metadata_aligned_and_filters_extra() -> None:
    frame = _frame()
    frame.channels[0].label = "renamed"

    selected = frame.get_channel([1, 0])

    assert selected._xr.coords["channel"].values.tolist() == ["c1", "c0"]
    assert selected._xr.coords["channel_label"].values.tolist() == ["right", "renamed"]
    assert selected._xr.coords["channel_unit"].values.tolist() == ["V", "Pa"]
    assert selected._xr.coords["channel_ref"].values.tolist() == [0.5, 2e-5]
    assert selected._xr.attrs["channel_extra"] == {
        "c1": {"sensitivity": 48.5},
        "c0": {"sensitivity": 50.0},
    }

    only_right = frame.get_channel(query="right")

    assert only_right._xr.coords["channel"].values.tolist() == ["c1"]
    assert only_right._xr.attrs["channel_extra"] == {"c1": {"sensitivity": 48.5}}


def test_add_remove_and_rename_preserve_stable_ids_and_extra_keys() -> None:
    frame = _frame()
    added = frame.add_channel(np.array([5.0, 6.0]), label="center")

    assert added._xr.coords["channel"].values.tolist() == ["c0", "c1", "c2"]
    assert added._xr.coords["channel_label"].values.tolist() == ["left", "right", "center"]
    assert added._xr.attrs["channel_extra"]["c2"] == {}

    removed = added.remove_channel("right")

    assert removed._xr.coords["channel"].values.tolist() == ["c0", "c2"]
    assert removed._xr.coords["channel_label"].values.tolist() == ["left", "center"]
    assert removed._xr.attrs["channel_extra"] == {"c0": {"sensitivity": 50.0}, "c2": {}}

    renamed = removed.rename_channels({"left": "front-left"})

    assert renamed._xr.coords["channel"].values.tolist() == ["c0", "c2"]
    assert renamed._xr.coords["channel_label"].values.tolist() == ["front-left", "center"]
    assert renamed._xr.attrs["channel_extra"] == {"c0": {"sensitivity": 50.0}, "c2": {}}


def test_duplicate_channel_selection_generates_unique_ids() -> None:
    frame = _frame()

    duplicated = frame.get_channel([0, 0])

    assert duplicated._xr.coords["channel"].values.tolist() == ["c0", "c2"]
    assert duplicated._xr.coords["channel_label"].values.tolist() == ["left", "left"]
    assert duplicated._xr.attrs["channel_extra"] == {
        "c0": {"sensitivity": 50.0},
        "c2": {"sensitivity": 50.0},
    }


def test_channel_metadata_snapshots_are_value_objects() -> None:
    frame = _frame()

    metadata = frame.channels[0].to_metadata()
    metadata_list = frame.channels.to_list()

    assert isinstance(metadata, ChannelMetadata)
    assert isinstance(metadata_list, list)
    assert all(isinstance(item, ChannelMetadata) for item in metadata_list)
    assert [item.label for item in metadata_list] == ["left", "right"]

    metadata.extra["sensitivity"] = 1.0
    assert frame.channels[0].extra["sensitivity"] == 50.0


def test_private_storage_retains_channel_metadata_coordinates() -> None:
    frame = _frame()

    assert frame._xr.coords["channel"].values.tolist() == ["c0", "c1"]
    assert frame._xr.coords["channel_label"].values.tolist() == ["left", "right"]
    assert frame._xr.coords["channel_unit"].values.tolist() == ["Pa", "V"]
    assert frame._xr.coords["channel_ref"].values.tolist() == [2e-5, 0.5]
    assert frame._xr.attrs["channel_extra"] == {
        "c0": {"sensitivity": 50.0},
        "c1": {"sensitivity": 48.5},
    }


def test_data_applies_private_storage_calibration_factors() -> None:
    frame = _frame().with_calibration([2.0, 0.5])

    np.testing.assert_allclose(frame.data, [[2.0, 4.0], [1.5, 2.0]])
    assert frame._xr.coords["channel_calibration_factor"].values.tolist() == [2.0, 0.5]


def test_channel_metadata_view_falls_back_to_value_object_attributes() -> None:
    view = ChannelMetadataView.__new__(ChannelMetadataView)
    object.__setattr__(view, "label", "snapshot")
    object.__setattr__(view, "unit", "V")
    object.__setattr__(view, "ref", 1.0)
    object.__setattr__(view, "extra", {"gain": 1.5})

    assert view.label == "snapshot"
    assert view.unit == "V"
    assert view.extra == {"gain": 1.5}

    with pytest.raises(AttributeError, match="id"):
        _ = view.id


def test_channel_metadata_view_normalizes_corrupt_extra_storage() -> None:
    frame = _frame()
    frame._xr.attrs["channel_extra"]["c0"] = "bad"

    assert frame.channels[0].extra == {}
    assert frame._xr.attrs["channel_extra"]["c0"] == {}


def test_channel_metadata_view_item_access_and_copy_semantics() -> None:
    frame = _frame()

    frame.channels[0]["label"] = "front"
    frame.channels[0]["ref"] = 0.25
    frame.channels[0]["extra"] = {"gain": {"db": 3}}

    assert frame.channels[0]["label"] == "front"
    assert frame.channels[0]["ref"] == 0.25
    assert frame.channels[0]["extra"] == {"gain": {"db": 3}}

    shallow = frame.channels[0].model_copy()
    deep = frame.channels[0].model_copy(deep=True)
    frame.channels[0].extra["gain"]["db"] = 6

    assert shallow.extra["gain"]["db"] == 3
    assert deep.extra["gain"]["db"] == 3


def test_channel_metadata_view_extra_setter_validates_mapping() -> None:
    frame = _frame()

    with pytest.raises(TypeError, match="Channel extra must be a mapping"):
        frame.channels[0].extra = "bad"  # ty: ignore[invalid-assignment]

    frame.channels[0].extra = {"ok": True}
    assert frame._xr.attrs["channel_extra"]["c0"] == {"ok": True}


def test_channel_metadata_view_empty_unit_does_not_overwrite_ref() -> None:
    frame = _frame()
    frame.channels[0].ref = 0.25

    frame.channels[0].unit = ""

    assert frame.channels[0].unit == ""
    assert frame.channels[0].ref == 0.25


def test_channel_metadata_indexer_supports_ids_labels_slices_and_validation() -> None:
    frame = _frame()

    assert frame.channels[-1].label == "right"
    assert [ch.label for ch in frame.channels[:1]] == ["left"]
    assert frame.channels["c0"].label == "left"
    assert frame.channels["right"].id == "c1"

    with pytest.raises(IndexError, match="out of range"):
        _ = frame.channels[10]
    with pytest.raises(KeyError, match="not found"):
        _ = frame.channels["missing"]
    with pytest.raises(TypeError, match="Invalid channel metadata key type"):
        _ = frame.channels[cast(Any, 1.5)]


def test_channel_metadata_indexer_equality_and_list_concatenation() -> None:
    frame = _frame()
    snapshots = frame.channels.to_list()

    assert frame.channels == snapshots
    assert frame.channels == [view for view in frame.channels]
    assert frame.channels != "left"
    assert frame.channels != [object()]
    assert frame.channels + [ChannelMetadata(label="center")] == snapshots + [ChannelMetadata(label="center")]
    assert [ChannelMetadata(label="front")] + frame.channels == [ChannelMetadata(label="front")] + snapshots
    assert isinstance(frame.channels, ChannelMetadataIndexer)


def test_channel_metadata_view_getattr_and_custom_attributes() -> None:
    frame = _frame()

    assert ChannelMetadataView.__getattr__(frame.channels[0], "label") == "left"
    with pytest.raises(AttributeError, match="missing"):
        ChannelMetadataView.__getattr__(frame.channels[0], "missing")

    view = frame.channels[0]
    view.custom_note = "local"
    assert object.__getattribute__(view, "custom_note") == "local"
    assert view["missing"] is None
