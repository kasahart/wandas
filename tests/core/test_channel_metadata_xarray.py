import copy
from collections.abc import Sequence

import numpy as np
import pytest

from wandas.core.metadata import ChannelMetadata
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
    assert frame._xr.attrs["channel_extra"] == {
        "c0": {"sensitivity": 50.0},
        "c1": {"sensitivity": 48.5},
    }


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
    with pytest.raises(TypeError, match="unit must be a string"):
        setattr(frame.channels[0], "unit", 123)
    with pytest.raises(TypeError, match="ref must be a number"):
        setattr(frame.channels[0], "ref", True)
    with pytest.raises(TypeError, match="ref must be a number"):
        setattr(frame.channels[0], "ref", "1.0")

    assert frame.channels[0].label == "left"
    assert frame.channels[0].unit == "Pa"
    assert frame.channels[0].ref == 2e-5


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


def test_to_xarray_exports_channel_metadata_without_sharing_attrs() -> None:
    frame = _frame()

    exported = frame.to_xarray()

    assert exported is not frame._xr
    assert exported.coords["channel"].values.tolist() == ["c0", "c1"]
    assert exported.coords["channel_label"].values.tolist() == ["left", "right"]
    assert exported.coords["channel_unit"].values.tolist() == ["Pa", "V"]
    assert exported.coords["channel_ref"].values.tolist() == [2e-5, 0.5]
    assert exported.attrs["channel_extra"] == copy.deepcopy(frame._xr.attrs["channel_extra"])

    exported.attrs["channel_extra"]["c0"]["sensitivity"] = 1.0
    assert frame.channels[0].extra["sensitivity"] == 50.0
