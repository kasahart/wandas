import numpy as np
import pytest
import xarray as xr
from dask import array as da

from wandas import ChannelFrame
from wandas.core.metadata import ChannelMetadata, FrameMetadata


def test_base_frame_owns_xarray_dataarray() -> None:
    data = da.ones((2, 8), chunks=(2, 4))
    frame = ChannelFrame(
        data=data,
        sampling_rate=4.0,
        label="signal",
        channel_metadata=[
            ChannelMetadata(label="left", unit="Pa"),
            ChannelMetadata(label="right", unit="Pa"),
        ],
    )

    assert isinstance(frame._xr, xr.DataArray)  # ty: ignore[unresolved-attribute]
    assert frame._xr.dims == ("channel", "time")  # ty: ignore[unresolved-attribute]
    assert frame._data is frame._xr.data  # ty: ignore[unresolved-attribute]
    assert frame._data.chunks == ((1, 1), (8,))
    assert list(frame._xr.coords["channel"].values) == [  # ty: ignore[unresolved-attribute]
        "left",
        "right",
    ]
    assert frame._xr.coords["time"].values.tolist() == [  # ty: ignore[unresolved-attribute]
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
    ]


def test_data_alias_is_read_only() -> None:
    frame = ChannelFrame.from_numpy(np.array([1.0, 2.0, 3.0]), sampling_rate=3.0)

    with pytest.raises(AttributeError):
        frame._data = da.zeros((1, 3), chunks=(1, -1))  # type: ignore[misc]


def test_replace_data_updates_xarray_container_only() -> None:
    metadata = FrameMetadata({"source": "test"}, source_file="input.wav")
    frame = ChannelFrame.from_numpy(
        np.array([[1.0, 2.0, 3.0]]),
        sampling_rate=3.0,
        label="original",
        metadata=metadata,
    )
    original_metadata = frame.metadata
    original_history = frame.operation_history
    replacement = da.full((1, 4), 2.0, chunks=(1, -1))

    frame._replace_data(replacement)  # ty: ignore[unresolved-attribute]

    assert frame._data is frame._xr.data  # ty: ignore[unresolved-attribute]
    assert frame._data.shape == (1, 4)
    assert frame._data.chunks == ((1,), (4,))
    assert frame.metadata is original_metadata
    assert frame.operation_history is original_history
    assert frame.label == "original"
    assert frame.sampling_rate == 3.0


def test_internal_xarray_attrs_do_not_own_wandas_metadata() -> None:
    frame = ChannelFrame.from_numpy(
        np.array([[1.0, 2.0]]),
        sampling_rate=2.0,
        label="owned-by-wandas",
        metadata={"gain": 1.5},
    )

    frame._xr.attrs["label"] = "not-authoritative"  # ty: ignore[unresolved-attribute]
    frame._xr.attrs["metadata"] = {"gain": 999}  # ty: ignore[unresolved-attribute]

    assert frame.label == "owned-by-wandas"
    assert frame.metadata["gain"] == 1.5
