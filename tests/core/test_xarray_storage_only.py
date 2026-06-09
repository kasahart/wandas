import numpy as np
import pytest
import xarray as xr
from dask import array as da

from wandas import ChannelFrame
from wandas.core.metadata import ChannelMetadata, FrameMetadata
from wandas.frames.roughness import RoughnessFrame
from wandas.frames.spectrogram import SpectrogramFrame


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

    assert isinstance(frame._xr, xr.DataArray)
    assert frame._xr.dims == ("channel", "time")
    assert frame._data is frame._xr.data
    assert frame._data.chunks == ((1, 1), (8,))
    assert list(frame._xr.coords["channel"].values) == [
        "left",
        "right",
    ]
    assert "time" not in frame._xr.coords


def test_data_alias_is_read_only() -> None:
    frame = ChannelFrame.from_numpy(np.array([1.0, 2.0, 3.0]), sampling_rate=3.0)

    with pytest.raises(AttributeError):
        setattr(frame, "_data", da.zeros((1, 3), chunks=(1, -1)))


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

    frame._replace_data(replacement)

    assert frame._data is frame._xr.data
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

    frame._xr.attrs["label"] = "not-authoritative"
    frame._xr.attrs["metadata"] = {"gain": 999}

    assert frame.label == "owned-by-wandas"
    assert frame.metadata["gain"] == 1.5


def test_base_frame_keeps_roughness_mono_dims_neutral() -> None:
    bark_axis = np.linspace(0.5, 23.5, 47)
    data = da.ones((47, 5), chunks=(47, 5))

    frame = RoughnessFrame(
        data=data,
        sampling_rate=10.0,
        bark_axis=bark_axis,
        overlap=0.5,
        channel_metadata=[ChannelMetadata(label="mono", unit="asper")],
    )

    assert frame.n_channels == 1
    assert frame._xr.dims == ("dim_0", "dim_1")
    assert "channel" not in frame._xr.coords


def test_base_frame_does_not_invent_spectrogram_time_coords() -> None:
    data = da.ones((513, 6), chunks=(513, 3)) + 0j

    frame = SpectrogramFrame(
        data=data,
        sampling_rate=48_000.0,
        n_fft=1024,
        hop_length=256,
    )

    assert frame._xr.dims == ("dim_0", "dim_1", "dim_2")
    assert "time" not in frame._xr.coords


def test_default_spectrogram_channel_metadata_matches_n_channels() -> None:
    data = da.ones((2, 513, 6), chunks=(1, 513, 3)) + 0j

    frame = SpectrogramFrame(
        data=data,
        sampling_rate=48_000.0,
        n_fft=1024,
        hop_length=256,
    )

    assert frame.n_channels == 2
    assert len(frame.channels) == 2


def test_default_roughness_mono_channel_metadata_matches_n_channels() -> None:
    bark_axis = np.linspace(0.5, 23.5, 47)

    frame = RoughnessFrame(
        data=da.ones((47, 5), chunks=(47, 5)),
        sampling_rate=10.0,
        bark_axis=bark_axis,
        overlap=0.5,
    )

    assert frame.n_channels == 1
    assert len(frame.channels) == 1


def test_default_roughness_stereo_channel_metadata_matches_n_channels() -> None:
    bark_axis = np.linspace(0.5, 23.5, 47)

    frame = RoughnessFrame(
        data=da.ones((2, 47, 5), chunks=(1, 47, 5)),
        sampling_rate=10.0,
        bark_axis=bark_axis,
        overlap=0.5,
    )

    assert frame.n_channels == 2
    assert len(frame.channels) == 2


def test_channel_frame_refreshes_xarray_channel_coord_after_label_update() -> None:
    frame = ChannelFrame.from_numpy(
        np.ones((2, 4)),
        sampling_rate=4.0,
        ch_labels=["L", "R"],
    )

    assert list(frame._xr.coords["channel"].values) == ["L", "R"]


def test_large_lazy_channel_frame_does_not_create_internal_time_coord() -> None:
    data = da.ones((2, 1_000_000), chunks=(1, 100_000))

    frame = ChannelFrame(data=data, sampling_rate=48_000.0)

    assert frame._xr.dims == ("channel", "time")
    assert "channel" in frame._xr.coords
    assert "time" not in frame._xr.coords


def test_rename_channels_inplace_refreshes_xarray_channel_coord() -> None:
    frame = ChannelFrame.from_numpy(
        np.ones((2, 4)),
        sampling_rate=4.0,
        ch_labels=["L", "R"],
    )

    frame.rename_channels({"L": "Left"}, inplace=True)

    assert frame.labels == ["Left", "R"]
    assert list(frame._xr.coords["channel"].values) == ["Left", "R"]
