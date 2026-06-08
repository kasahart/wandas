"""xarray conversion helpers for Wandas frames."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from wandas.core.metadata import ChannelMetadata, FrameMetadata
from wandas.utils.dask_helpers import da_from_array

if TYPE_CHECKING:
    from wandas.core.base_frame import BaseFrame


def frame_to_xarray(frame: BaseFrame[Any], *, copy_dataarray: bool = True) -> xr.DataArray:
    """Convert a Wandas frame to an xarray DataArray using the Wandas schema."""
    frame_type = type(frame).__name__
    dims = _dims_for_frame(frame)
    coords = _coords_for_frame(frame, dims)
    attrs = _attrs_for_frame(frame, frame_type)

    data_array = xr.DataArray(
        frame._data,
        dims=dims,
        coords=coords,
        attrs=attrs,
        name=frame.label,
    )
    if copy_dataarray:
        return data_array.copy(deep=False)
    return data_array


def from_xarray(data_array: xr.DataArray) -> BaseFrame[Any]:
    """Create a Wandas frame from a Wandas-schema xarray DataArray."""
    frame_type = str(data_array.attrs.get("wandas_frame_type", ""))
    if not frame_type:
        frame_type = _infer_frame_type(data_array)

    _validate_dims(data_array, frame_type)

    sampling_rate = float(data_array.attrs["sampling_rate"])
    label_attr = data_array.attrs.get("label")
    label = str(label_attr) if label_attr is not None else None
    metadata = _metadata_from_attrs(data_array.attrs)
    operation_history = copy.deepcopy(data_array.attrs.get("operation_history", []))
    channel_metadata = _channel_metadata_from_coords(data_array)
    dask_data = _as_dask_channelwise(data_array)

    if frame_type == "ChannelFrame":
        from wandas.frames.channel import ChannelFrame

        return ChannelFrame(
            data=dask_data,
            sampling_rate=sampling_rate,
            label=label,
            metadata=metadata,
            operation_history=operation_history,
            channel_metadata=channel_metadata,
        )

    if frame_type == "SpectralFrame":
        from wandas.frames.spectral import SpectralFrame

        return SpectralFrame(
            data=dask_data,
            sampling_rate=sampling_rate,
            n_fft=int(data_array.attrs["n_fft"]),
            window=str(data_array.attrs.get("window", "hann")),
            label=label,
            metadata=metadata,
            operation_history=operation_history,
            channel_metadata=channel_metadata,
        )

    if frame_type == "SpectrogramFrame":
        from wandas.frames.spectrogram import SpectrogramFrame

        return SpectrogramFrame(
            data=dask_data,
            sampling_rate=sampling_rate,
            n_fft=int(data_array.attrs["n_fft"]),
            hop_length=int(data_array.attrs["hop_length"]),
            win_length=int(data_array.attrs.get("win_length", data_array.attrs["n_fft"])),
            window=str(data_array.attrs.get("window", "hann")),
            label=label,
            metadata=metadata,
            operation_history=operation_history,
            channel_metadata=channel_metadata,
        )

    raise ValueError(f"Unsupported Wandas xarray frame type: {frame_type!r}")


def _dims_for_frame(frame: BaseFrame[Any]) -> tuple[str, ...]:
    frame_type = type(frame).__name__
    if frame_type == "ChannelFrame":
        return ("channel", "time")
    if frame_type == "SpectralFrame":
        return ("channel", "frequency")
    if frame_type == "SpectrogramFrame":
        return ("channel", "frequency", "time")
    if frame_type == "NOctFrame":
        return ("channel", "band")
    if frame_type == "RoughnessFrame":
        return ("channel", "time")
    return tuple(f"dim_{idx}" for idx in range(frame._data.ndim))


def _coords_for_frame(frame: BaseFrame[Any], dims: tuple[str, ...]) -> dict[str, Any]:
    coords: dict[str, Any] = {}
    if "channel" in dims:
        coords["channel"] = np.asarray(frame.labels, dtype=object)
        coords["unit"] = ("channel", np.asarray([ch.unit for ch in frame.channels], dtype=object))
        coords["ref"] = ("channel", np.asarray([ch.ref for ch in frame.channels], dtype=float))

    if "time" in dims:
        if hasattr(frame, "times"):
            coords["time"] = np.asarray(getattr(frame, "times"), dtype=float)
        else:
            coords["time"] = np.asarray(getattr(frame, "time"), dtype=float)

    if "frequency" in dims:
        coords["frequency"] = np.asarray(getattr(frame, "freqs"), dtype=float)

    if "band" in dims and hasattr(frame, "bands"):
        coords["band"] = np.asarray(getattr(frame, "bands"), dtype=float)

    return coords


def _attrs_for_frame(frame: BaseFrame[Any], frame_type: str) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "wandas_frame_type": frame_type,
        "sampling_rate": frame.sampling_rate,
        "label": frame.label,
        "metadata": copy.deepcopy(dict(frame.metadata)),
        "operation_history": copy.deepcopy(frame.operation_history),
    }
    source_file = getattr(frame.metadata, "source_file", None)
    if source_file is not None:
        attrs["source_file"] = source_file

    for name in ("n_fft", "hop_length", "win_length", "window"):
        if hasattr(frame, name):
            attrs[name] = getattr(frame, name)

    return attrs


def _infer_frame_type(data_array: xr.DataArray) -> str:
    dims = tuple(data_array.dims)
    if dims == ("channel", "time"):
        return "ChannelFrame"
    if dims == ("channel", "frequency"):
        return "SpectralFrame"
    if dims == ("channel", "frequency", "time"):
        return "SpectrogramFrame"
    raise ValueError(f"Cannot infer Wandas frame type from dims: {dims!r}")


def _validate_dims(data_array: xr.DataArray, frame_type: str) -> None:
    expected = {
        "ChannelFrame": ("channel", "time"),
        "SpectralFrame": ("channel", "frequency"),
        "SpectrogramFrame": ("channel", "frequency", "time"),
    }.get(frame_type)
    if expected is None:
        return
    if tuple(data_array.dims) != expected:
        raise ValueError(f"Invalid dims for {frame_type}: expected {expected}, got {tuple(data_array.dims)}")


def _metadata_from_attrs(attrs: dict[Any, Any]) -> FrameMetadata:
    metadata = FrameMetadata(copy.deepcopy(attrs.get("metadata", {})))
    source_file = attrs.get("source_file")
    if source_file is not None:
        metadata.source_file = str(source_file)
    return metadata


def _channel_metadata_from_coords(data_array: xr.DataArray) -> list[ChannelMetadata]:
    n_channels = int(data_array.sizes.get("channel", 1))
    labels = _coord_values(data_array, "channel", [f"ch{i}" for i in range(n_channels)])
    units = _coord_values(data_array, "unit", [""] * n_channels)
    refs = _coord_values(data_array, "ref", [1.0] * n_channels)

    return [
        ChannelMetadata(label=str(labels[idx]), unit=str(units[idx]), ref=float(refs[idx])) for idx in range(n_channels)
    ]


def _coord_values(data_array: xr.DataArray, name: str, default: list[Any]) -> list[Any]:
    if name not in data_array.coords:
        return default
    values = data_array.coords[name].values
    if np.ndim(values) == 0:
        return [values.item()] * len(default)
    return list(values)


def _as_dask_channelwise(data_array: xr.DataArray) -> Any:
    data = data_array.data
    if hasattr(data, "rechunk"):
        dask_data = data
    else:
        dask_data = da_from_array(np.asarray(data), chunks=tuple([1] + [-1] * (data_array.ndim - 1)))

    if data_array.ndim >= 2:
        chunks = tuple([1] + [-1] * (data_array.ndim - 1))
        return dask_data.rechunk(chunks)
    return dask_data.rechunk((-1,))
