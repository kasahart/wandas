"""xarray conversion helpers for Wandas frames."""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from wandas.core.metadata import ChannelMetadata, FrameMetadata
from wandas.utils.dask_helpers import da_from_array

if TYPE_CHECKING:
    from wandas.core.base_frame import BaseFrame


_COMPLEX_COMPONENT_DIM = "complex_component"


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
    if data_array.attrs.get("wandas_attrs_encoding") == "json-v1" or data_array.attrs.get("wandas_complex_encoding"):
        data_array = decode_attrs_from_netcdf(data_array)

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
    _validate_frame_shape(data_array, frame_type)
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

    if frame_type == "NOctFrame":
        from wandas.frames.noct import NOctFrame

        return NOctFrame(
            data=dask_data,
            sampling_rate=sampling_rate,
            fmin=float(data_array.attrs["fmin"]),
            fmax=float(data_array.attrs["fmax"]),
            n=int(data_array.attrs["n"]),
            G=int(data_array.attrs["G"]),
            fr=int(data_array.attrs["fr"]),
            label=label,
            metadata=metadata,
            operation_history=operation_history,
            channel_metadata=channel_metadata,
        )

    if frame_type == "RoughnessFrame":
        from wandas.frames.roughness import RoughnessFrame

        bark_axis = np.asarray(data_array.coords["bark"].values, dtype=float)
        overlap = float(data_array.attrs.get("overlap", metadata.get("overlap", 0.0)))
        return RoughnessFrame(
            data=dask_data,
            sampling_rate=sampling_rate,
            bark_axis=bark_axis,
            overlap=overlap,
            label=label,
            metadata=metadata,
            operation_history=operation_history,
            channel_metadata=channel_metadata if "channel" in data_array.dims else None,
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
        if frame._data.ndim == 2:
            return ("bark", "time")
        if frame._data.ndim == 3:
            return ("channel", "bark", "time")
    return tuple(f"dim_{idx}" for idx in range(frame._data.ndim))


def _coords_for_frame(frame: BaseFrame[Any], dims: tuple[str, ...]) -> dict[str, Any]:
    coords: dict[str, Any] = {}
    if "channel" in dims:
        channel_size = int(frame._data.shape[dims.index("channel")])
        labels = frame.labels if len(frame.labels) == channel_size else [f"ch{i}" for i in range(channel_size)]
        units = [ch.unit for ch in frame.channels] if len(frame.channels) == channel_size else [""] * channel_size
        refs = [ch.ref for ch in frame.channels] if len(frame.channels) == channel_size else [1.0] * channel_size
        coords["channel"] = np.asarray(labels, dtype=object)
        coords["unit"] = ("channel", np.asarray(units, dtype=object))
        coords["ref"] = ("channel", np.asarray(refs, dtype=float))

    if "time" in dims:
        if hasattr(frame, "times"):
            coords["time"] = np.asarray(getattr(frame, "times"), dtype=float)
        else:
            coords["time"] = np.asarray(getattr(frame, "time"), dtype=float)

    if "frequency" in dims:
        coords["frequency"] = np.asarray(getattr(frame, "freqs"), dtype=float)

    if "band" in dims and hasattr(frame, "bands"):
        coords["band"] = np.asarray(getattr(frame, "bands"), dtype=float)

    if "bark" in dims and hasattr(frame, "bark_axis"):
        coords["bark"] = np.asarray(getattr(frame, "bark_axis"), dtype=float)

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

    if len(frame.channels) == frame.n_channels:
        attrs["channel_extra"] = copy.deepcopy([ch.extra for ch in frame.channels])
        attrs["channel_extra_by_label"] = copy.deepcopy({ch.label: ch.extra for ch in frame.channels})

    for name in ("n_fft", "hop_length", "win_length", "window", "fmin", "fmax", "n", "G", "fr", "overlap"):
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
    if dims in {("bark", "time"), ("channel", "bark", "time")}:
        return "RoughnessFrame"
    raise ValueError(f"Cannot infer Wandas frame type from dims: {dims!r}")


def _validate_dims(data_array: xr.DataArray, frame_type: str) -> None:
    expected = {
        "ChannelFrame": ("channel", "time"),
        "SpectralFrame": ("channel", "frequency"),
        "SpectrogramFrame": ("channel", "frequency", "time"),
        "NOctFrame": ("channel", "band"),
    }.get(frame_type)
    if frame_type == "RoughnessFrame":
        if tuple(data_array.dims) not in {("bark", "time"), ("channel", "bark", "time")}:
            raise ValueError(
                "Invalid dims for RoughnessFrame: expected "
                f"('bark', 'time') or ('channel', 'bark', 'time'), got {tuple(data_array.dims)}"
            )
        return
    if expected is None:
        return
    if tuple(data_array.dims) != expected:
        raise ValueError(f"Invalid dims for {frame_type}: expected {expected}, got {tuple(data_array.dims)}")


def _validate_frame_shape(data_array: xr.DataArray, frame_type: str) -> None:
    _validate_time_coordinate(data_array, frame_type)
    if frame_type in {"SpectralFrame", "SpectrogramFrame"}:
        n_fft = int(data_array.attrs["n_fft"])
        expected_frequency = n_fft // 2 + 1
        actual_frequency = int(data_array.sizes["frequency"])
        if actual_frequency != expected_frequency:
            raise ValueError(
                f"Invalid frequency dimension length for {frame_type}: "
                f"expected {expected_frequency} from n_fft={n_fft}, got {actual_frequency}"
            )
        expected_coord = np.fft.rfftfreq(n_fft, d=1 / float(data_array.attrs["sampling_rate"]))
        actual_coord = np.asarray(data_array.coords["frequency"].values, dtype=float)
        if not np.allclose(actual_coord, expected_coord):
            raise ValueError(
                f"Invalid frequency coordinate for {frame_type}: "
                "expected canonical ascending rfftfreq values from sampling_rate and n_fft"
            )
    if frame_type == "NOctFrame":
        expected_bands = _expected_noct_band_count(data_array.attrs)
        actual_bands = int(data_array.sizes["band"])
        if actual_bands != expected_bands:
            raise ValueError(
                f"Invalid band dimension length for NOctFrame: "
                f"expected {expected_bands} from NOct attrs, got {actual_bands}"
            )


def _expected_noct_band_count(attrs: dict[Any, Any]) -> int:
    from wandas.frames.noct import _center_freq

    _, freqs = _center_freq(
        fmin=float(attrs["fmin"]),
        fmax=float(attrs["fmax"]),
        n=int(attrs["n"]),
        G=int(attrs["G"]),
        fr=int(attrs["fr"]),
    )
    return len(freqs)


def _validate_time_coordinate(data_array: xr.DataArray, frame_type: str) -> None:
    if "time" not in data_array.dims or "time" not in data_array.coords:
        return

    sampling_rate = float(data_array.attrs["sampling_rate"])
    time_size = int(data_array.sizes["time"])
    if frame_type == "SpectrogramFrame":
        step = int(data_array.attrs["hop_length"]) / sampling_rate
    else:
        step = 1 / sampling_rate
    expected = np.arange(time_size, dtype=float) * step
    actual = np.asarray(data_array.coords["time"].values, dtype=float)
    if not np.allclose(actual, expected):
        raise ValueError(
            f"Invalid time coordinate for {frame_type}: "
            "expected canonical values starting at 0 from sampling_rate/hop_length"
        )


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
    extras = _channel_extras_from_attrs(data_array.attrs, labels, n_channels)

    return [
        ChannelMetadata(label=str(labels[idx]), unit=str(units[idx]), ref=float(refs[idx]), extra=extras[idx])
        for idx in range(n_channels)
    ]


def _channel_extras_from_attrs(attrs: dict[Any, Any], labels: list[Any], n_channels: int) -> list[dict[str, Any]]:
    extras_by_label = attrs.get("channel_extra_by_label")
    if isinstance(extras_by_label, dict):
        return [copy.deepcopy(extras_by_label.get(str(label), {})) for label in labels]

    extras = copy.deepcopy(attrs.get("channel_extra", [{} for _ in range(n_channels)]))
    if not isinstance(extras, list) or len(extras) != n_channels:
        return [{} for _ in range(n_channels)]
    return extras


def _coord_values(data_array: xr.DataArray, name: str, default: list[Any]) -> list[Any]:
    if name not in data_array.coords:
        return default
    values = data_array.coords[name].values
    if np.ndim(values) == 0:
        return [values.item()] * len(default)
    return list(values)


def _chunks_for_ndim(ndim: int) -> tuple[int, ...]:
    if ndim == 0:
        return ()
    if ndim == 1:
        return (-1,)
    return tuple([1] + [-1] * (ndim - 1))


def _as_dask_channelwise(data_array: xr.DataArray) -> Any:
    chunks = _chunks_for_ndim(data_array.ndim)
    data = data_array.data
    if hasattr(data, "rechunk"):
        return data.rechunk(chunks)
    return da_from_array(np.asarray(data), chunks=chunks)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


def encode_attrs_for_netcdf(data_array: xr.DataArray) -> xr.DataArray:
    """Return a DataArray with NetCDF-safe Wandas attrs."""
    encoded = _encode_complex_for_netcdf(data_array)
    attrs = copy.deepcopy(dict(encoded.attrs))
    for key in ("metadata", "operation_history", "channel_extra", "channel_extra_by_label"):
        if key in attrs:
            attrs[key] = json.dumps(_json_safe(attrs[key]))
    attrs["wandas_attrs_encoding"] = "json-v1"
    encoded.attrs = attrs
    return encoded


def decode_attrs_from_netcdf(data_array: xr.DataArray) -> xr.DataArray:
    """Decode Wandas attrs loaded from NetCDF."""
    decoded = data_array.copy(deep=False)
    attrs = copy.deepcopy(dict(decoded.attrs))
    if attrs.get("wandas_attrs_encoding") == "json-v1":
        for key in ("metadata", "operation_history", "channel_extra", "channel_extra_by_label"):
            if key in attrs and isinstance(attrs[key], str):
                attrs[key] = json.loads(attrs[key])
        attrs.pop("wandas_attrs_encoding", None)
    decoded.attrs = attrs
    return _decode_complex_from_netcdf(decoded)


def _encode_complex_for_netcdf(data_array: xr.DataArray) -> xr.DataArray:
    if not np.issubdtype(data_array.dtype, np.complexfloating):
        return data_array.copy(deep=False)

    encoded = xr.concat(
        [data_array.real, data_array.imag],
        dim=xr.IndexVariable(_COMPLEX_COMPONENT_DIM, ["real", "imag"]),
    ).transpose(*data_array.dims, _COMPLEX_COMPONENT_DIM)
    encoded.attrs = copy.deepcopy(dict(data_array.attrs))
    encoded.attrs["wandas_complex_encoding"] = "real-imag-v1"
    return encoded


def _decode_complex_from_netcdf(data_array: xr.DataArray) -> xr.DataArray:
    if data_array.attrs.get("wandas_complex_encoding") != "real-imag-v1":
        return data_array

    attrs = copy.deepcopy(dict(data_array.attrs))
    attrs.pop("wandas_complex_encoding", None)
    decoded = data_array.sel({_COMPLEX_COMPONENT_DIM: "real"}) + 1j * data_array.sel({_COMPLEX_COMPONENT_DIM: "imag"})
    decoded = decoded.drop_vars(_COMPLEX_COMPONENT_DIM, errors="ignore")
    decoded.attrs = attrs
    return decoded


def open_netcdf(path: str | Any) -> BaseFrame[Any]:
    """Open a Wandas NetCDF file as a frame."""
    data_array = xr.open_dataarray(path).load()
    try:
        return from_xarray(decode_attrs_from_netcdf(data_array))
    finally:
        data_array.close()
