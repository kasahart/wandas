"""Strict xarray-backed persistence for typed WDF 0.4 artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import xarray as xr

from wandas.core.base_frame import BaseFrame
from wandas.core.metadata import ChannelCalibration, ChannelMetadata
from wandas.utils.optional_imports import require_h5netcdf

from .wdf_frames import decode_frame, encode_frame, frame_dimension_coordinates, restore_frame_coordinates

WDF_FORMAT_VERSION = "0.4"

_ROOT_ATTRS = frozenset(
    {
        "version",
        "frame_type",
        "sampling_rate",
        "label",
        "constructor_json",
        "metadata_json",
        "operation_history_json",
    }
)
_DATA_VARIABLES = frozenset(
    {
        "data",
        "channel_label",
        "channel_unit",
        "channel_ref",
        "channel_calibration_factor",
        "source_time_offset",
        "channel_extra_json",
    }
)


def _dump_json(value: object, *, field: str) -> str:
    """Encode strict JSON with field-aware save diagnostics."""
    try:
        return json.dumps(value, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "WDF field is not strict-JSON serializable\n"
            f"  Field: {field}\n"
            f"  Cause: {exc}\n"
            "Replace non-finite numbers and unsupported objects with strict JSON values before saving."
        ) from exc


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite number found: {value}")


def _load_json(value: object, *, field: str) -> Any:
    """Decode strict JSON from one required text attribute or variable."""
    if not isinstance(value, str):
        raise ValueError(f"Invalid WDF text field {field}; expected text, got {type(value).__name__}")
    try:
        return json.loads(value, parse_constant=_reject_nonfinite)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Invalid strict JSON in WDF\n"
            f"  Field: {field}\n"
            f"  Cause: {exc}\n"
            "Resave the file with a compatible Wandas version."
        ) from exc


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"Invalid WDF numeric field {field}; got {value!r}")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"Invalid WDF numeric field {field}; expected a finite value")
    return normalized


def _validate_history(value: object) -> list[dict[str, Any]]:
    expected = {"operation", "version", "params"}
    if not isinstance(value, list):
        raise ValueError("Invalid WDF operation history JSON; expected canonical history records")
    records: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Invalid WDF operation history JSON; expected canonical history records")
        record = cast(dict[str, Any], item)
        if (
            set(record) != expected
            or not isinstance(record["operation"], str)
            or not record["operation"].strip()
            or type(record["version"]) is not int
            or not isinstance(record["params"], dict)
        ):
            raise ValueError("Invalid WDF operation history JSON; expected canonical history records")
        records.append(record)
    return records


def _normalized_path(path: str | Path) -> Path:
    target = Path(path)
    return target if target.suffix == ".wdf" else target.with_suffix(".wdf")


def _build_dataset(frame: BaseFrame[Any]) -> xr.Dataset:
    """Build WDF from raw internal data, keeping calibration separate."""
    frame_type, constructor = encode_frame(frame)
    channels = frame.channels.to_list()
    extras = [_dump_json(channel.extra, field=f"channel_extra_json[{index}]") for index, channel in enumerate(channels)]
    attrs: dict[str, Any] = {
        "version": WDF_FORMAT_VERSION,
        "frame_type": frame_type,
        "sampling_rate": float(frame.sampling_rate),
        "label": _dump_json(frame.label, field="label"),
        "constructor_json": _dump_json(constructor, field="constructor_json"),
        "metadata_json": _dump_json(dict(frame.metadata), field="metadata_json"),
        "operation_history_json": _dump_json(frame.operation_history, field="operation_history_json"),
    }

    # Public to_xarray() contains calibrated values. WDF stores the raw tensor and
    # calibration independently so a loaded Frame applies each factor exactly once.
    data_array = xr.DataArray(frame._data, dims=frame._xr.dims)
    variables: dict[str, Any] = {
        "data": data_array,
        "channel_label": ("channel", [channel.label for channel in channels]),
        "channel_unit": ("channel", [channel.unit for channel in channels]),
        "channel_ref": ("channel", [channel.ref for channel in channels]),
        "channel_calibration_factor": ("channel", [channel.calibration.factor for channel in channels]),
        "source_time_offset": ("channel", np.asarray(frame.source_time_offset, dtype=float)),
        "channel_extra_json": ("channel", extras),
    }
    coords: dict[str, Any] = {"channel": ("channel", frame._channel_ids)}
    coords.update({name: (name, values) for name, values in frame_dimension_coordinates(frame).items()})
    return xr.Dataset(variables, coords=coords, attrs=attrs)


def save(
    frame: BaseFrame[Any],
    path: str | Path,
    *,
    compress: str | None = "gzip",
    overwrite: bool = False,
) -> None:
    """Save an exact built-in Frame as WDF 0.4.

    Dask data is handed directly to xarray and is written synchronously; Wandas does
    not first materialize the complete tensor with ``frame._data.compute()``.
    """
    target = _normalized_path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"File {target} already exists. Set overwrite=True to overwrite.")

    # Validate all Frame and JSON state before either importing the storage backend
    # or opening the destination, so invalid state cannot leave a partial artifact.
    dataset = _build_dataset(frame)
    require_h5netcdf("WDF save")
    encoding = {"data": {"compression": compress}} if compress else None
    dataset.to_netcdf(
        target,
        engine="h5netcdf",
        encoding=encoding,
        invalid_netcdf=True,
    )


def _require_exact_schema(dataset: xr.Dataset) -> None:
    missing_attrs = _ROOT_ATTRS - set(dataset.attrs)
    unexpected_attrs = set(dataset.attrs) - _ROOT_ATTRS
    if missing_attrs or unexpected_attrs:
        raise ValueError(
            "Invalid WDF root attribute schema\n"
            f"  Missing: {sorted(missing_attrs)}\n"
            f"  Unexpected: {sorted(unexpected_attrs)}"
        )
    missing_variables = _DATA_VARIABLES - set(dataset.data_vars)
    unexpected_variables = set(dataset.data_vars) - _DATA_VARIABLES
    if missing_variables or unexpected_variables:
        raise ValueError(
            "Invalid WDF data variable schema\n"
            f"  Missing: {sorted(missing_variables)}\n"
            f"  Unexpected: {sorted(unexpected_variables)}"
        )
    coordinates = set(dataset.coords)
    allowed_coordinates = {"channel"} | (set(dataset["data"].dims) - {"frequency", "time"})
    if (
        "channel" not in coordinates
        or not coordinates <= allowed_coordinates
        or any(dataset.coords[name].dims != (name,) for name in coordinates)
    ):
        raise ValueError(
            "Invalid WDF coordinate schema\n"
            f"  Got: {sorted(coordinates)}\n"
            "  Expected: channel and optional one-dimensional data-dimension coordinates"
        )


def _text_vector(dataset: xr.Dataset, name: str, channel_count: int) -> list[str]:
    variable = dataset[name]
    if variable.dims != ("channel",) or variable.shape != (channel_count,):
        raise ValueError(f"Invalid WDF channel variable {name}; expected one value per channel")
    values = variable.values.tolist()
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"Invalid WDF channel variable {name}; expected text values")
    return values


def _number_vector(dataset: xr.Dataset, name: str, channel_count: int) -> np.ndarray[Any, Any]:
    variable = dataset[name]
    if (
        variable.dims != ("channel",)
        or variable.shape != (channel_count,)
        or not np.issubdtype(variable.dtype, np.number)
    ):
        raise ValueError(f"Invalid WDF channel variable {name}; expected one numeric value per channel")
    values = np.asarray(variable.values)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Invalid WDF channel variable {name}; expected finite values")
    return values


def load(path: str | Path) -> BaseFrame[Any]:
    """Load a local WDF 0.4 artifact as its exact built-in Frame type.

    The returned tensor is a backend-backed Dask array. Until it is computed or
    persisted, do not move, delete, or overwrite the source WDF path.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"File not found: {source}")
    require_h5netcdf("WDF load")

    # CF decoding is disabled because WDF owns dtype/value semantics; foreign CF
    # scale, offset, fill-value, and time attributes must never transform raw data.
    dataset = xr.open_dataset(
        source,
        engine="h5netcdf",
        chunks={},
        decode_cf=False,
        mask_and_scale=False,
        backend_kwargs={"phony_dims": "access"},
    )
    version = dataset.attrs.get("version")
    if version != WDF_FORMAT_VERSION:
        got = "missing" if version is None else repr(version)
        raise ValueError(
            "Unsupported WDF format version\n"
            f"  Got: {got}\n"
            f"  Supported: {WDF_FORMAT_VERSION!r}\n"
            "Use a compatible Wandas version or resave the file."
        )
    _require_exact_schema(dataset)

    frame_type = dataset.attrs["frame_type"]
    if not isinstance(frame_type, str):
        raise ValueError("Invalid WDF frame_type; expected text")
    constructor = _load_json(dataset.attrs["constructor_json"], field="constructor_json")
    metadata = _load_json(dataset.attrs["metadata_json"], field="metadata_json")
    label = _load_json(dataset.attrs["label"], field="label")
    history = _validate_history(_load_json(dataset.attrs["operation_history_json"], field="operation_history_json"))
    if not isinstance(constructor, Mapping):
        raise ValueError("Invalid WDF constructor_json; expected an object")
    if not isinstance(metadata, dict):
        raise ValueError("Invalid WDF metadata_json; expected an object")
    if label is not None and not isinstance(label, str):
        raise ValueError("Invalid WDF label; expected a string or null")
    sampling_rate = _finite_number(dataset.attrs["sampling_rate"], field="sampling_rate")

    data = dataset["data"]
    channel_count = int(dataset.sizes["channel"])
    channel_ids = _text_vector(dataset, "channel", channel_count)
    labels = _text_vector(dataset, "channel_label", channel_count)
    units = _text_vector(dataset, "channel_unit", channel_count)
    refs = _number_vector(dataset, "channel_ref", channel_count)
    factors = _number_vector(dataset, "channel_calibration_factor", channel_count)
    offsets = _number_vector(dataset, "source_time_offset", channel_count)
    extras_json = _text_vector(dataset, "channel_extra_json", channel_count)
    extras = [_load_json(value, field=f"channel_extra_json[{index}]") for index, value in enumerate(extras_json)]
    if not all(isinstance(extra, dict) for extra in extras):
        raise ValueError("Invalid WDF channel_extra_json; expected JSON objects")
    channels = [
        ChannelMetadata(
            label=labels[index],
            calibration=ChannelCalibration(factor=float(factors[index]), unit=units[index], ref=float(refs[index])),
            extra=extras[index],
        )
        for index in range(channel_count)
    ]
    common: dict[str, Any] = {
        "sampling_rate": sampling_rate,
        "label": label,
        "metadata": metadata,
        "channel_metadata": channels,
        "channel_ids": channel_ids,
        "source_time_offset": offsets,
        "operation_history_prefix": history,
    }
    frame = decode_frame(
        frame_type,
        constructor,
        data=data.data,
        common=common,
        stored_dims=cast(tuple[str, ...], tuple(data.dims)),
    )
    coordinates = {
        str(name): np.asarray(coordinate.values) for name, coordinate in dataset.coords.items() if name != "channel"
    }
    restore_frame_coordinates(frame, coordinates)
    return frame


__all__ = ["WDF_FORMAT_VERSION", "load", "save"]
