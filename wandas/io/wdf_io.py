"""Typed WDF (Wandas Data File) persistence based on HDF5.

WDF 0.3 preserves built-in Frame type, semantic dimensions, domain constructor
state, represented coordinates, channel metadata, frame metadata, and display
history. Runtime lineage and Dask graphs are deliberately not reconstructed.
"""

import json
import logging
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast

import numpy as np

# Import BaseFrame from core module
from wandas.utils.dask_helpers import da_from_array as _da_from_array
from wandas.utils.optional_imports import require_h5py

from ..core.base_frame import BaseFrame
from .readers import download_url_to_temporary_file
from .wdf_frames import (
    decode_frame_state,
    encode_frame_state,
    frame_dimension_coordinates,
    restore_frame_coordinates,
    validate_frame_save_dtype,
)

logger = logging.getLogger(__name__)

# Constants for version management
WDF_FORMAT_VERSION = "0.3"
SUPPORTED_WDF_FORMAT_VERSIONS = frozenset({"0.1", "0.2", WDF_FORMAT_VERSION})
FRAME_STATE_SCHEMA_VERSION = 1
FRAME_STATE_SCHEMA_ATTR = "frame_state_schema"
FRAME_STATE_JSON_ATTR = "frame_state_json"
OPERATION_HISTORY_SCHEMA_VERSION = 1
OPERATION_HISTORY_SCHEMA_ATTR = "operation_history_schema"
OPERATION_HISTORY_JSON_ATTR = "operation_history_json"
LEGACY_OPERATION_SUMMARIES_SCHEMA_VERSION = 1
LEGACY_OPERATION_SUMMARIES_SCHEMA_ATTR = "operation_summaries_schema"
LEGACY_OPERATION_SUMMARIES_JSON_ATTR = "operation_summaries_json"
LEGACY_OPERATION_HISTORY_GROUP = "operation_history"
_WDF_V03_CHANNEL_ATTRS = frozenset({"label", "unit", "ref", "calibration_factor", "source_time_offset"})


def _dump_wdf_json(value: object, *, field: str) -> str:
    """Encode one strict JSON field with WDF-specific failure context."""
    try:
        return json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "WDF field is not strict-JSON serializable\n"
            f"  Field: {field}\n"
            f"  Cause: {exc}\n"
            "Replace non-finite numbers and unsupported objects with strict JSON values before saving."
        ) from exc


def _load_wdf_json(value: object, *, field: str) -> Any:
    """Decode one strict JSON field with WDF-specific failure context."""
    try:
        return json.loads(
            _decode_hdf5_str(value),
            parse_constant=_reject_nonfinite_json_number,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Invalid strict JSON in WDF\n"
            f"  Field: {field}\n"
            f"  Cause: {exc}\n"
            "Resave the file with a compatible Wandas version."
        ) from exc


def _validate_v03_channel_keys(keys: list[str], channel_ids: list[str]) -> list[int]:
    """Validate exact contiguous channel groups for WDF 0.3."""
    expected = [str(index) for index in range(len(channel_ids))]
    if set(keys) != set(expected):
        raise ValueError(
            "Invalid WDF 0.3 channel groups\n"
            f"  Got: {sorted(keys)}\n"
            f"  Expected: {expected}\n"
            "The channel metadata layout is incomplete; resave the file with Wandas 0.6."
        )
    return list(range(len(expected)))


def _validate_v03_channel_attrs(channel: Any, index: int) -> None:
    """Require every channel attribute written by WDF 0.3."""
    missing = _WDF_V03_CHANNEL_ATTRS - set(channel.attrs)
    if missing:
        raise ValueError(
            "Incomplete WDF 0.3 channel metadata\n"
            f"  Channel: {index}\n"
            f"  Missing attributes: {sorted(missing)}\n"
            "Resave the file with Wandas 0.6; channel metadata cannot be reconstructed safely."
        )


def _load_schema_version(value: object, *, field: str) -> int:
    """Decode one integer schema attribute with field-aware context."""
    try:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("boolean values are not schema integers")
        normalized = int(cast(Any, value))
        if isinstance(value, (float, np.floating)) and not float(value).is_integer():
            raise ValueError("fractional values are not schema integers")
        return normalized
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Invalid WDF schema version attribute\n"
            f"  Field: {field}\n"
            f"  Got: {value!r}\n"
            f"  Cause: {exc}\n"
            "Resave the file with a compatible Wandas version using an integer schema version."
        ) from exc


def _decode_hdf5_str(value: object) -> str:
    """Decode an HDF5 attribute value to a Python string.

    HDF5 may return ``bytes``, ``numpy.bytes_``, or plain ``str``.
    """
    if isinstance(value, (bytes, np.bytes_)):
        try:
            return value.decode("utf-8")
        except (UnicodeDecodeError, AttributeError):
            return str(value)
    return str(value)


def _reject_nonfinite_json_number(value: str) -> None:
    """Reject non-finite constants while decoding strict history JSON."""
    raise ValueError(f"WDF operation history must use strict JSON; non-finite number found: {value}")


def _migrate_legacy_history(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert pre-0.5 display records to the canonical history shape."""
    migrated: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        operation = record.get("operation")
        if not isinstance(operation, str) or not operation.strip():
            raise ValueError(
                "Invalid legacy WDF operation history record\n"
                f"  Record: {index}\n"
                "  Expected: a non-blank 'operation' string\n"
                "Resave the file with a compatible pre-0.5 Wandas version."
            )

        stored_params = record.get("params", {})
        params = dict(stored_params) if isinstance(stored_params, Mapping) else {"legacy_params": stored_params}
        for field, value in record.items():
            if field in {"operation", "params"}:
                continue
            if field in params:
                raise ValueError(
                    "Invalid legacy WDF operation history record\n"
                    f"  Record: {index}\n"
                    f"  Got: duplicate field {field!r} in params and the record\n"
                    "Resave the file with a compatible pre-0.5 Wandas version."
                )
            params[field] = value
        migrated.append({"operation": operation, "version": 1, "params": params})
    return migrated


def _load_legacy_history(h5_file: Any) -> list[dict[str, Any]] | None:
    """Read the two history encodings written before WDF 0.2."""
    if LEGACY_OPERATION_SUMMARIES_JSON_ATTR in h5_file.attrs:
        schema = int(h5_file.attrs.get(LEGACY_OPERATION_SUMMARIES_SCHEMA_ATTR, 0))
        if schema != LEGACY_OPERATION_SUMMARIES_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported WDF operation summaries schema\n"
                f"  Got: {schema}\n"
                f"  Supported: {LEGACY_OPERATION_SUMMARIES_SCHEMA_VERSION}\n"
                "Use a compatible Wandas version or resave the file."
            )
        parsed = json.loads(
            _decode_hdf5_str(h5_file.attrs[LEGACY_OPERATION_SUMMARIES_JSON_ATTR]),
            parse_constant=_reject_nonfinite_json_number,
        )
        if not isinstance(parsed, list) or not all(isinstance(record, dict) for record in parsed):
            raise ValueError(
                "Invalid WDF operation summaries JSON\n"
                f"  Expected: JSON array of objects\n  Got: {type(parsed).__name__}"
            )
        return parsed

    if LEGACY_OPERATION_HISTORY_GROUP not in h5_file:
        return None
    operation_group = h5_file[LEGACY_OPERATION_HISTORY_GROUP]
    if not all(key.startswith("operation_") and key.removeprefix("operation_").isdigit() for key in operation_group):
        raise ValueError(
            "Invalid legacy WDF operation history group\n"
            "  Expected: groups named operation_<non-negative integer>\n"
            "Resave the file with a compatible pre-0.5 Wandas version."
        )

    records: list[dict[str, Any]] = []
    for key in sorted(operation_group, key=lambda item: int(item.removeprefix("operation_"))):
        stored = operation_group[key]
        record: dict[str, Any] = {}
        for name, value in stored.attrs.items():
            decoded = _decode_hdf5_str(value) if isinstance(value, (str, bytes, np.bytes_)) else value
            if isinstance(decoded, str):
                try:
                    decoded = json.loads(decoded, parse_constant=_reject_nonfinite_json_number)
                except json.JSONDecodeError:
                    pass
            record[name] = decoded
        records.append(record)
    return records


def _load_operation_history(h5_file: Any) -> list[dict[str, Any]]:
    """Load and structurally validate display history from an open WDF file."""
    if OPERATION_HISTORY_JSON_ATTR not in h5_file.attrs:
        legacy_history = _load_legacy_history(h5_file)
        return [] if legacy_history is None else _migrate_legacy_history(legacy_history)
    schema = int(h5_file.attrs.get(OPERATION_HISTORY_SCHEMA_ATTR, 0))
    if schema != OPERATION_HISTORY_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported WDF operation history schema\n"
            f"  Got: {schema}\n"
            f"  Supported: {OPERATION_HISTORY_SCHEMA_VERSION}\n"
            "Use a compatible Wandas version or resave the file."
        )
    parsed = json.loads(
        _decode_hdf5_str(h5_file.attrs[OPERATION_HISTORY_JSON_ATTR]),
        parse_constant=_reject_nonfinite_json_number,
    )
    expected_fields = {"operation", "version", "params"}
    if not isinstance(parsed, list) or not all(
        isinstance(record, dict) and set(record) == expected_fields for record in parsed
    ):
        raise ValueError(
            f"Invalid WDF operation history JSON\n  Expected: canonical history records\n  Got: {type(parsed).__name__}"
        )
    return parsed


def save(
    frame: BaseFrame[Any],
    path: str | Path,
    *,
    format: str = "hdf5",
    compress: str | None = "gzip",
    overwrite: bool = False,
    dtype: str | np.dtype[Any] | None = None,
) -> None:
    """Save a frame to a file.

    Args:
        frame: The frame to save.
        path: Path to save the file. '.wdf' extension will be added if not present.
        format: Format to use (currently only 'hdf5' is supported)
        compress: Compression method ('gzip' by default, None for no compression)
        overwrite: Whether to overwrite existing file
        dtype: Optional data type conversion before saving (e.g. 'float32')

    Raises:
        FileExistsError: If the file exists and overwrite=False.
        NotImplementedError: For unsupported formats.
    """
    # Handle path
    path = Path(path)
    if path.suffix != ".wdf":
        path = path.with_suffix(".wdf")

    # Check if file exists
    if path.exists() and not overwrite:
        raise FileExistsError(f"File {path} already exists. Set overwrite=True to overwrite.")

    # Currently only HDF5 is supported
    if format.lower() != "hdf5":
        raise NotImplementedError(f"Format {format} not supported. Only 'hdf5' is currently implemented.")

    h5py = require_h5py("WDF save")

    operation_history = frame.operation_history
    frame_state_json = _dump_wdf_json(encode_frame_state(frame), field=FRAME_STATE_JSON_ATTR)
    operation_history_json = _dump_wdf_json(operation_history, field=OPERATION_HISTORY_JSON_ATTR)
    dimension_coordinates = frame_dimension_coordinates(frame)
    channel_metadata = frame.channels.to_list()
    channel_ids_json = _dump_wdf_json(frame._channel_ids, field="channel_ids_json")
    channel_extra_json = [
        _dump_wdf_json(ch_meta.extra, field=f"channels/{index}/metadata_json") if ch_meta.extra else None
        for index, ch_meta in enumerate(channel_metadata)
    ]
    frame_metadata = dict(frame.metadata)
    frame_metadata_json = _dump_wdf_json(frame_metadata, field="meta/json") if frame_metadata else None

    target_dtype = np.dtype(dtype) if dtype is not None else None
    if target_dtype is not None:
        validate_frame_save_dtype(frame, target_dtype)

    # Compute data arrays (this triggers actual computation)
    logger.info("Computing data arrays for saving...")
    # Persist raw samples together with their calibration metadata. Persisting
    # ``frame.compute()`` here would apply calibration before save and apply it
    # a second time after load.
    computed_data = frame._data.compute()
    if target_dtype is not None:
        computed_data = computed_data.astype(target_dtype)

    # Create file
    logger.info(f"Creating HDF5 file at {path}...")
    with h5py.File(path, "w") as f:
        # Set file version
        f.attrs["version"] = WDF_FORMAT_VERSION

        # Store frame metadata
        f.attrs["sampling_rate"] = frame.sampling_rate
        f.attrs["label"] = frame.label or ""
        f.attrs["channel_ids_json"] = channel_ids_json
        f.attrs[FRAME_STATE_SCHEMA_ATTR] = FRAME_STATE_SCHEMA_VERSION
        f.attrs[FRAME_STATE_JSON_ATTR] = frame_state_json
        f.attrs[OPERATION_HISTORY_SCHEMA_ATTR] = OPERATION_HISTORY_SCHEMA_VERSION
        f.attrs[OPERATION_HISTORY_JSON_ATTR] = operation_history_json

        # WDF 0.3 stores the complete typed Frame tensor once. Per-channel
        # groups carry channel metadata only and remain independent of rank.
        if compress:
            f.create_dataset("data", data=computed_data, compression=compress)
        else:
            f.create_dataset("data", data=computed_data)

        # Create channels group
        channels_grp = f.create_group("channels")

        # Store each channel
        for i, (ch_meta, extra_json) in enumerate(zip(channel_metadata, channel_extra_json, strict=True)):
            ch_grp = channels_grp.create_group(f"{i}")

            # Store metadata
            ch_grp.attrs["label"] = ch_meta.label
            ch_grp.attrs["unit"] = ch_meta.unit
            ch_grp.attrs["ref"] = ch_meta.ref
            ch_grp.attrs["calibration_factor"] = ch_meta.calibration.factor
            ch_grp.attrs["source_time_offset"] = frame.source_time_offset[i]

            # Store extra metadata as JSON
            if extra_json is not None:
                ch_grp.attrs["metadata_json"] = extra_json

        if dimension_coordinates:
            coordinates_group = f.create_group("coordinates")
            for name, values in dimension_coordinates.items():
                coordinates_group.create_dataset(name, data=values)

        # Store frame metadata
        if frame_metadata_json is not None:
            meta_grp = f.create_group("meta")
            # Store metadata dict content as JSON
            meta_grp.attrs["json"] = frame_metadata_json

            # Also store individual metadata items as attributes for compatibility
            for k, v in frame_metadata.items():
                if isinstance(v, (str, int, float, bool, np.number)):
                    meta_grp.attrs[k] = v

    logger.info(f"Frame saved to {path}")


def load(path: str | Path, *, format: str = "hdf5", timeout: float = 10.0) -> BaseFrame[Any]:
    """Load a typed Frame object from a WDF (Wandas Data File) file or URL.

    Args:
        path: Path to the WDF file to load, or an HTTP/HTTPS URL pointing to
            a remote WDF file. URL input is streamed into a temporary file in
            bounded chunks and rejected when it exceeds
            `wandas.io.readers.MAX_URL_DOWNLOAD_BYTES`. Call
            `wandas.io.readers.download_url_to_temporary_file` directly with a
            larger `max_bytes` value when a trusted remote WDF exceeds the
            default limit.
        format: Format of the file. Currently only "hdf5" is supported.
        timeout: Timeout in seconds for HTTP/HTTPS URL downloads. Default is
            10.0 seconds. Has no effect for local file paths.

    Returns:
        A new built-in Frame with data, domain state, axes, and metadata restored.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        NotImplementedError: If format is not "hdf5".
        ValueError: If the file format is invalid or incompatible.

    Example:
        >>> import wandas as wd
        >>> cf = wd.load("audio_data.wdf")
        >>> cf = wd.load("https://example.com/audio_data.wdf")
    """
    # Ensure ChannelFrame is imported here to avoid circular imports
    from ..core.metadata import ChannelCalibration, ChannelMetadata
    from ..frames.channel import ChannelFrame

    if format.lower() != "hdf5":
        raise NotImplementedError(f"Format '{format}' is not supported")

    h5py = require_h5py("WDF load")

    with ExitStack() as downloads:
        h5_source: str | Path
        if isinstance(path, str) and path.lower().startswith(("http://", "https://")):
            logger.debug(f"Downloading WDF from URL: {path}")
            download = downloads.enter_context(
                download_url_to_temporary_file(
                    path,
                    timeout=timeout,
                    suffix=".wdf",
                    resource_name="WDF file",
                )
            )
            h5_source = download.path
        else:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            h5_source = path

        logger.debug(f"Loading WDF Frame from {h5_source!r}")

        with h5py.File(h5_source, "r") as f:
            # Check format version for compatibility
            version = _decode_hdf5_str(f.attrs.get("version", "unknown"))
            if version not in SUPPORTED_WDF_FORMAT_VERSIONS:
                raise ValueError(
                    "Unsupported WDF format version\n"
                    f"  Got: {version!r}\n"
                    f"  Supported: {sorted(SUPPORTED_WDF_FORMAT_VERSIONS)}\n"
                    "Use a compatible Wandas version or resave the file."
                )
            if version == WDF_FORMAT_VERSION and ("data" not in f or FRAME_STATE_JSON_ATTR not in f.attrs):
                raise ValueError(
                    "Incomplete WDF 0.3 typed Frame state\n"
                    "  Expected: /data and frame_state_json\n"
                    "  Got: one or both fields are missing\n"
                    "Resave the file with Wandas 0.6 or load the correctly versioned legacy file."
                )
            if version != WDF_FORMAT_VERSION:
                logger.info(f"Loading legacy WDF format: file={version}, current={WDF_FORMAT_VERSION}")

            # Get global attributes
            sampling_rate = float(f.attrs["sampling_rate"])
            frame_label = _decode_hdf5_str(f.attrs.get("label", ""))
            legacy_source_time_offset = f.attrs.get("source_time_offset", None)

            # Get frame metadata
            frame_metadata: dict[str, Any] = {}
            if "meta" in f:
                meta_json = f["meta"].attrs.get("json", "{}")
                if isinstance(meta_json, (bytes, np.bytes_)):
                    meta_json = _decode_hdf5_str(meta_json)
                parsed_metadata = _load_wdf_json(meta_json, field="meta/json")
                if not isinstance(parsed_metadata, dict):
                    raise ValueError("WDF meta/json must decode to a JSON object")
                frame_metadata = parsed_metadata
                source_file = f["meta"].attrs.get("source_file", None)
                if source_file is not None:
                    frame_metadata.setdefault("_source_file", _decode_hdf5_str(source_file))

            operation_history = _load_operation_history(f)

            # Load channel metadata independently from the typed tensor rank.
            all_channel_data: list[np.ndarray[Any, Any]] = []
            channel_metadata_list = []
            channel_source_time_offsets = []
            channel_ids: list[str] | None = None
            if "channel_ids_json" in f.attrs:
                parsed_channel_ids = _load_wdf_json(f.attrs["channel_ids_json"], field="channel_ids_json")
                if version == WDF_FORMAT_VERSION:
                    if not isinstance(parsed_channel_ids, list) or not all(
                        isinstance(channel_id, str) for channel_id in parsed_channel_ids
                    ):
                        raise ValueError(
                            "Invalid WDF 0.3 channel identifiers\n"
                            f"  Got: {parsed_channel_ids!r}\n"
                            "  Expected: a JSON array of strings\n"
                            "Resave the file with Wandas 0.6."
                        )
                    channel_ids = [str(channel_id) for channel_id in parsed_channel_ids]
                elif isinstance(parsed_channel_ids, list):
                    channel_ids = [str(channel_id) for channel_id in parsed_channel_ids]
            elif version == WDF_FORMAT_VERSION:
                raise ValueError(
                    "Incomplete WDF 0.3 channel identifiers\n"
                    "  Missing: channel_ids_json\n"
                    "Resave the file with Wandas 0.6; channel identity cannot be reconstructed safely."
                )

            if "channels" in f:
                channels_group = f["channels"]
                if not isinstance(channels_group, h5py.Group):
                    raise ValueError("Invalid WDF channels layout; expected /channels to be an HDF5 group")
                channel_keys = list(channels_group)
                if version == WDF_FORMAT_VERSION:
                    assert channel_ids is not None
                    channel_indices = _validate_v03_channel_keys(channel_keys, channel_ids)
                else:
                    # Sort legacy channel indices numerically.
                    channel_indices = sorted([int(key) for key in channel_keys])

                for idx in channel_indices:
                    ch_group = channels_group[f"{idx}"]
                    if not isinstance(ch_group, h5py.Group):
                        raise ValueError(f"Invalid WDF channel layout; expected /channels/{idx} to be an HDF5 group")
                    if version == WDF_FORMAT_VERSION:
                        _validate_v03_channel_attrs(ch_group, idx)

                    # WDF 0.1/0.2 stored one data dataset per channel.
                    if "data" in ch_group:
                        all_channel_data.append(ch_group["data"][()])

                    # Load channel metadata
                    label = _decode_hdf5_str(ch_group.attrs.get("label", f"Ch{idx}"))
                    unit = _decode_hdf5_str(ch_group.attrs.get("unit", ""))
                    ref = float(ch_group.attrs["ref"]) if "ref" in ch_group.attrs else None
                    factor = float(ch_group.attrs.get("calibration_factor", 1.0))
                    if "source_time_offset" in ch_group.attrs:
                        channel_source_time_offsets.append(float(ch_group.attrs["source_time_offset"]))

                    # Load additional metadata if present
                    ch_extra = {}
                    if "metadata_json" in ch_group.attrs:
                        ch_extra = _load_wdf_json(
                            ch_group.attrs["metadata_json"],
                            field=f"channels/{idx}/metadata_json",
                        )
                        if not isinstance(ch_extra, dict):
                            raise ValueError("WDF channel metadata JSON must decode to an object")

                    # Create ChannelMetadata object
                    calibration = (
                        ChannelCalibration(factor=factor, unit=unit)
                        if ref is None
                        else ChannelCalibration(factor=factor, unit=unit, ref=ref)
                    )
                    channel_metadata = ChannelMetadata(
                        label=label,
                        calibration=calibration,
                        extra=ch_extra,
                    )
                    channel_metadata_list.append(channel_metadata)
            elif version == WDF_FORMAT_VERSION:
                raise ValueError(
                    "Incomplete WDF 0.3 channel metadata\n"
                    "  Missing: /channels\n"
                    "Resave the file with Wandas 0.6; channel metadata cannot be reconstructed safely."
                )

            # WDF 0.3 stores one rank-preserving tensor. Older WDF versions
            # continue to reconstruct a ChannelFrame from per-channel arrays.
            if "data" in f:
                combined_data = f["data"][()]
            elif all_channel_data:
                combined_data = np.stack(all_channel_data, axis=0)
            else:
                raise ValueError("No channel data found in the file")

            if channel_source_time_offsets:
                source_time_offset = channel_source_time_offsets
            elif legacy_source_time_offset is not None:
                source_time_offset = legacy_source_time_offset
            else:
                source_time_offset = 0.0

            chunks = tuple([1] + [-1] * (combined_data.ndim - 1))
            dask_data = _da_from_array(combined_data, chunks=chunks)
            common: dict[str, Any] = {
                "sampling_rate": sampling_rate,
                "label": frame_label if frame_label else None,
                "metadata": frame_metadata,
                "channel_metadata": channel_metadata_list,
                "channel_ids": channel_ids,
                "source_time_offset": source_time_offset,
                "operation_history_prefix": operation_history,
            }

            if version == WDF_FORMAT_VERSION:
                schema = _load_schema_version(
                    f.attrs.get(FRAME_STATE_SCHEMA_ATTR, 0),
                    field=FRAME_STATE_SCHEMA_ATTR,
                )
                if schema != FRAME_STATE_SCHEMA_VERSION:
                    raise ValueError(
                        "Unsupported WDF Frame state schema\n"
                        f"  Got: {schema}\n"
                        f"  Supported: {FRAME_STATE_SCHEMA_VERSION}\n"
                        "Use a compatible Wandas version or resave the file."
                    )
                frame_state = _load_wdf_json(f.attrs[FRAME_STATE_JSON_ATTR], field=FRAME_STATE_JSON_ATTR)
                if not isinstance(frame_state, dict):
                    raise ValueError("WDF Frame state JSON must decode to an object")
                frame = decode_frame_state(frame_state, data=dask_data, common=common)
            else:
                frame = ChannelFrame(data=dask_data, **common)

            if version == WDF_FORMAT_VERSION:
                coordinates: dict[str, np.ndarray[Any, Any]] = {}
                if "coordinates" in f:
                    coordinates_group = f["coordinates"]
                    if not isinstance(coordinates_group, h5py.Group):
                        raise ValueError("Invalid WDF coordinates layout; expected /coordinates to be an HDF5 group")
                    for name, dataset in coordinates_group.items():
                        if not isinstance(dataset, h5py.Dataset):
                            raise ValueError(
                                f"Invalid WDF coordinate layout; expected /coordinates/{name} to be a dataset"
                            )
                        coordinates[name] = dataset[()]
                restore_frame_coordinates(frame, coordinates)

            logger.debug(f"{type(frame).__name__} loaded from {path}: shape={frame.shape}")
            return frame
