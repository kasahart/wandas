"""Typed WDF (Wandas Data File) persistence based on HDF5.

WDF 0.3 preserves built-in Frame type, semantic dimensions, domain constructor
state, represented coordinates, channel metadata, frame metadata, and display
history. Runtime lineage and Dask graphs are deliberately not reconstructed.
"""

import json
import logging
from contextlib import ExitStack
from pathlib import Path
from typing import Any

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
)

logger = logging.getLogger(__name__)

# WDF version and nested JSON-schema versions advance independently. The root
# version selects the complete HDF5 layout; nested versions identify the exact JSON
# contracts stored inside that layout.
WDF_FORMAT_VERSION = "0.3"
FRAME_STATE_SCHEMA_VERSION = 1
FRAME_STATE_SCHEMA_ATTR = "frame_state_schema"
FRAME_STATE_JSON_ATTR = "frame_state_json"
OPERATION_HISTORY_SCHEMA_VERSION = 1
OPERATION_HISTORY_SCHEMA_ATTR = "operation_history_schema"
OPERATION_HISTORY_JSON_ATTR = "operation_history_json"
_WDF_CHANNEL_ATTRS = frozenset({"label", "unit", "ref", "calibration_factor", "source_time_offset", "metadata_json"})
_WDF_ROOT_ATTRS = frozenset(
    {
        "version",
        "sampling_rate",
        "label_json",
        "channel_ids_json",
        FRAME_STATE_SCHEMA_ATTR,
        FRAME_STATE_JSON_ATTR,
        OPERATION_HISTORY_SCHEMA_ATTR,
        OPERATION_HISTORY_JSON_ATTR,
    }
)


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
            _decode_hdf5_str(value, field=field),
            parse_constant=_reject_nonfinite_json_number,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Invalid strict JSON in WDF\n"
            f"  Field: {field}\n"
            f"  Cause: {exc}\n"
            "Resave the file with a compatible Wandas version."
        ) from exc


def _validate_channel_keys(keys: list[str], channel_ids: list[str]) -> list[int]:
    """Validate exact contiguous channel groups for the current WDF schema."""
    expected = [str(index) for index in range(len(channel_ids))]
    if set(keys) != set(expected):
        raise ValueError(
            "Invalid WDF channel groups\n"
            f"  Got: {sorted(keys)}\n"
            f"  Expected: {expected}\n"
            "The channel metadata layout is incomplete; resave the file with Wandas 0.6."
        )
    return list(range(len(expected)))


def _validate_channel_attrs(channel: Any, index: int) -> None:
    """Require the exact channel attributes in the current WDF schema."""
    missing = _WDF_CHANNEL_ATTRS - set(channel.attrs)
    unexpected = set(channel.attrs) - _WDF_CHANNEL_ATTRS
    if missing or unexpected:
        raise ValueError(
            "Invalid WDF channel metadata schema\n"
            f"  Channel: {index}\n"
            f"  Missing attributes: {sorted(missing)}\n"
            f"  Unexpected attributes: {sorted(unexpected)}\n"
            "Resave the file with Wandas 0.6; channel metadata cannot be reconstructed safely."
        )


def _load_schema_version(value: object, *, field: str) -> int:
    """Decode one strict integer schema attribute with field-aware context."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            "Invalid WDF schema version attribute\n"
            f"  Field: {field}\n"
            f"  Got: {value!r}\n"
            "Resave the file with a compatible Wandas version using an integer schema version."
        )
    return int(value)


def _decode_hdf5_str(value: object, *, field: str = "value") -> str:
    """Decode an HDF5 attribute value to a Python string.

    HDF5 may return ``bytes``, ``numpy.bytes_``, or plain ``str``.
    """
    if isinstance(value, (bytes, np.bytes_)):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Invalid UTF-8 WDF string in {field}") from exc
    if isinstance(value, str):
        return value
    raise ValueError(f"Invalid WDF string field {field}; expected text, got {type(value).__name__}")


def _reject_nonfinite_json_number(value: str) -> None:
    """Reject non-finite constants while decoding strict WDF JSON."""
    raise ValueError(f"WDF fields must use strict JSON; non-finite number found: {value}")


def _load_finite_number(value: object, *, field: str) -> float:
    """Decode one finite real number from an external WDF field."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"Invalid WDF numeric field {field}; got {value!r}")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"Invalid WDF numeric field {field}; expected a finite value")
    return normalized


def _require_root_attrs(h5_file: Any) -> None:
    """Require the exact root attribute set for the current WDF schema."""
    missing = _WDF_ROOT_ATTRS - set(h5_file.attrs)
    unexpected = set(h5_file.attrs) - _WDF_ROOT_ATTRS
    if missing or unexpected:
        raise ValueError(
            "Invalid WDF root attribute schema\n"
            f"  Missing: {sorted(missing)}\n"
            f"  Unexpected: {sorted(unexpected)}\n"
            "Resave the file with the current Wandas WDF writer."
        )


def _require_root_members(h5_file: Any) -> None:
    """Require current WDF objects and reject unrelated layouts."""
    required = {"data", "channels", "meta"}
    allowed = required | {"coordinates"}
    missing = required - set(h5_file)
    unexpected = set(h5_file) - allowed
    if missing or unexpected:
        raise ValueError(
            "Invalid WDF root object schema\n"
            f"  Missing: {sorted(missing)}\n"
            f"  Unexpected: {sorted(unexpected)}\n"
            "Resave the file with the current Wandas WDF writer."
        )


def _load_operation_history(h5_file: Any) -> list[dict[str, Any]]:
    """Load and structurally validate display history from an open WDF file."""
    schema = _load_schema_version(
        h5_file.attrs[OPERATION_HISTORY_SCHEMA_ATTR],
        field=OPERATION_HISTORY_SCHEMA_ATTR,
    )
    if schema != OPERATION_HISTORY_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported WDF operation history schema\n"
            f"  Got: {schema}\n"
            f"  Supported: {OPERATION_HISTORY_SCHEMA_VERSION}\n"
            "Use a compatible Wandas version or resave the file."
        )
    parsed = _load_wdf_json(h5_file.attrs[OPERATION_HISTORY_JSON_ATTR], field=OPERATION_HISTORY_JSON_ATTR)
    expected_fields = {"operation", "version", "params"}
    if not isinstance(parsed, list) or not all(
        isinstance(record, dict)
        and set(record) == expected_fields
        and isinstance(record["operation"], str)
        and bool(record["operation"].strip())
        and type(record["version"]) is int
        and isinstance(record["params"], dict)
        for record in parsed
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
) -> None:
    """Save one exact built-in Frame using the current typed WDF schema.

    The writer persists a single raw tensor plus the constructor state and semantic
    dimensions required to reconstruct its concrete Frame type. Channel calibration
    remains metadata beside the raw tensor so loading does not apply it twice.

    Args:
        frame: Exact supported built-in Frame to persist.
        path: Destination path. The ``.wdf`` suffix is appended when absent.
        format: Storage format. Only ``"hdf5"`` is currently supported.
        compress: HDF5 dataset compression filter, or ``None`` for no compression.
        overwrite: Replace an existing artifact when true.

    Raises:
        FileExistsError: If the destination exists and ``overwrite`` is false.
        NotImplementedError: If ``format`` is not ``"hdf5"``.
        TypeError: If ``frame`` is not an exact registered built-in Frame type.
        ValueError: If Frame state is invalid or not strict-JSON serializable.
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
        _dump_wdf_json(ch_meta.extra, field=f"channels/{index}/metadata_json")
        for index, ch_meta in enumerate(channel_metadata)
    ]
    frame_metadata_json = _dump_wdf_json(dict(frame.metadata), field="meta/json")
    frame_label_json = _dump_wdf_json(frame.label, field="label_json")

    logger.info("Computing data arrays for saving...")
    # Persist raw samples together with their calibration metadata. Persisting
    # ``frame.compute()`` here would apply calibration before save and apply it
    # a second time after load.
    computed_data = frame._data.compute()

    # Create file
    logger.info(f"Creating HDF5 file at {path}...")
    with h5py.File(path, "w") as f:
        f.attrs["version"] = WDF_FORMAT_VERSION

        # Small schema records remain root attributes so the tensor and its type
        # contract can be validated before any Frame constructor is called.
        f.attrs["sampling_rate"] = frame.sampling_rate
        f.attrs["label_json"] = frame_label_json
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

        channels_grp = f.create_group("channels")

        # Channel groups are indexed by tensor channel position. Stable channel IDs
        # live at the root and are validated against this contiguous group layout.
        for i, (ch_meta, extra_json) in enumerate(zip(channel_metadata, channel_extra_json, strict=True)):
            ch_grp = channels_grp.create_group(f"{i}")

            ch_grp.attrs["label"] = ch_meta.label
            ch_grp.attrs["unit"] = ch_meta.unit
            ch_grp.attrs["ref"] = ch_meta.ref
            ch_grp.attrs["calibration_factor"] = ch_meta.calibration.factor
            ch_grp.attrs["source_time_offset"] = frame.source_time_offset[i]

            ch_grp.attrs["metadata_json"] = extra_json

        if dimension_coordinates:
            coordinates_group = f.create_group("coordinates")
            for name, values in dimension_coordinates.items():
                coordinates_group.create_dataset(name, data=values)

        meta_grp = f.create_group("meta")
        meta_grp.attrs["json"] = frame_metadata_json

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
        A new exact built-in Frame with data, domain state, axes, and metadata
        restored. Numerical operations remain Dask-lazy after loading.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        NotImplementedError: If format is not "hdf5".
        ValueError: If the file format is invalid or incompatible.

    Example:
        >>> import wandas as wd
        >>> cf = wd.load("audio_data.wdf")
        >>> cf = wd.load("https://example.com/audio_data.wdf")
    """
    from ..core.metadata import ChannelCalibration, ChannelMetadata

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
            if "version" not in f.attrs:
                raise ValueError("Unsupported WDF format version\n  Got: missing\n  Supported: '0.3'")
            version = _decode_hdf5_str(f.attrs["version"], field="version")
            if version != WDF_FORMAT_VERSION:
                raise ValueError(
                    "Unsupported WDF format version\n"
                    f"  Got: {version!r}\n"
                    f"  Supported: {WDF_FORMAT_VERSION!r}\n"
                    "Use a compatible Wandas version or resave the file."
                )
            _require_root_attrs(f)
            _require_root_members(f)

            sampling_rate = _load_finite_number(f.attrs["sampling_rate"], field="sampling_rate")
            frame_label = _load_wdf_json(f.attrs["label_json"], field="label_json")
            if frame_label is not None and not isinstance(frame_label, str):
                raise ValueError("WDF label_json must decode to a string or null")

            meta_group = f["meta"]
            if not isinstance(meta_group, h5py.Group) or set(meta_group.attrs) != {"json"}:
                raise ValueError("Invalid WDF metadata layout; expected /meta with only a json attribute")
            parsed_metadata = _load_wdf_json(meta_group.attrs["json"], field="meta/json")
            if not isinstance(parsed_metadata, dict):
                raise ValueError("WDF meta/json must decode to a JSON object")
            frame_metadata = parsed_metadata

            operation_history = _load_operation_history(f)

            parsed_channel_ids = _load_wdf_json(f.attrs["channel_ids_json"], field="channel_ids_json")
            if not isinstance(parsed_channel_ids, list) or not all(
                isinstance(channel_id, str) for channel_id in parsed_channel_ids
            ):
                raise ValueError(
                    "Invalid WDF channel identifiers\n"
                    f"  Got: {parsed_channel_ids!r}\n"
                    "  Expected: a JSON array of strings\n"
                    "Resave the file with the current Wandas WDF writer."
                )
            channel_ids = parsed_channel_ids

            if "channels" not in f:
                raise ValueError("Incomplete WDF channel metadata\n  Missing: /channels")
            channels_group = f["channels"]
            if not isinstance(channels_group, h5py.Group):
                raise ValueError("Invalid WDF channels layout; expected /channels to be an HDF5 group")

            channel_metadata_list: list[ChannelMetadata] = []
            channel_source_time_offsets: list[float] = []
            for idx in _validate_channel_keys(list(channels_group), channel_ids):
                ch_group = channels_group[str(idx)]
                if not isinstance(ch_group, h5py.Group):
                    raise ValueError(f"Invalid WDF channel layout; expected /channels/{idx} to be an HDF5 group")
                _validate_channel_attrs(ch_group, idx)

                label = _decode_hdf5_str(ch_group.attrs["label"], field=f"channels/{idx}/label")
                unit = _decode_hdf5_str(ch_group.attrs["unit"], field=f"channels/{idx}/unit")
                ref = _load_finite_number(ch_group.attrs["ref"], field=f"channels/{idx}/ref")
                factor = _load_finite_number(
                    ch_group.attrs["calibration_factor"],
                    field=f"channels/{idx}/calibration_factor",
                )
                channel_source_time_offsets.append(
                    _load_finite_number(
                        ch_group.attrs["source_time_offset"],
                        field=f"channels/{idx}/source_time_offset",
                    )
                )

                parsed_extra = _load_wdf_json(
                    ch_group.attrs["metadata_json"],
                    field=f"channels/{idx}/metadata_json",
                )
                if not isinstance(parsed_extra, dict):
                    raise ValueError("WDF channel metadata JSON must decode to an object")
                channel_metadata_list.append(
                    ChannelMetadata(
                        label=label,
                        calibration=ChannelCalibration(factor=factor, unit=unit, ref=ref),
                        extra=parsed_extra,
                    )
                )

            if "data" not in f or not isinstance(f["data"], h5py.Dataset):
                raise ValueError("Incomplete WDF tensor\n  Expected: /data dataset")

            # Read while the HDF5 file is open: the returned Frame must not retain a
            # live h5py dataset owned by this context manager.
            combined_data = f["data"][()]

            # Channel-sized chunks preserve the Frame convention that channel-wise
            # operations can remain independent in the reconstructed Dask graph.
            chunks = tuple([1] + [-1] * (combined_data.ndim - 1))
            dask_data = _da_from_array(combined_data, chunks=chunks)
            common: dict[str, Any] = {
                "sampling_rate": sampling_rate,
                "label": frame_label,
                "metadata": frame_metadata,
                "channel_metadata": channel_metadata_list,
                "channel_ids": channel_ids,
                "source_time_offset": channel_source_time_offsets,
                "operation_history_prefix": operation_history,
            }

            schema = _load_schema_version(
                f.attrs[FRAME_STATE_SCHEMA_ATTR],
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

            # Typed construction establishes xarray dimensions first. Optional
            # represented-axis coordinates are restored only after those dimensions
            # have been validated against the stored semantic names.
            coordinates: dict[str, np.ndarray[Any, Any]] = {}
            if "coordinates" in f:
                coordinates_group = f["coordinates"]
                if not isinstance(coordinates_group, h5py.Group):
                    raise ValueError("Invalid WDF coordinates layout; expected /coordinates to be an HDF5 group")
                for name, dataset in coordinates_group.items():
                    if not isinstance(dataset, h5py.Dataset):
                        raise ValueError(f"Invalid WDF coordinate layout; expected /coordinates/{name} to be a dataset")
                    coordinates[name] = dataset[()]
            restore_frame_coordinates(frame, coordinates)

            logger.debug(f"{type(frame).__name__} loaded from {path}: shape={frame.shape}")
            return frame
