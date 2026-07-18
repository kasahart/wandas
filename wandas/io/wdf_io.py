"""
WDF (Wandas Data File) I/O module for saving and loading ChannelFrame objects.

This module provides functionality to save and load ChannelFrame objects in the
WDF (Wandas Data File) format, which is based on HDF5. The format preserves
all metadata including sampling rate, channel labels, units, and frame metadata.
"""

import json
import logging
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..frames.channel import ChannelFrame

# Import BaseFrame from core module
from wandas.utils.dask_helpers import da_from_array as _da_from_array
from wandas.utils.optional_imports import require_h5py

from ..core.base_frame import BaseFrame
from .readers import download_url_to_temporary_file

logger = logging.getLogger(__name__)

# Constants for version management
WDF_FORMAT_VERSION = "0.3"
OPERATION_HISTORY_SCHEMA_VERSION = 1
OPERATION_HISTORY_SCHEMA_ATTR = "operation_history_schema"
OPERATION_HISTORY_JSON_ATTR = "operation_history_json"
LEGACY_OPERATION_SUMMARIES_SCHEMA_VERSION = 1
LEGACY_OPERATION_SUMMARIES_SCHEMA_ATTR = "operation_summaries_schema"
LEGACY_OPERATION_SUMMARIES_JSON_ATTR = "operation_summaries_json"
LEGACY_OPERATION_HISTORY_GROUP = "operation_history"


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
        dtype: Optional data type conversion before saving (e.g. 'float32').
            Frames carrying reader sample-scale provenance accept only safe
            widening conversions.

    Raises:
        FileExistsError: If the file exists and overwrite=False.
        NotImplementedError: For unsupported formats.
        ValueError: If dtype would invalidate reader sample-scale provenance.
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

    requested_dtype = None if dtype is None else np.dtype(dtype)
    provenance_channels = [channel.label for channel in frame.channels if channel.calibration.sample_scale is not None]

    operation_history = frame.operation_history

    # Compute data arrays (this triggers actual computation). Reader-backed Dask
    # arrays can carry an estimated dtype, so provenance safety must use the
    # materialized source dtype that will actually be written.
    logger.info("Computing data arrays for saving...")
    computed_data = frame._data.compute()
    representation_preserving_dtype = (
        requested_dtype is not None
        and requested_dtype.kind in "iuf"
        and np.can_cast(computed_data.dtype, requested_dtype, casting="safe")
    )
    if requested_dtype is not None and provenance_channels and not representation_preserving_dtype:
        raise ValueError(
            "WDF dtype conversion would invalidate calibration sample scale\n"
            f"  Source dtype: {computed_data.dtype}\n"
            f"  Requested dtype: {requested_dtype}\n"
            f"  Provenance-bearing channels: {provenance_channels!r}\n"
            "Save without dtype, use a safe widening dtype, or process the data explicitly before saving."
        )

    if requested_dtype is not None:
        computed_data = computed_data.astype(requested_dtype)

    # Create file
    logger.info(f"Creating HDF5 file at {path}...")
    with h5py.File(path, "w") as f:
        # Set file version
        f.attrs["version"] = WDF_FORMAT_VERSION

        # Store frame metadata
        f.attrs["sampling_rate"] = frame.sampling_rate
        f.attrs["label"] = frame.label or ""
        f.attrs["frame_type"] = type(frame).__name__
        f.attrs["channel_ids_json"] = json.dumps(frame._channel_ids)
        f.attrs[OPERATION_HISTORY_SCHEMA_ATTR] = OPERATION_HISTORY_SCHEMA_VERSION
        f.attrs[OPERATION_HISTORY_JSON_ATTR] = json.dumps(operation_history, allow_nan=False)

        # Create channels group
        channels_grp = f.create_group("channels")

        # Store each channel
        for i, (channel_data, ch_meta) in enumerate(zip(computed_data, frame.channels, strict=True)):
            ch_grp = channels_grp.create_group(f"{i}")

            # Store channel data
            if compress:
                ch_grp.create_dataset("data", data=channel_data, compression=compress)
            else:
                ch_grp.create_dataset("data", data=channel_data)

            # Store metadata
            ch_grp.attrs["label"] = ch_meta.label
            ch_grp.attrs["unit"] = ch_meta.unit
            ch_grp.attrs["ref"] = ch_meta.ref
            ch_grp.attrs["calibration_factor"] = ch_meta.calibration.factor
            if ch_meta.calibration.sample_scale is not None:
                ch_grp.attrs["calibration_sample_scale"] = ch_meta.calibration.sample_scale
            ch_grp.attrs["source_time_offset"] = frame.source_time_offset[i]

            # Store extra metadata as JSON
            if ch_meta.extra:
                ch_grp.attrs["metadata_json"] = json.dumps(ch_meta.extra)

        # Store frame metadata
        if frame.metadata:
            meta_grp = f.create_group("meta")
            # Store metadata dict content as JSON
            meta_grp.attrs["json"] = json.dumps(dict(frame.metadata))

            # Also store individual metadata items as attributes for compatibility
            for k, v in frame.metadata.items():
                if isinstance(v, (str, int, float, bool, np.number)):
                    meta_grp.attrs[k] = v

    logger.info(f"Frame saved to {path}")


def load(path: str | Path, *, format: str = "hdf5", timeout: float = 10.0) -> "ChannelFrame":
    """Load a ChannelFrame object from a WDF (Wandas Data File) file or URL.

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
        A new ChannelFrame object with data and metadata loaded from the file.

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

        logger.debug(f"Loading ChannelFrame from {h5_source!r}")

        with h5py.File(h5_source, "r") as f:
            # Check format version for compatibility
            version = f.attrs.get("version", "unknown")
            if version != WDF_FORMAT_VERSION:
                logger.warning(f"File format version mismatch: file={version}, current={WDF_FORMAT_VERSION}")

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
                parsed_metadata = json.loads(meta_json)
                if not isinstance(parsed_metadata, dict):
                    raise ValueError("WDF meta/json must decode to a JSON object")
                frame_metadata = parsed_metadata
                source_file = f["meta"].attrs.get("source_file", None)
                if source_file is not None:
                    frame_metadata.setdefault("_source_file", _decode_hdf5_str(source_file))

            operation_history = _load_operation_history(f)

            # Load channel data and metadata
            all_channel_data = []
            channel_metadata_list = []
            channel_source_time_offsets = []
            channel_ids = None
            if "channel_ids_json" in f.attrs:
                parsed_channel_ids = json.loads(_decode_hdf5_str(f.attrs["channel_ids_json"]))
                if isinstance(parsed_channel_ids, list):
                    channel_ids = [str(channel_id) for channel_id in parsed_channel_ids]

            if "channels" in f:
                channels_group = f["channels"]
                # Sort channel indices numerically
                channel_indices = sorted([int(key) for key in channels_group])

                for idx in channel_indices:
                    ch_group = channels_group[f"{idx}"]

                    # Load channel data
                    channel_data = ch_group["data"][()]

                    # Append to combined array
                    all_channel_data.append(channel_data)

                    # Load channel metadata
                    label = _decode_hdf5_str(ch_group.attrs.get("label", f"Ch{idx}"))
                    unit = _decode_hdf5_str(ch_group.attrs.get("unit", ""))
                    ref = float(ch_group.attrs["ref"]) if "ref" in ch_group.attrs else None
                    factor = float(ch_group.attrs.get("calibration_factor", 1.0))
                    sample_scale = (
                        _decode_hdf5_str(ch_group.attrs["calibration_sample_scale"])
                        if "calibration_sample_scale" in ch_group.attrs
                        else None
                    )
                    if "source_time_offset" in ch_group.attrs:
                        channel_source_time_offsets.append(float(ch_group.attrs["source_time_offset"]))

                    # Load additional metadata if present
                    ch_extra = {}
                    if "metadata_json" in ch_group.attrs:
                        ch_extra = json.loads(ch_group.attrs["metadata_json"])

                    # Create ChannelMetadata object
                    calibration = (
                        ChannelCalibration(factor=factor, unit=unit, sample_scale=sample_scale)
                        if ref is None
                        else ChannelCalibration(
                            factor=factor,
                            unit=unit,
                            ref=ref,
                            sample_scale=sample_scale,
                        )
                    )
                    channel_metadata = ChannelMetadata(
                        label=label,
                        calibration=calibration,
                        extra=ch_extra,
                    )
                    channel_metadata_list.append(channel_metadata)

            # Stack channel data into a single array
            if all_channel_data:
                combined_data = np.stack(all_channel_data, axis=0)
            else:
                raise ValueError("No channel data found in the file")

            if channel_source_time_offsets:
                source_time_offset = channel_source_time_offsets
            elif legacy_source_time_offset is not None:
                source_time_offset = legacy_source_time_offset
            else:
                source_time_offset = 0.0

            # Create a new ChannelFrame
            # Use channel-wise chunking: 1 for channel axis and -1 for samples
            dask_data = _da_from_array(combined_data, chunks=(1, -1))

            cf = ChannelFrame(
                data=dask_data,
                sampling_rate=sampling_rate,
                label=frame_label if frame_label else None,
                metadata=frame_metadata,
                channel_metadata=channel_metadata_list,
                channel_ids=channel_ids,
                source_time_offset=source_time_offset,
                operation_history_prefix=operation_history,
            )

            logger.debug(f"ChannelFrame loaded from {path}: {len(cf)} channels, {cf.n_samples} samples")
            return cf
