"""
WDF (Wandas Data File) I/O module for saving and loading ChannelFrame objects.

This module provides functionality to save and load ChannelFrame objects in the
WDF (Wandas Data File) format, which is based on HDF5. The format preserves
all metadata including sampling rate, channel labels, units, and frame metadata.
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dask.array as da
import h5py
import numpy as np

if TYPE_CHECKING:
    from ..frames.channel import ChannelFrame

# CoreモジュールからBaseFrameをインポート
from ..core.base_frame import BaseFrame

da_from_array = da.from_array  # type: ignore [unused-ignore]

logger = logging.getLogger(__name__)

# バージョン管理のための定数
WDF_FORMAT_VERSION = "0.1"


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
        raise FileExistsError(
            f"File {path} already exists. Set overwrite=True to overwrite."
        )

    # Currently only HDF5 is supported
    if format.lower() != "hdf5":
        raise NotImplementedError(
            f"Format {format} not supported. Only 'hdf5' is currently implemented."
        )

    # Compute data arrays (this triggers actual computation)
    logger.info("Computing data arrays for saving...")
    computed_data = frame.compute()
    if dtype is not None:
        computed_data = computed_data.astype(dtype)

    # Create file
    logger.info(f"Creating HDF5 file at {path}...")
    with h5py.File(path, "w") as f:
        # Set file version
        f.attrs["version"] = WDF_FORMAT_VERSION

        # Store frame metadata
        f.attrs["sampling_rate"] = frame.sampling_rate
        f.attrs["label"] = frame.label or ""
        f.attrs["frame_type"] = type(frame).__name__

        # Create channels group
        channels_grp = f.create_group("channels")

        # Store each channel
        for i, (channel_data, ch_meta) in enumerate(
            zip(computed_data, frame._channel_metadata)
        ):
            ch_grp = channels_grp.create_group(f"{i}")

            # Store channel data
            if compress:
                ch_grp.create_dataset("data", data=channel_data, compression=compress)
            else:
                ch_grp.create_dataset("data", data=channel_data)

            # Store metadata
            ch_grp.attrs["label"] = ch_meta.label
            ch_grp.attrs["unit"] = ch_meta.unit

            # Store extra metadata as JSON
            if ch_meta.extra:
                ch_grp.attrs["metadata_json"] = json.dumps(ch_meta.extra)

        # Store operation history
        if frame.operation_history:
            op_grp = f.create_group("operation_history")
            for i, op in enumerate(frame.operation_history):
                op_sub_grp = op_grp.create_group(f"operation_{i}")
                for k, v in op.items():
                    # Store simple attributes directly
                    if isinstance(v, str | int | float | bool | np.number):
                        op_sub_grp.attrs[k] = v
                    else:
                        # For complex types, serialize to JSON
                        try:
                            op_sub_grp.attrs[k] = json.dumps(v)
                        except (TypeError, OverflowError) as e:
                            logger.warning(
                                f"Could not serialize operation key '{k}': {e}"
                            )
                            op_sub_grp.attrs[k] = str(v)

        # Store frame metadata
        if frame.metadata:
            meta_grp = f.create_group("meta")
            # Store metadata as JSON
            meta_grp.attrs["json"] = json.dumps(frame.metadata)

            # Also store individual metadata items as attributes for compatibility
            for k, v in frame.metadata.items():
                if isinstance(v, str | int | float | bool | np.number):
                    meta_grp.attrs[k] = v

    logger.info(f"Frame saved to {path}")


def load(path: str | Path, *, format: str = "hdf5") -> "ChannelFrame":
    """Load a ChannelFrame object from a WDF (Wandas Data File) file.

    Args:
        path: Path to the WDF file to load.
        format: Format of the file. Currently only "hdf5" is supported.

    Returns:
        A new ChannelFrame object with data and metadata loaded from the file.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        NotImplementedError: If format is not "hdf5".
        ValueError: If the file format is invalid or incompatible.

    Example:
        >>> cf = ChannelFrame.load("audio_data.wdf")
    """
    # Ensure ChannelFrame is imported here to avoid circular imports
    from ..core.metadata import ChannelMetadata
    from ..frames.channel import ChannelFrame

    if format != "hdf5":
        raise NotImplementedError(f"Format '{format}' is not supported")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    logger.debug(f"Loading ChannelFrame from {path}")

    with h5py.File(path, "r") as f:
        # Check format version for compatibility
        version = f.attrs.get("version", "unknown")
        if version != WDF_FORMAT_VERSION:
            logger.warning(
                f"File format version mismatch: file={version}, current={WDF_FORMAT_VERSION}"  # noqa: E501
            )

        # Get global attributes
        sampling_rate = float(f.attrs["sampling_rate"])
        frame_label = f.attrs.get("label", "")

        # Get frame metadata
        frame_metadata = {}
        if "meta" in f:
            meta_json = f["meta"].attrs.get("json", "{}")
            frame_metadata = json.loads(meta_json)

        # Load operation history
        operation_history = []
        if "operation_history" in f:
            op_grp = f["operation_history"]
            # Sort operation indices numerically
            op_indices = sorted([int(key.split("_")[1]) for key in op_grp.keys()])

            for idx in op_indices:
                op_sub_grp = op_grp[f"operation_{idx}"]
                op_dict = {}
                for attr_name in op_sub_grp.attrs:
                    attr_value = op_sub_grp.attrs[attr_name]
                    # Try to deserialize JSON, fallback to string
                    try:
                        op_dict[attr_name] = json.loads(attr_value)
                    except (json.JSONDecodeError, TypeError):
                        op_dict[attr_name] = attr_value
                operation_history.append(op_dict)

        # Load channel data and metadata
        all_channel_data = []
        channel_metadata_list = []

        if "channels" in f:
            channels_group = f["channels"]
            # Sort channel indices numerically
            channel_indices = sorted([int(key) for key in channels_group.keys()])

            for idx in channel_indices:
                ch_group = channels_group[f"{idx}"]

                # Load channel data
                channel_data = ch_group["data"][()]

                # Append to combined array
                all_channel_data.append(channel_data)

                # Load channel metadata
                label = ch_group.attrs.get("label", f"Ch{idx}")
                unit = ch_group.attrs.get("unit", "")

                # Load additional metadata if present
                ch_extra = {}
                if "metadata_json" in ch_group.attrs:
                    ch_extra = json.loads(ch_group.attrs["metadata_json"])

                # Create ChannelMetadata object
                channel_metadata = ChannelMetadata(
                    label=label, unit=unit, extra=ch_extra
                )
                channel_metadata_list.append(channel_metadata)

        # Stack channel data into a single array
        if all_channel_data:
            combined_data = np.stack(all_channel_data, axis=0)
        else:
            raise ValueError("No channel data found in the file")

        # Create a new ChannelFrame
        dask_data = da_from_array(combined_data)

        cf = ChannelFrame(
            data=dask_data,
            sampling_rate=sampling_rate,
            label=frame_label if frame_label else None,
            metadata=frame_metadata,
            operation_history=operation_history,
            channel_metadata=channel_metadata_list,
        )

        logger.debug(
            f"ChannelFrame loaded from {path}: {len(cf)} channels, {cf.n_samples} samples"  # noqa: E501
        )
        return cf
