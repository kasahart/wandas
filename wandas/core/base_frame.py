import copy
import json
import logging
import numbers
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from re import Pattern
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast, overload

import numpy as np
import numpy.typing as npt
import xarray as xr
from dask.array.core import Array as DaArray

from wandas.utils import validate_sampling_rate
from wandas.utils.optional_imports import require_dependency, require_pandas
from wandas.utils.types import NDArrayComplex, NDArrayReal

from .channel_metadata import ChannelMetadataIndexer
from .metadata import ChannelMetadata

# IPython display types for visualize_graph return type
# Define as type alias under TYPE_CHECKING; use Any at runtime
if TYPE_CHECKING:
    from typing import TypeAlias

    import pandas as pd
    from IPython.display import Image as IPythonImage
    from matplotlib.axes import Axes

    from wandas.processing.base import AudioOperation, LineageNode

    VisualizeReturnType: TypeAlias = IPythonImage | None
else:
    # Use Any at runtime to avoid type checker errors
    VisualizeReturnType = Any

logger = logging.getLogger(__name__)

T = TypeVar("T", NDArrayComplex, NDArrayReal)
S = TypeVar("S", bound="BaseFrame[Any]")
S_Out = TypeVar("S_Out", bound="BaseFrame[Any]")
QueryType = str | Pattern[str] | Callable[["ChannelMetadata"], bool] | dict[str, Any]


class _LineageOperationName:
    """Operation view used when callers provide an explicit lineage name."""

    def __init__(self, operation: Any, name: str, params: Mapping[str, Any] | None = None) -> None:
        self._operation = operation
        self.name = name
        self._params = params

    @property
    def params(self) -> Any:
        return self._params if self._params is not None else getattr(self._operation, "params", {})

    def to_params(self) -> Mapping[str, Any]:
        if self._params is not None:
            return self._params
        if hasattr(self._operation, "to_params"):
            return cast(Mapping[str, Any], self._operation.to_params())
        return cast(Mapping[str, Any], getattr(self._operation, "params", {}))


def _mutable_config_value(value: Any) -> Any:
    """Convert operation config values to plain JSON-friendly containers for history."""
    if isinstance(value, DaArray):
        return {
            "type": "dask.array",
            "shape": [_mutable_config_value(item) for item in value.shape],
            "dtype": str(value.dtype),
            "chunks": [[_mutable_config_value(item) for item in chunk] for chunk in value.chunks],
        }
    if isinstance(value, np.ndarray):
        return _mutable_config_value(value.tolist())
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        numeric = float(value)
        if np.isfinite(numeric):
            return numeric
        if np.isnan(numeric):
            return {"type": "float", "value": "nan"}
        if numeric > 0:
            return {"type": "float", "value": "inf"}
        return {"type": "float", "value": "-inf"}
    if isinstance(value, numbers.Complex):
        return {
            "type": "complex",
            "real": _mutable_config_value(value.real),
            "imag": _mutable_config_value(value.imag),
        }
    if isinstance(value, Mapping):
        return {_mutable_config_key(key): _mutable_config_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_mutable_config_value(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted((_mutable_config_value(item) for item in value), key=_stable_json_sort_key)
    if value is None or isinstance(value, str | bool):
        return value
    return str(value)


def _mutable_config_key(key: Any) -> str:
    """Convert operation param keys to stable JSON object keys."""
    if isinstance(key, str):
        return key
    value = _mutable_config_value(key)
    if isinstance(value, str):
        return value
    return _stable_json_sort_key(value)


def _stable_json_sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class BaseFrame(ABC, Generic[T]):
    """
    Abstract base class for all signal frame types.

    This class provides the common interface and functionality for all frame types
    used in signal processing. It implements basic operations like indexing, iteration,
    and data manipulation that are shared across all frame types.

    Parameters
    ----------
    data : DaArray
        The signal data to process. Must be a dask array.
    sampling_rate : float
        The sampling rate of the signal in Hz.
    label : str, optional
        A label for the frame. If not provided, defaults to "unnamed_frame".
    metadata : dict, optional
        Additional metadata for the frame.
    lineage : LineageNode, optional
        Computation lineage for this frame.
    channel_metadata : list[ChannelMetadata | dict], optional
        Metadata for each channel in the frame. Can be ChannelMetadata objects
        or dicts that will be converted to ChannelMetadata objects.
    previous : BaseFrame, optional
        The frame that this frame was derived from.

    Attributes
    ----------
    sampling_rate : float
        The sampling rate of the signal in Hz.
    label : str
        The label of the frame.
    metadata : dict
        Additional metadata for the frame.
    lineage : LineageNode | None
        Runtime-only computation lineage. This is set during construction and
        propagated through ``_create_new_instance``.
    operation_history : list[dict]
        Flat read-only compatibility view derived from ``lineage``.
    """

    _CHANNEL_DIM: ClassVar[str] = "channel"
    # Fallback only for neutral-dim and legacy frames. Target frames should
    # prefer the xarray "channel" dimension when it is declared.
    _channel_axis: ClassVar[int | None] = -2
    _xarray_dim_suffix: ClassVar[tuple[str, ...]] = ()
    _xr: xr.DataArray
    _previous: "BaseFrame[Any] | None"

    def __init__(
        self,
        data: DaArray,
        sampling_rate: float,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
        channel_metadata: Sequence[ChannelMetadata | dict[str, Any]] | None = None,
        channel_ids: list[str] | None = None,
        previous: "BaseFrame[Any] | None" = None,
        source_time_offset: float | Sequence[float] | NDArrayReal = 0.0,
        lineage: "LineageNode | None" = None,
    ):
        normalized_data = self._normalize_data(data)
        frame_label = label or "unnamed_frame"
        channel_count = self._channel_size_from_xarray_dims(normalized_data)
        if channel_count is None:
            channel_count = self._channel_count_from_data(normalized_data)

        normalized_channel_metadata = self._normalize_channel_metadata_for_count(channel_metadata, channel_count)
        self._pending_channel_metadata = normalized_channel_metadata
        self._pending_channel_ids = (
            self._validate_channel_ids(channel_ids, channel_count)
            if channel_ids is not None
            else self._default_channel_ids(channel_count)
        )

        self._xr = self._build_xarray(normalized_data, name=frame_label)
        self.label = label
        self.sampling_rate = sampling_rate
        self.metadata = metadata
        self._lineage = lineage
        self._set_channel_metadata(normalized_channel_metadata, self._pending_channel_ids)
        self.source_time_offset = source_time_offset
        del self._pending_channel_metadata
        del self._pending_channel_ids
        self._previous = previous

        try:
            # Display information for newer dask versions
            logger.debug(f"Dask graph layers: {list(self._data.dask.layers.keys())}")
            logger.debug(f"Dask graph dependencies: {len(self._data.dask.dependencies)}")
        except Exception as e:
            logger.debug(f"Dask graph visualization details unavailable: {e}")

    @property
    def _data(self) -> DaArray:
        """Compatibility alias for the Dask array stored in ``_xr``."""
        data = self._xr.data
        if not isinstance(data, DaArray):
            raise TypeError(f"Internal xarray data is not a Dask array: {type(data).__name__}")
        return data

    def _replace_data(self, data: DaArray) -> None:
        """Replace the internal xarray data container without touching frame state."""
        old_channel_metadata = self.channels.to_list()
        old_channel_ids = self._channel_ids
        old_source_time_offset = self.source_time_offset
        normalized = self._normalize_data(data)
        attrs = copy.deepcopy(self._xr.attrs)
        self._xr = self._build_xarray(normalized, name=self.label)
        self._xr.attrs = attrs
        if len(old_channel_metadata) == self._n_channels and len(old_channel_ids) == self._n_channels:
            self._set_channel_metadata(old_channel_metadata, old_channel_ids)
            self.source_time_offset = old_source_time_offset

    def _normalize_data(self, data: DaArray) -> DaArray:
        """Normalize Dask data shape and chunks using Wandas channel-wise policy."""
        try:
            normalized = data.reshape((1, -1)) if data.ndim == 1 else data
            if normalized.ndim >= 2:
                chunks = tuple([1] + [-1] * (normalized.ndim - 1))
            else:
                chunks = tuple([-1] * normalized.ndim)
            return normalized.rechunk(chunks)
        except Exception as e:
            logger.warning(f"Rechunk failed: {e!r}. Falling back to chunks=-1.")
            return data.rechunk(chunks=-1)

    def _build_xarray(self, data: DaArray, *, name: str) -> xr.DataArray:
        """Build the internal xarray container for frame data, dims, and coords."""
        return xr.DataArray(
            data,
            dims=self._xarray_dims(data),
            coords=self._xarray_coords(data),
            name=name,
        )

    def _xarray_dims(self, data: DaArray) -> tuple[str, ...]:
        """Return semantic xarray dims only for exact suffix-shaped data."""
        suffix = self._xarray_dim_suffix
        if suffix and data.ndim == len(suffix):
            return suffix
        return tuple(f"dim_{i}" for i in range(data.ndim))

    def _xarray_coords(self, data: DaArray) -> dict[str, Any]:
        """Return conservative coordinates for declared xarray dimensions."""
        channel_size = self._channel_size_from_xarray_dims(data)
        if channel_size is None:
            return {}

        metadata = getattr(self, "_pending_channel_metadata", None)
        channel_ids = getattr(self, "_pending_channel_ids", None)
        if metadata is None or channel_ids is None or len(metadata) != channel_size:
            return {}
        return {
            self._CHANNEL_DIM: (self._CHANNEL_DIM, channel_ids),
            "channel_label": (self._CHANNEL_DIM, [ch.label for ch in metadata]),
            "channel_unit": (self._CHANNEL_DIM, [ch.unit for ch in metadata]),
            "channel_ref": (self._CHANNEL_DIM, [ch.ref for ch in metadata]),
        }

    def _channel_size_from_xarray_dims(self, data: DaArray) -> int | None:
        """Return the channel size implied by xarray dims, if present."""
        dims = self._xarray_dims(data)
        if self._CHANNEL_DIM not in dims:
            return None
        return int(data.shape[dims.index(self._CHANNEL_DIM)])

    def _channel_count_from_data(self, data: DaArray) -> int:
        """Return the frame channel count from the declared channel axis."""
        if self._channel_axis is None:
            return 1
        return int(data.shape[self._channel_axis])

    @property
    def _n_channels(self) -> int:
        """Returns the number of channels from the xarray channel dimension when available."""
        if self._CHANNEL_DIM in self._xr.sizes:
            return int(self._xr.sizes[self._CHANNEL_DIM])
        return self._channel_count_from_data(self._data)

    @staticmethod
    def _default_channel_ids(n_channels: int) -> list[str]:
        return [f"c{i}" for i in range(n_channels)]

    @staticmethod
    def _validate_channel_ids(channel_ids: Sequence[Any], n_channels: int) -> list[str]:
        ids = [str(channel_id) for channel_id in channel_ids]
        if len(ids) != n_channels:
            raise ValueError(
                f"Channel id length must match number of channels\n  Channel ids: {len(ids)}\n  Channels: {n_channels}"
            )
        if len(set(ids)) != len(ids):
            raise ValueError(f"Channel ids must be unique: {ids}")
        return ids

    def _normalize_channel_metadata_for_count(
        self,
        channel_metadata: Sequence[ChannelMetadata | dict[str, Any]] | None,
        channel_count: int,
    ) -> list[ChannelMetadata]:
        def _to_channel_metadata(ch: ChannelMetadata | dict[str, Any], index: int) -> ChannelMetadata:
            to_metadata = getattr(ch, "to_metadata", None)
            if callable(to_metadata):
                return copy.deepcopy(cast(Any, to_metadata)())
            if isinstance(ch, ChannelMetadata):
                return copy.deepcopy(ch)
            if isinstance(ch, dict):
                try:
                    return ChannelMetadata(**ch)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"Invalid channel_metadata at index {index}\n"
                        f"  Got: {ch}\n"
                        f"  Error: {e}\n"
                        f"Ensure all dict keys match ChannelMetadata fields "
                        f"(label, unit, ref, extra) and have correct types."
                    ) from e
            raise TypeError(
                f"Invalid type in channel_metadata at index {index}\n"
                f"  Got: {type(ch).__name__} ({ch!r})\n"
                f"  Expected: ChannelMetadata or dict\n"
                f"Use ChannelMetadata objects or dicts with valid fields."
            )

        if channel_metadata is None:
            result = [ChannelMetadata(label=f"ch{i}", unit="", extra={}) for i in range(channel_count)]
        else:
            result = [_to_channel_metadata(ch, i) for i, ch in enumerate(channel_metadata)]
        if len(result) > channel_count:
            raise ValueError(
                "Channel metadata length must not exceed number of channels\n"
                f"  Metadata entries: {len(result)}\n"
                f"  Channels: {channel_count}"
            )
        if len(result) < channel_count:
            result.extend(ChannelMetadata(label=f"ch{i}", unit="", extra={}) for i in range(len(result), channel_count))
        return result

    def _refresh_xarray_channel_coord(self) -> None:
        """Refresh auxiliary channel metadata coordinates after compatibility mutations."""
        self._set_channel_metadata(self.channels.to_list(), self._channel_ids)

    @property
    def _channel_ids(self) -> list[str]:
        if self._CHANNEL_DIM in self._xr.coords:
            return [str(value) for value in self._xr.coords[self._CHANNEL_DIM].values.tolist()]
        return [str(value) for value in self._xr.attrs.get("channel_ids", [])]

    def _channel_id_at(self, index: int) -> str:
        if self._CHANNEL_DIM in self._xr.coords:
            return str(self._xr.coords[self._CHANNEL_DIM].values[index])
        return str(self._xr.attrs["channel_ids"][index])

    def _get_channel_coord_value(self, coord_name: str, index: int) -> Any:
        if coord_name in self._xr.coords:
            return self._xr.coords[coord_name].values[index]
        return self._xr.attrs[coord_name][index]

    def _channel_ids_for_selection(self, indices: Sequence[int]) -> list[str]:
        selected_ids: list[str] = []
        used_ids: set[str] = set()
        for index in indices:
            channel_id = self._channel_id_at(index)
            if channel_id in used_ids:
                channel_id = self._next_channel_id([*self._channel_ids, *selected_ids])
            selected_ids.append(channel_id)
            used_ids.add(channel_id)
        return selected_ids

    @property
    def channels(self) -> ChannelMetadataIndexer:
        """Property to access channel metadata."""
        return ChannelMetadataIndexer(self)

    @property
    def _channel_metadata(self) -> list[ChannelMetadata]:
        """Compatibility list-like view over xarray-backed channel metadata."""
        return cast(list[ChannelMetadata], self.channels)

    @_channel_metadata.setter
    def _channel_metadata(self, value: Sequence[ChannelMetadata | dict[str, Any]]) -> None:
        self._set_channel_metadata(value)

    def _set_channel_coord_value(self, coord_name: str, index: int, value: Any) -> None:
        if coord_name in self._xr.coords:
            values = self._xr.coords[coord_name].values.tolist()
            values[index] = value
            self._xr = self._xr.assign_coords({coord_name: (self._CHANNEL_DIM, values)})
            return
        values = list(self._xr.attrs.get(coord_name, []))
        values[index] = value
        self._xr.attrs[coord_name] = values

    def _set_channel_metadata(
        self,
        channel_metadata: Sequence[ChannelMetadata | dict[str, Any]],
        channel_ids: Sequence[Any] | None = None,
    ) -> None:
        normalized = self._normalize_channel_metadata_for_count(channel_metadata, self._n_channels)
        ids = (
            self._validate_channel_ids(channel_ids, self._n_channels) if channel_ids is not None else self._channel_ids
        )
        if not ids:
            ids = self._default_channel_ids(self._n_channels)
        labels = [ch.label for ch in normalized]
        units = [ch.unit for ch in normalized]
        refs = [ch.ref for ch in normalized]
        channel_extra = {channel_id: copy.deepcopy(ch.extra) for channel_id, ch in zip(ids, normalized, strict=True)}
        self._xr.attrs["channel_extra"] = channel_extra
        if self._CHANNEL_DIM in self._xr.dims:
            self._xr = self._xr.assign_coords(
                {
                    self._CHANNEL_DIM: (self._CHANNEL_DIM, ids),
                    "channel_label": (self._CHANNEL_DIM, labels),
                    "channel_unit": (self._CHANNEL_DIM, units),
                    "channel_ref": (self._CHANNEL_DIM, refs),
                }
            )
            for name in ("channel_ids", "channel_label", "channel_unit", "channel_ref"):
                self._xr.attrs.pop(name, None)
            return
        self._xr.attrs.update(
            {
                "channel_ids": ids,
                "channel_label": labels,
                "channel_unit": units,
                "channel_ref": refs,
            }
        )

    def _next_channel_id(self, existing_ids: Sequence[str] | None = None) -> str:
        ids = set(existing_ids if existing_ids is not None else self._channel_ids)
        index = 0
        while f"c{index}" in ids:
            index += 1
        return f"c{index}"

    @property
    def n_channels(self) -> int:
        """Returns the number of channels."""
        return self._n_channels

    @property
    def previous(self) -> "BaseFrame[Any] | None":
        """
        Returns the previous frame.
        """
        return self._previous

    @property
    def sampling_rate(self) -> float:
        """Return the frame sampling rate from xarray attrs."""
        return float(self._xr.attrs["sampling_rate"])

    @sampling_rate.setter
    def sampling_rate(self, value: float) -> None:
        validate_sampling_rate(value)
        self._xr.attrs["sampling_rate"] = float(value)

    @property
    def label(self) -> str:
        """Return the frame label from xarray attrs."""
        value = self._xr.attrs.get("label", self._xr.name)
        if value is None or value == "":
            return "unnamed_frame"
        return str(value)

    @label.setter
    def label(self, value: str | None) -> None:
        if value is not None and not isinstance(value, str):
            raise TypeError("Label must be a string or None")
        label = value or "unnamed_frame"
        self._xr.attrs["label"] = label
        self._xr.name = label

    @property
    def metadata(self) -> dict[str, Any]:
        """Return mutable frame metadata stored in xarray attrs."""
        value = self._xr.attrs.get("metadata")
        if value is None:
            value = {}
            self._xr.attrs["metadata"] = value
        if not isinstance(value, dict):
            raise TypeError(f"Internal metadata attrs must be a dictionary, got {type(value).__name__}")
        return value

    @metadata.setter
    def metadata(self, value: dict[str, Any] | None) -> None:
        if value is None:
            self._xr.attrs["metadata"] = {}
            return
        if not isinstance(value, dict):
            raise TypeError("Metadata must be a dictionary")
        self._xr.attrs["metadata"] = copy.deepcopy(value)

    @property
    def operation_history(self) -> list[dict[str, Any]]:
        """Return a flat read-only view derived from ``lineage``."""
        return self._lineage_to_history(self.lineage)

    @property
    def lineage(self) -> "LineageNode | None":
        """Return runtime computation lineage for this frame."""
        return self._lineage

    @property
    def operation_graph(self) -> dict[str, Any] | None:
        """Return nested serializable computation lineage."""
        return self._lineage_to_graph(self.lineage)

    @property
    def operation_summaries(self) -> list[dict[str, Any]]:
        """Return lightweight display summaries derived from ``lineage``."""
        return self._lineage_to_summaries(self.lineage)

    @staticmethod
    def _operation_name(operation: Any) -> str:
        from wandas.processing.base import BinaryOperation

        if isinstance(operation, BinaryOperation):
            return cast(str, getattr(operation, "symbol"))
        return cast(str, getattr(operation, "name", type(operation).__name__))

    @staticmethod
    def _operation_params(operation: Any) -> dict[str, Any]:
        if hasattr(operation, "to_params"):
            params = operation.to_params()
        else:
            params = getattr(operation, "params", {})
        return cast(dict[str, Any], _mutable_config_value(params))

    @classmethod
    def _lineage_to_history(cls, lineage: "LineageNode | None") -> list[dict[str, Any]]:
        if lineage is None:
            return []
        records: list[dict[str, Any]] = []
        for input_lineage in lineage.inputs:
            records.extend(cls._lineage_to_history(input_lineage))
        params = cls._operation_params(lineage.operation)
        record: dict[str, Any] = {"operation": cls._operation_name(lineage.operation)}
        if params:
            record["params"] = params
        return [*records, record]

    @classmethod
    def _lineage_to_graph(cls, lineage: "LineageNode | None") -> dict[str, Any] | None:
        if lineage is None:
            return None
        graph = {
            "operation": cls._operation_name(lineage.operation),
            "params": cls._operation_params(lineage.operation),
            "inputs": [
                input_graph
                for input_lineage in lineage.inputs
                if (input_graph := cls._lineage_to_graph(input_lineage)) is not None
            ],
        }
        from wandas.processing.base import FrameMethodOperation

        if isinstance(lineage.operation, FrameMethodOperation):
            graph["kind"] = "method"
        return graph

    @classmethod
    def _operation_summary(cls, operation: Any) -> dict[str, Any]:
        if hasattr(operation, "to_summary"):
            return cast(dict[str, Any], operation.to_summary())
        from wandas.processing.base import _summary_value

        if hasattr(operation, "to_params"):
            params = operation.to_params()
        else:
            params = getattr(operation, "params", {})
        return {
            "operation": cls._operation_name(operation),
            "params": _summary_value(params),
        }

    @classmethod
    def _lineage_to_summaries(cls, lineage: "LineageNode | None") -> list[dict[str, Any]]:
        if lineage is None:
            return []
        records: list[dict[str, Any]] = []
        for input_lineage in lineage.inputs:
            records.extend(cls._lineage_to_summaries(input_lineage))
        records.append(cls._operation_summary(lineage.operation))
        return records

    def _lineage_with_operation(self, operation: Any, *inputs: "LineageNode | None") -> "LineageNode":
        from wandas.processing.base import LineageNode

        lineage_inputs = tuple(input_lineage for input_lineage in inputs if input_lineage is not None)
        return LineageNode(operation=operation, inputs=lineage_inputs)

    def _lineage_with_method(self, method: str, params: Mapping[str, Any]) -> "LineageNode":
        from wandas.processing.base import FrameMethodOperation

        return self._lineage_with_operation(FrameMethodOperation(method, params), self.lineage)

    def _lineage_with_unsupported_indexing(self, indexing: str) -> "LineageNode":
        return self._lineage_with_method("__getitem__", {"indexing": indexing})

    @property
    def operations(self) -> tuple["AudioOperation[Any, Any]", ...]:
        """Return Wandas operation instances found in the lazy Dask graph."""
        from wandas.lineage import extract_operations

        return extract_operations(self._data)

    @property
    def source_time_offset(self) -> NDArrayReal:
        """Return each channel's offset from local time axis to source time."""
        if "source_time_offset" in self._xr.coords:
            value = self._xr.coords["source_time_offset"].values
        else:
            value = self._xr.attrs.get("source_time_offset", 0.0)
        return self._normalize_source_time_offset(value, self.n_channels)

    @source_time_offset.setter
    def source_time_offset(self, value: float | Sequence[float] | NDArrayReal) -> None:
        offsets = self._normalize_source_time_offset(value, self.n_channels)
        if self._CHANNEL_DIM in self._xr.dims:
            self._xr = self._xr.assign_coords({"source_time_offset": (self._CHANNEL_DIM, offsets)})
            self._xr.attrs.pop("source_time_offset", None)
            return
        self._xr.attrs["source_time_offset"] = offsets

    @staticmethod
    def _normalize_source_time_offset(
        value: object,
        n_channels: int,
    ) -> NDArrayReal:
        try:
            offsets = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("source_time_offset must be a finite numeric value") from exc
        if offsets.ndim == 0:
            offsets = np.full(n_channels, float(offsets), dtype=float)
        elif offsets.ndim != 1:
            raise ValueError("source_time_offset must be a scalar or a 1D array")
        elif len(offsets) != n_channels:
            raise ValueError(
                "source_time_offset length must match number of channels\n"
                f"  Offsets: {len(offsets)}\n"
                f"  Channels: {n_channels}"
            )
        if not np.all(np.isfinite(offsets)):
            raise ValueError("source_time_offset must be finite")
        return offsets.astype(float, copy=True)

    def get_channel(
        self: S,
        channel_idx: int | list[int] | tuple[int, ...] | npt.NDArray[np.int_] | npt.NDArray[np.bool_] | None = None,
        query: QueryType | None = None,
        validate_query_keys: bool = True,
    ) -> S:
        """
        Get channel(s) by index.

        Parameters
        ----------
        channel_idx : int or sequence of int
            Single channel index or sequence of channel indices.
            Supports negative indices (e.g., -1 for the last channel).
        query : str, re.Pattern, callable, or dict, optional
            If a query is provided, use it to derive indices and ignore the positional channel_idx argument.
            Query to select channels based on metadata. Supported types:
            - str: exact label match
            - re.Pattern: regex search against label
            - callable(ChannelMetadata) -> bool: predicate on channel metadata
            - dict: attribute equality on ChannelMetadata (values may be re.Pattern)
        validate_query_keys : bool, default True
            If True (default), dict queries that contain unknown keys (neither
            model fields nor any channel `extra` keys) will raise `KeyError`.
            Set to False to disable this strict validation and allow callers
            to attempt matches without pre-validation.
        Returns
        -------
        S
            New instance containing the selected channel(s).

        Examples
        --------
        >>> frame.get_channel(0)  # Single channel
        >>> frame.get_channel([0, 2, 3])  # Multiple channels
        >>> frame.get_channel((-1, -2))  # Last two channels
        >>> frame.get_channel(np.array([1, 2]))  # NumPy array of indices
        """

        def _indices_from_query(q: Any) -> list[int]:
            if isinstance(q, str):
                return [i for i, ch in enumerate(self.channels) if ch.label == q]

            # re.Pattern compatibility
            if hasattr(q, "search") and callable(q.search):
                return [i for i, ch in enumerate(self.channels) if q.search(ch.label)]

            if callable(q):
                return [i for i, ch in enumerate(self.channels) if bool(q(ch))]

            if isinstance(q, dict):
                # Validate dict keys against known model fields + extra keys.
                if validate_query_keys:
                    model_keys = set(ChannelMetadata._MODEL_FIELDS)

                    extra_keys: set[str] = set()
                    for ch in self.channels:
                        if isinstance(ch.extra, dict):
                            extra_keys.update(ch.extra.keys())

                    unknown_keys = [k for k in q if k not in model_keys | extra_keys]
                    if unknown_keys:
                        names_str = ", ".join(map(str, unknown_keys))
                        raise KeyError("Unknown channel metadata key(s): " + names_str)

                return [i for i, ch in enumerate(self.channels) if ch.matches_query(q)]

            raise TypeError(f"Unsupported query type: {type(q).__name__}")

        if query is not None:
            indices = _indices_from_query(query)
            if not indices:
                raise KeyError(f"No channels match query: {query!r}")
            channel_idx_list = indices
            if isinstance(query, str):
                lineage_params: dict[str, Any] = {"query": query, "validate_query_keys": validate_query_keys}
            else:
                lineage_params = {
                    "channel_idx": channel_idx_list,
                    "query_kind": type(query).__name__,
                }
        else:
            if channel_idx is None:
                raise TypeError("Either 'channel_idx' or 'query' must be provided.")

            channel_idx_list = [channel_idx] if isinstance(channel_idx, int) else list(channel_idx)
            lineage_params = {
                "channel_idx": channel_idx_list[0] if len(channel_idx_list) == 1 else channel_idx_list,
            }

        new_data = self._data[channel_idx_list]
        new_channel_metadata = [self.channels[i].to_metadata() for i in channel_idx_list]
        new_channel_ids = self._channel_ids_for_selection(channel_idx_list)

        return self._create_new_instance(
            data=new_data,
            channel_metadata=new_channel_metadata,
            channel_ids=new_channel_ids,
            source_time_offset=self.source_time_offset[channel_idx_list],
            lineage=self._lineage_with_method("get_channel", lineage_params),
        )

    def __len__(self) -> int:
        """
        Returns the number of channels.
        """
        return len(self._channel_metadata)

    def __iter__(self: S) -> Iterator[S]:
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(
        self: S,
        key: int
        | str
        | slice
        | list[int]
        | list[str]
        | tuple[
            int | str | slice | list[int] | list[str] | npt.NDArray[np.int_] | npt.NDArray[np.bool_],
            ...,
        ]
        | npt.NDArray[np.int_]
        | npt.NDArray[np.bool_],
    ) -> S:
        """
        Get channel(s) by index, label, or advanced indexing.

        This method supports multiple indexing patterns similar to NumPy and pandas:

        - Single channel by index: `frame[0]`
        - Single channel by label: `frame["ch0"]`
        - Slice of channels: `frame[0:3]`
        - Multiple channels by indices: `frame[[0, 2, 5]]`
        - Multiple channels by labels: `frame[["ch0", "ch2"]]`
        - NumPy integer array: `frame[np.array([0, 2])]`
        - Boolean mask: `frame[mask]` where mask is a boolean array
        - Multidimensional indexing: `frame[0, 100:200]` (channel + time)

        Parameters
        ----------
        key : int, str, slice, list, tuple, or ndarray
            - int: Single channel index (supports negative indexing)
            - str: Single channel label
            - slice: Range of channels
            - list[int]: Multiple channel indices
            - list[str]: Multiple channel labels
            - tuple: Multidimensional indexing (channel_key, time_key, ...)
            - ndarray[int]: NumPy array of channel indices
            - ndarray[bool]: Boolean mask for channel selection

        Returns
        -------
        S
            New instance containing the selected channel(s).

        Raises
        ------
        ValueError
            If the key length is invalid for the shape or if boolean mask
            length doesn't match number of channels.
        IndexError
            If the channel index is out of range.
        TypeError
            If the key type is invalid or list contains mixed types.
        KeyError
            If a channel label is not found.

        Examples
        --------
        >>> # Single channel selection
        >>> frame[0]  # First channel
        >>> frame["acc_x"]  # By label
        >>> frame[-1]  # Last channel
        >>>
        >>> # Multiple channel selection
        >>> frame[[0, 2, 5]]  # Multiple indices
        >>> frame[["acc_x", "acc_z"]]  # Multiple labels
        >>> frame[0:3]  # Slice
        >>>
        >>> # NumPy array indexing
        >>> frame[np.array([0, 2, 4])]  # Integer array
        >>> mask = np.array([True, False, True])
        >>> frame[mask]  # Boolean mask
        >>>
        >>> # Time slicing (multidimensional)
        >>> frame[0, 100:200]  # Channel 0, samples 100-200
        >>> frame[[0, 1], ::2]  # Channels 0-1, every 2nd sample
        """

        # Single index (int)
        if isinstance(key, numbers.Integral):
            # Ensure we pass a plain Python int to satisfy the type checker
            return self.get_channel(int(key))

        # Single label (str)
        if isinstance(key, str):
            index = self.label2index(key)
            return self.get_channel(index)

        # Phase 2: NumPy array support (bool mask and int array)
        if isinstance(key, np.ndarray):
            if key.dtype in (bool, np.bool_):
                # Boolean mask
                if len(key) != self.n_channels:
                    raise ValueError(
                        f"Boolean mask length {len(key)} does not match number of channels {self.n_channels}"
                    )
                indices = np.where(cast(npt.NDArray[np.bool_], key))[0]
                result = self.get_channel(indices)
                return result._create_new_instance(
                    data=result._data,
                    channel_metadata=result.channels.to_list(),
                    channel_ids=result._channel_ids,
                    source_time_offset=result.source_time_offset,
                    lineage=self._lineage_with_unsupported_indexing("array"),
                )
            if np.issubdtype(key.dtype, np.integer):
                # Integer array
                result = self.get_channel(cast(npt.NDArray[np.int_], key))
                return result._create_new_instance(
                    data=result._data,
                    channel_metadata=result.channels.to_list(),
                    channel_ids=result._channel_ids,
                    source_time_offset=result.source_time_offset,
                    lineage=self._lineage_with_unsupported_indexing("array"),
                )
            raise TypeError(f"NumPy array must be of integer or boolean type, got {key.dtype}")

        # Phase 1: List support (int or str)
        if isinstance(key, list):
            if len(key) == 0:
                raise ValueError("Cannot index with an empty list")

            # Check if all elements are strings
            if all(isinstance(k, str) for k in key):
                # Multiple labels - type narrowing for type checker
                str_list = cast(list[str], key)
                indices_from_labels = [self.label2index(label) for label in str_list]
                new_data = self._data[indices_from_labels]
                new_channel_metadata = [self.channels[i].to_metadata() for i in indices_from_labels]
                new_channel_ids = [self._channel_ids[i] for i in indices_from_labels]
                return self._create_new_instance(
                    data=new_data,
                    channel_metadata=new_channel_metadata,
                    channel_ids=new_channel_ids,
                    source_time_offset=self.source_time_offset[indices_from_labels],
                    lineage=self._lineage_with_method(
                        "__getitem__",
                        {"indexing": "label_list", "labels": tuple(str_list)},
                    ),
                )

            # Check if all elements are integers
            if all(isinstance(k, int | np.integer) for k in key):
                # Multiple indices - convert to list[int] for type safety
                int_list = [int(k) for k in key]
                result = self.get_channel(int_list)
                return result._create_new_instance(
                    data=result._data,
                    channel_metadata=result.channels.to_list(),
                    channel_ids=result._channel_ids,
                    source_time_offset=result.source_time_offset,
                    lineage=self._lineage_with_unsupported_indexing("integer_list"),
                )

            raise TypeError(f"List must contain all str or all int, got mixed types: {[type(k).__name__ for k in key]}")

        # Tuple: multidimensional indexing
        if isinstance(key, tuple):
            return self._handle_multidim_indexing(key)

        # Slice
        if isinstance(key, slice):
            new_data = self._data[key]
            indices = list(range(self.n_channels))[key]
            new_channel_metadata = [self.channels[i].to_metadata() for i in indices]
            new_channel_ids = [self._channel_ids[i] for i in indices]
            return self._create_new_instance(
                data=new_data,
                channel_metadata=new_channel_metadata,
                channel_ids=new_channel_ids,
                source_time_offset=self.source_time_offset[indices],
                lineage=self._lineage_with_method(
                    "__getitem__",
                    {
                        "indexing": "channel_slice",
                        "start": key.start,
                        "stop": key.stop,
                        "step": key.step,
                    },
                ),
            )

        raise TypeError(f"Invalid key type: {type(key).__name__}. Expected int, str, slice, list, tuple, or ndarray.")

    def _handle_multidim_indexing(
        self: S,
        key: tuple[
            int | str | slice | list[int] | list[str] | npt.NDArray[np.int_] | npt.NDArray[np.bool_],
            ...,
        ],
    ) -> S:
        """
        Handle multidimensional indexing (channel + time axis).

        Parameters
        ----------
        key : tuple
            Tuple of indices where the first element selects channels
            and subsequent elements select along other dimensions (e.g., time).

        Returns
        -------
        S
            New instance with selected channels and time range.

        Raises
        ------
        ValueError
            If the key length exceeds the data dimensions.
        """
        if len(key) > self._data.ndim:
            raise ValueError(f"Invalid key length: {len(key)} for shape {self.shape}")

        # First element: channel selection
        channel_key = key[0]
        time_keys = key[1:] if len(key) > 1 else ()

        # Select channels first (recursively call __getitem__)
        if isinstance(channel_key, list | np.ndarray | int | str | slice):
            selected = self[channel_key]
        else:
            raise TypeError(f"Invalid channel key type in tuple: {type(channel_key).__name__}")

        # Apply time indexing if present
        if time_keys:
            new_data = selected._data[(slice(None),) + time_keys]  # noqa: RUF005
            source_time_offset = selected.source_time_offset
            time_slice_context = selected._source_time_slice_context(time_keys)
            if time_slice_context is not None:
                time_axis_key, time_axis_size, time_step = time_slice_context
                if not isinstance(time_axis_key, slice):
                    raise ValueError("Only continuous slicing on the time axis is supported for source time offsets.")
                if time_axis_key.step not in (None, 1):
                    raise ValueError("Stepped slicing on the time axis is not supported for source time offsets.")
                start, _, _ = time_axis_key.indices(time_axis_size)
                source_time_offset = source_time_offset + start * time_step
            return selected._create_new_instance(
                data=new_data,
                channel_metadata=selected.channels.to_list(),
                channel_ids=selected._channel_ids,
                source_time_offset=source_time_offset,
                lineage=selected._lineage_with_method(
                    "__getitem__",
                    {"indexing": "multidimensional"},
                ),
            )

        return selected

    def _source_time_slice_context(self, keys: tuple[Any, ...]) -> tuple[Any, int, float] | None:
        """Return the sliced time-axis key, axis size, and seconds per index."""
        dims = self._xr.dims
        if "time" not in dims:
            return None
        time_dim_index = dims.index("time")
        key_index = time_dim_index - 1
        if key_index < 0 or key_index >= len(keys):
            return None
        hop_length = getattr(self, "hop_length", 1)
        return keys[key_index], self._data.shape[time_dim_index], float(hop_length) / self.sampling_rate

    def label2index(self, label: str) -> int:
        """
        Get the index from a channel label.

        Parameters
        ----------
        label : str
            Channel label.

        Returns
        -------
        int
            Corresponding index.

        Raises
        ------
        KeyError
            If the channel label is not found.
        """
        for idx, ch in enumerate(self.channels):
            if ch.label == label:
                return idx
        raise KeyError(f"Channel label '{label}' not found.")

    @property
    def shape(self) -> tuple[int, ...]:
        _shape: tuple[int, ...] = self._data.shape
        if _shape[0] == 1:
            return _shape[1:]
        return _shape

    @property
    def data(self) -> T:
        """
        Returns the computed data.
        Calculation is executed the first time this is accessed.
        """
        data = self.compute()
        if self.n_channels == 1:
            return cast(T, data.squeeze(axis=0))
        return data

    @property
    def labels(self) -> list[str]:
        """Get a list of all channel labels."""
        return [ch.label for ch in self.channels]

    def compute(self) -> T:
        """
        Compute and return the data.
        This method materializes lazily computed data into a concrete NumPy array.

        Returns
        -------
        NDArrayReal
            The computed data.

        Raises
        ------
        ValueError
            If the computed result is not a NumPy array.
        """
        logger.debug("COMPUTING DASK ARRAY - This will trigger file reading and all processing")
        result = self._data.compute()

        if not isinstance(result, np.ndarray):
            raise ValueError(f"Computed result is not a np.ndarray: {type(result)}")

        logger.debug(f"Computation complete, result shape: {result.shape}")
        return cast(T, result)

    def to_xarray(self) -> xr.DataArray:
        """Return a public xarray view of this frame without changing Wandas ownership."""
        exported = self._xr.copy(deep=False)
        for coord_name in (self._CHANNEL_DIM, "channel_label", "channel_unit", "channel_ref", "source_time_offset"):
            if coord_name in exported.coords:
                coord = exported.coords[coord_name]
                exported = exported.assign_coords({coord_name: (coord.dims, coord.values.copy())})
        exported.name = self.label
        exported.attrs = copy.deepcopy(self._xr.attrs)
        exported.attrs.pop("operation_history", None)
        exported.attrs.pop("operation_graph", None)
        exported.attrs["wandas_frame_type"] = type(self).__name__
        return exported

    @property
    def xr(self) -> xr.DataArray:
        """Return a public xarray view of this frame."""
        return self.to_xarray()

    @abstractmethod
    def plot(self, plot_type: str = "default", ax: "Axes | None" = None, **kwargs: Any) -> "Axes | Iterator[Axes]":
        """Plot the data"""

    def persist(self: S) -> S:
        """Persist the data in memory"""
        persisted_data = self._data.persist()
        return self._create_new_instance(data=persisted_data)

    def _get_additional_init_kwargs(self) -> dict[str, Any]:
        """Return additional keyword arguments for ``_create_new_instance``.

        Subclasses that require extra constructor parameters (e.g. ``n_fft``,
        ``hop_length``) should override this method.  The default returns an
        empty dict, which is correct for frames with no extra init args
        (e.g. ``ChannelFrame``).
        """
        return {}

    def _create_new_instance(self: S, data: DaArray, **kwargs: Any) -> S:
        """
        Create a new channel instance based on an existing channel.
        Keyword arguments can override or extend the original attributes.
        """

        sampling_rate = kwargs.pop("sampling_rate", self.sampling_rate)

        label = kwargs.pop("label", self.label)
        if not isinstance(label, str):
            raise TypeError("Label must be a string")

        metadata = kwargs.pop("metadata", copy.deepcopy(self.metadata))
        if not isinstance(metadata, dict):
            raise TypeError("Metadata must be a dictionary")

        lineage = kwargs.pop("lineage", self.lineage)

        channel_metadata = kwargs.pop("channel_metadata", self.channels.to_list())
        if not isinstance(channel_metadata, list):
            raise TypeError("Channel metadata must be a list")

        channel_ids = kwargs.pop("channel_ids", None)
        if channel_ids is None:
            channel_ids = (
                self._channel_ids
                if len(channel_metadata) == self.n_channels
                else self._default_channel_ids(len(channel_metadata))
            )
        if not isinstance(channel_ids, list):
            raise TypeError("Channel ids must be a list")

        source_time_offset = kwargs.pop("source_time_offset", self.source_time_offset)

        # Get additional initialization arguments from derived classes
        additional_kwargs = self._get_additional_init_kwargs()
        kwargs.update(additional_kwargs)

        init_kwargs: dict[str, Any] = {
            "data": data,
            "sampling_rate": sampling_rate,
            "label": label,
            "metadata": metadata,
            "channel_metadata": channel_metadata,
            "channel_ids": channel_ids,
            "source_time_offset": source_time_offset,
            "previous": self,
            "lineage": lineage,
            **kwargs,
        }

        return type(self)(**init_kwargs)

    def __array__(self, dtype: npt.DTypeLike = None) -> NDArrayReal:
        """Implicit conversion to NumPy array"""
        result = self.compute()
        if dtype is not None:
            return result.astype(dtype)
        return result

    def visualize_graph(self, filename: str | None = None) -> VisualizeReturnType:
        """
        Visualize the computation graph and save it to a file.

        This method creates a visual representation of the Dask computation graph.
        In interactive Python environments, it returns an IPython.display.Image object
        that can be displayed inline. In other environments, it saves the graph to
        a file and returns None.

        Parameters
        ----------
        filename : str, optional
            Output filename for the graph image. If None, a unique filename
            is generated using UUID. The file is saved in the current working
            directory.

        Returns
        -------
        IPython.display.Image or None
            In interactive Python environments: Returns an IPython.display.Image object
            that can be displayed inline.
            In other environments: Returns None after saving the graph to file.

        Notes
        -----
        This method requires graphviz to be installed on your system:
        - Ubuntu/Debian: `sudo apt-get install graphviz`
        - macOS: `brew install graphviz`
        - Windows: Download from https://graphviz.org/download/

        The graph displays operation names (e.g., 'normalize', 'lowpass_filter')
        making it easier to understand the processing pipeline.

        Examples
        --------
        >>> import wandas as wd
        >>> signal = wd.read("audio.wav")
        >>> processed = signal.normalize().low_pass_filter(cutoff=1000)
        >>> # In interactive environments: displays graph inline
        >>> processed.visualize_graph()
        >>> # Save to specific file
        >>> processed.visualize_graph("my_graph.png")

        See Also
        --------
        debug_info : Print detailed debug information about the frame
        """
        try:
            filename = filename or f"graph_{uuid.uuid4().hex[:8]}.png"
            return self._data.visualize(filename=filename)
        except Exception as e:
            logger.warning(f"Failed to visualize the graph: {e}")
            return None

    def _binary_op(
        self: S,
        other: S | int | float | complex | NDArrayReal | DaArray,
        op: Callable[[DaArray, Any], DaArray],
        symbol: str,
    ) -> S:
        """Default implementation of binary operations using dask's lazy evaluation.

        Handles both frame-frame and frame-scalar/array operations with
        metadata propagation and history tracking. Frame-frame operations
        combine current array indices and preserve the left operand's source
        timeline; no source-time alignment is performed. Uses
        ``_create_new_instance`` so that subclass-specific constructor
        parameters are automatically forwarded.

        Subclasses may override this entirely (e.g. ``RoughnessFrame``).
        """
        logger.debug(f"Setting up {symbol} operation (lazy)")

        metadata = copy.deepcopy(self.metadata)
        if isinstance(other, type(self)):
            if self.sampling_rate != other.sampling_rate:
                raise ValueError(
                    f"Sampling rate mismatch\n"
                    f"  Left operand: {self.sampling_rate} Hz\n"
                    f"  Right operand: {other.sampling_rate} Hz\n"
                    f"Resample one frame to match the other before performing "
                    f"{symbol} operation."
                )
            if self.n_channels != other.n_channels:
                raise ValueError(
                    f"Channel count mismatch\n"
                    f"  Left operand: {self.n_channels} channels\n"
                    f"  Right operand: {other.n_channels} channels\n"
                    f"Binary frame operations require matching channel counts to keep "
                    f"channel metadata aligned.\n"
                    f"Select, duplicate, or remove channels so both operands match "
                    f"before performing {symbol} operation."
                )
            if self.shape != other.shape:
                raise ValueError(
                    f"Frame shape mismatch\n"
                    f"  Left operand: {self.shape}\n"
                    f"  Right operand: {other.shape}\n"
                    f"Binary frame operations require identical shapes to avoid "
                    f"unintended broadcasting.\n"
                    f"Align the frame shapes before performing {symbol} operation."
                )

            result_data = op(self._data, other._data)
            other_str = other.label
            other_labels = other.labels
            operand_kind = "frame"
            lineage_inputs = (self.lineage, other.lineage)
        else:
            result_data = op(self._data, other)
            other_str = self._format_operand_str(other)
            other_labels = [other_str] * self.n_channels
            operand_kind = "operand"
            lineage_inputs = (self.lineage,)

        # Build merged channel metadata
        new_channel_metadata: list[ChannelMetadata] = []
        for self_ch, other_label in zip(self.channels, other_labels, strict=True):
            ch = self_ch.to_metadata()
            ch.label = f"({self_ch.label} {symbol} {other_label})"
            new_channel_metadata.append(ch)

        from wandas.processing.base import BinaryOperation

        binary_operation = BinaryOperation(
            symbol=symbol,
            operand_kind=operand_kind,
            operand=other_str if operand_kind == "frame" else other,
        )
        lineage = self._lineage_with_operation(binary_operation, *lineage_inputs)

        return self._create_new_instance(
            data=result_data,
            label=f"({self.label} {symbol} {other_str})",
            metadata=metadata,
            lineage=lineage,
            channel_metadata=new_channel_metadata,
        )

    @staticmethod
    def _format_operand_str(other: object) -> str:
        """Return a short display string for a binary operand."""
        if isinstance(other, int | float):
            return str(other)
        if isinstance(other, complex):
            return f"complex({other.real}, {other.imag})"
        if isinstance(other, np.ndarray):
            return f"ndarray{other.shape}"
        if hasattr(other, "shape"):
            return f"dask.array{other.shape}"
        return str(type(other).__name__)

    def __add__(self: S, other: S | int | float | NDArrayReal) -> S:
        """Addition operator"""
        return self._binary_op(other, lambda x, y: x + y, "+")

    def __sub__(self: S, other: S | int | float | NDArrayReal) -> S:
        """Subtraction operator"""
        return self._binary_op(other, lambda x, y: x - y, "-")

    def __mul__(self: S, other: S | int | float | NDArrayReal) -> S:
        """Multiplication operator"""
        return self._binary_op(other, lambda x, y: x * y, "*")

    def __truediv__(self: S, other: S | int | float | NDArrayReal) -> S:
        """Division operator"""
        return self._binary_op(other, lambda x, y: x / y, "/")

    def __pow__(self: S, other: S | int | float | NDArrayReal) -> S:
        """Power operator"""
        return self._binary_op(other, lambda x, y: x**y, "**")

    def apply_operation(self: S, operation_name: str, **params: Any) -> S:
        """
        Apply a named operation.

        Parameters
        ----------
        operation_name : str
            Name of the operation to apply.
        **params : Any
            Parameters to pass to the operation.

        Returns
        -------
        S
            A new instance with the operation applied.
        """
        # Apply the operation through abstract method
        return self._apply_operation_impl(operation_name, **params)

    def _updated_metadata(
        self,
        operation_name: str,
        params: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return frame metadata for a derived frame.

        Operation parameters are owned by runtime lineage. Frame metadata only
        carries user/domain metadata and is deep-copied to avoid sharing mutable
        state between frames.
        """
        return copy.deepcopy(self.metadata)

    def _apply_operation_impl(self: S, operation_name: str, **params: Any) -> S:
        """Default implementation of operation application.

        Creates the named operation, applies it to the data, and returns
        a new frame with updated metadata and operation history.
        Derived classes may override this to add extra behaviour
        (e.g. channel relabelling).
        """
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")
        from wandas.processing import create_operation

        operation = create_operation(operation_name, self.sampling_rate, **params)
        ensure_dependencies = getattr(operation, "ensure_dependencies", None)
        if ensure_dependencies is not None:
            ensure_dependencies()
        processed_data = operation.process(self._data)

        new_metadata = self._updated_metadata(operation_name, params)

        creation_params: dict[str, Any] = {
            "data": processed_data,
            "metadata": new_metadata,
            "lineage": self._lineage_with_operation(
                operation
                if getattr(operation, "name", None) == operation_name
                else _LineageOperationName(operation, operation_name, params),
                self.lineage,
            ),
        }

        return self._create_new_instance(**creation_params)

    @overload
    def _apply_operation_instance(
        self: S,
        operation: Any,
        operation_name: str | None = None,
        output_frame_class: None = None,
        output_frame_kwargs: dict[str, Any] | None = None,
    ) -> S: ...

    @overload
    def _apply_operation_instance(
        self: S,
        operation: Any,
        operation_name: str | None = None,
        output_frame_class: type[S_Out] = ...,
        output_frame_kwargs: dict[str, Any] | None = None,
    ) -> S_Out: ...

    def _apply_operation_instance(
        self: S,
        operation: Any,
        operation_name: str | None = None,
        output_frame_class: type[S_Out] | None = None,
        output_frame_kwargs: dict[str, Any] | None = None,
    ) -> S | S_Out:
        """Apply an already-instantiated operation to the frame.

        This method processes data through the operation, updates metadata,
        operation history, and channel labels atomically.  It is the
        entry-point used by ``ChannelProcessingMixin`` and by
        ``ChannelFrame._apply_operation_impl``.

        Parameters
        ----------
        operation : AudioOperation
            Instantiated operation to apply.
        operation_name : str, optional
            Override for the operation name in history.
        output_frame_class : type, optional
            If provided, the result is wrapped in this frame class instead
            of the same type as ``self``.  Enables domain transitions
            (e.g. ChannelFrame -> SpectralFrame) from ``apply()``.
        output_frame_kwargs : dict, optional
            Extra constructor keyword arguments required by *output_frame_class*
            (e.g. ``{"n_fft": 1024, "window": "hann"}``).
        """
        if operation_name is None:
            operation_name = getattr(operation, "name", "unknown_operation")
        expected_input_count = getattr(operation, "_expected_input_count", 1)
        if isinstance(expected_input_count, int) and expected_input_count != 1:
            raise ValueError(
                "Operation requires multiple runtime inputs\n"
                f"  Operation: {operation_name}\n"
                f"  Expected inputs: {expected_input_count}\n"
                "  Got: apply_operation provides one frame input\n"
                "Use an operation-specific method that can pass all runtime inputs."
            )

        ensure_dependencies = getattr(operation, "ensure_dependencies", None)
        if ensure_dependencies is not None:
            ensure_dependencies()
        processed_data = operation.process(self._data)

        params = getattr(operation, "params", {})

        new_metadata = self._updated_metadata(operation_name, params)
        lineage_operation = (
            operation
            if getattr(operation, "name", None) == operation_name
            else _LineageOperationName(operation, operation_name, params)
        )
        lineage = self._lineage_with_operation(lineage_operation, self.lineage)

        metadata_updates = operation.get_metadata_updates()
        if operation_name == "trim":
            start_sample = int(float(params.get("start", 0.0)) * self.sampling_rate)
            metadata_updates["source_time_offset"] = self.source_time_offset + start_sample / self.sampling_rate

        display_name = operation.get_display_name()
        new_channel_metadata = self._relabel_channels(operation_name, display_name)

        if output_frame_class is not None:
            if not isinstance(output_frame_class, type) or not issubclass(output_frame_class, BaseFrame):
                raise TypeError(
                    "Invalid output_frame_class\n"
                    f"  Got: {output_frame_class!r}\n"
                    f"  Expected: a BaseFrame subclass\n"
                    f"Pass a compatible Wandas frame class such as "
                    f"SpectralFrame or SpectrogramFrame."
                )
            # Domain transition: build a different frame type
            kw: dict[str, Any] = {
                "data": processed_data,
                "sampling_rate": metadata_updates.pop("sampling_rate", self.sampling_rate),
                "label": self.label,
                "metadata": new_metadata,
                "channel_metadata": new_channel_metadata,
                "channel_ids": self._channel_ids,
                "source_time_offset": metadata_updates.pop("source_time_offset", self.source_time_offset),
                "previous": self,
                "lineage": lineage,
            }
            kw.update(metadata_updates)
            if output_frame_kwargs:
                kw.update(output_frame_kwargs)
            try:
                return output_frame_class(**kw)
            except TypeError as exc:
                provided_kwargs = ", ".join(sorted(kw)) or "none"
                raise TypeError(
                    "Invalid output_frame_class constructor\n"
                    f"  Frame class: {output_frame_class.__name__}\n"
                    f"  Provided keyword arguments: {provided_kwargs}\n"
                    f"Ensure output_frame_class accepts these parameters and "
                    f"use output_frame_kwargs to supply any required "
                    f"domain-specific constructor arguments."
                ) from exc

        creation_params: dict[str, Any] = {
            "data": processed_data,
            "metadata": new_metadata,
            "lineage": lineage,
            "channel_metadata": new_channel_metadata,
            "channel_ids": self._channel_ids,
        }
        creation_params.update(metadata_updates)

        return self._create_new_instance(**creation_params)

    def _relabel_channels(
        self,
        operation_name: str,
        display_name: str | None = None,
    ) -> list[ChannelMetadata]:
        """
        Update channel labels to reflect applied operation.

        This method creates new channel metadata with labels that include
        the operation name, making it easier to track processing history
        and distinguish frames in plots.

        Parameters
        ----------
        operation_name : str
            Name of the operation (e.g., "normalize", "lowpass_filter")
        display_name : str, optional
            Display name for the operation. If None, uses operation_name.
            This allows operations to provide custom, more readable labels.

        Returns
        -------
        list[ChannelMetadata]
            New channel metadata with updated labels.
            Original metadata is deep-copied and only labels are modified.

        Examples
        --------
        >>> # Original label: "ch0"
        >>> # After normalize: "normalize(ch0)"
        >>> # After chained ops: "lowpass_filter(normalize(ch0))"

        Notes
        -----
        Labels are nested for chained operations, allowing full
        traceability of the processing pipeline.
        """
        display = display_name or operation_name
        new_metadata = []
        for ch in self.channels:
            new_ch = ch.to_metadata()
            new_ch.label = f"{display}({ch.label})"
            new_metadata.append(new_ch)
        return new_metadata

    def debug_info(self) -> None:
        """Output detailed debug information"""
        logger.debug(f"=== {self.__class__.__name__} Debug Info ===")
        logger.debug(f"Label: {self.label}")
        logger.debug(f"Shape: {self.shape}")
        logger.debug(f"Sampling rate: {self.sampling_rate} Hz")
        logger.debug(f"Operation history: {len(self.operation_history)} operations")
        self._debug_info_impl()
        logger.debug("=== End Debug Info ===")

    def print_operation_history(self) -> None:
        """
        Print the operation history to standard output in a readable format.

        This method writes a human-friendly representation of the
        `operation_history` list to stdout. Each operation is printed on its
        own line with an index, the operation name (if available), and the
        parameters used.

        Examples
        --------
        >>> cf.print_operation_history()
        1: normalize {}
        2: low_pass_filter {'cutoff': 1000}
        """
        if not self.operation_history:
            print("Operation history: <empty>")
            return

        print(f"Operation history ({len(self.operation_history)}):")
        for i, record in enumerate(self.operation_history, start=1):
            # record is expected to be a dict with at least a 'operation' key
            op_name = record.get("operation") or record.get("name") or "<unknown>"
            # Copy params for display - exclude the 'operation'/'name' keys
            params = {k: v for k, v in record.items() if k not in ("operation", "name")}
            print(f"{i}: {op_name} {params}")

    def to_numpy(self) -> T:
        """Convert the frame data to a NumPy array.

        This method computes the Dask array and returns it as a concrete NumPy array.
        The returned array has the same shape as the frame's data.

        Returns
        -------
        T
            NumPy array containing the frame data.

        Examples
        --------
        >>> cf = wd.read("audio.wav")
        >>> data = cf.to_numpy()
        >>> print(f"Shape: {data.shape}")  # (n_channels, n_samples)
        """
        return self.data

    def to_tensor(self, framework: str = "torch", device: str | None = None) -> Any:
        """
        Convert the Dask array to a tensor in the specified framework.

        Parameters
        ----------
        framework : str, default="torch"
            The ML framework to use ("torch" or "tensorflow").
        device : str or None, optional
            Device to place the tensor on. For PyTorch, use "cpu", "cuda", "cuda:0",
            etc. For TensorFlow, use "/CPU:0", "/GPU:0", etc. If None, uses the default
            device.

        Returns
        -------
        torch.Tensor or tf.Tensor
            A tensor in the specified framework.

        Raises
        ------
        ImportError
            If the specified framework is not installed.
        ValueError
            If the framework is not supported.
        TypeError
            If self.data is not a Dask array.

        Examples
        --------
        >>> # PyTorch tensor on CPU
        >>> tensor = frame.to_tensor(framework="torch", device="cpu")
        >>> # PyTorch tensor on GPU
        >>> tensor = frame.to_tensor(framework="torch", device="cuda:0")
        >>> # TensorFlow tensor on GPU
        >>> tensor = frame.to_tensor(framework="tensorflow", device="/GPU:0")
        """

        if framework == "torch":
            torch = require_dependency("torch", feature="tensor conversion with framework='torch'")
            numpy_data = self.to_numpy()

            # Convert NumPy array to PyTorch tensor
            tensor = torch.from_numpy(numpy_data)

            # Move to specified device if provided
            if device is not None:
                tensor = tensor.to(device)

            return tensor

        elif framework == "tensorflow":
            tf = require_dependency(
                "tensorflow",
                feature="tensor conversion with framework='tensorflow'",
            )
            numpy_data = self.to_numpy()

            # Convert NumPy array to TensorFlow tensor
            if device is not None:
                with tf.device(device):
                    tensor = tf.convert_to_tensor(numpy_data)
            else:
                tensor = tf.convert_to_tensor(numpy_data)

            return tensor

        else:
            raise ValueError(
                f"Unsupported framework\n"
                f"  Got: '{framework}'\n"
                f"  Expected: 'torch' or 'tensorflow'\n"
                f"Use a supported framework for tensor conversion"
            )

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert the frame data to a pandas DataFrame.

        This method provides a common implementation for converting frame data
        to pandas DataFrame. Subclasses can override this method for custom behavior.

        Returns
        -------
        pd.DataFrame
            DataFrame with appropriate index and columns.

        Examples
        --------
        >>> cf = wd.read("audio.wav")
        >>> df = cf.to_dataframe()
        >>> print(df.head())
        """
        pd = require_pandas("BaseFrame.to_dataframe")

        # Get data as numpy array
        data = self.to_numpy()

        # Get column names from subclass
        columns = self._get_dataframe_columns()

        # Get index from subclass
        index = self._get_dataframe_index()

        # Create DataFrame
        if data.ndim == 1:
            # Single channel case - reshape to 2D
            df = pd.DataFrame(data.reshape(-1, 1), columns=columns, index=index)
        else:
            # Multi-channel case - transpose to (n_samples, n_channels)
            df = pd.DataFrame(data.T, columns=columns, index=index)

        return df

    def _get_dataframe_columns(self) -> list[str]:
        """Get column names for DataFrame.

        Returns channel labels by default. Override in subclasses
        if different column names are needed.

        Returns
        -------
        list[str]
            List of column names.
        """
        return self.labels

    @abstractmethod
    def _get_dataframe_index(self) -> "pd.Index[Any]":
        """Get index for DataFrame.

        This method should be implemented by subclasses to provide
        appropriate index for the DataFrame based on the frame type.

        Returns
        -------
        pd.Index
            Index for the DataFrame.
        """

    def _debug_info_impl(self) -> None:
        """Implement derived class-specific debug information"""

    def _print_operation_history(self) -> None:
        """Print the operation history information.

        This is a helper method for info() implementations to display
        the number of operations applied to the frame in a consistent format.
        """
        if self.operation_history:
            print(f"  Operations Applied: {len(self.operation_history)}")
        else:
            print("  Operations Applied: None")
