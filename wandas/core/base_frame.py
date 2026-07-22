import copy
import logging
import numbers
import uuid
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from re import Pattern
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast, overload

import numpy as np
import numpy.typing as npt
import xarray as xr
from dask.array.core import Array as DaArray

from wandas.pipeline.decorators import OperationCapture, recipe_operation
from wandas.processing.calibration import apply_channel_factors
from wandas.processing.semantic import (
    ImmutableList,
    InputBinding,
    LineageNode,
    active_semantic_lineage,
    lineage_history,
    source_lineage,
)
from wandas.utils import validate_sampling_rate
from wandas.utils.optional_imports import require_dependency, require_pandas
from wandas.utils.types import NDArrayComplex, NDArrayReal

from ._channel_schema import (
    _CHANNEL_CALIBRATION_FACTOR_KEY,
    _CHANNEL_COORD_FALLBACK_ATTRS,
    _CHANNEL_EXTRA_ATTR,
    _CHANNEL_IDS_ATTR,
    _CHANNEL_LABEL_KEY,
    _CHANNEL_REF_KEY,
    _CHANNEL_UNIT_KEY,
)
from ._deprecated_mutable import is_wrapped_mutable, wrap_mutable
from .channel_metadata import ChannelMetadataIndexer, ChannelMetadataView
from .metadata import ChannelCalibration, ChannelMetadata

# IPython display types for visualize_graph return type
# Define as type alias under TYPE_CHECKING; use Any at runtime
if TYPE_CHECKING:
    from typing import TypeAlias

    import pandas as pd
    from IPython.display import Image as IPythonImage
    from matplotlib.axes import Axes

    VisualizeReturnType: TypeAlias = IPythonImage | None
else:
    # Use Any at runtime to avoid type checker errors
    VisualizeReturnType = Any

logger = logging.getLogger(__name__)

T = TypeVar("T", NDArrayComplex, NDArrayReal)
S = TypeVar("S", bound="BaseFrame[Any]")
S_Out = TypeVar("S_Out", bound="BaseFrame[Any]")
QueryType = str | Pattern[str] | Callable[["ChannelMetadata"], bool] | dict[str, Any]


def _get_channel_semantic_params(params: Mapping[str, Any]) -> Mapping[str, Any]:
    """Normalize replayable selection intent without resolving runtime searches."""
    query = params.get("query")
    if query is not None:
        return {
            "query": query,
            **({"validate_query_keys": params["validate_query_keys"]} if "validate_query_keys" in params else {}),
        }
    channel_idx = params.get("channel_idx")
    if isinstance(channel_idx, np.ndarray):
        return {"channel_idx": [int(value) for value in channel_idx.tolist()]}
    return params


def _capture_get_channel(args: tuple[Any, ...], params: Mapping[str, Any]) -> OperationCapture:
    """Capture portable channel-selection intent or an atomic rejection reason."""
    receiver = cast("BaseFrame[Any]", args[0])
    channel_idx = params.get("channel_idx")
    if params.get("query") is None and isinstance(channel_idx, np.ndarray) and channel_idx.dtype in (bool, np.bool_):
        normalized = {"channel_idx": receiver._semantic_index_params(channel_idx)}
    else:
        normalized = _get_channel_semantic_params(params)
    query = normalized.get("query")
    error = None
    if query is not None and not isinstance(query, str | Mapping):
        error = "Compiled regex and callable channel queries are not portable"
        normalized = {"unsupported_query_type": type(query).__name__}
    return OperationCapture(
        (InputBinding("frame", "frame"),),
        (receiver.lineage,),
        normalized,
        error,
    )


def _thaw_recipe_query(value: Any) -> Any:
    """Restore immutable Recipe query containers to canonical public values."""
    if isinstance(value, Mapping):
        return {key: _thaw_recipe_query(item) for key, item in value.items()}
    if isinstance(value, ImmutableList):
        return [_thaw_recipe_query(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_thaw_recipe_query(item) for item in value)
    return value


def _apply_get_channel_recipe(inputs: tuple[Any, ...], params: Mapping[str, Any]) -> Any:
    """Replay channel selection while restoring portable boolean-mask intent."""
    call_params = dict(params)
    if "query" in call_params:
        call_params["query"] = _thaw_recipe_query(call_params["query"])
    channel_idx = call_params.get("channel_idx")
    if isinstance(channel_idx, Mapping) and channel_idx.get("indexing") == "boolean_mask":
        call_params["channel_idx"] = inputs[0]._selector_from_intent(channel_idx)
    return inputs[0].get_channel(**call_params)


def _capture_index(args: tuple[Any, ...], params: Mapping[str, Any]) -> OperationCapture:
    """Capture one public indexing call using the canonical selector grammar."""
    receiver = cast("BaseFrame[Any]", args[0])
    intent = receiver._semantic_index_params(params["key"])
    indexing = intent.get("indexing")
    if indexing in {"unsupported", "multidimensional"}:
        return OperationCapture(
            (InputBinding("frame", "frame"),),
            (receiver.lineage,),
            {"unsupported_index_type": type(params["key"]).__name__},
            "Index is outside the portable Wandas selector grammar",
        )
    return OperationCapture(
        (InputBinding("frame", "frame"),),
        (receiver.lineage,),
        {"selector": intent},
    )


def _apply_index_recipe(inputs: tuple[Any, ...], params: Mapping[str, Any]) -> Any:
    """Replay canonical indexing intent through the public indexing entrypoint."""
    return inputs[0]._apply_index_intent(params["selector"])


def _capture_binary(args: tuple[Any, ...], params: Mapping[str, Any]) -> OperationCapture:
    """Capture ordered Frame, array, or immutable scalar binary operands."""
    receiver = cast("BaseFrame[Any]", args[0])
    other = params["other"]
    if isinstance(other, BaseFrame):
        return OperationCapture(
            (InputBinding("left", "frame"), InputBinding("right", "frame")),
            (receiver.lineage, other.lineage),
            {},
        )
    if isinstance(other, np.ndarray | DaArray):
        return OperationCapture(
            (InputBinding("left", "frame"), InputBinding("right", "array")),
            (receiver.lineage, None),
            {},
        )
    return OperationCapture(
        (InputBinding("left", "frame"),),
        (receiver.lineage,),
        {"operand": other},
    )


def _capture_reverse_binary(args: tuple[Any, ...], params: Mapping[str, Any]) -> OperationCapture:
    """Capture a reverse scalar operation while preserving operand order."""
    receiver = cast("BaseFrame[Any]", args[0])
    return OperationCapture(
        (InputBinding("right", "frame"),),
        (receiver.lineage,),
        {"operand": params["other"]},
    )


def _binary_recipe_handler(method_name: str) -> Callable[[tuple[Any, ...], Mapping[str, Any]], Any]:
    """Build a replay handler for one forward binary special method."""

    def apply(inputs: tuple[Any, ...], params: Mapping[str, Any]) -> Any:
        """Invoke the binary method with a runtime or canonical scalar operand."""
        operand = inputs[1] if len(inputs) == 2 else params["operand"]
        return getattr(inputs[0], method_name)(operand)

    return apply


_FORWARD_BINARY_PATTERNS = (
    (InputBinding("left", "frame"),),
    (InputBinding("left", "frame"), InputBinding("right", "frame")),
    (InputBinding("left", "frame"), InputBinding("right", "array")),
)
_REVERSE_BINARY_PATTERNS = ((InputBinding("right", "frame"),),)


def _capture_rename_channels(args: tuple[Any, ...], params: Mapping[str, Any]) -> OperationCapture:
    mapping = params["mapping"]
    if not isinstance(mapping, Mapping):
        raise TypeError("rename_channels mapping must be a mapping")
    entries = []
    for key, label in mapping.items():
        if type(key) is int:
            encoded_key: Mapping[str, Any] = {"type": "integer", "value": key}
        elif isinstance(key, str):
            encoded_key = {"type": "label", "value": key}
        else:
            raise TypeError("rename_channels keys must be integers or strings")
        entries.append([encoded_key, label])
    receiver = cast("BaseFrame[Any]", args[0])
    return OperationCapture((InputBinding("frame", "frame"),), (receiver.lineage,), {"entries": entries})


def _rename_channels_recipe(inputs: tuple[Any, ...], params: Mapping[str, Any]) -> Any:
    mapping: dict[int | str, str] = {}
    for raw_key, label in params["entries"]:
        key = int(raw_key["value"]) if raw_key["type"] == "integer" else str(raw_key["value"])
        mapping[key] = str(label)
    return inputs[0].rename_channels(mapping)


def _capture_source_time_offset(args: tuple[Any, ...], params: Mapping[str, Any]) -> OperationCapture:
    receiver = cast("BaseFrame[Any]", args[0])
    offsets = receiver._normalize_source_time_offset(params["value"], receiver.n_channels)
    return OperationCapture(
        (InputBinding("frame", "frame"),),
        (receiver.lineage,),
        {"value": offsets.tolist()},
    )


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
        Constructor override for the initial runtime lineage. When omitted, the
        constructor creates a source node; every constructed frame therefore has
        exactly one lineage authority. ``operation_history`` is its public JSON-safe
        projection.
    channel_metadata : list[ChannelMetadata | dict], optional
        Metadata for each channel in the frame. Can be ChannelMetadata objects
        or dicts that will be converted to ChannelMetadata objects.
    previous : BaseFrame, optional
        Compatibility/debug pointer to the immediate prior frame. This strong
        reference is not the source of truth for processing history.

    Attributes
    ----------
    sampling_rate : float
        The sampling rate of the signal in Hz.
    label : str
        The label of the frame.
    metadata : dict
        Additional metadata for the frame.
    lineage : LineageNode
        Runtime computation lineage. This is always set during construction and
        propagated through ``_create_new_instance``.
    operation_history : list[dict]
        Flat read-only compatibility view derived from ``lineage``.
    """

    _CHANNEL_DIM: ClassVar[str] = "channel"
    # Fallback only for neutral-dim and legacy frames. Target frames should
    # prefer the xarray "channel" dimension when it is declared.
    _channel_axis: ClassVar[int | None] = -2
    _xarray_dim_suffix: ClassVar[tuple[str, ...]] = ()
    _array_ufunc_reverse_methods: ClassVar[Mapping[str, str]] = {
        "add": "__radd__",
        "subtract": "__rsub__",
        "multiply": "__rmul__",
        "divide": "__rtruediv__",
        "true_divide": "__rtruediv__",
        "power": "__rpow__",
    }
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
        lineage: LineageNode | None = None,
        operation_history_prefix: Sequence[Mapping[str, Any]] = (),
    ):
        """Initialize immutable Frame data, metadata, channel state, and lineage."""
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
        self._write_label(label)
        self._write_sampling_rate(sampling_rate)
        self._write_metadata(metadata)
        if lineage is not None and operation_history_prefix:
            raise ValueError("operation_history_prefix is valid only for a new source Frame")
        self._lineage = lineage if lineage is not None else source_lineage(operation_history_prefix)
        self._write_normalized_channel_metadata(normalized_channel_metadata, self._pending_channel_ids)
        self._write_source_time_offset(source_time_offset)
        del self._pending_channel_metadata
        del self._pending_channel_ids
        self._previous = previous

        try:
            # Display information for newer dask versions
            effective_data = self._effective_data
            logger.debug(f"Dask graph layers: {list(effective_data.dask.layers.keys())}")
            logger.debug(f"Dask graph dependencies: {len(effective_data.dask.dependencies)}")
        except Exception as e:
            logger.debug(f"Dask graph visualization details unavailable: {e}")

    @property
    def _data(self) -> DaArray:
        """Compatibility alias for the Dask array stored in ``_xr``."""
        data = self._xr.data
        if not isinstance(data, DaArray):
            raise TypeError(f"Internal xarray data is not a Dask array: {type(data).__name__}")
        return data

    @property
    def _effective_data(self) -> DaArray:
        """Return lazily calibrated data used by numerical public APIs."""
        factors = tuple(channel.calibration.factor for channel in self.channels)
        if all(factor == 1.0 for factor in factors):
            return self._data
        return apply_channel_factors(self._data, factors)

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
            self._write_normalized_channel_metadata(old_channel_metadata, old_channel_ids)
            self._write_source_time_offset(old_source_time_offset)

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
            _CHANNEL_LABEL_KEY: (self._CHANNEL_DIM, [ch.label for ch in metadata]),
            _CHANNEL_UNIT_KEY: (self._CHANNEL_DIM, [ch.unit for ch in metadata]),
            _CHANNEL_REF_KEY: (self._CHANNEL_DIM, [ch.ref for ch in metadata]),
            _CHANNEL_CALIBRATION_FACTOR_KEY: (
                self._CHANNEL_DIM,
                [ch.calibration.factor for ch in metadata],
            ),
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
        """Return deterministic source channel identifiers."""
        return [f"c{i}" for i in range(n_channels)]

    @staticmethod
    def _validate_channel_ids(channel_ids: Sequence[Any], n_channels: int) -> list[str]:
        """Normalize unique channel identifiers and enforce channel-count agreement."""
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
        """Return defensive metadata values matching an exact channel count."""

        def _to_channel_metadata(ch: ChannelMetadata | dict[str, Any], index: int) -> ChannelMetadata:
            """Decode one metadata-like value with index-aware errors."""
            if type(ch) is ChannelMetadataView:
                return ch.to_metadata()
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
        self._set_channel_metadata(list(self.channels), self._channel_ids)

    @property
    def _channel_ids(self) -> list[str]:
        """Return channel identifiers from xarray coordinates or legacy attrs."""
        if self._CHANNEL_DIM in self._xr.coords:
            return [str(value) for value in self._xr.coords[self._CHANNEL_DIM].values.tolist()]
        return [str(value) for value in self._xr.attrs.get(_CHANNEL_IDS_ATTR, [])]

    def _channel_id_at(self, index: int) -> str:
        """Return the stable identifier for one channel position."""
        if self._CHANNEL_DIM in self._xr.coords:
            return str(self._xr.coords[self._CHANNEL_DIM].values[index])
        return str(self._xr.attrs[_CHANNEL_IDS_ATTR][index])

    def _get_channel_coord_value(self, coord_name: str, index: int) -> Any:
        """Read one channel metadata value from coordinates or legacy attrs."""
        if coord_name in self._xr.coords:
            return self._xr.coords[coord_name].values[index]
        return self._xr.attrs[coord_name][index]

    def _channel_ids_for_selection(self, indices: Sequence[int]) -> list[str]:
        """Return collision-free stable identifiers for a channel selection."""
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

    def _borrowed_channel_metadata_descriptors(
        self,
        indices: Sequence[int] | None = None,
        *,
        calibrations: Mapping[str, ChannelCalibration] | None = None,
    ) -> list[dict[str, Any]]:
        """Describe channel state for immediate ownership by a Frame constructor.

        The returned ``extra`` dictionaries remain owned by this Frame. Callers must
        pass the descriptors directly to a Frame constructor, whose normalization
        boundary takes one defensive copy.
        """
        selected = range(self.n_channels) if indices is None else indices
        descriptors: list[dict[str, Any]] = []
        for index in selected:
            view = self.channels[index]
            channel_id = self._channel_id_at(index)
            calibration = (
                calibrations[channel_id]
                if calibrations is not None and channel_id in calibrations
                else view.calibration
            )
            descriptors.append(
                {
                    "label": view.label,
                    "calibration": calibration,
                    "extra": view.extra,
                }
            )
        return descriptors

    @property
    def _channel_metadata(self) -> list[ChannelMetadata]:
        """Compatibility list-like view over xarray-backed channel metadata."""
        return cast(list[ChannelMetadata], self.channels)

    @_channel_metadata.setter
    def _channel_metadata(self, value: Sequence[ChannelMetadata | dict[str, Any]]) -> None:
        """Replace the compatibility metadata view through xarray-backed state."""
        self._set_channel_metadata(value)

    def _set_channel_coord_value(self, coord_name: str, index: int, value: Any) -> None:
        """Update one xarray-backed channel metadata coordinate defensively."""
        if coord_name in self._xr.coords:
            values = self._xr.coords[coord_name].values.tolist()
            values[index] = value
            self._xr = self._xr.assign_coords({coord_name: (self._CHANNEL_DIM, values)})
            return
        values = list(self._xr.attrs.get(coord_name, []))
        values[index] = value
        self._xr.attrs[coord_name] = values

    def _set_channel_calibration(self, index: int, calibration: ChannelCalibration) -> None:
        """Atomically replace one channel's factor and physical-domain coordinates."""
        if not isinstance(calibration, ChannelCalibration):
            raise TypeError("calibration must be a ChannelCalibration")
        updates = {
            _CHANNEL_CALIBRATION_FACTOR_KEY: calibration.factor,
            _CHANNEL_UNIT_KEY: calibration.unit,
            _CHANNEL_REF_KEY: calibration.ref,
        }
        if self._CHANNEL_DIM in self._xr.dims:
            coords: dict[str, Any] = {}
            for name, value in updates.items():
                values = self._xr.coords[name].values.tolist()
                values[index] = value
                coords[name] = (self._CHANNEL_DIM, values)
            self._xr = self._xr.assign_coords(coords)
            return
        for name, value in updates.items():
            values = list(self._xr.attrs[name])
            values[index] = value
            self._xr.attrs[name] = values

    def _set_channel_metadata(
        self,
        channel_metadata: Sequence[ChannelMetadata | dict[str, Any]],
        channel_ids: Sequence[Any] | None = None,
    ) -> None:
        """Take ownership of metadata-like values and write synchronized storage."""
        normalized = self._normalize_channel_metadata_for_count(channel_metadata, self._n_channels)
        self._write_normalized_channel_metadata(normalized, channel_ids)

    def _write_normalized_channel_metadata(
        self,
        channel_metadata: Sequence[ChannelMetadata],
        channel_ids: Sequence[Any] | None = None,
    ) -> None:
        """Write exclusively owned, normalized channel metadata without copying."""
        ids = (
            self._validate_channel_ids(channel_ids, self._n_channels) if channel_ids is not None else self._channel_ids
        )
        if not ids:
            ids = self._default_channel_ids(self._n_channels)
        if len(channel_metadata) != self._n_channels:
            raise ValueError(
                "Normalized channel metadata length must match number of channels\n"
                f"  Metadata entries: {len(channel_metadata)}\n"
                f"  Channels: {self._n_channels}"
            )
        labels = [ch.label for ch in channel_metadata]
        units = [ch.unit for ch in channel_metadata]
        refs = [ch.ref for ch in channel_metadata]
        factors = [ch.calibration.factor for ch in channel_metadata]
        channel_extra = {
            channel_id: (
                ch.extra
                if is_wrapped_mutable(ch.extra)
                else wrap_mutable(
                    ch.extra,
                    "Direct frame.channels[i].extra mutation is deprecated; use frame.with_channel_extra().",
                )
            )
            for channel_id, ch in zip(ids, channel_metadata, strict=True)
        }
        self._xr.attrs[_CHANNEL_EXTRA_ATTR] = channel_extra
        if self._CHANNEL_DIM in self._xr.dims:
            self._xr = self._xr.assign_coords(
                {
                    self._CHANNEL_DIM: (self._CHANNEL_DIM, ids),
                    _CHANNEL_LABEL_KEY: (self._CHANNEL_DIM, labels),
                    _CHANNEL_UNIT_KEY: (self._CHANNEL_DIM, units),
                    _CHANNEL_REF_KEY: (self._CHANNEL_DIM, refs),
                    _CHANNEL_CALIBRATION_FACTOR_KEY: (self._CHANNEL_DIM, factors),
                }
            )
            for name in _CHANNEL_COORD_FALLBACK_ATTRS:
                self._xr.attrs.pop(name, None)
            return
        self._xr.attrs.update(
            {
                _CHANNEL_IDS_ATTR: ids,
                _CHANNEL_LABEL_KEY: labels,
                _CHANNEL_UNIT_KEY: units,
                _CHANNEL_REF_KEY: refs,
                _CHANNEL_CALIBRATION_FACTOR_KEY: factors,
            }
        )

    def _next_channel_id(self, existing_ids: Sequence[str] | None = None) -> str:
        """Return the first unused deterministic channel identifier."""
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
        """Return the immediate prior frame for compatibility/debug inspection.

        This strong reference is not the source of truth for processing
        history. Runtime lineage drives ``operation_history`` and Recipe extraction.
        """
        return self._previous

    @property
    def sampling_rate(self) -> float:
        """Return the frame sampling rate from xarray attrs."""
        return float(self._xr.attrs["sampling_rate"])

    @sampling_rate.setter
    def sampling_rate(self, value: float) -> None:
        warnings.warn(
            "Direct frame.sampling_rate mutation is deprecated; resample the source ChannelFrame with "
            "ChannelFrame.resampling(). Derived Frame sampling rates cannot be reassigned.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._write_sampling_rate(value)

    def _write_sampling_rate(self, value: float) -> None:
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
        warnings.warn(
            "Direct frame.label mutation is deprecated; use frame.with_label().", DeprecationWarning, stacklevel=2
        )
        self._write_label(value)

    def _write_label(self, value: str | None) -> None:
        if value is not None and not isinstance(value, str):
            raise TypeError("Label must be a string or None")
        label = value or "unnamed_frame"
        self._xr.attrs["label"] = label
        self._xr.name = label

    @property
    def metadata(self) -> dict[str, Any]:
        """Return compatibility metadata; mutation warns and remains effective in v0.7."""
        value = self._xr.attrs.get("metadata")
        if value is None:
            value = wrap_mutable({}, "Direct frame.metadata mutation is deprecated; use frame.with_metadata().")
            self._xr.attrs["metadata"] = value
        if not isinstance(value, dict):
            raise TypeError(f"Internal metadata attrs must be a dictionary, got {type(value).__name__}")
        if not is_wrapped_mutable(value):
            value = wrap_mutable(value, "Direct frame.metadata mutation is deprecated; use frame.with_metadata().")
            self._xr.attrs["metadata"] = value
        return value

    @metadata.setter
    def metadata(self, value: dict[str, Any] | None) -> None:
        warnings.warn(
            "Direct frame.metadata mutation is deprecated; use frame.with_metadata().", DeprecationWarning, stacklevel=2
        )
        self._write_metadata(value)

    def _write_metadata(self, value: dict[str, Any] | None) -> None:
        if value is None:
            self._xr.attrs["metadata"] = wrap_mutable(
                {}, "Direct frame.metadata mutation is deprecated; use frame.with_metadata()."
            )
            return
        if not isinstance(value, dict):
            raise TypeError("Metadata must be a dictionary")
        self._xr.attrs["metadata"] = wrap_mutable(
            value, "Direct frame.metadata mutation is deprecated; use frame.with_metadata()."
        )

    @property
    def operation_history(self) -> list[dict[str, Any]]:
        """Return the sole public, JSON-safe provenance projection."""
        return lineage_history(self._lineage)

    @property
    def lineage(self) -> LineageNode:
        """Return runtime computation lineage for this frame."""
        return self._lineage

    def _required_semantic_lineage(self) -> LineageNode:
        """Return the active authoritative lineage or reject an internal bypass."""
        lineage = active_semantic_lineage()
        if not isinstance(lineage, LineageNode):
            raise RuntimeError("Public semantic lineage capture is not active")
        return lineage

    def _semantic_index_params(self, key: Any) -> Mapping[str, Any]:
        """Encode one public index as portable selector intent when supported."""
        if isinstance(key, tuple) and len(key) == 1:
            key = key[0]
        if isinstance(key, numbers.Integral) and not isinstance(key, bool | np.bool_):
            return {"indexing": "integer", "index": int(key)}
        if isinstance(key, str):
            return {"indexing": "label", "label": key}
        if isinstance(key, slice):
            bounds = self._slice_for_lineage(key)
            return {"indexing": "unsupported"} if bounds is None else {"indexing": "channel_slice", **bounds}
        if isinstance(key, np.ndarray) and key.ndim == 1 and key.dtype in (bool, np.bool_):
            return {"indexing": "boolean_mask", "mask": tuple(bool(value) for value in key.tolist())}
        if isinstance(key, np.ndarray) and key.ndim == 1 and np.issubdtype(key.dtype, np.integer):
            return {"indexing": "integer_array", "indices": tuple(int(value) for value in key.tolist())}
        if isinstance(key, list) and key and all(isinstance(value, str) for value in key):
            return {"indexing": "label_list", "labels": tuple(key)}
        if (
            isinstance(key, list)
            and key
            and all(isinstance(value, numbers.Integral) and not isinstance(value, bool | np.bool_) for value in key)
        ):
            return {"indexing": "integer_list", "indices": tuple(int(value) for value in key)}
        if isinstance(key, tuple) and key:
            channel = self._channel_selector_for_lineage(key[0])
            axis_slices = self._axis_slices_for_lineage(key[1:])
            if channel is not None and axis_slices is not None:
                return {"indexing": "multidimensional_slice", "channel": channel, "axis_slices": axis_slices}
            return {"indexing": "multidimensional"}
        return {"indexing": "unsupported"}

    @staticmethod
    def _slice_from_intent(value: Mapping[str, Any]) -> slice:
        """Decode canonical slice bounds into a fresh Python slice."""
        return slice(value.get("start"), value.get("stop"), value.get("step"))

    @classmethod
    def _selector_from_intent(cls, value: Mapping[str, Any]) -> Any:
        """Decode canonical channel-selector intent for public Recipe replay."""
        kind = value.get("indexing")
        if kind == "integer":
            return int(value["index"])
        if kind == "label":
            return value["label"]
        if kind == "channel_slice":
            return cls._slice_from_intent(value)
        if kind in {"integer_list", "integer_array"}:
            indices = [int(item) for item in value["indices"]]
            return indices if kind == "integer_list" else np.asarray(indices, dtype=int)
        if kind == "label_list":
            return list(value["labels"])
        if kind == "boolean_mask":
            return np.asarray(value["mask"], dtype=bool)
        if kind == "multidimensional_slice":
            channel = cls._selector_from_intent(cast(Mapping[str, Any], value["channel"]))
            axes = tuple(cls._slice_from_intent(item) for item in value["axis_slices"])
            return (channel, *axes)
        raise TypeError(f"Unsupported canonical selector: {kind!r}")

    def _apply_index_intent(self: S, intent: Mapping[str, Any]) -> S:
        """Apply one already-validated canonical selector."""
        return self[self._selector_from_intent(intent)]

    @staticmethod
    def _slice_bound_for_lineage(value: Any) -> int | None:
        """Normalize an optional integral slice bound for semantic capture."""
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            return None
        return int(value)

    @classmethod
    def _slice_for_lineage(cls, key: slice) -> dict[str, int | None] | None:
        """Encode a portable integral slice or return ``None`` when unsupported."""
        start = cls._slice_bound_for_lineage(key.start)
        stop = cls._slice_bound_for_lineage(key.stop)
        step = cls._slice_bound_for_lineage(key.step)
        if (key.start is not None and start is None) or (key.stop is not None and stop is None):
            return None
        if key.step is not None and step is None:
            return None
        return {"start": start, "stop": stop, "step": step}

    @classmethod
    def _axis_slices_for_lineage(cls, keys: tuple[Any, ...]) -> tuple[dict[str, int | None], ...] | None:
        """Encode portable non-channel axis slices for multidimensional indexing."""
        axis_slices: list[dict[str, int | None]] = []
        for key in keys:
            if not isinstance(key, slice):
                return None
            axis_slice = cls._slice_for_lineage(key)
            if axis_slice is None:
                return None
            axis_slices.append(axis_slice)
        return tuple(axis_slices)

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
        warnings.warn(
            "Direct frame.source_time_offset mutation is deprecated; use frame.with_source_time_offset().",
            DeprecationWarning,
            stacklevel=2,
        )
        self._write_source_time_offset(value)

    def _write_source_time_offset(self, value: float | Sequence[float] | NDArrayReal) -> None:
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
        """Return a defensive finite per-channel source-time offset array."""
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

    def with_label(self: S, label: str | None) -> S:
        """Return an annotation-only copy with a replacement Frame label."""
        return self._with_annotations(label=label, label_is_set=True)

    def with_metadata(self: S, updates: Mapping[str, Any], *, replace: bool = False) -> S:
        """Return an annotation-only copy with merged or replaced metadata."""
        return self._with_annotations(metadata=updates, replace=replace)

    def _resolve_one_channel(self, selector: str | int) -> int:
        if isinstance(selector, bool) or not isinstance(selector, str | int):
            raise TypeError("Channel selector must be a stable ID, label, or integer index")
        if isinstance(selector, int):
            if selector < -self.n_channels or selector >= self.n_channels:
                raise IndexError(f"Channel index out of range: {selector}")
            return selector % self.n_channels
        if selector in self._channel_ids:
            return self._channel_ids.index(selector)
        matches = [index for index, label in enumerate(self.labels) if label == selector]
        if not matches:
            raise KeyError(f"Channel selector not found: {selector!r}")
        if len(matches) > 1:
            raise ValueError(f"Channel label is ambiguous: {selector!r}; use a stable ID or index")
        return matches[0]

    def with_channel_extra(
        self: S,
        channel: str | int,
        updates: Mapping[str, Any],
        *,
        replace: bool = False,
    ) -> S:
        """Return an annotation-only copy with one channel's extra metadata updated."""
        return self._with_annotations(channel_extra={channel: updates}, replace=replace)

    def with_annotations(
        self: S,
        *,
        label: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        channel_extra: Mapping[str | int, Mapping[str, Any]] | None = None,
        replace: bool = False,
    ) -> S:
        """Atomically apply Frame annotations without adding lineage or Recipe intent."""
        return self._with_annotations(
            label=label,
            label_is_set=label is not None,
            metadata=metadata,
            channel_extra=channel_extra,
            replace=replace,
        )

    def _with_annotations(
        self: S,
        *,
        label: str | None = None,
        label_is_set: bool = False,
        metadata: Mapping[str, Any] | None = None,
        channel_extra: Mapping[str | int, Mapping[str, Any]] | None = None,
        replace: bool = False,
    ) -> S:
        """Apply normalized annotation intent through one reconstruction engine."""
        if label is not None and not isinstance(label, str):
            raise TypeError("Label must be a string or None")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("Metadata updates must be a mapping")
        new_metadata = {} if replace and metadata is not None else copy.deepcopy(dict(self.metadata))
        if metadata is not None:
            new_metadata.update(copy.deepcopy(dict(metadata)))
        descriptors = self._borrowed_channel_metadata_descriptors()
        if channel_extra is not None:
            if not isinstance(channel_extra, Mapping):
                raise TypeError("channel_extra must map channel selectors to update mappings")
            resolved: set[int] = set()
            for selector, updates in channel_extra.items():
                if not isinstance(updates, Mapping):
                    raise TypeError("Channel extra updates must be mappings")
                index = self._resolve_one_channel(selector)
                if index in resolved:
                    raise ValueError(f"Duplicate channel selector resolves to index {index}")
                resolved.add(index)
                extra = {} if replace else copy.deepcopy(dict(descriptors[index]["extra"]))
                extra.update(copy.deepcopy(dict(updates)))
                descriptors[index]["extra"] = extra
        return self._create_new_instance(
            self._data,
            label=label if label_is_set else self.label,
            metadata=new_metadata,
            channel_metadata=descriptors,
            channel_ids=self._channel_ids,
            source_time_offset=self.source_time_offset,
            lineage=self.lineage,
        )

    @recipe_operation("wandas.frame.with_source_time_offset", capture=_capture_source_time_offset)
    def with_source_time_offset(self: S, value: float | Sequence[float] | NDArrayReal) -> S:
        """Return a Recipe-capable copy with normalized per-channel source offsets."""
        offsets = self._normalize_source_time_offset(value, self.n_channels)
        return self._create_new_instance(
            self._data,
            source_time_offset=offsets,
            lineage=self._required_semantic_lineage(),
        )

    @recipe_operation(
        "wandas.channel.rename_channels",
        capture=_capture_rename_channels,
        handler=_rename_channels_recipe,
    )
    def rename_channels(self: S, mapping: Mapping[int | str, str]) -> S:
        """Return a copy with channel labels renamed by index or current label."""
        if not isinstance(mapping, Mapping):
            raise TypeError("rename_channels mapping must be a mapping")
        labels = self.labels
        new_labels = labels.copy()
        resolved: dict[int, str] = {}
        for key, new_label in mapping.items():
            if not isinstance(new_label, str):
                raise TypeError("Channel labels must be strings")
            if type(key) is int:
                if not 0 <= key < self.n_channels:
                    raise KeyError(f"Channel index out of range: {key}")
                index = key
            elif isinstance(key, str):
                matches = [i for i, label in enumerate(labels) if label == key]
                if not matches:
                    raise KeyError(f"Channel label not found: {key!r}")
                if len(matches) > 1:
                    raise ValueError(f"Channel label is ambiguous: {key!r}")
                index = matches[0]
            else:
                raise TypeError("rename_channels keys must be integers or strings")
            if index in resolved:
                raise ValueError(f"Duplicate channel rename mapping for index {index}")
            resolved[index] = new_label
            new_labels[index] = new_label
        if len(set(new_labels)) != len(new_labels):
            raise ValueError(f"Duplicate channel label after rename: {new_labels}")
        descriptors = self._borrowed_channel_metadata_descriptors()
        for descriptor, new_label in zip(descriptors, new_labels, strict=True):
            descriptor["label"] = new_label
        return self._create_new_instance(
            self._data,
            channel_metadata=descriptors,
            channel_ids=self._channel_ids,
            lineage=self._required_semantic_lineage(),
        )

    def _channel_indices_from_query(self, query: QueryType, validate_keys: bool) -> list[int]:
        """Resolve a public metadata query to ordered channel indices."""
        if isinstance(query, str):
            return [index for index, channel in enumerate(self.channels) if channel.label == query]
        if isinstance(query, Pattern):
            return [index for index, channel in enumerate(self.channels) if query.search(channel.label)]
        if callable(query):
            predicate = cast(Callable[[ChannelMetadata], bool], query)
            return [index for index, channel in enumerate(self.channels) if predicate(channel)]
        if not isinstance(query, Mapping):
            raise TypeError(f"Unsupported query type: {type(query).__name__}")
        if validate_keys:
            known_keys = set(ChannelMetadata._MODEL_FIELDS)
            for channel in self.channels:
                known_keys.update(channel.extra)
            unknown_keys = [key for key in query if key not in known_keys]
            if unknown_keys:
                raise KeyError("Unknown channel metadata key(s): " + ", ".join(map(str, unknown_keys)))
        return [index for index, channel in enumerate(self.channels) if channel.matches_query(dict(query))]

    def _channel_indices(self, selector: Any) -> list[int]:
        """Normalize and validate one channel selector as an index list."""
        if isinstance(selector, numbers.Integral) and not isinstance(selector, bool | np.bool_):
            index = int(selector)
            if index < -self.n_channels or index >= self.n_channels:
                raise IndexError(f"Channel index out of range: {index}")
            return [index]
        if isinstance(selector, str):
            return [self.label2index(selector)]
        if isinstance(selector, slice):
            return list(range(self.n_channels))[selector]
        if isinstance(selector, np.ndarray):
            if selector.ndim != 1:
                raise ValueError(f"Channel selector must be 1-D, got shape {selector.shape}")
            if np.issubdtype(selector.dtype, np.bool_):
                if len(selector) != self.n_channels:
                    raise ValueError(
                        f"Boolean mask length {len(selector)} does not match number of channels {self.n_channels}"
                    )
                return [int(index) for index in np.flatnonzero(selector)]
            if np.issubdtype(selector.dtype, np.integer):
                return self._channel_indices(selector.tolist())
            raise TypeError(f"NumPy selector must have integer or boolean dtype, got {selector.dtype}")
        if isinstance(selector, tuple):
            selector = list(selector)
        if isinstance(selector, list):
            if not selector:
                raise ValueError("Cannot index with an empty list")
            if all(isinstance(item, str) for item in selector):
                return [self.label2index(item) for item in selector]
            if all(isinstance(item, numbers.Integral) and not isinstance(item, bool | np.bool_) for item in selector):
                indices = [int(item) for item in selector]
                for index in indices:
                    if index < -self.n_channels or index >= self.n_channels:
                        raise IndexError(f"Channel index out of range: {index}")
                return indices
            raise TypeError(f"Channel list contains mixed or unsupported values: {selector!r}")
        raise TypeError(f"Invalid channel selector type: {type(selector).__name__}")

    def _select_channels(self: S, indices: list[int], lineage: LineageNode) -> S:
        """Create a channel subset that preserves metadata, offsets, and lineage."""
        return self._create_new_instance(
            data=self._data[indices],
            channel_metadata=self._borrowed_channel_metadata_descriptors(indices),
            channel_ids=self._channel_ids_for_selection(indices),
            source_time_offset=self.source_time_offset[indices],
            lineage=lineage,
        )

    @recipe_operation(
        "wandas.frame.get_channel",
        capture=_capture_get_channel,
        handler=_apply_get_channel_recipe,
    )
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

        if query is not None:
            indices = self._channel_indices_from_query(query, validate_query_keys)
            if not indices:
                raise KeyError(f"No channels match query: {query!r}")
        else:
            if channel_idx is None:
                raise TypeError("Either 'channel_idx' or 'query' must be provided.")
            indices = self._channel_indices(channel_idx)
        return self._select_channels(indices, self._required_semantic_lineage())

    def __len__(self) -> int:
        """
        Returns the number of channels.
        """
        return len(self._channel_metadata)

    def __iter__(self: S) -> Iterator[S]:
        """Yield immutable single-channel Frame selections in channel order."""
        for idx in range(len(self)):
            yield self[idx]

    @recipe_operation(
        "wandas.frame.index",
        capture=_capture_index,
        handler=_apply_index_recipe,
    )
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
        - Multidimensional indexing: `frame[0, 100:200]` (channel + axis slice)

        Parameters
        ----------
        key : int, str, slice, list, tuple, or ndarray
            - int: Single channel index (supports negative indexing)
            - str: Single channel label
            - slice: Range of channels
            - list[int]: Multiple channel indices
            - list[str]: Multiple channel labels
            - tuple: A channel selector followed by slices for semantic axes such
              as frequency or time
            - ndarray[int]: NumPy array of channel indices
            - ndarray[bool]: Boolean mask for channel selection

        Returns
        -------
        S
            New instance containing the selected channel(s).

        Raises
        ------
        ValueError
            If the key length is invalid for the shape, a non-channel selector
            is not a slice, a time slice is stepped or reversed, or a boolean mask
            length doesn't match the channels.
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
        >>> frame[[0, 1], 100:200]  # Channels 0-1, continuous sample slice
        """
        if isinstance(key, tuple) and len(key) == 1:
            key = key[0]
        lineage = self._required_semantic_lineage()
        if isinstance(key, tuple):
            return self._handle_multidim_indexing(cast(tuple[Any, ...], key))
        return self._select_channels(self._channel_indices(key), lineage)

    def _handle_multidim_indexing(self: S, key: tuple[Any, ...]) -> S:
        """Handle rank-preserving channel and non-channel axis selection.

        Parameters
        ----------
        key : tuple
            The first element selects channels. Each remaining element must be a
            slice along the corresponding semantic axis, such as frequency or time.

        Returns
        -------
        S
            New instance with the selected channels and axis ranges.

        Raises
        ------
        ValueError
            If the key length exceeds the data dimensions, a non-channel selector
            is not a slice, or a time-axis slice is stepped or reversed.
        """
        if len(key) > self._data.ndim:
            raise ValueError(f"Invalid key length: {len(key)} for shape {self.shape}")

        indices = self._channel_indices(key[0])
        axis_selectors = key[1:]
        selected_data = self._data[indices]
        source_time_offset = self.source_time_offset[indices]
        if axis_selectors:
            if not all(isinstance(selector, slice) for selector in axis_selectors):
                raise ValueError(
                    "Only slice selectors on non-channel axes are supported; "
                    "use a one-element slice for point selection"
                )
            time_slice_context = self._source_time_slice_context(axis_selectors)
            if time_slice_context is not None:
                time_axis_key, time_axis_size, time_step = time_slice_context
                if time_axis_key.step not in (None, 1):
                    raise ValueError("Only continuous forward slicing on the time axis is supported")
                start, _, _ = time_axis_key.indices(time_axis_size)
                source_time_offset = source_time_offset + start * time_step
            selected_data = selected_data[(slice(None),) + axis_selectors]  # noqa: RUF005
        return self._create_new_instance(
            data=selected_data,
            channel_metadata=self._borrowed_channel_metadata_descriptors(indices),
            channel_ids=self._channel_ids_for_selection(indices),
            source_time_offset=source_time_offset,
            lineage=self._required_semantic_lineage(),
        )

    def _channel_selector_for_lineage(self, key: Any) -> dict[str, Any] | None:
        """Encode the channel part of a multidimensional index when portable."""
        if isinstance(key, numbers.Integral) and not isinstance(key, bool | np.bool_):
            return {"indexing": "integer", "index": int(key)}
        if isinstance(key, str):
            return {"indexing": "label", "label": key}
        if isinstance(key, slice):
            bounds = self._slice_for_lineage(key)
            return None if bounds is None else {"indexing": "channel_slice", **bounds}
        if isinstance(key, list) and key and all(isinstance(item, str) for item in key):
            return {"indexing": "label_list", "labels": tuple(key)}
        if (
            isinstance(key, list)
            and key
            and all(isinstance(item, numbers.Integral) and not isinstance(item, bool | np.bool_) for item in key)
        ):
            return {"indexing": "integer_list", "indices": tuple(int(item) for item in key)}
        if isinstance(key, np.ndarray) and key.ndim == 1 and np.issubdtype(key.dtype, np.bool_):
            return {"indexing": "boolean_mask", "mask": tuple(bool(item) for item in key.tolist())}
        if isinstance(key, np.ndarray) and key.ndim == 1 and np.issubdtype(key.dtype, np.integer):
            return {"indexing": "integer_array", "indices": tuple(int(item) for item in key.tolist())}
        return None

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
        """Return data shape with the singleton channel dimension suppressed."""
        _shape: tuple[int, ...] = self._data.shape
        if _shape[0] == 1:
            return _shape[1:]
        return _shape

    @property
    def data(self) -> T:
        """Return the frame's calibrated values as a NumPy array.

        Channel calibration factors are applied automatically. A single-channel
        frame returns an array without the singleton channel axis; multichannel
        frames preserve the channel axis.
        """
        data = self._compute()
        if self.n_channels == 1:
            return cast(T, data.squeeze(axis=0))
        return data

    @property
    def labels(self) -> list[str]:
        """Get a list of all channel labels."""
        return [ch.label for ch in self.channels]

    def _compute(self) -> T:
        """Return calibrated values while preserving every frame dimension.

        This private materialization boundary is for internal code that requires
        the channel-first representation, including its singleton channel axis.

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
        result = self._effective_data.compute()

        if not isinstance(result, np.ndarray):
            raise ValueError(f"Computed result is not a np.ndarray: {type(result)}")

        logger.debug(f"Computation complete, result shape: {result.shape}")
        return cast(T, result)

    @abstractmethod
    def plot(self, plot_type: str = "default", ax: "Axes | None" = None, **kwargs: Any) -> "Axes | Iterator[Axes]":
        """Plot the data"""

    def save(
        self,
        path: str | Path,
        *,
        compress: str | None = "gzip",
        overwrite: bool = False,
    ) -> None:
        """Save this exact built-in Frame type as a WDF 0.4 artifact.

        WDF stores the raw tensor together with the constructor state, semantic
        dimensions, channel calibration, metadata, and display history needed to
        reconstruct the same Frame type. Runtime lineage and the Dask task graph are
        intentionally outside the persistence boundary.

        Args:
            path: Destination path. The ``.wdf`` suffix is appended when absent.
            compress: HDF5 dataset compression filter, or ``None`` for no
                compression.
            overwrite: Replace an existing artifact when true.

        Raises:
            FileExistsError: If the destination exists and ``overwrite`` is false.
            TypeError: If this is not an exact supported built-in Frame type.
            ValueError: If Frame state cannot be represented by the current schema.
        """
        from wandas.io.wdf_io import save as wdf_save

        wdf_save(
            self,
            path,
            compress=compress,
            overwrite=overwrite,
        )

    def _get_additional_init_kwargs(self) -> dict[str, Any]:
        """Return additional keyword arguments for ``_create_new_instance``.

        Subclasses that require extra constructor parameters (e.g. ``n_fft``,
        ``hop_length``) should override this method.  The default returns an
        empty dict, which is correct for frames with no extra init args
        (e.g. ``ChannelFrame``).
        """
        return {}

    def _create_new_instance(self: S, data: DaArray, **kwargs: Any) -> S:
        """Reconstruct this Frame type around new lazy data.

        Keyword arguments override copied Frame state. Subclass constructor state is
        supplied by :meth:`_get_additional_init_kwargs`, and compatible represented
        xarray dimension coordinates are restored after construction.
        """

        sampling_rate = kwargs.pop("sampling_rate", self.sampling_rate)

        label = kwargs.pop("label", self.label)
        if label is not None and not isinstance(label, str):
            raise TypeError("Label must be a string or None")

        metadata = kwargs.pop("metadata") if "metadata" in kwargs else self.metadata
        if not isinstance(metadata, dict):
            raise TypeError("Metadata must be a dictionary")

        lineage = kwargs.pop("lineage", self.lineage)

        channel_metadata = (
            kwargs.pop("channel_metadata")
            if "channel_metadata" in kwargs
            else self._borrowed_channel_metadata_descriptors()
        )
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
        result = type(self)(**init_kwargs)

        # Constructors create canonical xarray dimensions and coordinates. Preserve
        # a represented axis (for example, a sliced quefrency axis) only when it is a
        # one-dimensional coordinate attached to the same dimension and the new
        # tensor kept that dimension's size. Channel coordinates are rebuilt from
        # channel metadata above and must not be overwritten here.
        for dim in self._xr.dims:
            if (
                dim != self._CHANNEL_DIM
                and dim in self._xr.coords
                and dim in result._xr.dims
                and int(result._xr.sizes[dim]) == int(self._xr.sizes[dim])
            ):
                coordinate = self._xr.coords[dim]
                if coordinate.dims == (dim,):
                    result._xr = result._xr.assign_coords({dim: (dim, coordinate.values.copy())})
        return result

    def _metadata_after_analysis(
        self,
        channel_metadata: Sequence[ChannelMetadata] | None = None,
    ) -> list[dict[str, Any]]:
        """Describe channel metadata after consuming each calibration factor."""
        source = self.channels if channel_metadata is None else channel_metadata
        result: list[dict[str, Any]] = []
        for channel in source:
            result.append(
                {
                    "label": channel.label,
                    "calibration": ChannelCalibration(
                        factor=1.0,
                        unit=channel.unit,
                        ref=channel.ref,
                    ),
                    "extra": channel.extra,
                }
            )
        return result

    def __array__(self, dtype: npt.DTypeLike = None, copy: bool | None = None) -> NDArrayReal:
        """Implicit conversion to NumPy array"""
        if copy is False:
            raise ValueError("A Dask-backed Frame cannot provide a zero-copy NumPy array.")

        result = self.data
        if dtype is not None:
            result = result.astype(dtype, copy=copy is True)
        elif copy is True:
            result = result.copy()
        return cast(NDArrayReal, result)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any) -> Any:
        """Handle NumPy scalar-left operators without forcing eager arrays."""
        if method == "__call__" and len(inputs) == 2 and inputs[1] is self and isinstance(inputs[0], np.generic):
            reverse_method_name = self._array_ufunc_reverse_methods.get(ufunc.__name__)
            if reverse_method_name is not None:
                result = getattr(self, reverse_method_name)(inputs[0])
                return result
        array_inputs = tuple(np.asarray(input_value) if input_value is self else input_value for input_value in inputs)
        return getattr(ufunc, method)(*array_inputs, **kwargs)

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
            return self._effective_data.visualize(filename=filename)
        except Exception as e:
            logger.warning(f"Failed to visualize the graph: {e}")
            return None

    def _binary_op(
        self: S,
        other: S | int | float | complex | NDArrayReal | DaArray,
        op: Callable[[DaArray, Any], DaArray],
        symbol: str,
    ) -> S:
        """Apply a forward lazy binary operation through the shared implementation."""
        return self._binary_operand_op(other, op, symbol, reverse=False)

    def _binary_operand_op(
        self: S,
        other: S | int | float | complex | NDArrayReal | DaArray,
        op: Callable[[Any, Any], Any],
        symbol: str,
        *,
        reverse: bool = False,
    ) -> S:
        """Default implementation of binary operations using dask's lazy evaluation.

        Handles both frame-frame and frame-scalar/array operations with
        metadata propagation and runtime lineage tracking. Frame-frame operations are
        index-wise: they combine current array positions without using the right
        operand's coordinates for alignment or relabeling. They do not compare
        ``source_time_offset`` values and do not perform source-time alignment,
        trimming, or padding. Results preserve the left operand's source-time offset
        and compatible dimension coordinates through ``_create_new_instance``. Uses
        ``_create_new_instance`` so that subclass-specific constructor parameters are
        automatically forwarded.

        Subclasses may override this entirely (e.g. ``RoughnessFrame``).
        """
        logger.debug(f"Setting up {symbol} operation (lazy)")

        metadata = self.metadata
        if isinstance(other, BaseFrame):
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
            if self._data.shape != other._data.shape or self._xr.dims != other._xr.dims:
                raise ValueError(
                    f"Frame shape mismatch\n"
                    f"  Left operand: {self._data.shape} with axes {self._xr.dims}\n"
                    f"  Right operand: {other._data.shape} with axes {other._xr.dims}\n"
                    f"Binary frame operations require identical semantic shapes."
                )
            result_data = op(self._effective_data, other._effective_data)
            other_str = other.label
            other_labels = other.labels
        else:
            result_data = op(other, self._effective_data) if reverse else op(self._effective_data, other)
            other_str = self._format_operand_str(other)
            other_labels = [other_str] * self.n_channels

        # Build borrowed constructor descriptors and consume calibration once.
        new_channel_metadata = self._metadata_after_analysis()
        for descriptor, self_ch, other_label in zip(
            new_channel_metadata,
            self.channels,
            other_labels,
            strict=True,
        ):
            if reverse and not isinstance(other, BaseFrame):
                descriptor["label"] = f"({other_label} {symbol} {self_ch.label})"
            else:
                descriptor["label"] = f"({self_ch.label} {symbol} {other_label})"

        label = (
            f"({other_str} {symbol} {self.label})"
            if reverse and not isinstance(other, BaseFrame)
            else f"({self.label} {symbol} {other_str})"
        )
        return self._create_new_instance(
            data=result_data,
            label=label,
            metadata=metadata,
            lineage=self._required_semantic_lineage(),
            channel_metadata=new_channel_metadata,
        )

    @staticmethod
    def _is_supported_reverse_scalar(value: object) -> bool:
        """Return whether base reverse arithmetic accepts this scalar."""
        return isinstance(value, numbers.Real) and not isinstance(value, bool)

    def _supports_base_reverse_scalar_op(self) -> bool:
        """Return whether the concrete class retains BaseFrame binary semantics."""
        return type(self)._binary_op is BaseFrame._binary_op

    @staticmethod
    def _format_operand_str(other: object) -> str:
        """Return a short display string for a binary operand."""
        if isinstance(other, int | float):
            return str(other)
        if isinstance(other, np.bool_):
            return str(bool(other))
        if isinstance(other, complex):
            return f"complex({other.real}, {other.imag})"
        if isinstance(other, np.ndarray):
            return f"ndarray{other.shape}"
        if hasattr(other, "shape"):
            return f"dask.array{other.shape}"
        return str(type(other).__name__)

    @recipe_operation(
        "wandas.operator.add",
        binding_patterns=_FORWARD_BINARY_PATTERNS,
        capture=_capture_binary,
        handler=_binary_recipe_handler("__add__"),
    )
    def __add__(self: S, other: S | int | float | complex | NDArrayReal) -> S:
        """Addition operator"""
        return self._binary_op(other, lambda x, y: x + y, "+")

    @recipe_operation(
        "wandas.operator.subtract",
        binding_patterns=_FORWARD_BINARY_PATTERNS,
        capture=_capture_binary,
        handler=_binary_recipe_handler("__sub__"),
    )
    def __sub__(self: S, other: S | int | float | complex | NDArrayReal) -> S:
        """Subtraction operator"""
        return self._binary_op(other, lambda x, y: x - y, "-")

    @recipe_operation(
        "wandas.operator.multiply",
        binding_patterns=_FORWARD_BINARY_PATTERNS,
        capture=_capture_binary,
        handler=_binary_recipe_handler("__mul__"),
    )
    def __mul__(self: S, other: S | int | float | complex | NDArrayReal) -> S:
        """Multiplication operator"""
        return self._binary_op(other, lambda x, y: x * y, "*")

    @recipe_operation(
        "wandas.operator.divide",
        binding_patterns=_FORWARD_BINARY_PATTERNS,
        capture=_capture_binary,
        handler=_binary_recipe_handler("__truediv__"),
    )
    def __truediv__(self: S, other: S | int | float | complex | NDArrayReal) -> S:
        """Division operator"""
        return self._binary_op(other, lambda x, y: x / y, "/")

    @recipe_operation(
        "wandas.operator.power",
        binding_patterns=_FORWARD_BINARY_PATTERNS,
        capture=_capture_binary,
        handler=_binary_recipe_handler("__pow__"),
    )
    def __pow__(self: S, other: S | int | float | complex | NDArrayReal) -> S:
        """Power operator"""
        return self._binary_op(other, lambda x, y: x**y, "**")

    @recipe_operation(
        "wandas.operator.reverse_add",
        binding_patterns=_REVERSE_BINARY_PATTERNS,
        capture=_capture_reverse_binary,
        handler=_binary_recipe_handler("__radd__"),
    )
    def __radd__(self: S, other: int | float) -> S:
        """Reverse addition operator."""
        if not self._is_supported_reverse_scalar(other) or not self._supports_base_reverse_scalar_op():
            return NotImplemented
        return self._binary_operand_op(other, lambda x, y: x + y, "+", reverse=True)

    @recipe_operation(
        "wandas.operator.reverse_subtract",
        binding_patterns=_REVERSE_BINARY_PATTERNS,
        capture=_capture_reverse_binary,
        handler=_binary_recipe_handler("__rsub__"),
    )
    def __rsub__(self: S, other: int | float) -> S:
        """Reverse subtraction operator."""
        if not self._is_supported_reverse_scalar(other) or not self._supports_base_reverse_scalar_op():
            return NotImplemented
        return self._binary_operand_op(other, lambda x, y: x - y, "-", reverse=True)

    @recipe_operation(
        "wandas.operator.reverse_multiply",
        binding_patterns=_REVERSE_BINARY_PATTERNS,
        capture=_capture_reverse_binary,
        handler=_binary_recipe_handler("__rmul__"),
    )
    def __rmul__(self: S, other: int | float) -> S:
        """Reverse multiplication operator."""
        if not self._is_supported_reverse_scalar(other) or not self._supports_base_reverse_scalar_op():
            return NotImplemented
        return self._binary_operand_op(other, lambda x, y: x * y, "*", reverse=True)

    @recipe_operation(
        "wandas.operator.reverse_divide",
        binding_patterns=_REVERSE_BINARY_PATTERNS,
        capture=_capture_reverse_binary,
        handler=_binary_recipe_handler("__rtruediv__"),
    )
    def __rtruediv__(self: S, other: int | float) -> S:
        """Reverse division operator."""
        if not self._is_supported_reverse_scalar(other) or not self._supports_base_reverse_scalar_op():
            return NotImplemented
        return self._binary_operand_op(other, lambda x, y: x / y, "/", reverse=True)

    @recipe_operation(
        "wandas.operator.reverse_power",
        binding_patterns=_REVERSE_BINARY_PATTERNS,
        capture=_capture_reverse_binary,
        handler=_binary_recipe_handler("__rpow__"),
    )
    def __rpow__(self: S, other: int | float) -> S:
        """Reverse power operator."""
        if not self._is_supported_reverse_scalar(other) or not self._supports_base_reverse_scalar_op():
            return NotImplemented
        return self._binary_operand_op(other, lambda x, y: x**y, "**", reverse=True)

    def _apply_named_operation(self: S, operation_name: str, **params: Any) -> S:
        """Execute a numerical operation inside a decorated public boundary."""
        self._required_semantic_lineage()
        return self._apply_operation_impl(operation_name, **params)

    def _updated_metadata(
        self,
        operation_name: str,
        params: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return frame metadata for a derived frame.

        Operation parameters are owned by runtime lineage. Frame metadata only
        carries user/domain metadata; the receiving Frame constructor takes its
        one defensive ownership copy.
        """
        return self.metadata

    def _apply_operation_impl(self: S, operation_name: str, **params: Any) -> S:
        """Default implementation of operation application.

        Creates the named operation, applies it to the data, and returns
        a new frame with updated metadata and runtime lineage.
        Derived classes may override this to add extra behaviour
        (e.g. channel relabelling).
        """
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")
        from wandas.processing import create_operation

        operation = create_operation(operation_name, self.sampling_rate, **params)
        ensure_dependencies = getattr(operation, "ensure_dependencies", None)
        if ensure_dependencies is not None:
            ensure_dependencies()
        processed_data = operation.process(self._effective_data)

        new_metadata = self._updated_metadata(operation_name, params)

        creation_params: dict[str, Any] = {
            "data": processed_data,
            "metadata": new_metadata,
            "lineage": self._required_semantic_lineage(),
            "channel_metadata": self._metadata_after_analysis(),
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
        runtime lineage, and channel labels atomically.  It is the
        entry-point used by ``ChannelProcessingMixin`` and by
        ``ChannelFrame._apply_operation_impl``.

        Parameters
        ----------
        operation : AudioOperation
            Instantiated operation to apply.
        operation_name : str, optional
            Numerical operation name used for metadata and display handling. The
            enclosing ``@recipe_operation`` declaration owns the history operation ID.
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
                "  Got: this helper provides one frame input\n"
                "Use an operation-specific method that can pass all runtime inputs."
            )

        ensure_dependencies = getattr(operation, "ensure_dependencies", None)
        if ensure_dependencies is not None:
            ensure_dependencies()
        processed_data = operation.process(self._effective_data)

        params = getattr(operation, "params", {})

        new_metadata = self._updated_metadata(operation_name, params)
        lineage = self._required_semantic_lineage()

        metadata_updates = operation.get_metadata_updates()
        if operation_name == "trim":
            start_sample = int(float(params.get("start", 0.0)) * self.sampling_rate)
            metadata_updates["source_time_offset"] = self.source_time_offset + start_sample / self.sampling_rate

        display = operation.get_display_name() or operation_name
        new_channel_metadata = self._metadata_after_analysis()
        for descriptor, channel in zip(new_channel_metadata, self.channels, strict=True):
            descriptor["label"] = f"{display}({channel.label})"

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
        try:
            effective_data = self._effective_data
            logger.debug(f"Dask graph layers: {list(effective_data.dask.layers.keys())}")
            logger.debug(f"Dask graph dependencies: {len(effective_data.dask.dependencies)}")
        except Exception as e:
            logger.debug(f"Dask graph details unavailable: {e}")
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

        This method is equivalent to accessing :attr:`data`.

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
