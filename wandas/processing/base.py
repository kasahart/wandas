import copy
import importlib
import inspect
import json
import logging
import numbers
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass
from functools import wraps
from typing import Any, ClassVar, Generic, NoReturn, TypeVar, cast

import dask.array as da
import numpy as np
from dask.array.core import Array as DaArray
from dask.delayed import delayed

from wandas.utils.types import NDArrayComplex, NDArrayReal

logger = logging.getLogger(__name__)

_da_from_delayed = da.from_delayed

# Define TypeVars for input and output array types
InputArrayType = TypeVar("InputArrayType", NDArrayReal, NDArrayComplex)
OutputArrayType = TypeVar("OutputArrayType", NDArrayReal, NDArrayComplex)
OperationSummary = dict[str, Any]


def _execute_wandas_operation(operation: "AudioOperation[Any, Any]", *inputs: Any) -> Any:
    """Execute a Wandas operation from a Dask task."""
    return operation._process(*inputs)


def _mark_wandas_operation(data: Any, operation: "AudioOperation[Any, Any]") -> Any:
    """Mark a Dask-native task as a Wandas operation without changing the data."""
    return data


def _validate_channel_first_array(value: Any, label: str, *, ndim: int | None = None) -> None:
    if not hasattr(value, "ndim"):
        return
    if ndim is not None and value.ndim == ndim:
        return
    if ndim is None and value.ndim >= 2:
        return
    expected = (
        f"a Dask array shaped (channels, ...) with ndim={ndim}"
        if ndim is not None
        else "a Dask array shaped (channels, ...)"
    )
    raise ValueError(
        "AudioOperation.process requires channel-first data\n"
        f"  Got: {label} with shape {value.shape}\n"
        f"  Expected: {expected}\n"
        "Use Frame operations or reshape direct lazy inputs to include a channel axis."
    )


def _unimplemented_process(_self: object, *inputs: NDArrayReal | NDArrayComplex) -> NoReturn:
    """Fallback concrete kernel for subclasses that do not implement one."""
    raise NotImplementedError("Subclasses must implement this method.")


@dataclass(frozen=True)
class LineageNode:
    """Serializable computation provenance node.

    ``operation`` is the same operation object used to build the Dask task
    whenever the computation is represented by an ``AudioOperation``. This keeps
    lineage parameters and compute parameters tied to one immutable operation
    snapshot.
    """

    operation: Any
    inputs: tuple["LineageNode", ...] = ()


def _operand_descriptor(value: Any) -> dict[str, Any]:
    if isinstance(value, DaArray):
        return {
            "type": "dask.array",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "chunks": [list(chunk) for chunk in value.chunks],
        }
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, complex):
        return {"type": "complex", "real": value.real, "imag": value.imag}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int | float | str) or value is None:
        return {"type": type(value).__name__, "value": value}
    if hasattr(value, "shape"):
        return {"type": type(value).__name__, "shape": list(value.shape)}
    return {"type": type(value).__name__}


def _summary_sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _summary_value(value: Any) -> Any:
    """Return a lightweight, JSON-safe value for display summaries."""
    if callable(value):
        return {"type": "callable", "name": getattr(value, "__qualname__", type(value).__name__)}
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, np.timedelta64 | np.datetime64):
        return {"type": type(value).__name__}
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Rational):
        return {"type": type(value).__name__}
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
            "real": _summary_value(value.real),
            "imag": _summary_value(value.imag),
        }
    if isinstance(value, Mapping):
        return {str(key): _summary_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_summary_value(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted((_summary_value(item) for item in value), key=_summary_sort_key)
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": [_summary_value(item) for item in value.shape], "dtype": str(value.dtype)}
    if isinstance(value, DaArray):
        return {
            "type": "dask.array",
            "shape": [_summary_value(item) for item in value.shape],
            "dtype": str(value.dtype),
            "chunks": [[_summary_value(item) for item in chunk] for chunk in value.chunks],
        }
    return {"type": type(value).__name__}


@dataclass(frozen=True)
class BinaryOperation:
    """Lightweight operation record for frame binary computations."""

    symbol: str
    operand_kind: str
    operand: Any | None = None
    name: str = "binary_operation"

    @property
    def params(self) -> Mapping[str, Any]:
        return self.to_params()

    def to_params(self) -> Mapping[str, Any]:
        params: dict[str, Any] = {
            "symbol": self.symbol,
            "operand_kind": self.operand_kind,
        }
        if self.operand_kind == "frame" and self.operand is not None:
            params["operand"] = {"type": "frame", "label": str(self.operand)}
        elif self.operand_kind != "frame":
            params["operand"] = _operand_descriptor(self.operand)
        return params

    def to_summary(self) -> OperationSummary:
        """Return a lightweight display summary for this operation."""
        params = self.to_params()
        return {
            "operation": self.symbol,
            "params": {key: _summary_value(value) for key, value in params.items()},
        }


@dataclass(frozen=True)
class FrameMethodOperation:
    """Lightweight lineage record for replayable frame method calls."""

    name: str
    method_params: Mapping[str, Any]

    @property
    def params(self) -> Mapping[str, Any]:
        return self.to_params()

    def to_params(self) -> Mapping[str, Any]:
        return _snapshot_config_value(self.method_params)


def _snapshot_config_value(value: Any) -> Any:
    """Return an operation-owned snapshot of user supplied config values."""
    if value is None or isinstance(value, bool | int | float | str | bytes | complex):
        return value
    if isinstance(value, DaArray):
        return value
    if isinstance(value, np.ndarray):
        try:
            return value.copy()
        except Exception:
            return np.array(value, copy=True, subok=True)
    if isinstance(value, Mapping):
        items = [(key, _snapshot_config_value(item)) for key, item in value.items()]
        if isinstance(value, defaultdict):
            return defaultdict(value.default_factory, items)
        if type(value) is dict:
            return dict(items)
        if isinstance(value, MutableMapping):
            try:
                snapshot = copy.copy(value)
                snapshot.clear()
                snapshot.update(dict(items))
                return snapshot
            except Exception:
                pass
        return dict(items)
    if isinstance(value, tuple):
        items = tuple(_snapshot_config_value(item) for item in value)
        if type(value) is tuple:
            return items
        try:
            return type(value)(*items)
        except TypeError:
            return type(value)(items)
    if isinstance(value, list):
        return [_snapshot_config_value(item) for item in value]
    if isinstance(value, set | frozenset):
        return type(value)(_snapshot_config_value(item) for item in value)
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _config_values_equal(left: Any, right: Any) -> bool:
    """Compare captured config values without NumPy's ambiguous bool errors."""
    if isinstance(left, DaArray) or isinstance(right, DaArray):
        return (
            isinstance(left, DaArray)
            and isinstance(right, DaArray)
            and left.name == right.name
            and left.shape == right.shape
            and left.chunks == right.chunks
            and left.dtype == right.dtype
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return bool(np.array_equal(left, right))
        except Exception:
            return False
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return False
        return all(_config_values_equal(left[key], right[key]) for key in left)
    if isinstance(left, tuple | list) and isinstance(right, tuple | list):
        if type(left) is not type(right) or len(left) != len(right):
            return False
        return all(_config_values_equal(left_item, right_item) for left_item, right_item in zip(left, right))
    try:
        result = left == right
    except Exception:
        return False
    if isinstance(result, DaArray):
        return False
    if isinstance(result, np.ndarray):
        return bool(np.all(result))
    try:
        return bool(result)
    except Exception:
        return False


class _ParamsSnapshot(Mapping[str, Any]):
    def __init__(self, params: Mapping[str, Any]) -> None:
        self._params = {key: _snapshot_config_value(value) for key, value in params.items()}

    def __getitem__(self, key: str) -> Any:
        return _snapshot_config_value(self._params[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._params)

    def __len__(self) -> int:
        return len(self._params)

    def __repr__(self) -> str:
        return repr(dict(self.items()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return False
        other_mapping = cast(Mapping[str, Any], other)
        if set(self) != set(other_mapping):
            return False
        return all(_config_values_equal(self[key], other_mapping[key]) for key in self)

    def copy(self) -> dict[str, Any]:
        return dict(self.items())


class AudioOperation(Generic[InputArrayType, OutputArrayType]):
    """Abstract runtime lineage object for audio processing operations.

    Operation parameters are captured as operation-owned snapshots for lineage
    metadata. ``params`` is a read-only defensive snapshot; create a new
    operation when configuration needs to change. Subclasses can rely on the
    default ``to_params()`` when lineage parameters match constructor
    parameters.
    """

    # Class variable: operation name
    name: ClassVar[str]
    _expected_input_count: ClassVar[int | None] = 1

    _config: dict[str, Any]
    _process: Callable[..., OutputArrayType] = _unimplemented_process

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Ensure subclass ``process`` overrides keep the base input contract."""
        super().__init_subclass__(**kwargs)
        process = cls.__dict__.get("process")
        if process is None or getattr(process, "_wandas_validates_process_inputs", False):
            return

        @wraps(process)
        def validated_process(self: "AudioOperation[Any, Any]", data: Any, *inputs: Any) -> Any:
            self._validate_process_inputs(data, *inputs)
            return process(self, data, *inputs)

        setattr(validated_process, "_wandas_validates_process_inputs", True)
        cls.process = cast(Any, validated_process)

    def __init__(self, sampling_rate: float, *, pure: bool = True, **params: Any):
        """
        Initialize AudioOperation.

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        pure : bool, default=True
            Whether the operation is pure (deterministic with no side effects).
            When True, Dask can cache results for identical inputs.
            Set to False only if the operation has side effects or is non-deterministic.
        **params : Any
            Operation-specific parameters
        """
        object.__setattr__(self, "_sampling_rate", float(sampling_rate))
        object.__setattr__(
            self,
            "_config",
            {key: _snapshot_config_value(value) for key, value in params.items()},
        )
        self.pure = pure

        # Validate parameters during initialization
        self.validate_params()

        # Create processor function (lazy initialization possible)
        self._setup_processor()

        logger.debug(f"Initialized {self.__class__.__name__} operation with params: {params}")

    @property
    def sampling_rate(self) -> float:
        """Sampling rate captured at operation construction time."""
        return object.__getattribute__(self, "_sampling_rate")

    @property
    def params(self) -> _ParamsSnapshot:
        """Return a read-only defensive snapshot of operation parameters."""
        return _ParamsSnapshot(self.to_params())

    def to_params(self) -> Mapping[str, Any]:
        """Return operation parameters used for lineage and display."""
        return self._config_snapshot()

    def to_summary(self) -> OperationSummary:
        """Return a lightweight display summary for this operation."""
        params = self.to_params()
        return {
            "operation": self.name,
            "params": {str(key): _summary_value(value) for key, value in params.items()},
        }

    def _config_snapshot(self) -> dict[str, Any]:
        """Return a defensive copy of base-managed constructor config."""
        return {key: _snapshot_config_value(value) for key, value in self._config.items()}

    def _config_value(self, key: str) -> Any:
        """Return a defensive snapshot for one base-managed config value."""
        return _snapshot_config_value(self._config[key])

    def validate_params(self) -> None:
        """Validate parameters (raises exception if invalid)"""

    def _setup_processor(self) -> None:
        """Set up processor function (implemented by subclasses)"""

    def get_metadata_updates(self) -> dict[str, Any]:
        """
        Get metadata updates to apply after processing.

        This method allows operations to specify how metadata should be
        updated after processing. By default, no metadata is updated.

        Returns
        -------
        dict
            Dictionary of metadata updates. Can include:
            - 'sampling_rate': New sampling rate (float)
            - Other metadata keys as needed

        Examples
        --------
        Return empty dict for operations that don't change metadata:

        >>> return {}

        Return new sampling rate for operations that resample:

        >>> return {"sampling_rate": self.target_sr}

        Notes
        -----
        This method is called by the framework after processing to update
        the frame metadata. Subclasses should override this method if they
        need to update metadata (e.g., changing sampling rate).

        Design principle: Operations should use parameters provided at
        initialization (via __init__). All necessary information should be
        available as instance variables.
        """
        return {}

    def get_display_name(self) -> str | None:
        """
        Get display name for the operation for use in channel labels.

        Returns ``_display`` if the subclass sets it, otherwise ``None``
        (which tells the framework to fall back to the ``name`` class
        variable).  Subclasses with dynamic display names can still
        override this method.
        """
        return getattr(self, "_display", None)

    def _validate_process_input_count(self, input_count: int) -> None:
        """Validate process input arity using the operation class contract."""
        expected = self._expected_input_count
        if expected is None or input_count == expected:
            return

        noun = "input" if expected == 1 else "inputs"
        expected_text = "one" if expected == 1 else str(expected)
        raise ValueError(
            f"Expected exactly {expected_text} {noun} for {self.__class__.__name__}; "
            f"got {input_count}. Use an operation-specific method when multiple "
            "runtime inputs are required."
        )

    def _mark_array(self, data: DaArray) -> DaArray:
        """Attach an explicit operation marker to a Dask-native array result."""
        marker = cast(Any, _mark_wandas_operation)
        return data.map_blocks(marker, self, dtype=data.dtype)

    def _validate_process_inputs(self, data: DaArray, *inputs: DaArray, ndim: int | None = None) -> None:
        """Validate Frame-internal lazy inputs before building a process graph."""
        self._validate_process_input_count(1 + len(inputs))
        _validate_channel_first_array(data, "data", ndim=ndim)
        for index, input_data in enumerate(inputs, start=1):
            _validate_channel_first_array(input_data, f"input {index}")

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Calculate output data shape after operation.

        The default returns *input_shape* unchanged, which is correct for the
        majority of operations (filters, effects, weighting, etc.).
        Subclasses that alter the shape (e.g. FFT, STFT, resampling) **must**
        override this method.

        Parameters
        ----------
        input_shape : tuple
            Input data shape

        Returns
        -------
        tuple
            Output data shape
        """
        return input_shape

    def calculate_output_dtype(self, input_dtype: np.dtype[Any], *input_dtypes: np.dtype[Any]) -> np.dtype[Any]:
        """Calculate output dtype metadata after operation."""
        return np.result_type(input_dtype, *input_dtypes)

    def process(self, data: DaArray, *inputs: DaArray) -> DaArray:
        """
        Execute operation lazily on Frame-internal channel-first Dask arrays.

        ``data`` must be the lazy array held by a Frame, with a leading channel
        axis such as ``(channels, samples)``. Direct 1-D lazy input is not part
        of this API; use a Frame operation or reshape direct lazy inputs to add
        a channel axis before calling ``process()``. Multi-input operations
        pass additional channel-first Dask arrays through ``*inputs``.
        """
        self._validate_process_inputs(data, *inputs)
        logger.debug("Adding delayed operation to computation graph")
        delayed_result = delayed(_execute_wandas_operation, name=self.name, pure=self.pure)(self, data, *inputs)
        output_shape = self.calculate_output_shape(data.shape)
        output_dtype = self.calculate_output_dtype(data.dtype, *(input_data.dtype for input_data in inputs))
        return _da_from_delayed(delayed_result, shape=output_shape, dtype=output_dtype)


# Automatically collect operation types and corresponding classes
_OPERATION_REGISTRY: dict[str, type[AudioOperation[Any, Any]]] = {}
_OPERATION_MODULES: dict[str, str] = {}


def register_operation(operation_class: type) -> None:
    """Register a new operation type"""

    if not issubclass(operation_class, AudioOperation):
        raise TypeError("Strategy class must inherit from AudioOperation.")
    if inspect.isabstract(operation_class):
        raise TypeError("Cannot register abstract AudioOperation class.")

    existing = _OPERATION_REGISTRY.get(operation_class.name)
    if (
        existing is not None
        and existing.__module__ == operation_class.__module__
        and existing.__qualname__ == operation_class.__qualname__
    ):
        return

    _OPERATION_REGISTRY[operation_class.name] = operation_class


def register_lazy_operation(name: str, module_name: str) -> None:
    """Register an operation that can be loaded from *module_name* on demand."""
    _OPERATION_MODULES[name] = module_name


def get_operation(name: str) -> type[AudioOperation[Any, Any]]:
    """Get operation class by name"""
    if name not in _OPERATION_REGISTRY and name in _OPERATION_MODULES:
        importlib.import_module(_OPERATION_MODULES[name])
    if name not in _OPERATION_REGISTRY:
        raise ValueError(f"Unknown operation type: {name}")
    return _OPERATION_REGISTRY[name]


def create_operation(name: str, sampling_rate: float, **params: Any) -> AudioOperation[Any, Any]:
    """Create operation instance from name and parameters"""
    operation_class = get_operation(name)
    return operation_class(sampling_rate, **params)
