import copy
import importlib
import inspect
import logging
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from enum import Enum
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


class _ExecutionStrategy(Enum):
    """Internal lazy graph strategy for an ``AudioOperation`` kernel."""

    WHOLE_FRAME = "whole-frame"
    CHANNEL_WISE = "channel-wise"


def _execute_wandas_operation(operation: "AudioOperation[Any, Any]", *inputs: Any) -> Any:
    """Execute a Wandas operation from a Dask task."""
    return operation._process(*inputs)


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


class _DefensiveParamsMapping(Mapping[str, Any]):
    """Own parameter snapshots and return a fresh defensive value on every read."""

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
    """Numerical Dask operation with an operation-owned config snapshot."""

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
    def params(self) -> _DefensiveParamsMapping:
        """Return a read-only defensive snapshot of operation parameters."""
        return _DefensiveParamsMapping(self.to_params())

    def to_params(self) -> Mapping[str, Any]:
        """Return operation parameters used for lineage and display."""
        return self._config_snapshot()

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

    def _execution_strategy(self) -> _ExecutionStrategy:
        """Select the internal lazy graph strategy for this operation instance."""
        return _ExecutionStrategy.WHOLE_FRAME

    def _build_whole_frame_graph(
        self,
        data: DaArray,
        inputs: tuple[DaArray, ...],
        *,
        output_shape: tuple[int, ...],
        output_dtype: np.dtype[Any],
    ) -> DaArray:
        """Wrap the complete channel-first tensor in one delayed kernel call."""
        delayed_result = delayed(_execute_wandas_operation, name=self.name, pure=self.pure)(self, data, *inputs)
        return _da_from_delayed(delayed_result, shape=output_shape, dtype=output_dtype)

    def _build_channel_wise_graph(
        self,
        data: DaArray,
        *,
        output_shape: tuple[int, ...],
        output_dtype: np.dtype[Any],
    ) -> DaArray:
        """Build one delayed kernel call for each complete input channel."""
        channel_count = int(data.shape[0])
        if channel_count <= 0:
            raise ValueError("Channel-wise execution requires at least one channel")
        if not output_shape or int(output_shape[0]) != channel_count:
            raise ValueError(
                "Channel-wise execution must preserve the channel axis\n"
                f"  Input channels: {channel_count}\n"
                f"  Output shape: {output_shape}\n"
                "Use whole-frame execution for channel reductions or expansions."
            )

        per_channel_shape = (1, *output_shape[1:])
        channel_results = []
        for channel_index in range(channel_count):
            channel_data = data[channel_index : channel_index + 1]
            delayed_result = delayed(
                _execute_wandas_operation,
                name=f"{self.name}-channel",
                pure=self.pure,
            )(self, channel_data)
            channel_results.append(
                _da_from_delayed(
                    delayed_result,
                    shape=per_channel_shape,
                    dtype=output_dtype,
                )
            )
        return da.concatenate(channel_results, axis=0)

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
        output_shape = self.calculate_output_shape(data.shape)
        output_dtype = self.calculate_output_dtype(data.dtype, *(input_data.dtype for input_data in inputs))
        strategy = self._execution_strategy()
        if strategy is _ExecutionStrategy.CHANNEL_WISE:
            if inputs:
                raise ValueError("Channel-wise execution currently supports unary operations only")
            return self._build_channel_wise_graph(
                data,
                output_shape=output_shape,
                output_dtype=output_dtype,
            )
        if strategy is not _ExecutionStrategy.WHOLE_FRAME:
            raise TypeError(f"Unsupported AudioOperation execution strategy: {strategy!r}")
        return self._build_whole_frame_graph(
            data,
            inputs,
            output_shape=output_shape,
            output_dtype=output_dtype,
        )


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
