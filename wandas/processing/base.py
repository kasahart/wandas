import copy
import importlib
import inspect
import logging
from collections.abc import Iterator, Mapping, MutableMapping
from typing import Any, ClassVar, Generic, TypeVar, cast

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


def _snapshot_config_value(value: Any) -> Any:
    """Return an operation-owned snapshot of user supplied config values."""
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, Mapping):
        return {key: _snapshot_config_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot_config_value(item) for item in value)
    if isinstance(value, list):
        return [_snapshot_config_value(item) for item in value]
    if isinstance(value, set | frozenset):
        return type(value)(_snapshot_config_value(item) for item in value)
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _is_public_config_attr(
    name: str,
    value: Any,
    params: Mapping[str, Any],
    grouped_config_attrs: frozenset[str],
) -> bool:
    """Return whether a public attribute mirrors captured operation config."""
    if name in params:
        return True
    return name in grouped_config_attrs and isinstance(value, Mapping)


def _config_values_equal(left: Any, right: Any) -> bool:
    """Compare captured config values without NumPy's ambiguous bool errors."""
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
    if isinstance(result, np.ndarray):
        return bool(np.all(result))
    return bool(result)


class _ParamsView(MutableMapping[str, Any]):
    def __init__(self, operation: "AudioOperation[Any, Any]") -> None:
        self._operation = operation

    def _params(self) -> dict[str, Any]:
        return object.__getattribute__(self._operation, "_params")

    def __getitem__(self, key: str) -> Any:
        return _snapshot_config_value(self._params()[key])

    def __setitem__(self, key: str, value: Any) -> None:
        self._params()[key] = _snapshot_config_value(value)
        self._operation._snapshot_public_config_attributes()

    def __delitem__(self, key: str) -> None:
        del self._params()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._params())

    def __len__(self) -> int:
        return len(self._params())

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
    metadata. ``params`` and public mutable config attributes are exposed as
    defensive snapshots. Public attributes remain ordinary Python attributes for
    custom operation compatibility, so reflective/manual mutation is outside the
    public lineage contract.
    """

    # Class variable: operation name
    name: ClassVar[str]

    # Optional attributes used by some subclasses (e.g., FFT)
    n_fft: int | None
    window: str
    _grouped_config_attrs: ClassVar[frozenset[str]] = frozenset()

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
        object.__setattr__(self, "_wandas_initialized", False)
        self.sampling_rate = sampling_rate
        self.pure = pure
        self._params = {key: _snapshot_config_value(value) for key, value in params.items()}

        # Validate parameters during initialization
        self.validate_params()

        # Create processor function (lazy initialization possible)
        self._setup_processor()

        self._snapshot_public_config_attributes()

        logger.debug(f"Initialized {self.__class__.__name__} operation with params: {params}")
        object.__setattr__(self, "_wandas_initialized", True)

    def _snapshot_public_config_attributes(self) -> None:
        """Replace public mutable config attrs with operation-owned snapshots."""
        params = object.__getattribute__(self, "_params")
        grouped_config_attrs = object.__getattribute__(self, "_grouped_config_attrs")
        for name, value in list(object.__getattribute__(self, "__dict__").items()):
            if name.startswith("_") or not _is_public_config_attr(name, value, params, grouped_config_attrs):
                continue
            object.__setattr__(self, name, _snapshot_config_value(value))

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_wandas_initialized", False) and not name.startswith("_"):
            try:
                params = object.__getattribute__(self, "_params")
            except AttributeError:
                params = {}
            grouped_config_attrs = object.__getattribute__(self, "_grouped_config_attrs")
            if _is_public_config_attr(name, value, params, grouped_config_attrs):
                if name in object.__getattribute__(self, "__dict__"):
                    return
                value = _snapshot_config_value(value)
        object.__setattr__(self, name, value)

    def __getattribute__(self, name: str) -> Any:
        value = object.__getattribute__(self, name)
        if name.startswith("_"):
            return value
        try:
            initialized = object.__getattribute__(self, "_wandas_initialized")
        except AttributeError:
            return value
        if not initialized:
            return value
        try:
            params = object.__getattribute__(self, "_params")
        except AttributeError:
            return value
        grouped_config_attrs = object.__getattribute__(self, "_grouped_config_attrs")
        if _is_public_config_attr(name, value, params, grouped_config_attrs):
            return _snapshot_config_value(value)
        return value

    @property
    def params(self) -> _ParamsView:
        """Return a write-through view of operation parameters."""
        return _ParamsView(self)

    @params.setter
    def params(self, value: Mapping[str, Any]) -> None:
        """Set captured operation parameters for custom operation compatibility."""
        if not isinstance(value, Mapping):
            raise TypeError("Operation params must be a mapping")
        object.__setattr__(self, "_params", {key: _snapshot_config_value(item) for key, item in value.items()})
        try:
            initialized = object.__getattribute__(self, "_wandas_initialized")
        except AttributeError:
            initialized = False
        if initialized:
            self._snapshot_public_config_attributes()

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

    def _process_array(self, x: InputArrayType) -> OutputArrayType:
        """Processing function (implemented by subclasses)"""
        # Default is no-op function
        raise NotImplementedError("Subclasses must implement this method.")

    def _create_named_wrapper(self) -> Any:
        """
        Create a named wrapper function for better Dask graph visualization.

        Returns
        -------
        callable
            A wrapper function with the operation name set as __name__.
        """

        def operation_wrapper(x: InputArrayType) -> OutputArrayType:
            return self._process_array(x)

        # Set the function name to the operation name for better visualization
        operation_wrapper.__name__ = self.name
        return operation_wrapper

    def _delayed(self, data: Any) -> Any:
        """Create a ``dask.delayed`` result for *data* using the named wrapper."""
        wrapper = self._create_named_wrapper()
        return delayed(wrapper, pure=self.pure)(data)

    def process_array(self, x: Any) -> Any:
        """
        Processing function wrapped with @dask.delayed.

        This method returns a Delayed object that can be computed later.
        The operation name is used in the Dask task graph for better visualization.

        Parameters
        ----------
        x : InputArrayType
            Input array to process.

        Returns
        -------
        dask.delayed.Delayed
            A Delayed object representing the computation.
        """
        logger.debug(f"Creating delayed operation on data with shape: {x.shape}")
        return self._delayed(x)

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

    def process(self, data: DaArray) -> DaArray:
        """
        Execute operation and return result
        data shape is (channels, samples)
        """
        logger.debug("Adding delayed operation to computation graph")
        delayed_result = self._delayed(data)
        output_shape = self.calculate_output_shape(data.shape)
        return _da_from_delayed(delayed_result, shape=output_shape, dtype=data.dtype)


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
