import inspect
import logging
from typing import Any, ClassVar, Generic, TypeVar

import dask.array as da
from dask.array.core import Array as DaArray
from dask.delayed import delayed

from wandas.utils.types import NDArrayComplex, NDArrayReal

logger = logging.getLogger(__name__)

_da_from_delayed = da.from_delayed  # type: ignore [unused-ignore]

# Define TypeVars for input and output array types
InputArrayType = TypeVar("InputArrayType", NDArrayReal, NDArrayComplex)
OutputArrayType = TypeVar("OutputArrayType", NDArrayReal, NDArrayComplex)


class AudioOperation(Generic[InputArrayType, OutputArrayType]):
    """Abstract base class for audio processing operations."""

    # Class variable: operation name
    name: ClassVar[str]

    def __init__(self, sampling_rate: float, **params: Any):
        """
        Initialize AudioOperation.

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        **params : Any
            Operation-specific parameters
        """
        self.sampling_rate = sampling_rate
        self.params = params

        # Validate parameters during initialization
        self.validate_params()

        # Create processor function (lazy initialization possible)
        self._setup_processor()

        logger.debug(
            f"Initialized {self.__class__.__name__} operation with params: {params}"
        )

    def validate_params(self) -> None:
        """Validate parameters (raises exception if invalid)"""
        pass

    def _setup_processor(self) -> None:
        """Set up processor function (implemented by subclasses)"""
        pass

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

    def process_array(self, x: InputArrayType) -> Any:
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
        # Create wrapper with operation name and wrap it with dask.delayed
        wrapper = self._create_named_wrapper()
        delayed_func = delayed(wrapper, pure=False)
        return delayed_func(x)

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Calculate output data shape after operation.

        This method can be overridden by subclasses for efficiency.
        If not overridden, it will execute _process_array on a small test array
        to determine the output shape.

        Parameters
        ----------
        input_shape : tuple
            Input data shape

        Returns
        -------
        tuple
            Output data shape

        Notes
        -----
        The default implementation creates a minimal test array and processes it
        to determine output shape. For performance-critical code, subclasses should
        override this method with a direct calculation.
        """
        # Try to infer shape by executing _process_array on test data
        import numpy as np

        try:
            # Create minimal test array with input shape
            if len(input_shape) == 0:
                return input_shape

            # Create test input with correct dtype
            # Try complex first, fall back to float if needed
            test_input: Any = np.zeros(input_shape, dtype=np.complex128)

            # Process test input
            test_output: Any = self._process_array(test_input)

            # Return the shape of the output
            if isinstance(test_output, np.ndarray):
                return tuple(int(s) for s in test_output.shape)
            return input_shape
        except Exception as e:
            logger.warning(
                f"Failed to infer output shape for {self.__class__.__name__}: {e}. "
                "Please implement calculate_output_shape method."
            )
            raise NotImplementedError(
                f"Subclass {self.__class__.__name__} must implement "
                f"calculate_output_shape or ensure _process_array can be "
                f"called with test data."
            ) from e

    def process(self, data: DaArray) -> DaArray:
        """
        Execute operation and return result
        data shape is (channels, samples)
        """
        # Add task as delayed processing with custom name for visualization
        logger.debug("Adding delayed operation to computation graph")

        # Create a wrapper function with the operation name
        # This allows Dask to use the operation name in the task graph
        wrapper = self._create_named_wrapper()
        delayed_func = delayed(wrapper, pure=False)
        delayed_result = delayed_func(data)

        # Convert delayed result to dask array and return
        output_shape = self.calculate_output_shape(data.shape)
        return _da_from_delayed(delayed_result, shape=output_shape, dtype=data.dtype)


# Automatically collect operation types and corresponding classes
_OPERATION_REGISTRY: dict[str, type[AudioOperation[Any, Any]]] = {}


def register_operation(operation_class: type) -> None:
    """Register a new operation type"""

    if not issubclass(operation_class, AudioOperation):
        raise TypeError("Strategy class must inherit from AudioOperation.")
    if inspect.isabstract(operation_class):
        raise TypeError("Cannot register abstract AudioOperation class.")

    _OPERATION_REGISTRY[operation_class.name] = operation_class


def get_operation(name: str) -> type[AudioOperation[Any, Any]]:
    """Get operation class by name"""
    if name not in _OPERATION_REGISTRY:
        raise ValueError(f"Unknown operation type: {name}")
    return _OPERATION_REGISTRY[name]


def create_operation(
    name: str, sampling_rate: float, **params: Any
) -> AudioOperation[Any, Any]:
    """Create operation instance from name and parameters"""
    operation_class = get_operation(name)
    return operation_class(sampling_rate, **params)
