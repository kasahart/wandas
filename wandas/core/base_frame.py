import copy
import logging
import numbers
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Callable, Generic, Optional, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt
from dask.array.core import Array as DaArray
from matplotlib.axes import Axes

from wandas.utils.types import NDArrayComplex, NDArrayReal

from .metadata import ChannelMetadata

logger = logging.getLogger(__name__)

T = TypeVar("T", NDArrayComplex, NDArrayReal)
S = TypeVar("S", bound="BaseFrame[Any]")


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
    operation_history : list[dict], optional
        History of operations performed on this frame.
    channel_metadata : list[ChannelMetadata], optional
        Metadata for each channel in the frame.
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
    operation_history : list[dict]
        History of operations performed on this frame.
    """

    _data: DaArray
    sampling_rate: float
    label: str
    metadata: dict[str, Any]
    operation_history: list[dict[str, Any]]
    _channel_metadata: list[ChannelMetadata]
    _previous: Optional["BaseFrame[Any]"]

    def __init__(
        self,
        data: DaArray,
        sampling_rate: float,
        label: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        operation_history: Optional[list[dict[str, Any]]] = None,
        channel_metadata: Optional[list[ChannelMetadata]] = None,
        previous: Optional["BaseFrame[Any]"] = None,
    ):
        self._data = data.rechunk(chunks=-1)  # type: ignore [unused-ignore]
        if self._data.ndim == 1:
            self._data = self._data.reshape((1, -1))
        self.sampling_rate = sampling_rate
        self.label = label or "unnamed_frame"
        self.metadata = metadata or {}
        self.operation_history = operation_history or []
        self._previous = previous

        if channel_metadata:
            self._channel_metadata = copy.deepcopy(channel_metadata)
        else:
            self._channel_metadata = [
                ChannelMetadata(label=f"ch{i}", unit="", extra={})
                for i in range(self._n_channels)
            ]

        try:
            # Display information for newer dask versions
            logger.debug(f"Dask graph layers: {list(self._data.dask.layers.keys())}")
            logger.debug(
                f"Dask graph dependencies: {len(self._data.dask.dependencies)}"
            )
        except Exception as e:
            logger.debug(f"Dask graph visualization details unavailable: {e}")

    @property
    @abstractmethod
    def _n_channels(self) -> int:
        """Returns the number of channels."""

    @property
    def n_channels(self) -> int:
        """Returns the number of channels."""
        return self._n_channels

    @property
    def channels(self) -> list[ChannelMetadata]:
        """Property to access channel metadata."""
        return self._channel_metadata

    @property
    def previous(self) -> Optional["BaseFrame[Any]"]:
        """
        Returns the previous frame.
        """
        return self._previous

    def get_channel(self: S, channel_idx: int) -> S:
        n_channels = len(self)
        if channel_idx < 0 or channel_idx >= n_channels:
            range_max = n_channels - 1
            raise ValueError(
                f"Channel index out of range: {channel_idx} "
                f"(valid range: 0-{range_max})"
            )
        logger.debug(f"Extracting channel index={channel_idx} (lazy operation).")
        channel_data = self._data[channel_idx : channel_idx + 1]

        return self._create_new_instance(
            data=channel_data,
            operation_history=self.operation_history,
            channel_metadata=[self._channel_metadata[channel_idx]],
        )

    def __len__(self) -> int:
        """
        Returns the number of channels.
        """
        return len(self._channel_metadata)

    def __iter__(self: S) -> Iterator[S]:
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self: S, key: Union[str, int, slice, tuple[slice, ...]]) -> S:
        """
        Method to get a channel by name or index.

        Parameters
        ----------
        key : str, int, slice, or tuple of slices
            Channel name (label) or index number.

        Returns
        -------
        BaseFrame
            The corresponding channel.

        Raises
        ------
        ValueError
            If the key length is invalid for the shape.
        IndexError
            If the channel index is out of range.
        TypeError
            If the key type is invalid.
        """
        if isinstance(key, str):
            index = self.label2index(key)
            return self.get_channel(index)

        elif isinstance(key, tuple):
            # When key is a tuple, treat the first element as an index
            if len(key) > len(self.shape):
                raise ValueError(
                    f"Invalid key length: {len(key)} for shape {self.shape}"
                )
            new_data = self._data[key]
            new_channel_metadata = self._channel_metadata[key[0]]
            if isinstance(new_channel_metadata, ChannelMetadata):
                new_channel_metadata = [new_channel_metadata]
            return self._create_new_instance(
                data=new_data,
                operation_history=self.operation_history,
                channel_metadata=new_channel_metadata,
            )
        elif isinstance(key, slice):
            new_data = self._data[key]
            new_channel_metadata = self._channel_metadata[key]
            if isinstance(new_channel_metadata, ChannelMetadata):
                new_channel_metadata = [new_channel_metadata]
            return self._create_new_instance(
                data=new_data,
                operation_history=self.operation_history,
                channel_metadata=new_channel_metadata,
            )
        elif isinstance(key, numbers.Integral):
            # Access by index number
            if key < 0 or key >= len(self):
                raise IndexError(f"Channel index {key} out of range.")
            return self.get_channel(key)
        else:
            raise TypeError(
                f"Invalid key type: {type(key)}. Expected str, int, or tuple."
            )

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
        for idx, ch in enumerate(self._channel_metadata):
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
            return data.squeeze(axis=0)
        return data

    @property
    def labels(self) -> list[str]:
        """Get a list of all channel labels."""
        return [ch.label for ch in self._channel_metadata]

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
        logger.debug(
            "COMPUTING DASK ARRAY - This will trigger file reading and all processing"
        )
        result = self._data.compute()

        if not isinstance(result, np.ndarray):
            raise ValueError(f"Computed result is not a np.ndarray: {type(result)}")

        logger.debug(f"Computation complete, result shape: {result.shape}")
        return cast(T, result)

    @abstractmethod
    def plot(
        self, plot_type: str = "default", ax: Optional[Axes] = None, **kwargs: Any
    ) -> Union[Axes, Iterator[Axes]]:
        """Plot the data"""
        pass

    def persist(self: S) -> S:
        """Persist the data in memory"""
        persisted_data = self._data.persist()
        return self._create_new_instance(data=persisted_data)

    @abstractmethod
    def _get_additional_init_kwargs(self) -> dict[str, Any]:
        """
        Abstract method for derived classes to provide
        additional initialization arguments.
        """
        pass

    def _create_new_instance(self: S, data: DaArray, **kwargs: Any) -> S:
        """
        Create a new channel instance based on an existing channel.
        Keyword arguments can override or extend the original attributes.
        """

        sampling_rate = kwargs.pop("sampling_rate", self.sampling_rate)
        # if not isinstance(sampling_rate, int):
        #     raise TypeError("Sampling rate must be an integer")

        label = kwargs.pop("label", self.label)
        if not isinstance(label, str):
            raise TypeError("Label must be a string")

        metadata = kwargs.pop("metadata", copy.deepcopy(self.metadata))
        if not isinstance(metadata, dict):
            raise TypeError("Metadata must be a dictionary")

        channel_metadata = kwargs.pop(
            "channel_metadata", copy.deepcopy(self._channel_metadata)
        )
        if not isinstance(channel_metadata, list):
            raise TypeError("Channel metadata must be a list")

        # Get additional initialization arguments from derived classes
        additional_kwargs = self._get_additional_init_kwargs()
        kwargs.update(additional_kwargs)

        return type(self)(
            data=data,
            sampling_rate=sampling_rate,
            label=label,
            metadata=metadata,
            channel_metadata=channel_metadata,
            previous=self,
            **kwargs,
        )

    def __array__(self, dtype: npt.DTypeLike = None) -> NDArrayReal:
        """Implicit conversion to NumPy array"""
        result = self.compute()
        if dtype is not None:
            return result.astype(dtype)
        return result

    def visualize_graph(self, filename: Optional[str] = None) -> Optional[str]:
        """Visualize the computation graph and save it to a file"""
        try:
            filename = filename or f"graph_{uuid.uuid4().hex[:8]}.png"
            self._data.visualize(filename=filename)
            return filename
        except Exception as e:
            logger.warning(f"Failed to visualize the graph: {e}")
            return None

    @abstractmethod
    def _binary_op(
        self: S,
        other: Union[S, int, float, NDArrayReal, DaArray],
        op: Callable[[DaArray, Any], DaArray],
        symbol: str,
    ) -> S:
        """Basic implementation of binary operations"""
        # Basic logic
        # Actual implementation is left to derived classes
        pass

    def __add__(self: S, other: Union[S, int, float, NDArrayReal]) -> S:
        """Addition operator"""
        return self._binary_op(other, lambda x, y: x + y, "+")

    def __sub__(self: S, other: Union[S, int, float, NDArrayReal]) -> S:
        """Subtraction operator"""
        return self._binary_op(other, lambda x, y: x - y, "-")

    def __mul__(self: S, other: Union[S, int, float, NDArrayReal]) -> S:
        """Multiplication operator"""
        return self._binary_op(other, lambda x, y: x * y, "*")

    def __truediv__(self: S, other: Union[S, int, float, NDArrayReal]) -> S:
        """Division operator"""
        return self._binary_op(other, lambda x, y: x / y, "/")

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

    @abstractmethod
    def _apply_operation_impl(self: S, operation_name: str, **params: Any) -> S:
        """Implementation of operation application"""
        pass

    def debug_info(self) -> None:
        """Output detailed debug information"""
        logger.debug(f"=== {self.__class__.__name__} Debug Info ===")
        logger.debug(f"Label: {self.label}")
        logger.debug(f"Shape: {self.shape}")
        logger.debug(f"Sampling rate: {self.sampling_rate} Hz")
        logger.debug(f"Operation history: {len(self.operation_history)} operations")
        self._debug_info_impl()
        logger.debug("=== End Debug Info ===")

    def _debug_info_impl(self) -> None:
        """Implement derived class-specific debug information"""
        pass
