"""Common protocol definition module.

This module contains common protocols used by mixin classes.
"""

import logging
from typing import Any, Protocol, TypeVar, runtime_checkable

from dask.array.core import Array as DaArray

from wandas.core.metadata import ChannelMetadata
from wandas.utils.types import NDArrayReal

logger = logging.getLogger(__name__)


T_Base = TypeVar("T_Base", bound="BaseFrameProtocol")


@runtime_checkable
class BaseFrameProtocol(Protocol):
    """Protocol that defines basic frame operations.

    Defines the basic methods and properties provided by all frame classes.
    """

    _data: DaArray
    sampling_rate: float
    _channel_metadata: list[ChannelMetadata]
    metadata: dict[str, Any]
    operation_history: list[dict[str, Any]]
    label: str

    @property
    def duration(self) -> float:
        """Returns the duration in seconds."""
        ...

    @property
    def data(self) -> NDArrayReal:
        """Returns the computed data as a NumPy array.

        Implementations should materialize any lazy computation (e.g. Dask)
        and return a concrete NumPy array.
        """
        ...

    def label2index(self, label: str) -> int:
        """
        Get the index from a channel label.
        """
        ...

    def apply_operation(
        self, operation_name: str, **params: Any
    ) -> "BaseFrameProtocol":
        """Apply a named operation.

        Args:
            operation_name: Name of the operation to apply
            **params: Parameters to pass to the operation

        Returns:
            A new frame instance with the operation applied
        """
        ...

    def _create_new_instance(self: T_Base, data: DaArray, **kwargs: Any) -> T_Base:
        """Create a new instance of the frame with updated data and metadata.
        Args:
            data: The new data for the frame
            metadata: The new metadata for the frame
            operation_history: The new operation history for the frame
            channel_metadata: The new channel metadata for the frame
        Returns:
            A new instance of the frame with the updated data and metadata
        """
        ...


@runtime_checkable
class ProcessingFrameProtocol(BaseFrameProtocol, Protocol):
    """Protocol that defines operations related to signal processing.

    Defines methods that provide frame operations related to signal processing.
    """

    pass


@runtime_checkable
class TransformFrameProtocol(BaseFrameProtocol, Protocol):
    """Protocol related to transform operations.

    Defines methods that provide operations such as frequency analysis and
    spectral transformation.
    """

    pass


# Type variable definitions
T_Processing = TypeVar("T_Processing", bound=ProcessingFrameProtocol)
T_Transform = TypeVar("T_Transform", bound=TransformFrameProtocol)

__all__ = [
    "BaseFrameProtocol",
    "ProcessingFrameProtocol",
    "TransformFrameProtocol",
    "T_Processing",
]
