"""Common protocol definition module.

This module contains common protocols used by mixin classes.
"""

import logging
from typing import Any, Protocol, TypeVar, runtime_checkable

from dask.array.core import Array as DaArray

from wandas.core.metadata import ChannelMetadata

logger = logging.getLogger(__name__)


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
T_Base = TypeVar("T_Base", bound=BaseFrameProtocol)
T_Processing = TypeVar("T_Processing", bound=ProcessingFrameProtocol)
T_Transform = TypeVar("T_Transform", bound=TransformFrameProtocol)

__all__ = [
    "BaseFrameProtocol",
    "ProcessingFrameProtocol",
    "TransformFrameProtocol",
    "T_Base",
    "T_Processing",
    "T_Transform",
]
