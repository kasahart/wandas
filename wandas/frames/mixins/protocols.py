"""Common protocol definition module.

This module contains common protocols used by mixin classes.
"""

import logging
from typing import Any, Protocol, TypeVar, runtime_checkable

from dask.array.core import Array as DaArray

from wandas.core.metadata import ChannelMetadata, FrameMetadata
from wandas.utils.types import NDArrayReal

logger = logging.getLogger(__name__)


# Protocol for operation objects that provide process_array method
class OperationProtocol(Protocol):
    """Protocol defining the interface expected by _compute_scalar_metric."""

    def process_array(self, x: Any) -> Any: ...


T_Base = TypeVar("T_Base", bound="BaseFrameProtocol")


@runtime_checkable
class BaseFrameProtocol(Protocol):
    """Protocol that defines basic frame operations.

    Defines the basic methods and properties provided by all frame classes.
    """

    _data: DaArray
    sampling_rate: float
    _channel_metadata: list[ChannelMetadata]
    metadata: FrameMetadata
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

    def apply_operation(self, operation_name: str, **params: Any) -> "BaseFrameProtocol":
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


from wandas.core.base_frame import BaseFrame  # noqa: E402


@runtime_checkable
class ProcessingFrameProtocol(BaseFrameProtocol, Protocol):
    """Protocol that defines operations related to signal processing.

    Defines methods that provide frame operations related to signal processing.
    """

    def _updated_metadata_and_history(
        self,
        operation_name: str,
        params: dict[str, Any],
    ) -> tuple[FrameMetadata, list[dict[str, Any]]]: ...

    def _get_ref_values(self, *, require_non_default: bool = False) -> list[float]: ...

    def _compute_scalar_metric(
        self,
        operation: OperationProtocol,
    ) -> NDArrayReal: ...

    def _hpss(
        self,
        operation_name: str,
        kernel_size: Any = ...,
        power: float = ...,
        margin: Any = ...,
        n_fft: int = ...,
        hop_length: int | None = ...,
        win_length: int | None = ...,
        window: Any = ...,
        center: bool = ...,
        pad_mode: Any = ...,
    ) -> "ProcessingFrameProtocol": ...


from wandas.frames.spectral import SpectralFrame  # noqa: E402


@runtime_checkable
class TransformFrameProtocol(BaseFrameProtocol, Protocol):
    """Protocol related to transform operations.

    Defines methods that provide operations such as frequency analysis and
    spectral transformation.
    """

    @property
    def _as_base_frame(self) -> BaseFrame[Any]: ...

    def _cross_channel_spectral_transform(
        self,
        operation_name: str,
        label_prefix: str,
        label_template: str,
        **params: Any,
    ) -> SpectralFrame: ...


# Type variable definitions
T_Processing = TypeVar("T_Processing", bound=ProcessingFrameProtocol)
T_Transform = TypeVar("T_Transform", bound=TransformFrameProtocol)

__all__ = [
    "BaseFrameProtocol",
    "ProcessingFrameProtocol",
    "T_Processing",
    "TransformFrameProtocol",
]
