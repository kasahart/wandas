"""
ChannelCollectionMixin: Common functionality for adding/removing channels in
ChannelFrame
"""

from collections.abc import Sequence
from typing import Any, Literal, TypeVar

import dask.array as da
import numpy as np

from wandas.utils.types import NDArrayReal

T = TypeVar("T", bound="ChannelCollectionMixin")


class ChannelCollectionMixin:
    def add_channel(
        self: T,
        data: np.ndarray[Any, Any] | da.Array | T,
        label: str | None = None,
        align: Literal["strict", "pad", "truncate"] = "strict",
        suffix_on_dup: str | None = None,
        source_time_offset: float | Sequence[float] | NDArrayReal | None = None,
        **kwargs: Any,
    ) -> T:
        """
        Add a channel
        Args:
            data: Channel to add (1ch ndarray/dask/ChannelFrame)
            label: Label for the added channel
            align: Behavior when lengths don't match
            suffix_on_dup: Suffix when label is duplicated
            source_time_offset: Offset for raw ndarray/dask input
        Returns:
            New Frame
        Raises:
            ValueError, TypeError
        """
        raise NotImplementedError("add_channel() must be implemented in subclasses")

    def remove_channel(
        self: T,
        key: int | str,
    ) -> T:
        """
        Remove a channel
        Args:
            key: Target to remove (index or label)
        Returns:
            New Frame
        Raises:
            ValueError, KeyError, IndexError
        """
        raise NotImplementedError("remove_channel() must be implemented in subclasses")
