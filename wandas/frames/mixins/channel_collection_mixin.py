"""
ChannelCollectionMixin: Common functionality for adding/removing channels in
ChannelFrame
"""

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import dask.array as da
import numpy as np

if TYPE_CHECKING:
    pass

T = TypeVar("T", bound="ChannelCollectionMixin")


class ChannelCollectionMixin:
    def add_channel(
        self: T,
        data: np.ndarray[Any, Any] | da.Array | T,
        label: str | None = None,
        align: Literal["strict", "pad", "truncate"] = "strict",
        suffix_on_dup: str | None = None,
        inplace: bool = False,
        **kwargs: Any,
    ) -> T:
        """
        Add a channel
        Args:
            data: Channel to add (1ch ndarray/dask/ChannelFrame)
            label: Label for the added channel
            align: Behavior when lengths don't match
            suffix_on_dup: Suffix when label is duplicated
            inplace: True for self-modification
        Returns:
            New Frame or self
        Raises:
            ValueError, TypeError
        """
        raise NotImplementedError("add_channel() must be implemented in subclasses")

    def remove_channel(
        self: T,
        key: int | str,
        inplace: bool = False,
    ) -> T:
        """
        Remove a channel
        Args:
            key: Target to remove (index or label)
            inplace: True for self-modification
        Returns:
            New Frame or self
        Raises:
            ValueError, KeyError, IndexError
        """
        raise NotImplementedError("remove_channel() must be implemented in subclasses")
