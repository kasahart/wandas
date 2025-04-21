from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Union, cast

import numpy as np

from wandas.utils.types import NDArrayReal

if TYPE_CHECKING:
    from wandas.core.channel import Channel


class ArithmeticMixin(ABC):
    """
    A mixin class providing arithmetic operations for channel classes.

    Classes using this mixin must implement:
    - data: Property returning channel data as NDArrayReal
    - sampling_rate: Property returning channel sampling rate as int
    - label: Property returning channel label as str
    """

    # Define abstract properties to ensure that any subclass provides these.
    @property
    @abstractmethod
    def data(self) -> NDArrayReal: ...

    @property
    @abstractmethod
    def sampling_rate(self) -> int: ...

    @property
    @abstractmethod
    def label(self) -> str: ...

    def _binary_op(
        self,
        other: Union["Channel", int, float, NDArrayReal],
        op: Callable[[Any, Any], Any],
        symbol: str,
    ) -> "Channel":
        """
        Apply a binary operation between this channel and another object.

        Parameters
        ----------
        other : Channel, int, float, or NDArrayReal
            The other operand for the binary operation.
        op : Callable
            The operation function to apply.
        symbol : str
            Symbol representing the operation (for labeling).

        Returns
        -------
        Channel
            A new Channel with the result of the operation.

        Raises
        ------
        ValueError
            If the sampling rates of the channels don't match.
        TypeError
            If the other operand has an unsupported type.
        """
        # Delayed import to avoid circular dependencies.
        from wandas.core.channel import Channel

        if isinstance(other, Channel):
            if self.sampling_rate != other.sampling_rate:
                raise ValueError("Sampling rates must be the same.")
            new_data = op(self.data, other.data)
            new_label = f"({self.label} {symbol} {other.label})"
        elif isinstance(other, (int, float, np.ndarray)):
            new_data = op(self.data, other)
            new_label = f"({self.label} {symbol} {other})"
        else:
            raise TypeError(
                f"Unsupported type for operation with Channel: {type(other)}"
            )
        result = dict(data=new_data, sampling_rate=self.sampling_rate, label=new_label)

        return Channel.from_channel(cast(Channel, self), **result)

    def __add__(self, other: Union["Channel", int, float, NDArrayReal]) -> "Channel":
        """
        Add another channel, scalar, or array to this channel.

        Parameters
        ----------
        other : Channel, int, float, or NDArrayReal
            The value to add to this channel.

        Returns
        -------
        Channel
            A new Channel with the result of the addition.
        """
        return self._binary_op(other, lambda a, b: a + b, "+")

    def __sub__(self, other: Union["Channel", int, float, NDArrayReal]) -> "Channel":
        """
        Subtract another channel, scalar, or array from this channel.

        Parameters
        ----------
        other : Channel, int, float, or NDArrayReal
            The value to subtract from this channel.

        Returns
        -------
        Channel
            A new Channel with the result of the subtraction.
        """
        return self._binary_op(other, lambda a, b: a - b, "-")

    def __mul__(self, other: Union["Channel", int, float, NDArrayReal]) -> "Channel":
        """
        Multiply this channel by another channel, scalar, or array.

        Parameters
        ----------
        other : Channel, int, float, or NDArrayReal
            The value to multiply this channel by.

        Returns
        -------
        Channel
            A new Channel with the result of the multiplication.
        """
        return self._binary_op(other, lambda a, b: a * b, "*")

    def __truediv__(
        self, other: Union["Channel", int, float, NDArrayReal]
    ) -> "Channel":
        """
        Divide this channel by another channel, scalar, or array.

        Parameters
        ----------
        other : Channel, int, float, or NDArrayReal
            The value to divide this channel by.

        Returns
        -------
        Channel
            A new Channel with the result of the division.
        """
        return self._binary_op(other, lambda a, b: a / b, "/")
