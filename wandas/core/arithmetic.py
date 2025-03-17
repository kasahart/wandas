from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Union, cast

import numpy as np

from wandas.utils.types import NDArrayReal

if TYPE_CHECKING:
    from wandas.core.channel import Channel


class ArithmeticMixin(ABC):
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
        return self._binary_op(other, lambda a, b: a + b, "+")

    def __sub__(self, other: Union["Channel", int, float, NDArrayReal]) -> "Channel":
        return self._binary_op(other, lambda a, b: a - b, "-")

    def __mul__(self, other: Union["Channel", int, float, NDArrayReal]) -> "Channel":
        return self._binary_op(other, lambda a, b: a * b, "*")

    def __truediv__(
        self, other: Union["Channel", int, float, NDArrayReal]
    ) -> "Channel":
        return self._binary_op(other, lambda a, b: a / b, "/")
