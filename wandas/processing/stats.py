import logging

import numpy as np
from dask.array.core import Array as DaArray

from wandas.processing.base import AudioOperation, register_operation
from wandas.utils.types import NDArrayReal

logger = logging.getLogger(__name__)


class ABS(AudioOperation[NDArrayReal, NDArrayReal]):
    """Absolute value operation"""

    name = "abs"
    _display = "abs"

    def __init__(self, sampling_rate: float):
        """
        Initialize absolute value operation

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        """
        super().__init__(sampling_rate)

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        result: NDArrayReal = np.abs(x)
        return result


class Power(AudioOperation[NDArrayReal, NDArrayReal]):
    """Power operation"""

    name = "power"
    _display = "pow"

    def __init__(self, sampling_rate: float, exponent: float):
        """
        Initialize power operation

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        exponent : float
            Power exponent
        """
        self.exp = exponent
        super().__init__(sampling_rate, exponent=exponent)

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        result: NDArrayReal = np.power(x, self.exp)
        return result


class Sum(AudioOperation[NDArrayReal, NDArrayReal]):
    """Sum calculation"""

    name = "sum"
    _display = "sum"

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (1, *input_shape[1:])

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        result: NDArrayReal = x.sum(axis=0, keepdims=True)
        return result


class Mean(AudioOperation[NDArrayReal, NDArrayReal]):
    """Mean calculation"""

    name = "mean"
    _display = "mean"

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (1, *input_shape[1:])

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        result: NDArrayReal = x.mean(axis=0, keepdims=True)
        return result


class ChannelDifference(AudioOperation[NDArrayReal, NDArrayReal]):
    """Channel difference calculation operation"""

    name = "channel_difference"
    _display = "diff"
    other_channel: int

    def __init__(self, sampling_rate: float, other_channel: int = 0):
        """
        Initialize channel difference calculation

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        other_channel : int
            Channel to calculate difference with, default is 0
        """
        self.other_channel = other_channel
        super().__init__(sampling_rate, other_channel=other_channel)

    def process(self, data: DaArray) -> DaArray:
        if self.other_channel < 0 or self.other_channel >= data.shape[0]:
            raise IndexError("Channel index out of range")
        return super().process(data)

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        result: NDArrayReal = x - x[self.other_channel]
        return result


# Register all operations
for op_class in [ABS, Power, Sum, Mean, ChannelDifference]:
    register_operation(op_class)
