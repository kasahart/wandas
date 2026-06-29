import logging

import dask.array as da
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

    def process(self, data: DaArray, *inputs: DaArray) -> DaArray:
        self._validate_process_input_count(1 + len(inputs))
        return self._mark_array(da.abs(data))


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
        super().__init__(sampling_rate, exponent=exponent)

    @property
    def exponent(self) -> float:
        """Exponent captured at operation construction time."""
        return self._config_value("exponent")

    @property
    def exp(self) -> float:
        """Backward-compatible read-only alias for the captured exponent."""
        return self.exponent

    def process(self, data: DaArray, *inputs: DaArray) -> DaArray:
        self._validate_process_input_count(1 + len(inputs))
        return self._mark_array(da.power(data, self.exponent))


class Sum(AudioOperation[NDArrayReal, NDArrayReal]):
    """Sum calculation"""

    name = "sum"
    _display = "sum"

    def process(self, data: DaArray, *inputs: DaArray) -> DaArray:
        self._validate_process_input_count(1 + len(inputs))
        return self._mark_array(data.sum(axis=0, keepdims=True))


class Mean(AudioOperation[NDArrayReal, NDArrayReal]):
    """Mean calculation"""

    name = "mean"
    _display = "mean"

    def process(self, data: DaArray, *inputs: DaArray) -> DaArray:
        self._validate_process_input_count(1 + len(inputs))
        return self._mark_array(data.mean(axis=0, keepdims=True))


class ChannelDifference(AudioOperation[NDArrayReal, NDArrayReal]):
    """Channel difference calculation operation"""

    name = "channel_difference"
    _display = "diff"

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
        super().__init__(sampling_rate, other_channel=other_channel)

    @property
    def other_channel(self) -> int:
        """Other channel index captured at operation construction time."""
        return self._config_value("other_channel")

    def process(self, data: DaArray, *inputs: DaArray) -> DaArray:
        self._validate_process_input_count(1 + len(inputs))
        other_channel = self.other_channel
        if not -data.shape[0] <= other_channel < data.shape[0]:
            raise IndexError("Channel index out of range")
        return self._mark_array(data - data[other_channel])


# Register all operations
for op_class in [ABS, Power, Sum, Mean, ChannelDifference]:
    register_operation(op_class)
