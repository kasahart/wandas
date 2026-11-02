import logging

import dask.array as da
from dask.array.core import Array as DaArray

from wandas.processing.base import AudioOperation, _register_module_operations
from wandas.processing.conversion import integer_arithmetic_as_float64
from wandas.utils.types import NDArrayReal

logger = logging.getLogger(__name__)


class ABS(AudioOperation[NDArrayReal, NDArrayReal]):
    """Absolute value operation"""

    name = "abs"
    _display = "abs"

    def __init__(self, sampling_rate: float):
        """
        Initialize absolute value operation

        Args:
            sampling_rate: float. Sampling rate (Hz)
        """
        super().__init__(sampling_rate)

    def process(self, data: DaArray, *inputs: DaArray) -> DaArray:
        (values,) = integer_arithmetic_as_float64(data)
        return da.abs(values)


class Power(AudioOperation[NDArrayReal, NDArrayReal]):
    """Power operation"""

    name = "power"
    _display = "pow"

    def __init__(self, sampling_rate: float, exponent: float):
        """
        Initialize power operation

        Args:
            sampling_rate: float. Sampling rate (Hz)
            exponent: float. Power exponent
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
        values, exponent = integer_arithmetic_as_float64(data, self.exponent)
        return da.power(values, exponent)


class Sum(AudioOperation[NDArrayReal, NDArrayReal]):
    """Sum calculation"""

    name = "sum"
    _display = "sum"

    def process(self, data: DaArray, *inputs: DaArray) -> DaArray:
        return data.sum(axis=0, keepdims=True)


class Mean(AudioOperation[NDArrayReal, NDArrayReal]):
    """Mean calculation"""

    name = "mean"
    _display = "mean"

    def process(self, data: DaArray, *inputs: DaArray) -> DaArray:
        return data.mean(axis=0, keepdims=True)


class ChannelDifference(AudioOperation[NDArrayReal, NDArrayReal]):
    """Channel difference calculation operation"""

    name = "channel_difference"
    _display = "diff"

    def __init__(self, sampling_rate: float, other_channel: int = 0):
        """
        Initialize channel difference calculation

        Args:
            sampling_rate: float. Sampling rate (Hz)
            other_channel: int. Channel to calculate difference with, default is 0
        """
        super().__init__(sampling_rate, other_channel=other_channel)

    @property
    def other_channel(self) -> int:
        """Other channel index captured at operation construction time."""
        return self._config_value("other_channel")

    def process(self, data: DaArray, *inputs: DaArray) -> DaArray:
        other_channel = self.other_channel
        if not -data.shape[0] <= other_channel < data.shape[0]:
            raise IndexError("Channel index out of range")
        (values,) = integer_arithmetic_as_float64(data)
        return values - values[other_channel]


def spectral_kurtosis(stft: DaArray) -> DaArray:
    """Return the spectral kurtosis of STFT data over its last (time) axis.

    The spectral kurtosis of a frequency bin is::

        SK(f) = mean(|X(f, t)| ** 4) / mean(|X(f, t)| ** 2) ** 2 - 2

    It is 0 for stationary Gaussian noise and rises in bins where the signal
    is impulsive. Real-valued bins (0 Hz and Nyquist) have the Gaussian
    reference value 1 instead of 0. A bin with no energy gives NaN.

    Args:
        stft: Complex STFT data whose last axis is time.

    Returns:
        Real array with the time axis removed.
    """
    power = da.real(stft) ** 2 + da.imag(stft) ** 2
    # Normalise by the mean power of each bin first: the moment ratio is the
    # same, and |X|^4 cannot overflow or underflow in single precision.
    normalized = power / da.mean(power, axis=-1, keepdims=True)
    return da.mean(normalized**2, axis=-1) - 2.0


_register_module_operations(__name__)
