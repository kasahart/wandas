import logging

import dask.array as da
import numpy as np
from dask.array.core import Array as DaArray

from wandas.processing.base import AudioOperation, register_operation
from wandas.utils.types import NDArrayReal

logger = logging.getLogger(__name__)


def _reduction_data(data: DaArray) -> DaArray:
    """Return channel-first data in a dtype safe for amplitude reductions."""
    if data.ndim != 2:
        raise ValueError("Channel reductions require 2-D channel-first data")
    if data.shape[1] == 0:
        raise ValueError("Channel reductions require at least one sample per channel")
    if not np.issubdtype(data.dtype, np.inexact):
        return data.astype(np.float64)
    return data


def _channel_peak(data: DaArray) -> DaArray:
    """Return the lazy per-channel absolute peak reduction."""
    values = _reduction_data(data)
    return da.max(da.absolute(values), axis=1)


def _channel_rms(data: DaArray) -> DaArray:
    """Return lazy per-channel RMS using a scale-normalized reduction."""
    values = _reduction_data(data)
    magnitude = da.absolute(values)
    scale = da.max(magnitude, axis=1, keepdims=True)
    finite_nonzero = da.isfinite(scale) & (scale != 0)
    safe_scale = da.where(finite_nonzero, scale, 1.0)
    normalized_rms = da.sqrt(da.mean((magnitude / safe_scale) ** 2, axis=1))
    squeezed_scale = scale[:, 0]
    scaled_rms = squeezed_scale * normalized_rms
    return da.where(squeezed_scale == 0, 0.0, da.where(da.isinf(squeezed_scale), squeezed_scale, scaled_rms))


def _channel_crest_factor(data: DaArray) -> DaArray:
    """Return the lazy per-channel peak-to-RMS ratio."""
    peak = _channel_peak(data)
    rms = _channel_rms(data)
    finite_nonzero = da.isfinite(peak) & da.isfinite(rms) & (rms != 0)
    ratio = da.where(finite_nonzero, peak, 1.0) / da.where(finite_nonzero, rms, 1.0)
    return da.where(rms == 0, 1.0, da.where(finite_nonzero, ratio, np.nan))


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
        return da.abs(data)


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
        return da.power(data, self.exponent)


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
        return data - data[other_channel]


# Register all operations
for op_class in [ABS, Power, Sum, Mean, ChannelDifference]:
    register_operation(op_class)
