import logging

import dask.array as da
import numpy as np
from dask.array.core import Array as DaArray

from wandas.processing.base import AudioOperation, register_operation
from wandas.utils.types import NDArrayReal

logger = logging.getLogger(__name__)


class ABS(AudioOperation[NDArrayReal, NDArrayReal]):
    """Absolute value operation"""

    name = "abs"

    def __init__(self, sampling_rate: float):
        """
        Initialize absolute value operation

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        """
        super().__init__(sampling_rate)

    def get_display_name(self) -> str:
        """Get display name for the operation for use in channel labels."""
        return "abs"

    def process(self, data: DaArray) -> DaArray:
        # map_blocksを使わず、直接Daskの集約関数を使用
        return da.abs(data)  # type: ignore [unused-ignore]


class Power(AudioOperation[NDArrayReal, NDArrayReal]):
    """Power operation"""

    name = "power"

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
        super().__init__(sampling_rate)
        self.exp = exponent

    def get_display_name(self) -> str:
        """Get display name for the operation for use in channel labels."""
        return "pow"

    def process(self, data: DaArray) -> DaArray:
        # map_blocksを使わず、直接Daskの集約関数を使用
        return da.power(data, self.exp)  # type: ignore [unused-ignore]


class Sum(AudioOperation[NDArrayReal, NDArrayReal]):
    """Sum calculation"""

    name = "sum"

    def get_display_name(self) -> str:
        """Get display name for the operation for use in channel labels."""
        return "sum"

    def process(self, data: DaArray) -> DaArray:
        # Use Dask's aggregate function directly without map_blocks
        return data.sum(axis=0, keepdims=True)


class Mean(AudioOperation[NDArrayReal, NDArrayReal]):
    """Mean calculation"""

    name = "mean"

    def get_display_name(self) -> str:
        """Get display name for the operation for use in channel labels."""
        return "mean"

    def process(self, data: DaArray) -> DaArray:
        # Use Dask's aggregate function directly without map_blocks
        return data.mean(axis=0, keepdims=True)


class ChannelDifference(AudioOperation[NDArrayReal, NDArrayReal]):
    """Channel difference calculation operation"""

    name = "channel_difference"
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

    def get_display_name(self) -> str:
        """Get display name for the operation for use in channel labels."""
        return "diff"

    def process(self, data: DaArray) -> DaArray:
        # map_blocksを使わず、直接Daskの集約関数を使用
        result = data - data[self.other_channel]
        return result


class CrestFactor(AudioOperation[NDArrayReal, NDArrayReal]):
    """Crest factor (peak-to-RMS ratio) calculation per channel.

    The crest factor is defined as the ratio of the peak amplitude to the
    root-mean-square (RMS) amplitude:

        crest_factor = max(|x|) / RMS(x)

    For a pure sine wave the crest factor is sqrt(2) ≈ 1.414.

    When the RMS is zero (all-zero channel) the result is NaN to avoid a
    division-by-zero error.
    """

    name = "crest_factor"

    def get_display_name(self) -> str:
        """Get display name for the operation for use in channel labels."""
        return "crest_factor"

    def process(self, data: DaArray) -> DaArray:
        """Compute the crest factor for each channel.

        Parameters
        ----------
        data : DaArray
            Input array of shape ``(channels, samples)``.

        Returns
        -------
        DaArray
            Array of shape ``(channels, 1)`` containing the crest factor for
            each channel.  Channels with zero RMS yield NaN.
        """
        peak: DaArray = da.max(da.abs(data), axis=-1, keepdims=True)
        rms: DaArray = da.sqrt((data**2).mean(axis=-1, keepdims=True))
        # Avoid division by zero: return NaN when RMS is 0
        result: DaArray = da.where(rms == 0, np.nan, peak / rms)
        return result


# Register all operations
for op_class in [ABS, Power, Sum, Mean, ChannelDifference, CrestFactor]:
    register_operation(op_class)
