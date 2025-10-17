import logging
from typing import Literal

import dask.array as da
import numpy as np
from dask.array.core import Array as DaArray
from mosqito.sq_metrics.speech_intelligibility.sii_ansi import comp_sii

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

    def process(self, data: DaArray) -> DaArray:
        # map_blocksを使わず、直接Daskの集約関数を使用
        return da.power(data, self.exp)  # type: ignore [unused-ignore]


class Sum(AudioOperation[NDArrayReal, NDArrayReal]):
    """Sum calculation"""

    name = "sum"

    def process(self, data: DaArray) -> DaArray:
        # Use Dask's aggregate function directly without map_blocks
        return data.sum(axis=0, keepdims=True)


class Mean(AudioOperation[NDArrayReal, NDArrayReal]):
    """Mean calculation"""

    name = "mean"

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

    def process(self, data: DaArray) -> DaArray:
        # map_blocksを使わず、直接Daskの集約関数を使用
        result = data - data[self.other_channel]
        return result


class SpeechIntelligibilityIndex(AudioOperation[NDArrayReal, NDArrayReal]):
    """
    Speech Intelligibility Index (SII) calculation.

    Computes the Speech Intelligibility Index according to ANSI S3.5 standard
    using the MoSQITo library. The SII is a measure of speech intelligibility
    that ranges from 0.0 (completely unintelligible) to 1.0 (perfectly intelligible).

    This operation returns a scalar value representing the overall SII,
    along with frequency-specific SII values and the corresponding frequency axis.

    Note:
        This operation is designed for mono signals. Multi-channel inputs will be
        automatically converted to mono by averaging across channels before computing SII.

    References:
        ANSI S3.5-1997: Methods for Calculation of the Speech Intelligibility Index
        MoSQITo documentation: https://mosqito.readthedocs.io/
    """

    name = "speech_intelligibility_index"

    def __init__(
        self,
        sampling_rate: float,
        method_band: Literal["octave", "third octave", "critical"] = "octave",
    ):
        """
        Initialize Speech Intelligibility Index operation.

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        method_band : {'octave', 'third octave', 'critical'}, default='octave'
            Frequency band method for SII calculation:
            - 'octave': Use octave bands (faster, less detailed)
            - 'third octave': Use third-octave bands (balanced)
            - 'critical': Use critical bands (most detailed, slower)
        """
        self.method_band = method_band
        super().__init__(sampling_rate, method_band=method_band)

    def validate_params(self) -> None:
        """Validate parameters."""
        valid_methods = ["octave", "third octave", "critical"]
        if self.method_band not in valid_methods:
            raise ValueError(
                f"method_band must be one of {valid_methods}, got '{self.method_band}'"
            )

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """
        Process the input array to compute SII.

        Parameters
        ----------
        x : NDArrayReal
            Input signal array with shape (n_channels, n_samples).
            Multi-channel signals will be converted to mono.

        Returns
        -------
        NDArrayReal
            Array containing [sii_value, len(specific_sii), len(freq_axis)]
            where:
            - sii_value: Overall SII value (0.0 to 1.0)
            - The other values are used to maintain array structure
        """
        # Convert to mono if multi-channel
        if x.ndim > 1 and x.shape[0] > 1:
            signal = np.mean(x, axis=0)
        else:
            signal = x.flatten() if x.ndim > 1 else x

        # Compute SII using MoSQITo
        sii_value, specific_sii, freq_axis = comp_sii(
            signal, self.sampling_rate, method_band=self.method_band
        )

        # Store results for later retrieval
        # We return a simple array structure
        return np.array([[sii_value]])

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Calculate output data shape after operation.

        Parameters
        ----------
        input_shape : tuple
            Input data shape (channels, samples)

        Returns
        -------
        tuple
            Output data shape (1, 1) - contains the SII scalar value
        """
        return (1, 1)

    def process(self, data: DaArray) -> DaArray:
        """
        Execute operation and return result.

        The SII computation requires the full signal, so we compute it
        immediately rather than using delayed evaluation.

        Parameters
        ----------
        data : DaArray
            Input data with shape (channels, samples)

        Returns
        -------
        DaArray
            Dask array containing the SII value with shape (1, 1)
        """
        # Compute the data immediately since SII needs the full signal
        data_computed = data.compute()

        # Process and get the SII value
        result = self._process_array(data_computed)

        # Convert back to Dask array
        return da.from_array(result, chunks=-1)


# Register all operations
for op_class in [ABS, Power, Sum, Mean, ChannelDifference, SpeechIntelligibilityIndex]:
    register_operation(op_class)
