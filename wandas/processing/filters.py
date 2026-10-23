import logging
from typing import Any

import numpy as np
from scipy import signal

from wandas.processing.base import (
    AudioOperation,
    ChannelIndependentAudioOperation,
    register_operation,
)
from wandas.processing.weighting import A_weight
from wandas.utils.types import NDArrayReal

logger = logging.getLogger(__name__)


def _validate_cutoff(cutoff: float, sampling_rate: float, label: str = "Cutoff") -> None:
    """Validate a single cutoff frequency against the Nyquist limit.

    Args:
        cutoff: float. Cutoff frequency in Hz.
        sampling_rate: float. Sampling rate in Hz.
        label: str. Human-readable name for error messages (e.g. "Lower cutoff").

    Raises:
        ValueError: If cutoff is not in the open interval (0, Nyquist).
    """
    nyquist = sampling_rate / 2
    if cutoff <= 0 or cutoff >= nyquist:
        raise ValueError(
            f"{label} frequency out of valid range\n"
            f"  Got: {cutoff} Hz\n"
            f"  Valid range: 0 < cutoff < {nyquist} Hz (Nyquist frequency)\n"
            f"The Nyquist frequency is half the sampling rate\n"
            f"  ({sampling_rate} Hz).\n"
            f"Filters cannot work above this limit due to aliasing.\n"
            f"Solutions:\n"
            f"  - Use a cutoff frequency below {nyquist} Hz\n"
            f"  - Or increase sampling rate above {cutoff * 2} Hz\n"
            f"    using resample()"
        )


class _ButterworthFilter(ChannelIndependentAudioOperation[NDArrayReal, NDArrayReal]):
    """Shared base for single-cutoff Butterworth filters (high-pass/low-pass)."""

    _btype: str  # "high" or "low" — set by subclasses
    _display: str  # set by subclasses
    _sos: NDArrayReal

    def __init__(self, sampling_rate: float, cutoff: float, order: int = 4):
        super().__init__(sampling_rate, cutoff=cutoff, order=order)

    @property
    def cutoff(self) -> float:
        """Cutoff frequency captured at operation construction time."""
        return self._config_value("cutoff")

    @property
    def order(self) -> int:
        """Filter order captured at operation construction time."""
        return self._config_value("order")

    def validate_params(self) -> None:
        _validate_cutoff(self.cutoff, self.sampling_rate, "Cutoff")

    def _normalized_cutoff(self) -> float | list[float]:
        return self.cutoff / (0.5 * self.sampling_rate)

    def _setup_processor(self) -> None:
        self._sos = signal.butter(self.order, self._normalized_cutoff(), btype=self._btype, output="sos")
        logger.debug(f"{self._display} filter sections calculated: sos={self._sos}")

    def _transfer_function(self) -> tuple[NDArrayReal, NDArrayReal]:
        # Designed directly so the coefficients keep their minimal length
        # (sos2tf would carry the zero padding of an odd-order section).
        b, a = signal.butter(self.order, self._normalized_cutoff(), btype=self._btype)
        return b, a

    @property
    def sos(self) -> NDArrayReal:
        """Return a defensive copy of the second-order sections used for filtering."""
        return self._sos.copy()

    @property
    def b(self) -> NDArrayReal:
        """Return the numerator of the equivalent transfer function.

        Filtering uses :attr:`sos`; the expanded polynomial can be numerically
        inaccurate for high orders or low normalized cutoffs.
        """
        return self._transfer_function()[0]

    @property
    def a(self) -> NDArrayReal:
        """Return the denominator of the equivalent transfer function.

        Filtering uses :attr:`sos`; the expanded polynomial can be numerically
        inaccurate for high orders or low normalized cutoffs.
        """
        return self._transfer_function()[1]

    def calculate_output_dtype(self, input_dtype: np.dtype[Any], *input_dtypes: np.dtype[Any]) -> np.dtype[Any]:
        return np.dtype(np.float64)

    def _process(self, x: NDArrayReal) -> NDArrayReal:
        logger.debug(f"Applying {self._display} filter to array with shape: {x.shape}")
        result: NDArrayReal = signal.sosfiltfilt(self._sos, x, axis=-1)
        logger.debug(f"Filter applied, returning result with shape: {result.shape}")
        return result


class HighPassFilter(_ButterworthFilter):
    """High-pass filter operation"""

    name = "highpass_filter"
    _btype = "high"
    _display = "hpf"


class LowPassFilter(_ButterworthFilter):
    """Low-pass filter operation"""

    name = "lowpass_filter"
    _btype = "low"
    _display = "lpf"


class BandPassFilter(_ButterworthFilter):
    """Band-pass filter operation"""

    name = "bandpass_filter"
    _btype = "band"
    _display = "bpf"

    def __init__(
        self,
        sampling_rate: float,
        low_cutoff: float,
        high_cutoff: float,
        order: int = 4,
    ):
        """
        Initialize band-pass filter

        Args:
            sampling_rate: float. Sampling rate (Hz)
            low_cutoff: float. Lower cutoff frequency (Hz). Must be between 0 and Nyquist frequency.
            high_cutoff: float. Higher cutoff frequency (Hz). Must be between 0 and Nyquist frequency
                and greater than low_cutoff.
            order: int, optional. Order of the Butterworth prototype, default is 4.
                The resulting band-pass filter has order ``2 * order``, and because it
                is applied forward and backward the magnitude response is squared.

        Raises:
            ValueError: If either cutoff frequency is not within valid range (0 < cutoff < Nyquist),
                or if low_cutoff >= high_cutoff
        """
        # Skip single-cutoff _ButterworthFilter.__init__
        AudioOperation.__init__(self, sampling_rate, low_cutoff=low_cutoff, high_cutoff=high_cutoff, order=order)

    @property
    def low_cutoff(self) -> float:
        """Lower cutoff frequency captured at operation construction time."""
        return self._config_value("low_cutoff")

    @property
    def high_cutoff(self) -> float:
        """Higher cutoff frequency captured at operation construction time."""
        return self._config_value("high_cutoff")

    def validate_params(self) -> None:
        """Validate parameters"""
        low_cutoff = self.low_cutoff
        high_cutoff = self.high_cutoff
        _validate_cutoff(low_cutoff, self.sampling_rate, "Lower cutoff")
        _validate_cutoff(high_cutoff, self.sampling_rate, "Higher cutoff")
        if low_cutoff >= high_cutoff:
            raise ValueError(
                f"Invalid bandpass filter cutoff frequencies\n"
                f"  Lower cutoff: {low_cutoff} Hz\n"
                f"  Higher cutoff: {high_cutoff} Hz\n"
                f"  Problem: Lower cutoff must be less than higher cutoff\n"
                f"A bandpass filter passes frequencies between low and high\n"
                f"  cutoffs.\n"
                f"Ensure low_cutoff < high_cutoff\n"
                f"  (e.g., low_cutoff=100, high_cutoff=1000)"
            )

    def _normalized_cutoff(self) -> float | list[float]:
        nyquist = 0.5 * self.sampling_rate
        return [self.low_cutoff / nyquist, self.high_cutoff / nyquist]


def _released_filtfilt(operation: _ButterworthFilter, x: NDArrayReal) -> NDArrayReal:
    """Apply the released Recipe v1 transfer-function ``filtfilt`` kernel."""
    b, a = operation._transfer_function()
    result: NDArrayReal = signal.filtfilt(b, a, x, axis=1)
    return result


class _RecipeHighPassFilterV1(HighPassFilter):
    """Released Recipe v1 high-pass numerics retained for exact replay."""

    name = "_recipe_highpass_filter_v1"

    def _process(self, x: NDArrayReal) -> NDArrayReal:
        return _released_filtfilt(self, x)


class _RecipeLowPassFilterV1(LowPassFilter):
    """Released Recipe v1 low-pass numerics retained for exact replay."""

    name = "_recipe_lowpass_filter_v1"

    def _process(self, x: NDArrayReal) -> NDArrayReal:
        return _released_filtfilt(self, x)


class _RecipeBandPassFilterV1(BandPassFilter):
    """Released Recipe v1 band-pass numerics retained for exact replay."""

    name = "_recipe_bandpass_filter_v1"

    def _process(self, x: NDArrayReal) -> NDArrayReal:
        return _released_filtfilt(self, x)


class AWeighting(ChannelIndependentAudioOperation[NDArrayReal, NDArrayReal]):
    """Apply the implemented digital A-frequency-weighting curve.

    The output is a linear waveform. This operation does not calculate RMS,
    convert to dB, or establish sound-level-meter conformance.
    """

    name = "a_weighting"
    _display = "Aw"

    def __init__(self, sampling_rate: float):
        """
        Initialize A-weighting filter

        Args:
            sampling_rate: float. Sampling rate (Hz)
        """
        super().__init__(sampling_rate)

    def calculate_output_dtype(self, input_dtype: np.dtype[Any], *input_dtypes: np.dtype[Any]) -> np.dtype[Any]:
        return np.dtype(np.float64)

    def _process(self, x: NDArrayReal) -> NDArrayReal:
        """Create processor function for A-weighting filter"""
        logger.debug(f"Applying A-weighting to array with shape: {x.shape}")
        result = A_weight(x, self.sampling_rate)

        # Handle case where A_weight returns a tuple
        if isinstance(result, tuple):
            # Use the first element of the tuple
            result = result[0]

        logger.debug(f"A-weighting applied, returning result with shape: {result.shape}")
        return np.array(result)


# Register all operations, including the private Recipe v1 implementations
# used directly by replay handlers.
for op_class in [
    HighPassFilter,
    LowPassFilter,
    BandPassFilter,
    AWeighting,
    _RecipeHighPassFilterV1,
    _RecipeLowPassFilterV1,
    _RecipeBandPassFilterV1,
]:
    register_operation(op_class)
