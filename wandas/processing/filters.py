import logging

import numpy as np
from scipy import signal

from wandas.processing.base import AudioOperation, register_operation
from wandas.processing.weighting import A_weight
from wandas.utils.types import NDArrayReal

logger = logging.getLogger(__name__)


def _validate_cutoff(cutoff: float, sampling_rate: float, label: str = "Cutoff") -> None:
    """Validate a single cutoff frequency against the Nyquist limit.

    Parameters
    ----------
    cutoff : float
        Cutoff frequency in Hz.
    sampling_rate : float
        Sampling rate in Hz.
    label : str
        Human-readable name for error messages (e.g. "Lower cutoff").

    Raises
    ------
    ValueError
        If cutoff is not in the open interval (0, Nyquist).
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


class _ButterworthFilter(AudioOperation[NDArrayReal, NDArrayReal]):
    """Shared base for single-cutoff Butterworth filters (high-pass/low-pass)."""

    _btype: str  # "high" or "low" — set by subclasses
    _display: str  # set by subclasses
    a: NDArrayReal
    b: NDArrayReal

    def __init__(self, sampling_rate: float, cutoff: float, order: int = 4):
        self.cutoff = cutoff
        self.order = order
        super().__init__(sampling_rate, cutoff=cutoff, order=order)

    def validate_params(self) -> None:
        _validate_cutoff(self.cutoff, self.sampling_rate, "Cutoff")

    def _setup_processor(self) -> None:
        normal_cutoff = self.cutoff / (0.5 * self.sampling_rate)
        self.b, self.a = signal.butter(self.order, normal_cutoff, btype=self._btype)  # type: ignore [unused-ignore]
        logger.debug(f"{self._display} filter coefficients calculated: b={self.b}, a={self.a}")

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        logger.debug(f"Applying {self._display} filter to array with shape: {x.shape}")
        result: NDArrayReal = signal.filtfilt(self.b, self.a, x, axis=1)
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

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        low_cutoff : float
            Lower cutoff frequency (Hz). Must be between 0 and Nyquist frequency.
        high_cutoff : float
            Higher cutoff frequency (Hz). Must be between 0 and Nyquist frequency
            and greater than low_cutoff.
        order : int, optional
            Filter order, default is 4

        Raises
        ------
        ValueError
            If either cutoff frequency is not within valid range (0 < cutoff < Nyquist),
            or if low_cutoff >= high_cutoff
        """
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.order = order
        # Skip single-cutoff _ButterworthFilter.__init__
        AudioOperation.__init__(self, sampling_rate, low_cutoff=low_cutoff, high_cutoff=high_cutoff, order=order)

    def validate_params(self) -> None:
        """Validate parameters"""
        _validate_cutoff(self.low_cutoff, self.sampling_rate, "Lower cutoff")
        _validate_cutoff(self.high_cutoff, self.sampling_rate, "Higher cutoff")
        if self.low_cutoff >= self.high_cutoff:
            raise ValueError(
                f"Invalid bandpass filter cutoff frequencies\n"
                f"  Lower cutoff: {self.low_cutoff} Hz\n"
                f"  Higher cutoff: {self.high_cutoff} Hz\n"
                f"  Problem: Lower cutoff must be less than higher cutoff\n"
                f"A bandpass filter passes frequencies between low and high\n"
                f"  cutoffs.\n"
                f"Ensure low_cutoff < high_cutoff\n"
                f"  (e.g., low_cutoff=100, high_cutoff=1000)"
            )

    def _setup_processor(self) -> None:
        """Set up band-pass filter processor"""
        nyquist = 0.5 * self.sampling_rate
        low_normal_cutoff = self.low_cutoff / nyquist
        high_normal_cutoff = self.high_cutoff / nyquist

        # Precompute and save filter coefficients
        self.b, self.a = signal.butter(self.order, [low_normal_cutoff, high_normal_cutoff], btype="band")  # type: ignore [unused-ignore]
        logger.debug(f"Bandpass filter coefficients calculated: b={self.b}, a={self.a}")


class AWeighting(AudioOperation[NDArrayReal, NDArrayReal]):
    """A-weighting filter operation"""

    name = "a_weighting"
    _display = "Aw"

    def __init__(self, sampling_rate: float):
        """
        Initialize A-weighting filter

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        """
        super().__init__(sampling_rate)

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Create processor function for A-weighting filter"""
        logger.debug(f"Applying A-weighting to array with shape: {x.shape}")
        result = A_weight(x, self.sampling_rate)

        # Handle case where A_weight returns a tuple
        if isinstance(result, tuple):
            # Use the first element of the tuple
            result = result[0]

        logger.debug(f"A-weighting applied, returning result with shape: {result.shape}")
        return np.array(result)


# Register all operations
for op_class in [HighPassFilter, LowPassFilter, BandPassFilter, AWeighting]:
    register_operation(op_class)
