import logging

import numpy as np
from scipy import signal
from waveform_analysis import A_weight

from wandas.processing.base import AudioOperation, register_operation
from wandas.utils.types import NDArrayReal

logger = logging.getLogger(__name__)


class HighPassFilter(AudioOperation[NDArrayReal, NDArrayReal]):
    """High-pass filter operation"""

    name = "highpass_filter"

    def __init__(self, sampling_rate: float, cutoff: float, order: int = 4):
        """
        Initialize high-pass filter

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        cutoff : float
            Cutoff frequency (Hz). Must be in range (0, Nyquist frequency).
        order : int, optional
            Filter order, default is 4
            
        Raises
        ------
        ValueError
            If cutoff frequency is <= 0 or >= Nyquist frequency (sampling_rate / 2).
        """
        self.cutoff = cutoff
        self.order = order
        super().__init__(sampling_rate, cutoff=cutoff, order=order)

    def validate_params(self) -> None:
        """Validate parameters
        
        Raises:
            ValueError: If cutoff frequency is invalid (not in range (0, Nyquist)).
        """
        nyquist = self.sampling_rate / 2
        if self.cutoff <= 0:
            raise ValueError(
                f"Cutoff frequency is too low:\n"
                f"  Given: {self.cutoff} Hz\n"
                f"  Minimum: > 0 Hz\n"
                f"\n"
                f"Solution:\n"
                f"  - Use a positive cutoff frequency\n"
                f"  - For high-pass filter, typical range: 20 Hz to {nyquist} Hz\n"
                f"\n"
                f"Background:\n"
                f"  Cutoff frequency defines the frequency below which signals\n"
                f"  are attenuated. It must be positive and physical."
            )
        elif self.cutoff >= nyquist:
            raise ValueError(
                f"Cutoff frequency exceeds Nyquist limit:\n"
                f"  Given: {self.cutoff} Hz\n"
                f"  Nyquist frequency (limit): {nyquist} Hz\n"
                f"  Sampling rate: {self.sampling_rate} Hz\n"
                f"\n"
                f"Solution:\n"
                f"  - Use cutoff < {nyquist} Hz\n"
                f"  - Or increase sampling rate to at least {self.cutoff * 2} Hz using:\n"
                f"    frame.resample(target_sr={int(self.cutoff * 2.5)})\n"
                f"\n"
                f"Background:\n"
                f"  Nyquist frequency is half of the sampling rate.\n"
                f"  Filters cannot work above this limit due to aliasing.\n"
                f"  To filter higher frequencies, you need a higher sampling rate."
            )

    def _setup_processor(self) -> None:
        """Set up high-pass filter processor"""
        # Calculate filter coefficients (once) - safely retrieve from instance variables
        nyquist = 0.5 * self.sampling_rate
        normal_cutoff = self.cutoff / nyquist

        # Precompute and save filter coefficients
        self.b, self.a = signal.butter(self.order, normal_cutoff, btype="high")  # type: ignore [unused-ignore]
        logger.debug(f"Highpass filter coefficients calculated: b={self.b}, a={self.a}")

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Filter processing wrapped with @dask.delayed"""
        logger.debug(f"Applying highpass filter to array with shape: {x.shape}")
        result: NDArrayReal = signal.filtfilt(self.b, self.a, x, axis=1)
        logger.debug(f"Filter applied, returning result with shape: {result.shape}")
        return result


class LowPassFilter(AudioOperation[NDArrayReal, NDArrayReal]):
    """Low-pass filter operation"""

    name = "lowpass_filter"
    a: NDArrayReal
    b: NDArrayReal

    def __init__(self, sampling_rate: float, cutoff: float, order: int = 4):
        """
        Initialize low-pass filter

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        cutoff : float
            Cutoff frequency (Hz). Must be in range (0, Nyquist frequency).
        order : int, optional
            Filter order, default is 4
            
        Raises
        ------
        ValueError
            If cutoff frequency is <= 0 or >= Nyquist frequency (sampling_rate / 2).
        """
        self.cutoff = cutoff
        self.order = order
        super().__init__(sampling_rate, cutoff=cutoff, order=order)

    def validate_params(self) -> None:
        """Validate parameters
        
        Raises:
            ValueError: If cutoff frequency is invalid (not in range (0, Nyquist)).
        """
        nyquist = self.sampling_rate / 2
        if self.cutoff <= 0:
            raise ValueError(
                f"Cutoff frequency is too low:\n"
                f"  Given: {self.cutoff} Hz\n"
                f"  Minimum: > 0 Hz\n"
                f"\n"
                f"Solution:\n"
                f"  - Use a positive cutoff frequency\n"
                f"  - For low-pass filter, typical range: 20 Hz to {nyquist} Hz\n"
                f"\n"
                f"Background:\n"
                f"  Cutoff frequency defines the frequency above which signals\n"
                f"  are attenuated. It must be positive and physical."
            )
        elif self.cutoff >= nyquist:
            raise ValueError(
                f"Cutoff frequency exceeds Nyquist limit:\n"
                f"  Given: {self.cutoff} Hz\n"
                f"  Nyquist frequency (limit): {nyquist} Hz\n"
                f"  Sampling rate: {self.sampling_rate} Hz\n"
                f"\n"
                f"Solution:\n"
                f"  - Use cutoff < {nyquist} Hz\n"
                f"  - Or increase sampling rate to at least {self.cutoff * 2} Hz using:\n"
                f"    frame.resample(target_sr={int(self.cutoff * 2.5)})\n"
                f"\n"
                f"Background:\n"
                f"  Nyquist frequency is half of the sampling rate.\n"
                f"  Filters cannot work above this limit due to aliasing.\n"
                f"  To filter higher frequencies, you need a higher sampling rate."
            )

    def _setup_processor(self) -> None:
        """Set up low-pass filter processor"""
        nyquist = 0.5 * self.sampling_rate
        normal_cutoff = self.cutoff / nyquist

        # Precompute and save filter coefficients
        self.b, self.a = signal.butter(self.order, normal_cutoff, btype="low")  # type: ignore [unused-ignore]
        logger.debug(f"Lowpass filter coefficients calculated: b={self.b}, a={self.a}")

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Filter processing wrapped with @dask.delayed"""
        logger.debug(f"Applying lowpass filter to array with shape: {x.shape}")
        result: NDArrayReal = signal.filtfilt(self.b, self.a, x, axis=1)

        logger.debug(f"Filter applied, returning result with shape: {result.shape}")
        return result


class BandPassFilter(AudioOperation[NDArrayReal, NDArrayReal]):
    """Band-pass filter operation"""

    name = "bandpass_filter"
    a: NDArrayReal
    b: NDArrayReal

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
            Lower cutoff frequency (Hz). Must be in range (0, Nyquist frequency).
        high_cutoff : float
            Higher cutoff frequency (Hz). Must be in range (0, Nyquist frequency)
            and greater than low_cutoff.
        order : int, optional
            Filter order, default is 4
            
        Raises
        ------
        ValueError
            If low_cutoff or high_cutoff is <= 0 or >= Nyquist frequency,
            or if low_cutoff >= high_cutoff.
        """
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.order = order
        super().__init__(
            sampling_rate, low_cutoff=low_cutoff, high_cutoff=high_cutoff, order=order
        )

    def validate_params(self) -> None:
        """Validate parameters
        
        Raises:
            ValueError: If cutoff frequencies are invalid.
        """
        nyquist = self.sampling_rate / 2
        if self.low_cutoff <= 0:
            raise ValueError(
                f"Lower cutoff frequency is too low:\n"
                f"  Given: {self.low_cutoff} Hz\n"
                f"  Minimum: > 0 Hz\n"
                f"\n"
                f"Solution:\n"
                f"  - Use a positive lower cutoff frequency\n"
                f"  - For band-pass filter, typical range: 20 Hz to {nyquist} Hz\n"
                f"\n"
                f"Background:\n"
                f"  Lower cutoff frequency defines the lower edge of the passband.\n"
                f"  It must be positive and physical."
            )
        if self.low_cutoff >= nyquist:
            raise ValueError(
                f"Lower cutoff frequency exceeds Nyquist limit:\n"
                f"  Given: {self.low_cutoff} Hz\n"
                f"  Nyquist frequency (limit): {nyquist} Hz\n"
                f"  Sampling rate: {self.sampling_rate} Hz\n"
                f"\n"
                f"Solution:\n"
                f"  - Use lower cutoff < {nyquist} Hz\n"
                f"  - Or increase sampling rate to at least {self.low_cutoff * 2} Hz\n"
                f"\n"
                f"Background:\n"
                f"  Nyquist frequency is half of the sampling rate.\n"
                f"  Filters cannot work above this limit due to aliasing."
            )
        if self.high_cutoff <= 0:
            raise ValueError(
                f"Higher cutoff frequency is too low:\n"
                f"  Given: {self.high_cutoff} Hz\n"
                f"  Minimum: > 0 Hz\n"
                f"\n"
                f"Solution:\n"
                f"  - Use a positive higher cutoff frequency\n"
                f"  - For band-pass filter, typical range: 20 Hz to {nyquist} Hz"
            )
        if self.high_cutoff >= nyquist:
            raise ValueError(
                f"Higher cutoff frequency exceeds Nyquist limit:\n"
                f"  Given: {self.high_cutoff} Hz\n"
                f"  Nyquist frequency (limit): {nyquist} Hz\n"
                f"  Sampling rate: {self.sampling_rate} Hz\n"
                f"\n"
                f"Solution:\n"
                f"  - Use higher cutoff < {nyquist} Hz\n"
                f"  - Or increase sampling rate to at least {self.high_cutoff * 2} Hz\n"
                f"\n"
                f"Background:\n"
                f"  Nyquist frequency is half of the sampling rate.\n"
                f"  Filters cannot work above this limit due to aliasing."
            )
        if self.low_cutoff >= self.high_cutoff:
            raise ValueError(
                f"Invalid cutoff frequency range:\n"
                f"  Lower cutoff: {self.low_cutoff} Hz\n"
                f"  Higher cutoff: {self.high_cutoff} Hz\n"
                f"\n"
                f"Solution:\n"
                f"  - Ensure lower cutoff < higher cutoff\n"
                f"  - For band-pass filter, use: low_cutoff={self.high_cutoff/2} Hz, high_cutoff={self.high_cutoff} Hz\n"
                f"\n"
                f"Background:\n"
                f"  Band-pass filter requires a valid frequency range.\n"
                f"  Lower cutoff must be less than higher cutoff to define a passband."
            )

    def _setup_processor(self) -> None:
        """Set up band-pass filter processor"""
        nyquist = 0.5 * self.sampling_rate
        low_normal_cutoff = self.low_cutoff / nyquist
        high_normal_cutoff = self.high_cutoff / nyquist

        # Precompute and save filter coefficients
        self.b, self.a = signal.butter(
            self.order, [low_normal_cutoff, high_normal_cutoff], btype="band"
        )  # type: ignore [unused-ignore]
        logger.debug(f"Bandpass filter coefficients calculated: b={self.b}, a={self.a}")

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Filter processing wrapped with @dask.delayed"""
        logger.debug(f"Applying bandpass filter to array with shape: {x.shape}")
        result: NDArrayReal = signal.filtfilt(self.b, self.a, x, axis=1)
        logger.debug(f"Filter applied, returning result with shape: {result.shape}")
        return result


class AWeighting(AudioOperation[NDArrayReal, NDArrayReal]):
    """A-weighting filter operation"""

    name = "a_weighting"

    def __init__(self, sampling_rate: float):
        """
        Initialize A-weighting filter

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        """
        super().__init__(sampling_rate)

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Create processor function for A-weighting filter"""
        logger.debug(f"Applying A-weighting to array with shape: {x.shape}")
        result = A_weight(x, self.sampling_rate)

        # Handle case where A_weight returns a tuple
        if isinstance(result, tuple):
            # Use the first element of the tuple
            result = result[0]

        logger.debug(
            f"A-weighting applied, returning result with shape: {result.shape}"
        )
        return np.array(result)


# Register all operations
for op_class in [HighPassFilter, LowPassFilter, BandPassFilter, AWeighting]:
    register_operation(op_class)
