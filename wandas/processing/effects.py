import logging
from typing import Any

from dask.array.core import Array as DaArray
from librosa import effects  # type: ignore[attr-defined]

from wandas.processing.base import AudioOperation, register_operation
from wandas.utils import util
from wandas.utils.types import NDArrayReal

logger = logging.getLogger(__name__)


class HpssHarmonic(AudioOperation[NDArrayReal, NDArrayReal]):
    """HPSS Harmonic operation"""

    name = "hpss_harmonic"

    def __init__(
        self,
        sampling_rate: float,
        **kwargs: Any,
    ):
        """
        Initialize HPSS Harmonic

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        """
        self.kwargs = kwargs
        super().__init__(sampling_rate, **kwargs)

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Create processor function for HPSS Harmonic"""
        logger.debug(f"Applying HPSS Harmonic to array with shape: {x.shape}")
        result: NDArrayReal = effects.harmonic(x, **self.kwargs)
        logger.debug(
            f"HPSS Harmonic applied, returning result with shape: {result.shape}"
        )
        return result


class HpssPercussive(AudioOperation[NDArrayReal, NDArrayReal]):
    """HPSS Percussive operation"""

    name = "hpss_percussive"

    def __init__(
        self,
        sampling_rate: float,
        **kwargs: Any,
    ):
        """
        Initialize HPSS Percussive

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        """
        self.kwargs = kwargs
        super().__init__(sampling_rate, **kwargs)

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Create processor function for HPSS Percussive"""
        logger.debug(f"Applying HPSS Percussive to array with shape: {x.shape}")
        result: NDArrayReal = effects.percussive(x, **self.kwargs)
        logger.debug(
            f"HPSS Percussive applied, returning result with shape: {result.shape}"
        )
        return result


class AddWithSNR(AudioOperation[NDArrayReal, NDArrayReal]):
    """Addition operation considering SNR"""

    name = "add_with_snr"

    def __init__(self, sampling_rate: float, other: DaArray, snr: float = 1.0):
        """
        Initialize addition operation considering SNR

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        other : DaArray
            Noise signal to add (channel-frame format)
        snr : float
            Signal-to-noise ratio (dB)
        """
        super().__init__(sampling_rate, other=other, snr=snr)

        self.other = other
        self.snr = snr
        logger.debug(f"Initialized AddWithSNR operation with SNR: {snr} dB")

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Calculate output data shape after operation

        Parameters
        ----------
        input_shape : tuple
            Input data shape

        Returns
        -------
        tuple
            Output data shape (same as input)
        """
        return input_shape

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        """Perform addition processing considering SNR"""
        logger.debug(f"Applying SNR-based addition with shape: {x.shape}")
        other: NDArrayReal = self.other.compute()

        # Use multi-channel versions of calculate_rms and calculate_desired_noise_rms
        clean_rms = util.calculate_rms(x)
        other_rms = util.calculate_rms(other)

        # Adjust noise gain based on specified SNR (apply per channel)
        desired_noise_rms = util.calculate_desired_noise_rms(clean_rms, self.snr)

        # Apply gain per channel using broadcasting
        gain = desired_noise_rms / other_rms
        # Add adjusted noise to signal
        result: NDArrayReal = x + other * gain
        return result


# Register all operations
for op_class in [HpssHarmonic, HpssPercussive, AddWithSNR]:
    register_operation(op_class)
