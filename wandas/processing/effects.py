import logging
from typing import Any

import numpy as np
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


class Normalize(AudioOperation[NDArrayReal, NDArrayReal]):
    """Normalization operation to adjust signal to target RMS level"""

    name = "normalize"

    def __init__(
        self, sampling_rate: float, target_level: float = -20, channel_wise: bool = True
    ):
        """
        Initialize normalization operation

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        target_level : float
            Target RMS level in dB. Default is -20 dB.
        channel_wise : bool
            If True, normalize each channel individually.
            If False, apply the same scaling to all channels.
            Default is True.
        """
        super().__init__(
            sampling_rate, target_level=target_level, channel_wise=channel_wise
        )
        self.target_level = target_level
        self.channel_wise = channel_wise
        logger.debug(
            f"Initialized Normalize operation with target_level: {target_level} dB, "
            f"channel_wise: {channel_wise}"
        )

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
        """Perform normalization processing"""
        logger.debug(
            f"Applying normalization with shape: {x.shape}, "
            f"target_level: {self.target_level} dB"
        )

        # Calculate current RMS
        if self.channel_wise:
            # Calculate RMS per channel (axis=-1 is the sample axis)
            current_rms = util.calculate_rms(x)
        else:
            # Calculate RMS across all channels
            current_rms = np.sqrt(np.mean(np.square(x)))

        # Convert target level from dB to linear scale
        # Reference is 1.0 for normalized audio
        # Formula: amplitude = 10^(dB/20)
        target_rms = 10 ** (self.target_level / 20)

        # Calculate gain factor
        if self.channel_wise:
            # Avoid division by zero
            gain = np.where(
                current_rms > 1e-10, target_rms / current_rms, np.ones_like(current_rms)
            )
            # Apply gain per channel
            result: NDArrayReal = x * gain
        else:
            # Apply same gain to all channels
            gain_scalar = target_rms / current_rms if current_rms > 1e-10 else 1.0
            result = x * gain_scalar

        logger.debug(
            f"Normalization applied, current_rms: {current_rms}, "
            f"target_rms: {target_rms}"
        )
        return result


# Register all operations
for op_class in [HpssHarmonic, HpssPercussive, AddWithSNR, Normalize]:
    register_operation(op_class)
