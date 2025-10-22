import logging
from typing import Any, Union

import numpy as np
from dask.array.core import Array as DaArray
from librosa import effects  # type: ignore[attr-defined]
from librosa import util as librosa_util

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


class Normalize(AudioOperation[NDArrayReal, NDArrayReal]):
    """Signal normalization operation using librosa.util.normalize"""

    name = "normalize"

    def __init__(
        self,
        sampling_rate: float,
        norm: Union[float, None] = np.inf,
        axis: Union[int, None] = -1,
        threshold: Union[float, None] = None,
        fill: Union[bool, None] = None,
    ):
        """
        Initialize normalization operation

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        norm : float or np.inf, default=np.inf
            Norm type. Supported values:
            - np.inf: Maximum absolute value normalization
            - -np.inf: Minimum absolute value normalization
            - 0: Peak normalization
            - float: Lp norm
            - None: No normalization
        axis : int or None, default=-1
            Axis along which to normalize.
            - -1: Normalize along time axis (each channel independently)
            - None: Global normalization across all axes
            - int: Normalize along specified axis
        threshold : float or None, optional
            Threshold below which values are considered zero.
            If None, no threshold is applied.
        fill : bool or None, optional
            Value to fill when the norm is zero.
            If None, the zero vector remains zero.

        Raises
        ------
        ValueError
            If norm parameter is not a valid type
            If axis parameter is not a valid integer or None
        """
        # Validate norm parameter
        if norm is not None and not isinstance(norm, (int, float, np.number)):
            raise ValueError(
                f"Norm parameter must be a number, np.inf, -np.inf, or None.\n"
                f"Received type: {type(norm).__name__}\n"
                f"Received value: {norm}\n"
                f"Hint: Common values are np.inf (max normalization), "
                f"2 (L2 norm), or None (no normalization)."
            )

        # Validate axis parameter
        if axis is not None and not isinstance(axis, (int, np.integer)):
            raise ValueError(
                f"Axis parameter must be an integer or None.\n"
                f"Received type: {type(axis).__name__}\n"
                f"Received value: {axis}\n"
                f"Hint: Use -1 for time axis normalization (per channel) "
                f"or None for global normalization."
            )

        super().__init__(
            sampling_rate, norm=norm, axis=axis, threshold=threshold, fill=fill
        )
        self.norm = norm
        self.axis = axis
        self.threshold = threshold
        self.fill = fill
        logger.debug(
            f"Initialized Normalize operation with norm={norm}, "
            f"axis={axis}, threshold={threshold}, fill={fill}"
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
            f"Applying normalization to array with shape: {x.shape}, "
            f"norm={self.norm}, axis={self.axis}"
        )

        # Apply librosa.util.normalize
        result: NDArrayReal = librosa_util.normalize(
            x, norm=self.norm, axis=self.axis, threshold=self.threshold, fill=self.fill
        )

        logger.debug(
            f"Normalization applied, returning result with shape: {result.shape}"
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
for op_class in [HpssHarmonic, HpssPercussive, Normalize, AddWithSNR]:
    register_operation(op_class)
