import logging
from typing import Any, Union

import numpy as np
from dask.array.core import Array as DaArray
from librosa import effects  # type: ignore[attr-defined]
from librosa import util as librosa_util
from scipy.signal import windows as sp_windows

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
        """
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


class Fade(AudioOperation[NDArrayReal, NDArrayReal]):
    """Fade operation using a Tukey (tapered cosine) window.

    This operation applies symmetric fade-in and fade-out with the same
    duration. The Tukey window alpha parameter is computed from the fade
    duration so that the tapered portion equals the requested fade length
    at each end.
    """

    name = "fade"

    def __init__(self, sampling_rate: float, fade_ms: float = 50) -> None:
        self.fade_ms = float(fade_ms)
        # Precompute fade length in samples at construction time
        self.fade_len = int(round(self.fade_ms * float(sampling_rate) / 1000.0))
        super().__init__(sampling_rate, fade_ms=fade_ms)

    def validate_params(self) -> None:
        if self.fade_ms < 0:
            raise ValueError("fade_ms must be non-negative")

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        logger.debug(f"Applying Tukey Fade to array with shape: {x.shape}")

        arr = x
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        n_samples = int(arr.shape[-1])

        # If no fade requested, return input
        if self.fade_len <= 0:
            return arr

        if 2 * self.fade_len >= n_samples:
            raise ValueError(
                "Fade length too long: 2*fade_ms must be less than signal length"
            )

        # alpha is fraction of the window that is tapered (total), so that
        # each side taper length == fade_len -> alpha = 2*fade_len / n_samples
        alpha = float(2 * self.fade_len) / float(n_samples)
        alpha = min(1.0, alpha)

        # Create tukey window (numpy) and apply
        env = sp_windows.tukey(n_samples, alpha=alpha)

        result: NDArrayReal = arr * env[None, :]
        logger.debug("Tukey fade applied")
        return result


# Register all operations
for op_class in [HpssHarmonic, HpssPercussive, Normalize, AddWithSNR, Fade]:
    register_operation(op_class)
