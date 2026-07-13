import logging
from typing import Any

import numpy as np
from scipy.signal import windows as sp_windows

from wandas.processing.base import AudioOperation, register_operation
from wandas.utils import util
from wandas.utils.optional_imports import require_librosa_effects
from wandas.utils.types import NDArrayReal

logger = logging.getLogger(__name__)


def _normalize_array(
    x: NDArrayReal,
    norm: float | None,
    axis: int | None,
    threshold: float | None,
    fill: bool | None,
) -> NDArrayReal:
    if norm is None:
        return x
    if threshold is None:
        dtype = np.asarray(x).dtype
        threshold = float(np.finfo(dtype.name).tiny if dtype.kind == "f" else np.finfo(float).tiny)
    elif threshold <= 0:
        raise ValueError("threshold must be strictly positive")
    if norm == 0:
        length = np.sum(x != 0, axis=axis, keepdims=True).astype(float)
    elif norm in {np.inf, -np.inf}:
        dtype = np.asarray(x).dtype
        magnitude_input = x if dtype.kind == "f" else x.astype(float, copy=False)
        magnitude = np.abs(magnitude_input)
        if norm == np.inf:
            length = np.max(magnitude, axis=axis, keepdims=True)
        else:
            length = np.min(magnitude, axis=axis, keepdims=True)
    else:
        magnitude = np.abs(x.astype(float, copy=False))
        length = np.sum(magnitude**norm, axis=axis, keepdims=True) ** (1.0 / norm)

    small = length < threshold
    safe_length = np.where(small, 1.0, length)
    out = x / safe_length
    if fill is True:
        if norm == 0:
            raise ValueError("Cannot normalize with norm=0 and fill=True")
        fill_value = 1.0
        if norm not in {np.inf, -np.inf}:
            axis_length = x.size if axis is None else x.shape[axis]
            fill_value = axis_length ** (-1.0 / norm)
        out = np.where(np.broadcast_to(small, x.shape), fill_value, out)
    elif fill is False:
        out = np.where(np.broadcast_to(small, x.shape), 0.0, out)
    return np.asarray(out)


class _HpssBase(AudioOperation[NDArrayReal, NDArrayReal]):
    supports_generic_replay = True
    """Shared base for HPSS harmonic/percussive extraction."""

    _extract_func: str  # "harmonic" or "percussive" — set by subclasses
    _display: str  # set by subclasses

    def __init__(self, sampling_rate: float, **kwargs: Any):
        self._effects = require_librosa_effects(self.name)
        super().__init__(sampling_rate, **kwargs)

    @property
    def kwargs(self) -> dict[str, Any]:
        """Keyword arguments captured at operation construction time."""
        return self._config_snapshot()

    def _process(self, x: NDArrayReal) -> NDArrayReal:
        logger.debug(f"Applying HPSS {self._extract_func} to array with shape: {x.shape}")
        func = getattr(self._effects, self._extract_func)
        result: NDArrayReal = func(x, **self._config)
        logger.debug(f"HPSS {self._extract_func} applied, returning result with shape: {result.shape}")
        return result


class HpssHarmonic(_HpssBase):
    """HPSS Harmonic operation"""

    name = "hpss_harmonic"
    _extract_func = "harmonic"
    _display = "Hrm"


class HpssPercussive(_HpssBase):
    """HPSS Percussive operation"""

    name = "hpss_percussive"
    _extract_func = "percussive"
    _display = "Prc"


class Normalize(AudioOperation[NDArrayReal, NDArrayReal]):
    supports_generic_replay = True
    """Signal normalization operation."""

    name = "normalize"
    _display = "norm"

    @staticmethod
    def _output_dtype(input_dtype: np.dtype[Any], norm: float | None) -> np.dtype[Any]:
        dtype = np.dtype(input_dtype)
        if norm is None:
            return dtype
        if dtype.kind == "f" and norm in {np.inf, -np.inf}:
            return dtype
        return np.dtype(np.float64)

    def __init__(
        self,
        sampling_rate: float,
        norm: float | None = np.inf,
        axis: int | None = -1,
        threshold: float | None = None,
        fill: bool | None = None,
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
            - 0: Pseudo L0 normalization (divide by number of non-zero elements)
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
            If norm parameter is invalid or threshold is negative
        """
        # Validate norm parameter
        if norm is not None and not isinstance(norm, int | float):
            raise ValueError(
                f"Invalid normalization method\n"
                f"  Got: {type(norm).__name__} ({norm})\n"
                f"  Expected: float, int, np.inf, -np.inf, or None\n"
                f"Norm parameter must be a numeric value or None.\n"
                f"Common values: np.inf (max norm), 2 (L2 norm),\n"
                f"1 (L1 norm), 0 (pseudo L0)"
            )

        # Validate that norm is non-negative (except for -np.inf which is valid)
        if norm is not None and norm < 0 and not np.isneginf(norm):
            raise ValueError(
                f"Invalid normalization method\n"
                f"  Got: {norm}\n"
                f"  Expected: Non-negative value, np.inf, -np.inf, or None\n"
                f"Norm parameter must be non-negative (except -np.inf for min norm).\n"
                f"Common values: np.inf (max norm), 2 (L2 norm),\n"
                f"1 (L1 norm), 0 (pseudo L0)"
            )

        # Validate threshold
        if threshold is not None and threshold <= 0:
            raise ValueError(
                f"Invalid threshold for normalization\n"
                f"  Got: {threshold}\n"
                f"  Expected: Positive value or None\n"
                f"Threshold must be strictly positive.\n"
                f"Typical values: 1e-10 (small threshold), 1e-6 (larger threshold)"
            )

        super().__init__(sampling_rate, norm=norm, axis=axis, threshold=threshold, fill=fill)
        logger.debug(
            f"Initialized Normalize operation with norm={norm}, axis={axis}, threshold={threshold}, fill={fill}"
        )

    @property
    def norm(self) -> float | None:
        """Norm captured at operation construction time."""
        return self._config_value("norm")

    @property
    def axis(self) -> int | None:
        """Axis captured at operation construction time."""
        return self._config_value("axis")

    @property
    def threshold(self) -> float | None:
        """Threshold captured at operation construction time."""
        return self._config_value("threshold")

    @property
    def fill(self) -> bool | None:
        """Fill behavior captured at operation construction time."""
        return self._config_value("fill")

    def _process(self, x: NDArrayReal) -> NDArrayReal:
        """Perform normalization processing"""
        logger.debug(f"Applying normalization to array with shape: {x.shape}, norm={self.norm}, axis={self.axis}")

        result = _normalize_array(
            x,
            norm=self.norm,
            axis=self.axis,
            threshold=self.threshold,
            fill=self.fill,
        )

        logger.debug(f"Normalization applied, returning result with shape: {result.shape}")
        return result

    def calculate_output_dtype(self, input_dtype: np.dtype[Any], *input_dtypes: np.dtype[Any]) -> np.dtype[Any]:
        """Return normalization output dtype metadata."""
        return self._output_dtype(input_dtype, self.norm)


class RemoveDC(AudioOperation[NDArrayReal, NDArrayReal]):
    supports_generic_replay = True
    """Remove DC component (DC offset) from the signal.

    This operation removes the DC component by subtracting the mean value
    from each channel, centering the signal around zero.
    """

    name = "remove_dc"
    _display = "dcRM"

    def __init__(self, sampling_rate: float):
        """Initialize DC removal operation.

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        """
        super().__init__(sampling_rate)
        logger.debug("Initialized RemoveDC operation")

    def calculate_output_dtype(self, input_dtype: np.dtype[Any], *input_dtypes: np.dtype[Any]) -> np.dtype[Any]:
        if np.issubdtype(input_dtype, np.integer):
            return np.dtype(np.float64)
        return np.dtype(input_dtype)

    def _process(self, x: NDArrayReal) -> NDArrayReal:
        """Perform DC removal processing.

        Parameters
        ----------
        x : NDArrayReal
            Input signal array (channels, samples)

        Returns
        -------
        NDArrayReal
            Signal with DC component removed
        """
        logger.debug(f"Removing DC component from array with shape: {x.shape}")

        # Subtract mean along time axis (axis=1 for channel data)
        mean_values = x.mean(axis=-1, keepdims=True)
        result: NDArrayReal = x - mean_values

        logger.debug(f"DC removal applied, returning result with shape: {result.shape}")
        return result


class AddWithSNR(AudioOperation[NDArrayReal, NDArrayReal]):
    supports_generic_replay = False
    """Addition operation considering SNR"""

    name = "add_with_snr"
    _display = "+SNR"
    _expected_input_count = 2
    input_roles = ("signal", "noise")
    replay_handler_path = "wandas.pipeline.calls.apply_add_with_snr"

    def __init__(self, sampling_rate: float, snr: float = 1.0):
        """
        Initialize addition operation considering SNR

        Parameters
        ----------
        sampling_rate : float
            Sampling rate (Hz)
        snr : float
            Signal-to-noise ratio (dB)
        """
        super().__init__(sampling_rate, snr=snr)
        logger.debug(f"Initialized AddWithSNR operation with SNR: {snr} dB")

    @property
    def snr(self) -> float:
        """Signal-to-noise ratio captured at operation construction time."""
        return self._config_value("snr")

    def calculate_output_dtype(self, input_dtype: np.dtype[Any], *input_dtypes: np.dtype[Any]) -> np.dtype[Any]:
        """Promote SNR mixing to at least float32 precision."""
        return np.result_type(input_dtype, *input_dtypes, np.float32)

    def _process(self, x: NDArrayReal, other: NDArrayReal) -> NDArrayReal:
        """Perform addition processing considering SNR."""
        logger.debug(f"Applying SNR-based addition with shape: {x.shape}")
        output_dtype = self.calculate_output_dtype(x.dtype, other.dtype)
        clean = np.asarray(x, dtype=output_dtype)
        noise = np.asarray(other, dtype=output_dtype)

        clean_rms = util.calculate_rms(clean)
        other_rms = util.calculate_rms(noise)
        desired_noise_rms = util.calculate_desired_noise_rms(clean_rms, self.snr)
        gain = desired_noise_rms / other_rms
        result: NDArrayReal = clean + noise * gain
        return np.asarray(result, dtype=output_dtype)


class Fade(AudioOperation[NDArrayReal, NDArrayReal]):
    supports_generic_replay = True
    """Fade operation using a Tukey (tapered cosine) window.

    This operation applies symmetric fade-in and fade-out with the same
    duration. The Tukey window alpha parameter is computed from the fade
    duration so that the tapered portion equals the requested fade length
    at each end.
    """

    name = "fade"
    _display = "fade"

    def __init__(self, sampling_rate: float, fade_ms: float = 50) -> None:
        fade_ms = float(fade_ms)
        super().__init__(sampling_rate, fade_ms=fade_ms)

    @property
    def fade_ms(self) -> float:
        """Fade duration captured at operation construction time."""
        return self._config_value("fade_ms")

    def validate_params(self) -> None:
        if self.fade_ms < 0:
            raise ValueError("fade_ms must be non-negative")

    def _fade_len_for_sampling_rate(self) -> int:
        return round(self.fade_ms * self.sampling_rate / 1000.0)

    @staticmethod
    def calculate_tukey_alpha(fade_len: int, n_samples: int) -> float:
        """Calculate Tukey window alpha parameter from fade length.

        The alpha parameter determines what fraction of the window is tapered.
        For symmetric fade-in/fade-out, alpha = 2 * fade_len / n_samples ensures
        that each side's taper has exactly fade_len samples.

        Parameters
        ----------
        fade_len : int
            Desired fade length in samples for each end (in and out).
        n_samples : int
            Total number of samples in the signal.

        Returns
        -------
        float
            Alpha parameter for scipy.signal.windows.tukey, clamped to [0, 1].

        Examples
        --------
        >>> Fade.calculate_tukey_alpha(fade_len=20, n_samples=200)
        0.2
        >>> Fade.calculate_tukey_alpha(fade_len=100, n_samples=100)
        1.0
        """
        alpha = float(2 * fade_len) / float(n_samples)
        return min(1.0, alpha)

    def calculate_output_dtype(self, input_dtype: np.dtype[Any], *input_dtypes: np.dtype[Any]) -> np.dtype[Any]:
        del input_dtypes
        if self._fade_len_for_sampling_rate() <= 0:
            return np.dtype(input_dtype)
        return np.result_type(input_dtype, np.float64)

    def _process(self, x: NDArrayReal) -> NDArrayReal:
        logger.debug(f"Applying Tukey Fade to array with shape: {x.shape}")

        arr = x
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        n_samples = int(arr.shape[-1])

        # If no fade requested, return input
        fade_len = self._fade_len_for_sampling_rate()

        if fade_len <= 0:
            return arr

        if 2 * fade_len >= n_samples:
            raise ValueError("Fade length too long: 2*fade_ms must be less than signal length")

        # Calculate Tukey window alpha parameter
        alpha = self.calculate_tukey_alpha(fade_len, n_samples)

        # Create tukey window (numpy) and apply
        env = sp_windows.tukey(n_samples, alpha=alpha)

        result: NDArrayReal = arr * env[None, :]
        logger.debug("Tukey fade applied")
        return result


# Register all operations
for op_class in [HpssHarmonic, HpssPercussive, Normalize, RemoveDC, AddWithSNR, Fade]:
    register_operation(op_class)
