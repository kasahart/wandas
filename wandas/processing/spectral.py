import logging
import math
import numbers
from collections.abc import Mapping
from typing import Any, Literal, cast

import numpy as np
import scipy
from dask.array.core import Array as DaArray
from scipy import fft as sp_fft

from wandas.processing.base import AudioOperation, ChannelIndependentAudioOperation
from wandas.processing.spectral_contracts import (
    coherence_from_cross_spectra,
    cross_spectral_auto_power,
    flatten_output_input_pairs,
    transfer_function_ratio,
)
from wandas.utils.optional_imports import require_mosqito_center_freq, require_mosqito_sound_level_meter
from wandas.utils.types import NDArrayComplex, NDArrayReal

logger = logging.getLogger(__name__)


def noct_spectrum(*args: Any, **kwargs: Any) -> Any:
    return require_mosqito_sound_level_meter("noct_spectrum").noct_spectrum(*args, **kwargs)


def noct_synthesis(*args: Any, **kwargs: Any) -> Any:
    return require_mosqito_sound_level_meter("noct_synthesis").noct_synthesis(*args, **kwargs)


def _center_freq(*args: Any, **kwargs: Any) -> Any:
    return require_mosqito_center_freq("NOctFrame")(*args, **kwargs)


def noct_band_labels(f_exact: Any, f_nom: Any) -> NDArrayReal:
    """Return one center-frequency label per band that MoSQITo computes.

    MoSQITo computes a band for every exact center frequency between ``fmin``
    and ``fmax``, but its nominal labels cover only its fixed tables (25 Hz to
    20 kHz for 1/3 octave, 31.5 Hz to 16 kHz for octave). A band with a
    nominal label keeps it; a band outside the table is labelled with its
    exact center frequency.

    Args:
        f_exact: Exact center frequencies returned by MoSQITo.
        f_nom: Nominal center frequencies returned by MoSQITo, a contiguous
            run of the bands in ``f_exact``.

    Returns:
        Labels with the same length as ``f_exact``.
    """
    nominal = np.asarray(f_nom)
    exact = np.asarray(f_exact, dtype=np.float64)
    if nominal.size == exact.size:
        return nominal
    labels = exact.copy()
    if nominal.size:
        # Nominal values are within about 1 % of their exact band, while
        # adjacent bands are at least a third of an octave apart.
        start = int(np.argmin(np.abs(np.log(exact / float(nominal[0])))))
        labels[start : start + nominal.size] = nominal
    return labels


def _validate_noct_g(value: Any) -> None:
    """Validate the exact N-octave ratio convention accepted by MoSQITo."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(
            "Invalid N-octave ratio G\n"
            f"  Got: {value!r} ({type(value).__name__})\n"
            "  Expected: integer 2 or 10\n"
            "N-octave ratio conventions support only G=2 or G=10.\n"
            "Specify G=2 or G=10."
        )
    normalized = int(value)
    if normalized not in (2, 10):
        raise ValueError(
            "Invalid N-octave ratio G\n"
            f"  Got: {value!r}\n"
            "  Expected: integer 2 or 10\n"
            "N-octave ratio conventions support only G=2 or G=10.\n"
            "Specify G=2 or G=10."
        )


def validate_noct_recipe_params(params: Mapping[str, Any]) -> None:
    """Validate portable N-octave parameters without importing MoSQITo."""
    if "G" in params:
        _validate_noct_g(params["G"])


def validate_stft_recipe_params(params: Mapping[str, Any]) -> None:
    """Validate sparse opt-in before constructing a portable STFT graph."""
    unknown = set(params) - {"n_fft", "hop_length", "win_length", "window", "allow_sparse", "detrend"}
    if unknown:
        raise ValueError(f"Unsupported STFT Recipe parameters: {sorted(unknown)}")
    if type(params.get("allow_sparse", False)) is not bool:
        raise TypeError("allow_sparse must be a bool")
    detrend = params.get("detrend")
    if detrend is not None and detrend not in ("constant", "linear"):
        raise ValueError(
            "Invalid detrend for STFT\n"
            f"  Got: {detrend!r}\n"
            "  Expected: None, 'constant' or 'linear'\n"
            "Use 'constant' to remove each frame's mean or 'linear' to remove its linear trend."
        )


def _spectral_real_dtype(input_dtype: np.dtype[Any]) -> np.dtype[Any]:
    return np.dtype(np.result_type(input_dtype, np.float32))


def _spectral_complex_dtype(input_dtype: np.dtype[Any]) -> np.dtype[Any]:
    return np.dtype(np.result_type(_spectral_real_dtype(input_dtype), np.complex64))


def _normalize_rfft_amplitude(
    spectrum: NDArrayComplex,
    *,
    n_fft: int,
    window_gain: float,
    axis: int = -1,
) -> NDArrayComplex:
    """Return a one-sided spectrum with coherent-gain amplitude scaling.

    Every positive-frequency bin is doubled except the Nyquist bin, which exists
    only for an even FFT size. The input spectrum is not mutated.
    """
    normalized = np.asarray(spectrum).copy()
    normalized[_rfft_positive_frequency_bins(normalized.ndim, n_fft=n_fft, axis=axis)] *= 2.0
    normalized /= window_gain
    return normalized


def _denormalize_rfft_amplitude(
    spectrum: NDArrayComplex,
    *,
    n_fft: int,
    window_gain: float,
    axis: int = -1,
) -> NDArrayComplex:
    """Undo :func:`_normalize_rfft_amplitude` without mutating input."""
    restored = np.asarray(spectrum).copy()
    restored *= window_gain
    restored[_rfft_positive_frequency_bins(restored.ndim, n_fft=n_fft, axis=axis)] /= 2.0
    return restored


def _rfft_positive_frequency_bins(
    ndim: int,
    *,
    n_fft: int,
    axis: int,
) -> tuple[slice, ...]:
    """Return an index selecting one-sided positive bins except Nyquist."""
    axis_index = axis if axis >= 0 else ndim + axis
    if not 0 <= axis_index < ndim:
        raise ValueError(f"Frequency axis {axis} is invalid for {ndim}D spectrum.")
    bins = [slice(None)] * ndim
    bins[axis_index] = slice(1, -1 if n_fft % 2 == 0 else None)
    return tuple(bins)


def _validate_spectral_params(
    n_fft: int,
    win_length: int | None,
    hop_length: int | None,
    method_name: str,
    *,
    allow_sparse: bool = False,
) -> tuple[int, int]:
    """
    Validate and compute spectral analysis parameters.

    Args:
        n_fft: int. FFT size
        win_length: int or None. Window length (None means use n_fft)
        hop_length: int or None. Hop length (None means use win_length // 4)
        method_name: str. Name of the method for error messages (e.g., "STFT", "Welch method")

    Returns:
        tuple[int, int]: (actual_win_length, actual_hop_length)

    Raises:
        ValueError: If parameters are invalid
    """
    validate_stft_recipe_params({"allow_sparse": allow_sparse})

    # Validate n_fft
    if n_fft <= 0:
        raise ValueError(
            f"Invalid FFT size for {method_name}\n"
            f"  Got: {n_fft}\n"
            f"  Expected: Positive integer > 0\n"
            f"FFT size must be a positive integer.\n"
            f"Common values: 512, 1024, 2048, 4096 (powers of 2 are most efficient)"
        )

    # Set win_length with default
    actual_win_length = win_length if win_length is not None else n_fft

    # Validate win_length - check positive first, then relationship
    if actual_win_length <= 0:
        raise ValueError(
            f"Invalid window length for {method_name}\n"
            f"  Got: {actual_win_length}\n"
            f"  Expected: Positive integer > 0\n"
            f"Window length must be a positive integer.\n"
            f"Typical values: same as n_fft ({n_fft}) or slightly smaller"
        )

    if actual_win_length > n_fft:
        raise ValueError(
            f"Invalid window length for {method_name}\n"
            f"  Got: win_length={actual_win_length}\n"
            f"  Expected: win_length <= n_fft ({n_fft})\n"
            f"Window length cannot exceed FFT size.\n"
            f"Use win_length={n_fft} or smaller, or increase n_fft to\n"
            f"{actual_win_length} or larger"
        )

    # Set hop_length with default
    if hop_length is None:
        if actual_win_length < 4:
            raise ValueError(
                f"Window length too small to compute default hop length for\n"
                f"{method_name}\n"
                f"  Got: win_length={actual_win_length}\n"
                f"  Expected: win_length >= 4 when hop_length is not specified\n"
                f"Default hop_length is computed as win_length // 4, which would be\n"
                f"zero for win_length < 4.\n"
                f"Please specify a larger win_length or provide hop_length explicitly."
            )
        actual_hop_length = actual_win_length // 4
    else:
        actual_hop_length = hop_length

    # Validate hop_length
    if actual_hop_length <= 0:
        raise ValueError(
            f"Invalid hop length for {method_name}\n"
            f"  Got: {actual_hop_length}\n"
            f"  Expected: Positive integer > 0\n"
            f"Hop length must be a positive integer.\n"
            f"Typical value: win_length // 4 = {actual_win_length // 4}"
        )

    if actual_hop_length > actual_win_length and not allow_sparse:
        if method_name == "ISTFT":
            raise ValueError(
                "Invalid hop length for ISTFT\n"
                "ISTFT cannot reconstruct sparse STFT windows with gaps (hop_length > win_length)"
            )
        raise ValueError(
            f"Invalid hop length for {method_name}\n"
            f"  Got: hop_length={actual_hop_length}\n"
            f"  Expected: hop_length <= win_length ({actual_win_length})\n"
            f"Hop length cannot exceed window length (would create gaps).\n"
            f"Use hop_length={actual_win_length} or smaller for proper overlap"
        )

    return actual_win_length, actual_hop_length


class FFT(AudioOperation[NDArrayReal, NDArrayComplex]):
    """One-sided, coherent-gain-normalized peak-amplitude FFT.

    The input is truncated to ``n_fft``; the selected window spans the
    remaining ``min(len(x), n_fft)`` samples and is applied before zero-padding
    to ``n_fft``. The coherent gain is taken from that same window, so a short
    input keeps its amplitude. DC and Nyquist bins retain their real-FFT
    scaling; every other positive-frequency bin is doubled. The complex result
    therefore has the same physical unit as the input, and an on-bin sinusoid's
    magnitude is its peak amplitude.
    """

    name = "fft"
    _display = "FFT"

    def __init__(self, sampling_rate: float, n_fft: int | None = None, window: str = "hann"):
        """
        Initialize FFT operation

        Args:
            sampling_rate: float. Sampling rate (Hz)
            n_fft: int, optional. FFT size, default is None (determined by input size)
            window: str, optional. Window function type, default is 'hann'

        Raises:
            ValueError: If n_fft is not a positive integer
        """
        # Validate n_fft parameter
        if n_fft is not None and n_fft <= 0:
            raise ValueError(
                f"Invalid FFT size\n"
                f"  Got: {n_fft}\n"
                f"  Expected: Positive integer > 0\n"
                f"FFT size must be a positive integer.\n"
                f"Common values: 512, 1024, 2048, 4096,\n"
                f"8192 (powers of 2 are most efficient)"
            )

        super().__init__(sampling_rate, n_fft=n_fft, window=window)

    @property
    def n_fft(self) -> int | None:
        """FFT size captured at operation construction time."""
        return self._config_value("n_fft")

    @property
    def window(self) -> str:
        """Window name captured at operation construction time."""
        return self._config_value("window")

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Calculate output data shape after the operation.

        Args:
            input_shape: tuple. Input data shape (channels, samples).

        Returns:
            tuple: Output data shape (channels, freqs).
        """
        n_fft = self.n_fft
        n_freqs = n_fft // 2 + 1 if n_fft else input_shape[-1] // 2 + 1
        return (*input_shape[:-1], n_freqs)

    def calculate_output_dtype(self, input_dtype: np.dtype[Any], *input_dtypes: np.dtype[Any]) -> np.dtype[Any]:
        return np.dtype(np.complex128)

    def _process(self, x: NDArrayReal) -> NDArrayComplex:
        """Window the available samples, then zero-pad to ``n_fft``."""
        fft_size = int(x.shape[-1]) if self.n_fft is None else self.n_fft
        x = x[..., :fft_size]
        win = scipy.signal.get_window(self.window, x.shape[-1])
        result: NDArrayComplex = np.fft.rfft(x * win, n=fft_size, axis=-1)
        return _normalize_rfft_amplitude(
            result,
            n_fft=fft_size,
            window_gain=float(np.sum(win)),
        )


class _RecipeFFTV2(FFT):
    """Released Recipe v2 FFT: zero-pad to ``n_fft`` before an ``n_fft``-long window."""

    name = "_recipe_fft_v2"
    _display = "FFT Recipe v2"

    def _process(self, x: NDArrayReal) -> NDArrayComplex:
        """Apply the released zero-padding-before-window preparation order."""
        fft_size = int(x.shape[-1]) if self.n_fft is None else self.n_fft
        if x.shape[-1] >= fft_size:
            x = x[..., :fft_size]
        else:
            x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, fft_size - x.shape[-1])])

        win = scipy.signal.get_window(self.window, fft_size)
        x = x * win
        result: NDArrayComplex = np.fft.rfft(x, n=fft_size, axis=-1)
        scaling_factor = np.sum(win)
        return _normalize_rfft_amplitude(
            result,
            n_fft=fft_size,
            window_gain=float(scaling_factor),
        )


class IFFT(AudioOperation[NDArrayComplex, NDArrayReal]):
    """Inverse of Wandas' one-sided peak-amplitude FFT normalization.

    For a spectrum produced by :class:`FFT` with matching ``n_fft``,
    ``window`` and ``win_length``, the result is the analysed samples
    multiplied by that analysis window and zero-padded to ``n_fft``. A boxcar
    window therefore reconstructs the prepared input exactly; tapered windows
    intentionally reconstruct the windowed waveform rather than guessing
    samples discarded by the analysis window.
    """

    name = "ifft"
    _display = "iFFT"

    def __init__(
        self,
        sampling_rate: float,
        n_fft: int | None = None,
        window: str = "hann",
        win_length: int | None = None,
    ):
        """
        Initialize IFFT operation

        Args:
            sampling_rate: float. Sampling rate (Hz)
            n_fft: Optional[int], optional. IFFT size, default is None (determined based on input size)
            window: str, optional. Window function type, default is 'hann'
            win_length: Optional[int], optional. Length of the analysis window whose
                coherent gain is undone, default is None (the IFFT size)
        """
        config: dict[str, Any] = {"n_fft": n_fft, "window": window}
        # Omitted when unset so existing IFFT configurations stay identical.
        if win_length is not None:
            if (
                isinstance(win_length, bool)
                or not isinstance(win_length, int | np.integer)
                or win_length < 1
                or (n_fft is not None and win_length > n_fft)
            ):
                upper_bound = "" if n_fft is None else f" no larger than n_fft ({n_fft})"
                raise ValueError(
                    "Invalid win_length for IFFT\n"
                    f"  Got: {win_length!r}\n"
                    f"  Expected: a positive integer{upper_bound}\n"
                    "Pass the length of the analysis window applied before zero-padding."
                )
            config["win_length"] = int(win_length)
        super().__init__(sampling_rate, **config)

    @property
    def win_length(self) -> int | None:
        """Analysis window length captured at operation construction time."""
        return self._config.get("win_length")

    @property
    def n_fft(self) -> int | None:
        """IFFT size captured at operation construction time."""
        return self._config_value("n_fft")

    @property
    def window(self) -> str:
        """Window name captured at operation construction time."""
        return self._config_value("window")

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Calculate output data shape after operation

        Args:
            input_shape: tuple. Input data shape (channels, freqs)

        Returns:
            tuple: Output data shape (channels, samples)
        """
        n_fft = self.n_fft
        n_samples = 2 * (input_shape[-1] - 1) if n_fft is None else n_fft
        return (*input_shape[:-1], n_samples)

    def calculate_output_dtype(self, input_dtype: np.dtype[Any], *input_dtypes: np.dtype[Any]) -> np.dtype[Any]:
        return np.dtype(np.float64)

    def _process(self, x: NDArrayComplex) -> NDArrayReal:
        """Invert Wandas peak-amplitude scaling to a windowed waveform."""
        logger.debug(f"Applying IFFT to array with shape: {x.shape}")

        fft_size = 2 * (int(x.shape[-1]) - 1) if self.n_fft is None else self.n_fft
        if self.win_length is not None and self.win_length > fft_size:
            raise ValueError(
                "Invalid win_length for IFFT\n"
                f"  Got: {self.win_length}\n"
                f"  Expected: at most the IFFT size ({fft_size}) inferred from the input\n"
                "Pass n_fft, or a win_length that fits the spectrum."
            )
        win = scipy.signal.get_window(self.window, fft_size if self.win_length is None else self.win_length)
        _x = _denormalize_rfft_amplitude(
            x,
            n_fft=fft_size,
            window_gain=float(np.sum(win)),
        )

        result: NDArrayReal = np.fft.irfft(_x, n=fft_size, axis=-1)

        logger.debug(f"IFFT applied, returning result with shape: {result.shape}")
        return result


class _RecipeIFFTV2(IFFT):
    """Released Recipe v2 IFFT: undo the gain of an ``n_fft``-long window."""

    name = "_recipe_ifft_v2"
    _display = "iFFT Recipe v2"

    def _process(self, x: NDArrayComplex) -> NDArrayReal:
        """Apply the released normalization, which ignores ``win_length``."""
        fft_size = 2 * (int(x.shape[-1]) - 1) if self.n_fft is None else self.n_fft
        win = scipy.signal.get_window(self.window, fft_size)
        _x = _denormalize_rfft_amplitude(x, n_fft=fft_size, window_gain=float(np.sum(win)))
        result: NDArrayReal = np.fft.irfft(_x, n=fft_size, axis=-1)
        return result


class STFT(AudioOperation[NDArrayReal, NDArrayComplex]):
    """One-sided peak-amplitude Short-Time Fourier Transform.

    Each frame uses SciPy's coherent-gain magnitude scaling, with non-DC and
    non-Nyquist positive-frequency bins doubled. Values retain the input
    physical unit.
    """

    name = "stft"
    _display = "STFT"

    def __init__(
        self,
        sampling_rate: float,
        n_fft: int = 2048,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
        *,
        allow_sparse: bool = False,
        detrend: str | None = None,
    ):
        """
        Initialize STFT operation

        Args:
            sampling_rate: float. Sampling rate (Hz)
            n_fft: int. FFT size, default is 2048
            hop_length: int, optional. Number of samples between frames. Default is win_length // 4
            win_length: int, optional. Window length. Default is n_fft
            window: str. Window type, default is 'hann'
            allow_sparse: Permit gaps between windows when hop_length exceeds win_length.
            detrend: None, ``"constant"`` or ``"linear"``. Trend removed from each frame's
                samples before windowing, as in SciPy ``ShortTimeFFT.stft_detrend``.

        Raises:
            ValueError: If n_fft is not positive, win_length > n_fft, or hop_length is invalid
        """
        validate_stft_recipe_params({"detrend": detrend})
        # Validate and compute parameters
        actual_win_length, actual_hop_length = _validate_spectral_params(
            n_fft, win_length, hop_length, "STFT", allow_sparse=allow_sparse
        )

        self._SFT = scipy.signal.ShortTimeFFT(
            win=scipy.signal.get_window(window, actual_win_length),
            hop=actual_hop_length,
            fs=sampling_rate,
            mfft=n_fft,
            scale_to="magnitude",
        )
        config: dict[str, Any] = {
            "n_fft": n_fft,
            "win_length": actual_win_length,
            "hop_length": actual_hop_length,
            "window": window,
            "allow_sparse": allow_sparse,
        }
        # Omitted when unset so existing STFT configurations stay identical.
        if detrend is not None:
            config["detrend"] = detrend
        super().__init__(sampling_rate, **config)

    @property
    def n_fft(self) -> int:
        """FFT size captured at operation construction time."""
        return self._config_value("n_fft")

    @property
    def detrend(self) -> str | None:
        """Per-frame trend removal captured at operation construction time."""
        return self._config.get("detrend")

    @property
    def win_length(self) -> int:
        """Window length captured at operation construction time."""
        return self._config_value("win_length")

    @property
    def hop_length(self) -> int:
        """Hop length captured at operation construction time."""
        return self._config_value("hop_length")

    @property
    def window(self) -> str:
        """Window name captured at operation construction time."""
        return self._config_value("window")

    @property
    def frame_time_origin(self) -> float:
        """First actual frame center relative to the input start, in seconds.

        SciPy's padding-dependent minimum frame index is determined without
        computing signal samples. This derived domain state is not a Recipe
        parameter or a second copy of the window/hop configuration.
        """
        return float(self._SFT.p_min * self._SFT.delta_t)

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Calculate output data shape after operation

        Args:
            input_shape: tuple. Input data shape

        Returns:
            tuple: Output data shape
        """
        n_channels = input_shape[0]
        n_samples = input_shape[-1]
        n_f = len(self._SFT.f)
        n_t = len(self._SFT.t(n_samples))
        return (n_channels, n_f, n_t)

    def calculate_output_dtype(self, input_dtype: np.dtype[Any], *input_dtypes: np.dtype[Any]) -> np.dtype[Any]:
        return np.dtype(np.complex128)

    def _process(self, x: NDArrayReal) -> NDArrayComplex:
        """Apply SciPy STFT processing to multiple channels at once"""
        logger.debug(f"Applying SciPy STFT to array with shape: {x.shape}")

        # Convert 1D input to 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Apply STFT to all channels at once
        result: NDArrayComplex = (
            self._SFT.stft(x)
            if self.detrend is None
            else self._SFT.stft_detrend(x, cast(Literal["constant", "linear"], self.detrend))
        )
        result = _normalize_rfft_amplitude(
            result,
            n_fft=self.n_fft,
            window_gain=1.0,
            axis=-2,
        )
        logger.debug(f"SciPy STFT applied, returning result with shape: {result.shape}")
        return result


class ISTFT(AudioOperation[NDArrayComplex, NDArrayReal]):
    """Inverse Short-Time Fourier Transform operation"""

    name = "istft"
    _display = "iSTFT"

    def __init__(
        self,
        sampling_rate: float,
        n_fft: int = 2048,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
        length: int | None = None,
    ):
        """
        Initialize ISTFT operation

        Args:
            sampling_rate: float. Sampling rate (Hz)
            n_fft: int. FFT size, default is 2048
            hop_length: int, optional. Number of samples between frames. Default is win_length // 4
            win_length: int, optional. Window length. Default is n_fft
            window: str. Window type, default is 'hann'
            length: int, optional. Length of output signal. Default is None (determined from input)

        Raises:
            ValueError: If n_fft is not positive, win_length > n_fft, or hop_length is invalid
        """
        # Validate and compute parameters
        actual_win_length, actual_hop_length = _validate_spectral_params(n_fft, win_length, hop_length, "ISTFT")

        # Instantiate ShortTimeFFT for ISTFT calculation
        self._SFT = scipy.signal.ShortTimeFFT(
            win=scipy.signal.get_window(window, actual_win_length),
            hop=actual_hop_length,
            fs=sampling_rate,
            mfft=n_fft,
            scale_to="magnitude",  # Consistent scaling with STFT
        )

        super().__init__(
            sampling_rate,
            n_fft=n_fft,
            win_length=actual_win_length,
            hop_length=actual_hop_length,
            window=window,
            length=length,
        )

    @property
    def n_fft(self) -> int:
        """FFT size captured at operation construction time."""
        return self._config_value("n_fft")

    @property
    def win_length(self) -> int:
        """Window length captured at operation construction time."""
        return self._config_value("win_length")

    @property
    def hop_length(self) -> int:
        """Hop length captured at operation construction time."""
        return self._config_value("hop_length")

    @property
    def window(self) -> str:
        """Window name captured at operation construction time."""
        return self._config_value("window")

    @property
    def length(self) -> int | None:
        """Output length captured at operation construction time."""
        return self._config_value("length")

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Calculate output data shape after ISTFT operation.

        Uses the SciPy ShortTimeFFT calculation formula to compute the expected
        output length based on the input spectrogram dimensions and output range
        parameters (k0, k1).

        Args:
            input_shape: tuple. Input spectrogram shape (channels, n_freqs, n_frames)
                where n_freqs = n_fft // 2 + 1 and n_frames is the number of time frames.

        Returns:
            tuple: Output shape (channels, output_samples) where output_samples is the
                reconstructed signal length determined by the output range [k0, k1).

        Notes:
            The calculation follows SciPy's ShortTimeFFT.istft() implementation.
            When k1 is None (default), the maximum reconstructible signal length is
            computed as:

            .. math::

            q_{max} = n_{frames} + p_{min}

            k_{max} = (q_{max} - 1) \\cdot hop + m_{num} - m_{num\\_mid}

            The output length is then:

            .. math::

            output\\_samples = k_1 - k_0

            where k0 defaults to 0 and k1 defaults to k_max.

            Parameters that affect the calculation:
            - n_frames: number of time frames in the STFT
            - p_min: minimum frame index (ShortTimeFFT property)
            - hop: hop length (samples between frames)
            - m_num: window length
            - m_num_mid: window midpoint position
            - length: optional length override (if set, limits output)

        References:
            - SciPy ShortTimeFFT.istft:
          https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.istft.html
            - SciPy Source: https://github.com/scipy/scipy/blob/main/scipy/signal/_short_time_fft.py
        """
        n_channels = input_shape[0]
        n_frames = input_shape[-1]  # time_frames

        # Follow SciPy ShortTimeFFT formula
        # See: https://github.com/scipy/scipy/blob/main/scipy/signal/_short_time_fft.py
        q_max = n_frames + self._SFT.p_min
        k_max = (q_max - 1) * self._SFT.hop + self._SFT.m_num - self._SFT.m_num_mid

        # Default parameters: k0=0, k1=None (which becomes k_max)
        # The output length is k1 - k0 = k_max - 0 = k_max
        k0 = 0
        k1 = k_max

        # If length is specified, it acts as an override to limit the output
        length = self.length
        if length is not None:
            k1 = min(length, k1)

        output_samples = k1 - k0

        return (n_channels, output_samples)

    def calculate_output_dtype(self, input_dtype: np.dtype[Any], *input_dtypes: np.dtype[Any]) -> np.dtype[Any]:
        return np.dtype(np.float64)

    def _process(self, x: NDArrayComplex) -> NDArrayReal:
        """
        Apply SciPy ISTFT processing to multiple channels at once using ShortTimeFFT"""
        logger.debug(f"Applying SciPy ISTFT (ShortTimeFFT) to array with shape: {x.shape}")

        # Convert 2D input to 3D (assume single channel)
        if x.ndim == 2:
            x = x.reshape(1, *x.shape)

        # Adjust scaling back if STFT applied factor of 2
        _x = _denormalize_rfft_amplitude(
            x,
            n_fft=self.n_fft,
            window_gain=1.0,
            axis=-2,
        )

        # Apply ISTFT using the ShortTimeFFT instance
        result: NDArrayReal = self._SFT.istft(_x)

        # Trim to desired length if specified
        length = self.length
        if length is not None:
            result = result[..., :length]

        logger.debug(f"ShortTimeFFT applied, returning result with shape: {result.shape}")
        return result

    def process(self, data: DaArray, *inputs: DaArray) -> DaArray:
        """Execute ISTFT on Frame-internal channel-first spectrogram data."""
        self._validate_process_inputs(data, *inputs, ndim=3)
        return super().process(data, *inputs)


# Upper bound on segment-spectrum elements per ``scipy.signal.welch`` call
# (lanes x segments x n_fft); 2**22 is 32 MiB of float64 per array SciPy builds.
_WELCH_RUN_ELEMENTS = 2**22


class Welch(AudioOperation[NDArrayReal, NDArrayReal]):
    """Welch-averaged one-sided peak-amplitude spectrum.

    Segment power spectra are averaged with ``scaling="spectrum"`` and then
    converted to peak amplitude. Values retain the input physical unit; they
    are neither power spectral density nor expressed per hertz. For an on-bin
    sine wave with peak amplitude ``A``, the corresponding bin is approximately
    ``A``.

    Internally, this uses ``scipy.signal.welch`` with ``scaling="spectrum"``
    and converts the power spectrum to amplitude spectrum:

    - DC component (f=0): A = sqrt(P)
    - positive non-Nyquist components: A = sqrt(2*P)
    """

    name = "welch"
    _display = "Welch"

    def __init__(
        self,
        sampling_rate: float,
        n_fft: int = 2048,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
        average: str = "mean",
        detrend: str = "constant",
    ):
        """
        Initialize Welch operation

        Args:
            sampling_rate: float. Sampling rate (Hz)
            n_fft: int, optional. FFT size, default is 2048
            hop_length: int, optional. Number of samples between frames. Default is win_length // 4
            win_length: int, optional. Window length. Default is n_fft
            window: str, optional. Window function type, default is 'hann'
            average: str, optional. Averaging method, default is 'mean'
            detrend: str, optional. Detrend method, default is 'constant'

        Raises:
            ValueError: If n_fft, win_length, or hop_length are invalid
        """
        # Validate and compute parameters
        actual_win_length, actual_hop_length = _validate_spectral_params(n_fft, win_length, hop_length, "Welch method")

        super().__init__(
            sampling_rate,
            n_fft=n_fft,
            win_length=actual_win_length,
            hop_length=actual_hop_length,
            window=window,
            average=average,
            detrend=detrend,
        )

    @property
    def n_fft(self) -> int:
        """FFT size captured at operation construction time."""
        return self._config_value("n_fft")

    @property
    def win_length(self) -> int:
        """Window length captured at operation construction time."""
        return self._config_value("win_length")

    @property
    def hop_length(self) -> int:
        """Hop length captured at operation construction time."""
        return self._config_value("hop_length")

    @property
    def window(self) -> str:
        """Window name captured at operation construction time."""
        return self._config_value("window")

    @property
    def average(self) -> str:
        """Averaging method captured at operation construction time."""
        return self._config_value("average")

    @property
    def detrend(self) -> str:
        """Detrend method captured at operation construction time."""
        return self._config_value("detrend")

    @property
    def noverlap(self) -> int:
        """Overlap captured at operation construction time."""
        return self.win_length - self.hop_length

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Calculate output data shape after operation

        Args:
            input_shape: tuple. Input data shape (channels, samples)

        Returns:
            tuple: Output data shape (channels, freqs)
        """
        n_freqs = self.n_fft // 2 + 1
        return (*input_shape[:-1], n_freqs)

    def calculate_output_dtype(self, input_dtype: np.dtype[Any], *input_dtypes: np.dtype[Any]) -> np.dtype[Any]:
        return _spectral_real_dtype(input_dtype)

    def _process(self, x: NDArrayReal) -> NDArrayReal:
        """Return a Welch-averaged one-sided peak-amplitude spectrum.

        Converts ``scipy.signal.welch(..., scaling="spectrum")`` power to
        peak amplitude for consistency with FFT/STFT.
        """
        from scipy import signal as ss

        if not isinstance(x, np.ndarray):
            raise ValueError("Welch operation requires a numpy ndarray, but received a non-ndarray.")

        def welch_power(samples: NDArrayReal) -> NDArrayReal:
            _, power = ss.welch(
                samples,
                nperseg=self.win_length,
                noverlap=self.noverlap,
                nfft=self.n_fft,
                window=self.window,
                average=self.average,
                detrend=self.detrend,
                scaling="spectrum",
            )
            return power

        # SciPy holds every segment and its FFT at once, about ten times the input for a
        # long recording. For the mean, call it on runs of whole segments (overlapping by
        # ``noverlap`` so the segments are the same) and weight each run by its segment
        # count. The median of the runs is not the median of all segments, so it keeps one
        # call.
        n_samples = x.shape[-1]
        n_segments = (n_samples - self.win_length) // self.hop_length + 1 if n_samples >= self.win_length else 0
        lanes = max(1, int(np.prod(x.shape[:-1])))
        segments_per_run = max(1, _WELCH_RUN_ELEMENTS // (lanes * self.n_fft))
        if self.average != "mean" or n_segments <= segments_per_run:
            result = welch_power(x)
        else:
            weighted_sum: NDArrayReal | None = None
            for first in range(0, n_segments, segments_per_run):
                count = min(segments_per_run, n_segments - first)
                start = first * self.hop_length
                stop = start + (count - 1) * self.hop_length + self.win_length
                power = welch_power(x[..., start:stop])
                # Scale by a scalar of the spectrum's dtype: NumPy 1.x promotes float32 by a
                # Python int above 65535 to float64.
                run = power * power.dtype.type(count)
                weighted_sum = run if weighted_sum is None else weighted_sum + run
            assert weighted_sum is not None
            result = weighted_sum / weighted_sum.dtype.type(n_segments)

        # Convert power spectrum to amplitude spectrum for consistency with FFT/STFT.
        # scipy.signal.welch with scaling='spectrum' returns a one-sided power spectrum
        # where for a sine wave with amplitude A:
        #   - DC component (f=0): P = A^2 (no factor of 2 since DC is not mirrored)
        #   - AC components (f>0): P = A^2/2 (half power due to one-sided spectrum)
        # To recover amplitude A:
        #   - DC: A = sqrt(P)
        #   - AC: A = sqrt(2*P) = sqrt(2) * sqrt(P)
        result = np.sqrt(result)
        result[_rfft_positive_frequency_bins(result.ndim, n_fft=self.n_fft, axis=-1)] *= np.sqrt(2)

        return result


class _NOctBase(AudioOperation[NDArrayReal, NDArrayReal]):
    """Shared base for N-octave band operations (spectrum and synthesis).

    Handles common parameter storage and output shape calculation
    for operations on fractional octave bands.
    """

    _display: str  # set by subclasses

    def __init__(
        self,
        sampling_rate: float,
        fmin: float,
        fmax: float,
        n: int = 3,
        G: int = 10,
        fr: int = 1000,
    ):
        super().__init__(sampling_rate, fmin=fmin, fmax=fmax, n=n, G=G, fr=fr)

    @property
    def fmin(self) -> float:
        """Minimum band frequency captured at operation construction time."""
        return self._config_value("fmin")

    @property
    def fmax(self) -> float:
        """Maximum band frequency captured at operation construction time."""
        return self._config_value("fmax")

    @property
    def n(self) -> int:
        """Fractional octave denominator captured at operation construction time."""
        return self._config_value("n")

    @property
    def G(self) -> int:  # noqa: N802
        """Octave ratio base captured at operation construction time."""
        return self._config_value("G")

    @property
    def fr(self) -> int:
        """Reference frequency captured at operation construction time."""
        return self._config_value("fr")

    def ensure_dependencies(self) -> None:
        require_mosqito_center_freq("NOctFrame")

    def validate_params(self) -> None:
        """Validate common N-octave configuration before optional dependencies."""
        _validate_noct_g(self.G)

    def _validate_process_shape(self, data: Any, *inputs: Any) -> None:
        """Validate operation-specific input shape before dependency loading."""

    def _synthesize(self, x: NDArrayReal, *, n_fft: int) -> NDArrayReal:
        """Run MoSQITo synthesis for one explicit or legacy FFT size."""
        freqs = np.fft.rfftfreq(n_fft, d=1 / self.sampling_rate)
        result, _ = noct_synthesis(
            spectrum=np.abs(x).T,
            freqs=freqs,
            fmin=self.fmin,
            fmax=self.fmax,
            n=self.n,
            G=self.G,
            fr=self.fr,
        )
        return np.asarray(result).T

    def process(self, data: Any, *inputs: Any) -> Any:
        self._validate_process_shape(data, *inputs)
        self.ensure_dependencies()
        return super().process(data, *inputs)

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        f_exact, f_nom = _center_freq(
            fmin=self.fmin,
            fmax=self.fmax,
            n=self.n,
            G=self.G,
            fr=self.fr,
        )
        return (input_shape[0], noct_band_labels(f_exact, f_nom).shape[0])


class NOctSpectrum(_NOctBase, ChannelIndependentAudioOperation[NDArrayReal, NDArrayReal]):
    """N-octave spectrum operation"""

    name = "noct_spectrum"
    _display = "Oct"

    def calculate_output_dtype(self, input_dtype: np.dtype[Any], *input_dtypes: np.dtype[Any]) -> np.dtype[Any]:
        """Advertise the float64 output produced by MoSQITo."""
        return np.dtype(np.float64)

    def _process(self, x: NDArrayReal) -> NDArrayReal:
        """Create processor function for octave spectrum"""
        logger.debug(f"Applying NoctSpectrum to array with shape: {x.shape}")
        spec, f_nom = noct_spectrum(
            sig=x.T,
            fs=self.sampling_rate,
            fmin=self.fmin,
            fmax=self.fmax,
            n=self.n,
            G=self.G,
            fr=self.fr,
        )
        # MoSQITo returns one row per exact band, and its nominal labels can be
        # shorter, so the band count comes from the exact center frequencies.
        f_exact, _ = _center_freq(fmin=self.fmin, fmax=self.fmax, n=self.n, G=self.G, fr=self.fr)
        n_bands = noct_band_labels(f_exact, f_nom).size
        spec = np.asarray(spec).reshape(n_bands, x.shape[0]).T
        logger.debug(f"NoctSpectrum applied, returning result with shape: {spec.shape}")
        return np.array(spec)


class NOctSynthesis(_NOctBase):
    """N-octave synthesis operation using an explicit original FFT size.

    ``n_fft`` is required because a one-sided spectrum's bin count cannot
    distinguish an odd FFT size from the adjacent even size. The value is
    captured in the operation configuration and is used to construct the
    canonical real-FFT frequency grid.

    The input is Wandas' one-sided peak-amplitude spectrum. MoSQITo sums the
    squared bins of an RMS spectrum in each band, so every bin except DC and
    Nyquist is divided by ``sqrt(2)``, and every bin by the square root of the
    analysis window's equivalent noise bandwidth in bins,
    ``n_fft * sum(w**2) / sum(w)**2``. ``w`` is the ``win_length``-sample
    ``window`` that produced the spectrum.
    """

    name = "noct_synthesis"
    _display = "Octs"

    def __init__(
        self,
        sampling_rate: float,
        fmin: float,
        fmax: float,
        n: int = 3,
        G: int = 10,
        fr: int = 1000,
        *,
        n_fft: int,
        window: str = "hann",
        win_length: int | None = None,
    ) -> None:
        """Initialize N-octave synthesis with the source spectrum's FFT size.

        Args:
            sampling_rate: Sampling rate in Hz. The public synthesis Frame
                method requires 48000 Hz.
            fmin: Lower frequency bound in Hz.
            fmax: Upper frequency bound in Hz.
            n: Number of bands per octave.
            G: Exact center-frequency ratio convention, either 2 or 10.
            fr: Reference frequency in Hz.
            n_fft: Positive integer FFT size that produced the complete
                one-sided input spectrum.
            window: Analysis window that produced the input spectrum.
            win_length: Length of that window in samples. Default is ``n_fft``.

        Raises:
            TypeError: If ``n_fft`` is not an integer or ``G`` is not an
                integer ratio convention.
            ValueError: If ``n_fft`` is not positive, ``G`` is not 2 or 10, or
                ``win_length`` is not an integer with ``1 <= win_length <= n_fft``.
        """
        AudioOperation.__init__(
            self,
            sampling_rate,
            fmin=fmin,
            fmax=fmax,
            n=n,
            G=G,
            fr=fr,
            n_fft=n_fft,
            window=window,
            win_length=n_fft if win_length is None else win_length,
        )

    @property
    def n_fft(self) -> int:
        """Return the source FFT size captured by this operation."""
        return int(self._config_value("n_fft"))

    @property
    def window(self) -> str:
        """Analysis window of the source spectrum."""
        return self._config_value("window")

    @property
    def win_length(self) -> int:
        """Analysis window length of the source spectrum, in samples."""
        return int(self._config_value("win_length"))

    def validate_params(self) -> None:
        """Validate common N-octave parameters and the explicit FFT size."""
        super().validate_params()
        value = self._config_value("n_fft")
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError(
                "Invalid n_fft for NOctSynthesis\n"
                f"  Got: {value!r} ({type(value).__name__})\n"
                "  Expected: a positive integer\n"
                "Pass the positive integer n_fft stored by SpectralFrame."
            )
        normalized = int(value)
        if normalized <= 0:
            raise ValueError(
                "Invalid n_fft for NOctSynthesis\n"
                f"  Got: {normalized}\n"
                "  Expected: a positive integer\n"
                "Pass the positive integer n_fft stored by SpectralFrame."
            )
        win_length = self._config_value("win_length")
        if (
            isinstance(win_length, bool)
            or not isinstance(win_length, numbers.Integral)
            or not 1 <= int(win_length) <= normalized
        ):
            raise ValueError(
                "Invalid win_length for NOctSynthesis\n"
                f"  Got: {win_length!r}\n"
                f"  Expected: an integer with 1 <= win_length <= n_fft ({normalized})\n"
                "Pass the win_length stored by SpectralFrame."
            )

    def _validate_process_shape(self, data: Any, *inputs: Any) -> None:
        """Reject spectra whose bin count disagrees with the explicit FFT size."""
        del inputs
        expected_bins = self.n_fft // 2 + 1
        actual_bins = data.shape[-1]
        if actual_bins != expected_bins:
            raise ValueError(
                "Invalid frequency bin count for NOctSynthesis\n"
                f"  Got: {actual_bins} bins\n"
                f"  Expected: {expected_bins} bins for n_fft={self.n_fft}\n"
                "Pass the complete one-sided spectrum matching the explicit n_fft."
            )

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Count the bands MoSQITo synthesizes.

        MoSQITo drops the top bands whose upper edge ``fc * 2**(1 / (2 * n))``
        exceeds half its sampling rate, ``2 * max(freqs)``. It skips the drop
        when the only such band is the first one.
        """
        shape = super().calculate_output_shape(input_shape)
        f_exact, _ = _center_freq(fmin=self.fmin, fmax=self.fmax, n=self.n, G=self.G, fr=self.fr)
        nyquist = np.fft.rfftfreq(self.n_fft, d=1 / self.sampling_rate).max()
        dropped = np.flatnonzero(np.asarray(f_exact) * 2 ** (1 / (2 * self.n)) > nyquist)
        if dropped.any():
            return (shape[0], shape[1] - dropped.size)
        return shape

    def calculate_output_dtype(self, input_dtype: np.dtype[Any], *input_dtypes: np.dtype[Any]) -> np.dtype[Any]:
        """Advertise the real float64 output produced by MoSQITo."""
        return np.dtype(np.float64)

    def _process(self, x: NDArrayReal) -> NDArrayReal:
        """Convert the peak-amplitude bins to RMS band contributions, then synthesize."""
        logger.debug(f"Applying NoctSynthesis to array with shape: {x.shape}")
        window_values = scipy.signal.get_window(self.window, self.win_length)
        enbw_bins = self.n_fft * np.sum(window_values**2) / np.sum(window_values) ** 2
        rms = np.abs(x) / np.sqrt(enbw_bins)
        rms[_rfft_positive_frequency_bins(rms.ndim, n_fft=self.n_fft, axis=-1)] /= np.sqrt(2)
        result = self._synthesize(rms, n_fft=self.n_fft)
        logger.debug(f"NoctSynthesis applied, returning result with shape: {result.shape}")
        return np.array(result)


class _RecipeNOctSynthesisV2(NOctSynthesis):
    """Released Recipe v2 synthesis, which passed peak-amplitude bins unchanged."""

    name = "_recipe_noct_synthesis_v2"
    _display = "Octs Recipe v2"

    def _process(self, x: NDArrayReal) -> NDArrayReal:
        """Reproduce the released v2 numerics exactly."""
        logger.debug(f"Applying Recipe v2 NoctSynthesis to array with shape: {x.shape}")
        result = self._synthesize(x, n_fft=self.n_fft)
        logger.debug(f"Recipe v2 NoctSynthesis applied, returning result with shape: {result.shape}")
        return np.array(result)


# SciPy 1.16 moved ``signal.csd`` onto ``ShortTimeFFT``, which scales the window
# before the FFT instead of scaling the segment products afterwards.
_SCIPY_CSD_USES_SHORT_TIME_FFT = tuple(int(part) for part in scipy.__version__.split(".")[:2]) >= (1, 16)


class _CrossSpectralBase(AudioOperation[NDArrayReal, NDArrayReal]):
    """Shared base for cross-spectral operations (coherence, CSD, transfer function).

    Handles common parameter validation, storage, and output shape calculation
    for operations that produce (n_channels * n_channels, n_freqs) output.
    """

    _method_label: str  # human-readable label for _validate_spectral_params
    _display: str  # set by subclasses

    def __init__(
        self,
        sampling_rate: float,
        n_fft: int = 2048,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
        detrend: str = "constant",
        **extra_kwargs: Any,
    ):
        actual_win_length, actual_hop_length = _validate_spectral_params(
            n_fft, win_length, hop_length, self._method_label
        )
        super().__init__(
            sampling_rate,
            n_fft=n_fft,
            hop_length=actual_hop_length,
            win_length=actual_win_length,
            window=window,
            detrend=detrend,
            **extra_kwargs,
        )

    @property
    def n_fft(self) -> int:
        """FFT size captured at operation construction time."""
        return self._config_value("n_fft")

    @property
    def win_length(self) -> int:
        """Window length captured at operation construction time."""
        return self._config_value("win_length")

    @property
    def hop_length(self) -> int:
        """Hop length captured at operation construction time."""
        return self._config_value("hop_length")

    @property
    def window(self) -> str:
        """Window name captured at operation construction time."""
        return self._config_value("window")

    @property
    def detrend(self) -> str:
        """Detrend method captured at operation construction time."""
        return self._config_value("detrend")

    @property
    def noverlap(self) -> int:
        """Overlap captured at operation construction time."""
        return self.win_length - self.hop_length

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        n_channels = input_shape[0]
        n_freqs = self.n_fft // 2 + 1
        return (n_channels * n_channels, n_freqs)

    def _cross_spectral_matrix(self, x: NDArrayReal, *, scaling: str, average: str) -> NDArrayComplex:
        """Return every ``P_out_in`` as one ``(output, input, frequency)`` Welch matrix.

        The values are SciPy's ``csd`` for each channel pair.  Each channel's
        segments are transformed once and paired afterwards, so N channels cost
        N FFT passes and an ``(N, N, frequency)`` result instead of N² passes.
        """
        from scipy import signal as ss

        n_samples = x.shape[-1]
        if average != "mean" or n_samples < self.win_length:
            # The median needs every segment's product, and SciPy shortens the
            # segment for short inputs; one SciPy call per output channel keeps
            # those semantics while holding only N segment products at a time.
            columns = [
                ss.csd(
                    x=x,
                    y=x[output],
                    fs=self.sampling_rate,
                    nperseg=self.win_length,
                    noverlap=self.noverlap,
                    nfft=self.n_fft,
                    window=self.window,
                    detrend=self.detrend,
                    scaling=scaling,
                    average=average,
                )[1]
                for output in range(x.shape[0])
            ]
            # Each call returns one output channel's (input, frequency) slice.
            return np.stack(columns)

        if scaling not in ("spectrum", "density"):
            raise ValueError(f"Unknown scaling: {scaling!r}")
        out_dtype = np.result_type(x, np.complex64)
        # Summing the segment products in double precision keeps single-precision
        # inputs finite wherever SciPy's per-segment average is.
        sum_dtype = np.result_type(out_dtype, np.complex128)
        win = scipy.signal.get_window(self.window, self.win_length)
        # The segment products are formed in the installed SciPy's order, so even
        # bins that are pure rounding noise equal its ``csd`` output.
        if _SCIPY_CSD_USES_SHORT_TIME_FFT:
            transform = scipy.signal.ShortTimeFFT(
                win,
                self.hop_length,
                self.sampling_rate,
                fft_mode="onesided",
                mfft=self.n_fft,
                scale_to="psd" if scaling == "density" else "magnitude",
                phase_shift=None,
            )
            spectra = transform.stft_detrend(
                x,
                cast(Literal["linear", "constant"], self.detrend) if self.detrend else None,
                p0=0,
                p1=(n_samples - self.noverlap) // self.hop_length,
                k_offset=self.win_length // 2,
                axis=-1,
            )
            cross = np.einsum("ofp,ifp->oif", spectra, np.conj(spectra), dtype=sum_dtype) / spectra.shape[-1]
        else:
            if np.result_type(win, np.complex64) != out_dtype:
                win = win.astype(out_dtype)
            segments = np.lib.stride_tricks.sliding_window_view(x, self.win_length, axis=-1)[..., :: self.hop_length, :]
            if self.detrend:
                segments = ss.detrend(segments, type=cast(Literal["linear", "constant"], self.detrend), axis=-1)
            spectra = sp_fft.rfft((win * segments).real, n=self.n_fft)
            if scaling == "density":
                scale = 1.0 / (self.sampling_rate * (win * win).sum())
            else:
                scale = 1.0 / win.sum() ** 2
            cross = np.einsum("isf,osf->oif", np.conj(spectra), spectra, dtype=sum_dtype) * (scale / spectra.shape[1])
        if self.n_fft % 2:
            cross[..., 1:] *= 2
        else:
            cross[..., 1:-1] *= 2
        return cross.astype(out_dtype, copy=False)

    def calculate_output_dtype(self, input_dtype: np.dtype[Any], *input_dtypes: np.dtype[Any]) -> np.dtype[Any]:
        return _spectral_real_dtype(input_dtype)


class Coherence(_CrossSpectralBase):
    """Coherence estimation operation"""

    name = "coherence"
    _method_label = "Coherence"
    _display = "Coh"

    def _process(self, x: NDArrayReal) -> NDArrayReal:
        """Processor function for coherence estimation operation"""
        logger.debug(f"Applying coherence estimation to array with shape: {x.shape}")
        # Coherence is scaling-invariant; one CSD matrix supplies both the cross
        # and auto spectra instead of separate Welch passes per pair.
        cross = self._cross_spectral_matrix(x, scaling="density", average="mean")
        result: NDArrayReal = flatten_output_input_pairs(coherence_from_cross_spectra(cross))

        logger.debug(f"Coherence estimation applied, result shape: {result.shape}")
        return result


class _ScaledCrossSpectralBase(_CrossSpectralBase):
    """Cross-spectral base that adds scaling and averaging parameters.

    Used by CSD and TransferFunction which share the same extended
    parameter set on top of the common cross-spectral parameters.
    """

    def __init__(
        self,
        sampling_rate: float,
        n_fft: int = 2048,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: str = "hann",
        detrend: str = "constant",
        scaling: str = "spectrum",
        average: str = "mean",
    ):
        super().__init__(
            sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            detrend=detrend,
            scaling=scaling,
            average=average,
        )

    @property
    def scaling(self) -> str:
        """Scaling mode captured at operation construction time."""
        return self._config_value("scaling")

    @property
    def average(self) -> str:
        """Averaging method captured at operation construction time."""
        return self._config_value("average")

    def calculate_output_dtype(self, input_dtype: np.dtype[Any], *input_dtypes: np.dtype[Any]) -> np.dtype[Any]:
        return _spectral_complex_dtype(input_dtype)


class CSD(_ScaledCrossSpectralBase):
    """Cross-spectral density estimation operation"""

    name = "csd"
    _method_label = "CSD"
    _display = "CSD"

    def _process(self, x: NDArrayReal) -> NDArrayComplex:
        """Processor function for cross-spectral density estimation operation"""
        logger.debug(f"Applying CSD estimation to array with shape: {x.shape}")
        cross = self._cross_spectral_matrix(x, scaling=self.scaling, average=self.average)
        result: NDArrayComplex = flatten_output_input_pairs(cross)

        logger.debug(f"CSD estimation applied, result shape: {result.shape}")
        return result


class TransferFunction(_ScaledCrossSpectralBase):
    """Transfer function estimation operation"""

    name = "transfer_function"
    _method_label = "Transfer function"
    _display = "H"

    def _process(self, x: NDArrayReal) -> NDArrayComplex:
        """Processor function for transfer function estimation operation"""
        logger.debug(f"Applying transfer function estimation to array with shape: {x.shape}")
        # The CSD diagonal is the Welch auto-spectrum, so one matrix supplies
        # H[output, input] = P_out_in / P_in_in. Exact zero denominators remain
        # complex NaN; nonzero near-zero bins are untouched.
        cross = self._cross_spectral_matrix(x, scaling=self.scaling, average=self.average)
        h_f = transfer_function_ratio(cross, cross_spectral_auto_power(cross))
        result: NDArrayComplex = flatten_output_input_pairs(h_f)

        logger.debug(f"Transfer function estimation applied, result shape: {result.shape}")
        return result


class _CrossSpectralDerivation(AudioOperation[NDArrayComplex, Any]):
    """Shared base deriving a pairwise quantity from a complete flattened CSD matrix.

    Input rows are every ``P_out_in`` in output-major/input-minor order, as stored
    by :class:`CSD`.  The frequency axis is passed through unchanged, so the
    derivation does not repeat any FFT or segment averaging.
    """

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        self._source_channel_count(input_shape[0])
        return input_shape

    def _source_channel_count(self, n_rows: int) -> int:
        n_channels = math.isqrt(n_rows)
        if n_channels * n_channels != n_rows or n_channels == 0:
            raise ValueError(
                f"Invalid cross-spectral matrix for {self.name}\n"
                f"  Got: {n_rows} pair rows\n"
                "  Expected: every output/input pair of the source channels (n_channels ** 2 rows).\n"
                "Derive the quantity from the complete CSD before selecting pairs."
            )
        return n_channels

    def _matrix(self, x: NDArrayComplex) -> NDArrayComplex:
        n_channels = self._source_channel_count(x.shape[0])
        return x.reshape((n_channels, n_channels, *x.shape[1:]))


class TransferFunctionFromCSD(_CrossSpectralDerivation):
    """Canonical ``H_out_in = P_out_in / P_in_in`` from an existing CSD matrix."""

    name = "transfer_function_from_csd"
    _display = "H"

    def calculate_output_dtype(self, input_dtype: np.dtype[Any], *input_dtypes: np.dtype[Any]) -> np.dtype[Any]:
        return np.result_type(input_dtype, np.complex64)

    def _process(self, x: NDArrayComplex) -> NDArrayComplex:
        cross = self._matrix(x)
        result: NDArrayComplex = flatten_output_input_pairs(
            transfer_function_ratio(cross, cross_spectral_auto_power(cross))
        )
        return result


class CoherenceFromCSD(_CrossSpectralDerivation):
    """Magnitude-squared coherence from an existing CSD matrix."""

    name = "coherence_from_csd"
    _display = "Coh"

    def calculate_output_dtype(self, input_dtype: np.dtype[Any], *input_dtypes: np.dtype[Any]) -> np.dtype[Any]:
        return _spectral_real_dtype(np.finfo(input_dtype).dtype)

    def _process(self, x: NDArrayComplex) -> NDArrayReal:
        result: NDArrayReal = flatten_output_input_pairs(coherence_from_cross_spectra(self._matrix(x)))
        return result


class SpectrogramCSD(AudioOperation[NDArrayComplex, NDArrayComplex]):
    """Every ``P_out_in`` averaged over the frames of an amplitude-normalized STFT.

    The input is ``(channel, frequency, time)`` as produced by :class:`STFT`.
    Undoing its coherent-gain scaling gives SciPy's ``csd`` scaling for the same
    window and segments: ``scaling="spectrum"`` divides
    ``mean(conj(X_in) * X_out)`` by two in the doubled bins, and
    ``scaling="density"`` also multiplies by ``sum(w)**2 / (fs * sum(w**2))``.
    Frames are averaged with the mean and are not detrended.
    """

    name = "spectrogram_csd"
    _display = "CSD"

    def __init__(
        self,
        sampling_rate: float,
        n_fft: int = 2048,
        win_length: int | None = None,
        window: str = "hann",
        scaling: str = "spectrum",
    ):
        # The CSD averages existing frames, so any hop the STFT accepted is valid here.
        resolved_win_length = n_fft if win_length is None else win_length
        actual_win_length, _ = _validate_spectral_params(n_fft, win_length, resolved_win_length, "Spectrogram CSD")
        if scaling not in ("spectrum", "density"):
            raise ValueError(
                "Invalid scaling for spectrogram CSD\n"
                f"  Got: {scaling!r}\n"
                "  Expected: 'spectrum' or 'density'\n"
                "Use 'spectrum' for power per bin or 'density' for power per hertz."
            )
        super().__init__(
            sampling_rate,
            n_fft=n_fft,
            win_length=actual_win_length,
            window=window,
            scaling=scaling,
        )

    @property
    def n_fft(self) -> int:
        """FFT size captured at operation construction time."""
        return self._config_value("n_fft")

    @property
    def win_length(self) -> int:
        """Window length captured at operation construction time."""
        return self._config_value("win_length")

    @property
    def window(self) -> str:
        """Window name captured at operation construction time."""
        return self._config_value("window")

    @property
    def scaling(self) -> str:
        """CSD scaling captured at operation construction time."""
        return self._config_value("scaling")

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        expected_frequency_bins = self.n_fft // 2 + 1
        if len(input_shape) != 3 or int(input_shape[-2]) != expected_frequency_bins or input_shape[-1] == 0:
            raise ValueError(
                "Invalid spectrogram for CSD\n"
                f"  Got: shape {input_shape}\n"
                f"  Expected: (channel, {expected_frequency_bins}, time) with at least one time frame "
                f"for n_fft={self.n_fft}\n"
                "Select a non-empty frame range of the complete STFT before computing the CSD."
            )
        n_channels = input_shape[0]
        return (n_channels * n_channels, input_shape[1])

    def calculate_output_dtype(self, input_dtype: np.dtype[Any], *input_dtypes: np.dtype[Any]) -> np.dtype[Any]:
        if not np.issubdtype(input_dtype, np.complexfloating):
            raise ValueError(
                "Invalid spectrogram for CSD\n"
                f"  Got: {input_dtype} values\n"
                "  Expected: complex STFT values with phase\n"
                "Compute the CSD from the STFT before taking its magnitude."
            )
        return np.result_type(input_dtype, np.complex64)

    def _process(self, x: NDArrayComplex) -> NDArrayComplex:
        cross = np.einsum("ift,oft->oif", np.conj(x), x) / x.shape[-1]
        cross = _denormalize_rfft_amplitude(cross, n_fft=self.n_fft, window_gain=1.0)
        if self.scaling == "density":
            win = scipy.signal.get_window(self.window, self.win_length)
            cross *= np.sum(win) ** 2 / (self.sampling_rate * np.sum(win**2))
        result: NDArrayComplex = flatten_output_input_pairs(cross)
        return result
