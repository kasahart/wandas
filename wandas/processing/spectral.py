import logging
import numbers
from collections.abc import Mapping
from typing import Any

import numpy as np
from dask.array.core import Array as DaArray
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import get_window

from wandas.processing.base import AudioOperation, ChannelIndependentAudioOperation
from wandas.processing.spectral_contracts import (
    as_output_input_pairs,
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

    if actual_hop_length > actual_win_length:
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

    The input is truncated or zero-padded to ``n_fft`` before the selected
    window is applied. DC and Nyquist bins retain their real-FFT scaling; every
    other positive-frequency bin is doubled. The complex result therefore has
    the same physical unit as the input, and an on-bin sinusoid's magnitude is
    its peak amplitude.
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
        """Apply FFT to the input array."""
        fft_size = int(x.shape[-1]) if self.n_fft is None else self.n_fft
        if x.shape[-1] >= fft_size:
            x = x[..., :fft_size]
        else:
            x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, fft_size - x.shape[-1])])

        win = get_window(self.window, fft_size)
        x = x * win
        result: NDArrayComplex = np.fft.rfft(x, n=fft_size, axis=-1)
        scaling_factor = np.sum(win)
        return _normalize_rfft_amplitude(
            result,
            n_fft=fft_size,
            window_gain=float(scaling_factor),
        )


class _RecipeFFTV1(FFT):
    """Released Recipe v1 FFT padding and windowing contract."""

    name = "_recipe_fft_v1"
    _display = "FFT Recipe v1"

    def _process(self, x: NDArrayReal) -> NDArrayComplex:
        """Apply the released window-before-zero-padding preparation order."""
        n_fft = self.n_fft
        if n_fft is not None and x.shape[-1] > n_fft:
            x = x[..., :n_fft]

        win = get_window(self.window, x.shape[-1])
        x = x * win
        fft_size = int(x.shape[-1]) if n_fft is None else n_fft
        result: NDArrayComplex = np.fft.rfft(x, n=fft_size, axis=-1)
        scaling_factor = np.sum(win)
        return _normalize_rfft_amplitude(
            result,
            n_fft=fft_size,
            window_gain=float(scaling_factor),
        )


class IFFT(AudioOperation[NDArrayComplex, NDArrayReal]):
    """Inverse of Wandas' one-sided peak-amplitude FFT normalization.

    For a spectrum produced by :class:`FFT` with matching ``n_fft`` and
    ``window``, the result is the truncated-or-zero-padded input multiplied by
    that analysis window. A boxcar window therefore reconstructs the prepared
    input exactly; tapered windows intentionally reconstruct the windowed
    waveform rather than guessing samples discarded by the analysis window.
    """

    name = "ifft"
    _display = "iFFT"

    def __init__(self, sampling_rate: float, n_fft: int | None = None, window: str = "hann"):
        """
        Initialize IFFT operation

        Args:
            sampling_rate: float. Sampling rate (Hz)
            n_fft: Optional[int], optional. IFFT size, default is None (determined based on input size)
            window: str, optional. Window function type, default is 'hann'
        """
        super().__init__(sampling_rate, n_fft=n_fft, window=window)

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
        win = get_window(self.window, fft_size)
        _x = _denormalize_rfft_amplitude(
            x,
            n_fft=fft_size,
            window_gain=float(np.sum(win)),
        )

        result: NDArrayReal = np.fft.irfft(_x, n=fft_size, axis=-1)

        logger.debug(f"IFFT applied, returning result with shape: {result.shape}")
        return result


class _RecipeIFFTV1(IFFT):
    """Released Recipe v1 IFFT scaling retained for deterministic replay."""

    name = "_recipe_ifft_v1"
    _display = "iFFT Recipe v1"

    def _process(self, x: NDArrayComplex) -> NDArrayReal:
        """Apply the legacy normalization exactly as released."""
        logger.debug(f"Applying Recipe v1 IFFT to array with shape: {x.shape}")

        fft_size = 2 * (int(x.shape[-1]) - 1) if self.n_fft is None else self.n_fft
        _x = _denormalize_rfft_amplitude(x, n_fft=fft_size, window_gain=1.0)
        result: NDArrayReal = np.fft.irfft(_x, n=self.n_fft, axis=-1)
        win = get_window(self.window, result.shape[-1])
        scaling_factor = np.sum(win) / result.shape[-1]
        result = result / scaling_factor

        logger.debug(f"Recipe v1 IFFT applied, returning result with shape: {result.shape}")
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
    ):
        """
        Initialize STFT operation

        Args:
            sampling_rate: float. Sampling rate (Hz)
            n_fft: int. FFT size, default is 2048
            hop_length: int, optional. Number of samples between frames. Default is win_length // 4
            win_length: int, optional. Window length. Default is n_fft
            window: str. Window type, default is 'hann'

        Raises:
            ValueError: If n_fft is not positive, win_length > n_fft, or hop_length is invalid
        """
        # Validate and compute parameters
        actual_win_length, actual_hop_length = _validate_spectral_params(n_fft, win_length, hop_length, "STFT")

        self._SFT = ShortTimeFFT(
            win=get_window(window, actual_win_length),
            hop=actual_hop_length,
            fs=sampling_rate,
            mfft=n_fft,
            scale_to="magnitude",
        )
        super().__init__(
            sampling_rate,
            n_fft=n_fft,
            win_length=actual_win_length,
            hop_length=actual_hop_length,
            window=window,
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
        result: NDArrayComplex = self._SFT.stft(x)
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
        self._SFT = ShortTimeFFT(
            win=get_window(window, actual_win_length),
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

        _, result = ss.welch(
            x,
            nperseg=self.win_length,
            noverlap=self.noverlap,
            nfft=self.n_fft,
            window=self.window,
            average=self.average,
            detrend=self.detrend,
            scaling="spectrum",
        )

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


class _RecipeWelchV1(Welch):
    """Released Recipe v1 Welch positive-frequency scaling contract."""

    name = "_recipe_welch_v1"
    _display = "Welch Recipe v1"

    def _process(self, x: NDArrayReal) -> NDArrayReal:
        """Apply the released ``1:-1`` scaling, including its odd-size endpoint."""
        from scipy import signal as ss

        if not isinstance(x, np.ndarray):
            raise ValueError("Welch operation requires a numpy ndarray, but received a non-ndarray.")

        _, result = ss.welch(
            x,
            nperseg=self.win_length,
            noverlap=self.noverlap,
            nfft=self.n_fft,
            window=self.window,
            average=self.average,
            detrend=self.detrend,
            scaling="spectrum",
        )
        result = np.sqrt(result)
        result[..., 1:-1] *= np.sqrt(2)
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
        _, fpref = _center_freq(
            fmin=self.fmin,
            fmax=self.fmax,
            n=self.n,
            G=self.G,
            fr=self.fr,
        )
        return (input_shape[0], fpref.shape[0])


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
        spec, frequencies = noct_spectrum(
            sig=x.T,
            fs=self.sampling_rate,
            fmin=self.fmin,
            fmax=self.fmax,
            n=self.n,
            G=self.G,
            fr=self.fr,
        )
        spec = np.asarray(spec).reshape(np.asarray(frequencies).size, x.shape[0]).T
        logger.debug(f"NoctSpectrum applied, returning result with shape: {spec.shape}")
        return np.array(spec)


class NOctSynthesis(_NOctBase):
    """N-octave synthesis operation using an explicit original FFT size.

    ``n_fft`` is required because a one-sided spectrum's bin count cannot
    distinguish an odd FFT size from the adjacent even size. The value is
    captured in the operation configuration and is used to construct the
    canonical real-FFT frequency grid.
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

        Raises:
            TypeError: If ``n_fft`` is not an integer or ``G`` is not an
                integer ratio convention.
            ValueError: If ``n_fft`` is not positive or ``G`` is not 2 or 10.
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
        )

    @property
    def n_fft(self) -> int:
        """Return the source FFT size captured by this operation."""
        return int(self._config_value("n_fft"))

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

    def calculate_output_dtype(self, input_dtype: np.dtype[Any], *input_dtypes: np.dtype[Any]) -> np.dtype[Any]:
        """Advertise the real float64 output produced by MoSQITo."""
        return np.dtype(np.float64)

    def _process(self, x: NDArrayReal) -> NDArrayReal:
        """Create processor function for octave synthesis."""
        logger.debug(f"Applying NoctSynthesis to array with shape: {x.shape}")
        result = self._synthesize(x, n_fft=self.n_fft)
        logger.debug(f"NoctSynthesis applied, returning result with shape: {result.shape}")
        return np.array(result)


class _RecipeNOctSynthesisV1(_NOctBase):
    """Released Recipe v1 synthesis with its original bin-count inference."""

    name = "_recipe_noct_synthesis_v1"
    _display = "Octs Recipe v1"

    def _process(self, x: NDArrayReal) -> NDArrayReal:
        """Reproduce the released v1 frequency-axis inference exactly."""
        logger.debug(f"Applying Recipe v1 NoctSynthesis to array with shape: {x.shape}")
        n_bins = int(x.shape[-1])
        inferred_n_fft = n_bins * 2 - 1 if n_bins % 2 == 0 else (n_bins - 1) * 2
        result = self._synthesize(x, n_fft=inferred_n_fft)
        logger.debug(f"Recipe v1 NoctSynthesis applied, returning result with shape: {result.shape}")
        return np.array(result)


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
        from scipy import signal as ss

        _, coh = ss.coherence(
            x=x[:, np.newaxis],
            y=x[np.newaxis, :],
            fs=self.sampling_rate,
            nperseg=self.win_length,
            noverlap=self.noverlap,
            nfft=self.n_fft,
            window=self.window,
            detrend=self.detrend,
        )

        # SciPy returns (input, output, frequency); expose output-major pairs.
        result: NDArrayReal = flatten_output_input_pairs(as_output_input_pairs(coh))

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
        from scipy import signal as ss

        # Calculate all combinations using scipy's csd function
        _, csd_result = ss.csd(
            x=x[:, np.newaxis],
            y=x[np.newaxis, :],
            fs=self.sampling_rate,
            nperseg=self.win_length,
            noverlap=self.noverlap,
            nfft=self.n_fft,
            window=self.window,
            detrend=self.detrend,
            scaling=self.scaling,
            average=self.average,
        )

        # SciPy returns (input, output, frequency); expose output-major pairs.
        result: NDArrayComplex = flatten_output_input_pairs(as_output_input_pairs(csd_result))

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
        from scipy import signal as ss

        # Calculate cross-spectral density between all channels
        _f, p_yx = ss.csd(
            x=x[:, np.newaxis, :],
            y=x[np.newaxis, :, :],
            fs=self.sampling_rate,
            nperseg=self.win_length,
            noverlap=self.noverlap,
            nfft=self.n_fft,
            window=self.window,
            detrend=self.detrend,
            scaling=self.scaling,
            average=self.average,
            axis=-1,
        )
        # p_yx shape: (input, output, frequency)

        # Calculate power spectral density for each channel
        _f, p_xx = ss.welch(
            x=x,
            fs=self.sampling_rate,
            nperseg=self.win_length,
            noverlap=self.noverlap,
            nfft=self.n_fft,
            window=self.window,
            detrend=self.detrend,
            scaling=self.scaling,
            average=self.average,
            axis=-1,
        )
        # p_xx shape: (num_channels, num_frequencies)

        # Calculate H[output, input] = P_out_in / P_in_in. Exact zero
        # denominators remain complex NaN; nonzero near-zero bins are untouched.
        h_f = transfer_function_ratio(as_output_input_pairs(p_yx), p_xx)
        result: NDArrayComplex = flatten_output_input_pairs(h_f)

        logger.debug(f"Transfer function estimation applied, result shape: {result.shape}")
        return result


class _RecipeTransferFunctionV1(TransferFunction):
    """Released Recipe v1 transfer denominator contract.

    Recipe v1 exposed output-major pair labels but divided each cross-spectrum
    by the PSD selected from the output axis.  Keep that numerical behavior
    available only for replay; the public operation uses
    :class:`TransferFunction` above and divides by the input PSD.
    """

    name = "_recipe_transfer_function_v1"
    _display = "H Recipe v1"

    def _process(self, x: NDArrayReal) -> NDArrayComplex:
        """Reproduce the released output-axis broadcast exactly."""
        logger.debug(f"Applying Recipe v1 transfer function to array with shape: {x.shape}")
        from scipy import signal as ss

        _f, p_yx = ss.csd(
            x=x[:, np.newaxis, :],
            y=x[np.newaxis, :, :],
            fs=self.sampling_rate,
            nperseg=self.win_length,
            noverlap=self.noverlap,
            nfft=self.n_fft,
            window=self.window,
            detrend=self.detrend,
            scaling=self.scaling,
            average=self.average,
            axis=-1,
        )
        _f, p_xx = ss.welch(
            x=x,
            fs=self.sampling_rate,
            nperseg=self.win_length,
            noverlap=self.noverlap,
            nfft=self.n_fft,
            window=self.window,
            detrend=self.detrend,
            scaling=self.scaling,
            average=self.average,
            axis=-1,
        )
        h_f = p_yx / p_xx[np.newaxis, :, :]
        return h_f.transpose(1, 0, 2).reshape(-1, h_f.shape[-1])
