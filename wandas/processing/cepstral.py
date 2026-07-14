"""Real-cepstrum analysis, liftering, and spectral-envelope reconstruction."""

from __future__ import annotations

import logging
import numbers
from typing import Any, Literal

import numpy as np
from scipy.signal import get_window

from wandas.processing.base import AudioOperation, register_operation
from wandas.processing.spectral import _normalize_rfft_amplitude
from wandas.utils.types import NDArrayComplex, NDArrayReal

logger = logging.getLogger(__name__)

DEFAULT_LOG_FLOOR = 1e-12


def _normalize_log_floor(floor: float, *, analysis_name: str) -> float:
    """Return a positive finite log floor with a domain-specific error."""
    if isinstance(floor, bool) or not isinstance(floor, numbers.Real):
        raise TypeError(
            f"Invalid log floor for {analysis_name}\n"
            f"  Got: {type(floor).__name__}\n"
            "  Expected: a positive finite real number\n"
            f"Use a small value such as {DEFAULT_LOG_FLOOR}."
        )
    normalized_floor = float(floor)
    if not np.isfinite(normalized_floor) or normalized_floor <= 0:
        raise ValueError(
            f"Invalid log floor for {analysis_name}\n"
            f"  Got: {normalized_floor}\n"
            "  Expected: a positive finite real number\n"
            "The floor prevents log(0); choose a small positive value."
        )
    return normalized_floor


def _normalize_transform_axis(axis: int, *, operation_name: str) -> int:
    """Snapshot an integer transform axis without assuming an input rank."""
    if isinstance(axis, bool) or not isinstance(axis, numbers.Integral):
        raise TypeError(
            f"Invalid axis for {operation_name}\n"
            f"  Got: {type(axis).__name__}\n"
            "  Expected: a non-channel integer axis\n"
            "Pass an integer axis such as -1 or -2."
        )
    return int(axis)


def _resolve_transform_axis(axis: int, ndim: int, *, operation_name: str) -> int:
    """Resolve an axis and reject the leading channel axis."""
    resolved = axis + ndim if axis < 0 else axis
    if resolved < 0 or resolved >= ndim or resolved == 0:
        raise ValueError(
            f"Invalid axis for {operation_name}\n"
            f"  Got: axis={axis} for {ndim}D channel-first data\n"
            "  Expected: an existing non-channel axis\n"
            "Select the quefrency axis rather than the leading channel axis."
        )
    return resolved


def _resolve_fft_size(n_fft: int | None, sample_count: int) -> int:
    """Resolve an optional FFT size and reject empty implicit transforms."""
    resolved = sample_count if n_fft is None else n_fft
    if resolved <= 0:
        raise ValueError(
            "Cepstrum requires at least one sample\n"
            f"  Got: {sample_count} samples\n"
            "  Expected: a non-empty channel-first array\n"
            "Provide signal data or set n_fft to a positive integer."
        )
    return resolved


class Cepstrum(AudioOperation[NDArrayReal, NDArrayReal]):
    """Calculate a normalized real cepstrum.

    The operation windows each channel, calculates the one-sided FFT using the
    same amplitude normalization as :class:`wandas.processing.spectral.FFT`,
    applies a positive floor, and returns ``irfft(log(magnitude))``. Processing
    is lazy when called through :meth:`AudioOperation.process`.

    Parameters
    ----------
    sampling_rate : float
        Sampling rate in Hz.
    n_fft : int, optional
        FFT size. ``None`` uses the input sample count. A smaller value truncates
        the input and a larger value zero-pads it.
    window : str, default="hann"
        SciPy window name applied before the FFT.
    floor : float, default=1e-12
        Positive finite floor applied to normalized magnitudes before ``log``.

    Raises
    ------
    TypeError
        If ``n_fft`` is not an integer or ``window`` is not a non-empty string.
    ValueError
        If ``n_fft`` or ``floor`` is not positive and finite.
    """

    name = "cepstrum"
    _display = "cepstrum"

    def __init__(
        self,
        sampling_rate: float,
        n_fft: int | None = None,
        window: str = "hann",
        floor: float = DEFAULT_LOG_FLOOR,
    ) -> None:
        if n_fft is not None and (isinstance(n_fft, bool) or not isinstance(n_fft, numbers.Integral)):
            raise TypeError(
                "Invalid FFT size for cepstrum\n"
                f"  Got: {type(n_fft).__name__}\n"
                "  Expected: a positive integer or None\n"
                "Pass an integer FFT size, or omit n_fft to use the input length."
            )
        normalized_n_fft = None if n_fft is None else int(n_fft)
        if normalized_n_fft is not None and normalized_n_fft <= 0:
            raise ValueError(
                "Invalid FFT size for cepstrum\n"
                f"  Got: {normalized_n_fft}\n"
                "  Expected: a positive integer\n"
                "Use n_fft=None to match the input length automatically."
            )
        if not isinstance(window, str) or not window:
            raise TypeError(
                "Invalid window for cepstrum\n"
                f"  Got: {window!r}\n"
                "  Expected: a non-empty SciPy window name\n"
                "Use a name such as 'hann' or 'boxcar'."
            )
        normalized_floor = _normalize_log_floor(floor, analysis_name="cepstrum")
        super().__init__(
            sampling_rate,
            n_fft=normalized_n_fft,
            window=window,
            floor=normalized_floor,
        )

    @property
    def n_fft(self) -> int | None:
        """Return the configured FFT size, or ``None`` for input length."""
        return self._config_value("n_fft")

    @property
    def window(self) -> str:
        """Return the configured analysis-window name."""
        return self._config_value("window")

    @property
    def floor(self) -> float:
        """Return the positive log-magnitude floor."""
        return self._config_value("floor")

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Return ``(..., n_fft)`` without evaluating input data."""
        n_fft = _resolve_fft_size(self.n_fft, int(input_shape[-1]))
        return (*input_shape[:-1], n_fft)

    def calculate_output_dtype(
        self,
        input_dtype: np.dtype[Any],
        *input_dtypes: np.dtype[Any],
    ) -> np.dtype[Any]:
        """Return NumPy FFT's real output dtype."""
        return np.dtype(np.float64)

    def _process(self, data: NDArrayReal) -> NDArrayReal:
        """Calculate the eager real-cepstrum kernel for delayed execution."""
        if np.iscomplexobj(data):
            raise TypeError(
                "Cepstrum analysis requires real-valued input\n"
                f"  Got: {np.asarray(data).dtype}\n"
                "  Expected: real time-domain samples\n"
                "Use ChannelFrame time-domain data as the input."
            )
        n_fft = _resolve_fft_size(self.n_fft, int(data.shape[-1]))
        analysis = np.asarray(data[..., :n_fft], dtype=np.float64)
        window_values = get_window(self.window, analysis.shape[-1])
        window_gain = float(np.sum(window_values))
        if not np.isfinite(window_gain) or window_gain == 0:
            raise ValueError(
                "Invalid window gain for cepstrum\n"
                f"  Window: {self.window!r}\n"
                f"  Gain: {window_gain}\n"
                "Use a window with a finite non-zero coherent gain."
            )
        spectrum = np.fft.rfft(analysis * window_values, n=n_fft, axis=-1)
        normalized_spectrum = _normalize_rfft_amplitude(
            spectrum,
            n_fft=n_fft,
            window_gain=window_gain,
        )
        magnitude = np.abs(normalized_spectrum)
        log_magnitude = np.log(np.maximum(magnitude, self.floor))
        return np.asarray(np.fft.irfft(log_magnitude, n=n_fft, axis=-1), dtype=np.float64)


class SpectrogramCepstrum(AudioOperation[NDArrayComplex, NDArrayReal]):
    """Calculate a real cepstrum independently at every STFT time frame.

    Input data is a normalized one-sided spectrum shaped
    ``(channel, frequency, time)``. The operation discards phase, applies a
    positive log floor, and performs ``irfft`` along the frequency axis. The
    result is shaped ``(channel, quefrency, time)``.

    Parameters
    ----------
    sampling_rate : float
        Sampling rate in Hz.
    n_fft : int
        FFT size used to create the input spectrogram.
    floor : float, default=1e-12
        Positive finite floor applied to magnitude before ``log``.

    Raises
    ------
    TypeError
        If ``n_fft`` is not an integer or ``floor`` is not real.
    ValueError
        If ``n_fft`` or ``floor`` is not positive, or input shape disagrees
        with the FFT size.
    """

    name = "spectrogram_cepstrum"
    _display = "cepstrum"

    def __init__(
        self,
        sampling_rate: float,
        n_fft: int,
        floor: float = DEFAULT_LOG_FLOOR,
    ) -> None:
        if isinstance(n_fft, bool) or not isinstance(n_fft, numbers.Integral):
            raise TypeError(
                "Invalid FFT size for spectrogram cepstrum\n"
                f"  Got: {type(n_fft).__name__}\n"
                "  Expected: a positive integer\n"
                "Pass the FFT size used to create the spectrogram."
            )
        normalized_n_fft = int(n_fft)
        if normalized_n_fft <= 0:
            raise ValueError(
                "Invalid FFT size for spectrogram cepstrum\n"
                f"  Got: {normalized_n_fft}\n"
                "  Expected: a positive integer\n"
                "Pass the FFT size used to create the spectrogram."
            )
        normalized_floor = _normalize_log_floor(
            floor,
            analysis_name="spectrogram cepstrum",
        )
        super().__init__(
            sampling_rate,
            n_fft=normalized_n_fft,
            floor=normalized_floor,
        )

    @property
    def n_fft(self) -> int:
        """Return the FFT size of the input spectrogram."""
        return self._config_value("n_fft")

    @property
    def floor(self) -> float:
        """Return the positive log-magnitude floor."""
        return self._config_value("floor")

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Replace the frequency axis with a complete quefrency axis."""
        expected_frequency_bins = self.n_fft // 2 + 1
        if len(input_shape) != 3 or int(input_shape[-2]) != expected_frequency_bins:
            raise ValueError(
                "Invalid spectrogram shape for cepstrum\n"
                f"  Got: {input_shape}\n"
                f"  Expected: (channels, {expected_frequency_bins}, time) for n_fft={self.n_fft}\n"
                "Use the n_fft stored by the source SpectrogramFrame."
            )
        return (int(input_shape[0]), self.n_fft, int(input_shape[-1]))

    def calculate_output_dtype(
        self,
        input_dtype: np.dtype[Any],
        *input_dtypes: np.dtype[Any],
    ) -> np.dtype[Any]:
        """Return NumPy FFT's real output dtype."""
        return np.dtype(np.float64)

    def _process(self, data: NDArrayComplex) -> NDArrayReal:
        """Calculate the eager framewise real-cepstrum kernel."""
        self.calculate_output_shape(np.asarray(data).shape)
        magnitude = np.abs(np.asarray(data))
        log_magnitude = np.log(np.maximum(magnitude, self.floor))
        result = np.fft.irfft(log_magnitude, n=self.n_fft, axis=-2)
        return np.asarray(result, dtype=np.float64)


class Lifter(AudioOperation[NDArrayReal, NDArrayReal]):
    """Keep low- or high-quefrency real-cepstrum coefficients.

    Parameters
    ----------
    sampling_rate : float
        Sampling rate in Hz; its reciprocal is the quefrency-bin spacing.
    cutoff : float
        Positive quefrency boundary in seconds. The represented bin and its
        circularly mirrored negative-quefrency bins are included in low mode.
    mode : {"low", "high"}, default="low"
        ``"low"`` keeps the smooth spectral-envelope region. ``"high"`` keeps
        the complementary fine structure.
    axis : int, default=-1
        Non-channel quefrency axis. ``CepstrogramFrame`` uses ``-2``.

    Raises
    ------
    TypeError
        If ``cutoff`` is not a real number or ``axis`` is not an integer.
    ValueError
        If the cutoff is non-positive, non-finite, smaller than one bin, or
        overlaps the mirrored half of the concrete cepstrum; or if ``mode`` is
        unknown or ``axis`` does not identify a non-channel input axis.
    """

    name = "lifter"
    _display = "lifter"

    def __init__(
        self,
        sampling_rate: float,
        cutoff: float,
        mode: Literal["low", "high"] = "low",
        *,
        axis: int = -1,
    ) -> None:
        if isinstance(cutoff, bool) or not isinstance(cutoff, numbers.Real):
            raise TypeError(
                "Invalid lifter cutoff\n"
                f"  Got: {type(cutoff).__name__}\n"
                "  Expected: a positive finite duration in seconds\n"
                "Pass a real quefrency boundary such as 0.002."
            )
        normalized_cutoff = float(cutoff)
        if not np.isfinite(normalized_cutoff) or normalized_cutoff <= 0:
            raise ValueError(
                "Invalid lifter cutoff\n"
                f"  Got: {normalized_cutoff}\n"
                "  Expected: a positive finite duration in seconds\n"
                "Choose a small positive quefrency boundary."
            )
        if mode not in {"low", "high"}:
            raise ValueError(
                "Invalid lifter mode\n"
                f"  Got: {mode!r}\n"
                "  Expected: 'low' or 'high'\n"
                "Use 'low' for the envelope or 'high' for fine structure."
            )
        normalized_axis = _normalize_transform_axis(axis, operation_name="lifter")
        super().__init__(
            sampling_rate,
            cutoff=normalized_cutoff,
            mode=mode,
            axis=normalized_axis,
        )

    @property
    def cutoff(self) -> float:
        """Return the quefrency cutoff in seconds."""
        return self._config_value("cutoff")

    @property
    def mode(self) -> Literal["low", "high"]:
        """Return the selected low- or high-quefrency mode."""
        return self._config_value("mode")

    @property
    def axis(self) -> int:
        """Return the configured quefrency axis."""
        return self._config_value("axis")

    def calculate_output_dtype(
        self,
        input_dtype: np.dtype[Any],
        *input_dtypes: np.dtype[Any],
    ) -> np.dtype[Any]:
        """Preserve a real floating input dtype."""
        return np.dtype(np.result_type(input_dtype, np.float32))

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Validate the cutoff against the known cepstrum length."""
        axis = _resolve_transform_axis(self.axis, len(input_shape), operation_name="lifter")
        self._resolve_cutoff_bins(int(input_shape[axis]))
        return input_shape

    def _resolve_cutoff_bins(self, coefficient_count: int) -> int:
        """Return the represented cutoff after validating mirrored regions."""
        cutoff_bins = int(np.floor(self.cutoff * self.sampling_rate))
        if cutoff_bins < 1:
            raise ValueError(
                "Invalid lifter cutoff for this sampling rate\n"
                f"  Got: {self.cutoff} seconds\n"
                f"  Expected: at least one bin ({1 / self.sampling_rate:g} seconds)\n"
                "Increase cutoff so it reaches a represented quefrency bin."
            )
        if 2 * cutoff_bins >= coefficient_count:
            maximum_cutoff_bins = (coefficient_count - 1) // 2
            raise ValueError(
                "Invalid lifter cutoff for this cepstrum length\n"
                f"  Got: {self.cutoff} seconds ({cutoff_bins} bins)\n"
                f"  Expected: at most {maximum_cutoff_bins} non-overlapping bins\n"
                "Choose a smaller cutoff so mirrored regions do not overlap."
            )
        return cutoff_bins

    def _process(self, data: NDArrayReal) -> NDArrayReal:
        """Apply the eager symmetric lifter mask for delayed execution."""
        if np.iscomplexobj(data):
            raise TypeError("Lifter requires real-valued cepstral coefficients.")
        coefficients = np.asarray(data)
        axis = _resolve_transform_axis(self.axis, coefficients.ndim, operation_name="lifter")
        coefficient_count = int(coefficients.shape[axis])
        cutoff_bins = self._resolve_cutoff_bins(coefficient_count)
        keep = np.zeros(coefficient_count, dtype=bool)
        keep[: cutoff_bins + 1] = True
        keep[-cutoff_bins:] = True
        if self.mode == "high":
            keep = ~keep
        mask_shape = [1] * coefficients.ndim
        mask_shape[axis] = coefficient_count
        result = np.where(keep.reshape(mask_shape), coefficients, 0)
        return np.asarray(result, dtype=self.calculate_output_dtype(coefficients.dtype))


class SpectralEnvelope(AudioOperation[NDArrayReal, NDArrayComplex]):
    """Reconstruct a normalized one-sided spectral envelope.

    The input must be a complete, circularly symmetric real cepstrum. The
    operation calculates ``exp(real(rfft(cepstrum)))`` and returns complex data
    with zero phase so it can be represented by ``SpectralFrame``. Processing is
    lazy through :meth:`AudioOperation.process`.

    Parameters
    ----------
    sampling_rate : float
        Sampling rate in Hz.
    axis : int, default=-1
        Non-channel quefrency axis. ``CepstrogramFrame`` uses ``-2``.

    Raises
    ------
    TypeError
        If ``axis`` is not an integer or concrete input is complex-valued.
    ValueError
        If ``axis`` is invalid or concrete coefficients are not circularly
        symmetric.
    """

    name = "spectral_envelope"
    _display = "spectral envelope"

    def __init__(self, sampling_rate: float, *, axis: int = -1) -> None:
        normalized_axis = _normalize_transform_axis(
            axis,
            operation_name="spectral envelope",
        )
        super().__init__(sampling_rate, axis=normalized_axis)

    @property
    def axis(self) -> int:
        """Return the configured quefrency axis."""
        return self._config_value("axis")

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Replace the quefrency axis with its one-sided frequency axis."""
        axis = _resolve_transform_axis(
            self.axis,
            len(input_shape),
            operation_name="spectral envelope",
        )
        output_shape = list(input_shape)
        output_shape[axis] = int(input_shape[axis]) // 2 + 1
        return tuple(output_shape)

    def calculate_output_dtype(
        self,
        input_dtype: np.dtype[Any],
        *input_dtypes: np.dtype[Any],
    ) -> np.dtype[Any]:
        """Return the complex dtype used by ``SpectralFrame``."""
        return np.dtype(np.complex128)

    def _process(self, data: NDArrayReal) -> NDArrayComplex:
        """Calculate the eager spectral-envelope kernel for delayed execution."""
        if np.iscomplexobj(data):
            raise TypeError("SpectralEnvelope requires real-valued cepstral coefficients.")
        coefficients = np.asarray(data, dtype=np.float64)
        axis = _resolve_transform_axis(
            self.axis,
            coefficients.ndim,
            operation_name="spectral envelope",
        )
        transformed = np.moveaxis(coefficients, axis, -1)
        tolerance = 64 * np.finfo(np.float64).eps
        if not np.allclose(
            transformed[..., 1:],
            transformed[..., :0:-1],
            rtol=tolerance,
            atol=tolerance,
        ):
            raise ValueError("SpectralEnvelope requires symmetric real cepstral coefficients.")
        log_envelope = np.fft.rfft(transformed, axis=-1)
        envelope = np.exp(np.real(log_envelope))
        restored_axis = np.moveaxis(envelope, -1, axis)
        return np.asarray(restored_axis, dtype=np.complex128)


for _operation in (Cepstrum, SpectrogramCepstrum, Lifter, SpectralEnvelope):
    register_operation(_operation)


__all__ = [
    "DEFAULT_LOG_FLOOR",
    "Cepstrum",
    "Lifter",
    "SpectralEnvelope",
    "SpectrogramCepstrum",
]
