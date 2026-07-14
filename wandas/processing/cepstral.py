"""Real-cepstrum analysis, liftering, and spectral-envelope reconstruction."""

from __future__ import annotations

import logging
import numbers
from typing import Any, Literal

import numpy as np
from scipy.signal import get_window

from wandas.processing.base import AudioOperation, register_operation
from wandas.utils.types import NDArrayComplex, NDArrayReal

logger = logging.getLogger(__name__)

DEFAULT_LOG_FLOOR = 1e-12


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
        if isinstance(floor, bool) or not isinstance(floor, numbers.Real):
            raise TypeError(
                "Invalid log floor for cepstrum\n"
                f"  Got: {type(floor).__name__}\n"
                "  Expected: a positive finite real number\n"
                f"Use a small value such as {DEFAULT_LOG_FLOOR}."
            )
        normalized_floor = float(floor)
        if not np.isfinite(normalized_floor) or normalized_floor <= 0:
            raise ValueError(
                "Invalid log floor for cepstrum\n"
                f"  Got: {normalized_floor}\n"
                "  Expected: a positive finite real number\n"
                "The floor prevents log(0); choose a small positive value."
            )
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
        spectrum[..., 1:-1] *= 2.0
        magnitude = np.abs(spectrum / window_gain)
        log_magnitude = np.log(np.maximum(magnitude, self.floor))
        return np.asarray(np.fft.irfft(log_magnitude, n=n_fft, axis=-1), dtype=np.float64)


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

    Raises
    ------
    TypeError
        If ``cutoff`` is not a real number.
    ValueError
        If the cutoff is non-positive, non-finite, smaller than one bin, or
        overlaps the mirrored half of the concrete cepstrum; or if ``mode`` is
        unknown.
    """

    name = "lifter"
    _display = "lifter"

    def __init__(
        self,
        sampling_rate: float,
        cutoff: float,
        mode: Literal["low", "high"] = "low",
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
        super().__init__(sampling_rate, cutoff=normalized_cutoff, mode=mode)

    @property
    def cutoff(self) -> float:
        """Return the quefrency cutoff in seconds."""
        return self._config_value("cutoff")

    @property
    def mode(self) -> Literal["low", "high"]:
        """Return the selected low- or high-quefrency mode."""
        return self._config_value("mode")

    def calculate_output_dtype(
        self,
        input_dtype: np.dtype[Any],
        *input_dtypes: np.dtype[Any],
    ) -> np.dtype[Any]:
        """Preserve a real floating input dtype."""
        return np.dtype(np.result_type(input_dtype, np.float32))

    def _process(self, data: NDArrayReal) -> NDArrayReal:
        """Apply the eager symmetric lifter mask for delayed execution."""
        if np.iscomplexobj(data):
            raise TypeError("Lifter requires real-valued cepstral coefficients.")
        coefficient_count = int(data.shape[-1])
        cutoff_bins = int(np.floor(self.cutoff * self.sampling_rate))
        if cutoff_bins < 1:
            raise ValueError(
                "Invalid lifter cutoff for this sampling rate\n"
                f"  Got: {self.cutoff} seconds\n"
                f"  Expected: at least one bin ({1 / self.sampling_rate:g} seconds)\n"
                "Increase cutoff so it reaches a represented quefrency bin."
            )
        if cutoff_bins >= coefficient_count // 2:
            raise ValueError(
                "Invalid lifter cutoff for this cepstrum length\n"
                f"  Got: {self.cutoff} seconds ({cutoff_bins} bins)\n"
                f"  Expected: fewer than {coefficient_count // 2} bins\n"
                "Choose a smaller cutoff so mirrored regions do not overlap."
            )
        keep = np.zeros(coefficient_count, dtype=bool)
        keep[: cutoff_bins + 1] = True
        keep[-cutoff_bins:] = True
        if self.mode == "high":
            keep = ~keep
        result = np.where(keep, np.asarray(data), 0)
        return np.asarray(result, dtype=self.calculate_output_dtype(np.asarray(data).dtype))


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

    Raises
    ------
    TypeError
        If the concrete input is complex-valued.
    ValueError
        If the concrete coefficients are not circularly symmetric.
    """

    name = "spectral_envelope"
    _display = "spectral envelope"

    def __init__(self, sampling_rate: float) -> None:
        super().__init__(sampling_rate)

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Return the one-sided FFT shape without evaluating input data."""
        return (*input_shape[:-1], int(input_shape[-1]) // 2 + 1)

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
        tolerance = 64 * np.finfo(np.float64).eps
        if not np.allclose(
            coefficients[..., 1:],
            coefficients[..., :0:-1],
            rtol=tolerance,
            atol=tolerance,
        ):
            raise ValueError("SpectralEnvelope requires symmetric real cepstral coefficients.")
        log_envelope = np.fft.rfft(coefficients, axis=-1)
        envelope = np.exp(np.real(log_envelope))
        return np.asarray(envelope, dtype=np.complex128)


for _operation in (Cepstrum, Lifter, SpectralEnvelope):
    register_operation(_operation)


__all__ = ["DEFAULT_LOG_FLOOR", "Cepstrum", "Lifter", "SpectralEnvelope"]
