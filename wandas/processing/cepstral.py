import logging
from typing import Any

import dask.array as da
import numpy as np
from dask.array.core import Array as DaArray
from dask.delayed import delayed
from scipy.signal import get_window

from wandas.processing.base import AudioOperation, register_operation
from wandas.utils.types import NDArrayComplex, NDArrayReal

logger = logging.getLogger(__name__)

MIN_LOG_MAGNITUDE = 1e-12


def _resolve_n_fft(n_fft: int | None, length: int) -> int:
    """Resolve FFT size using the input length when not explicitly provided."""
    return length if n_fft is None else n_fft


def _real_output_dtype(dtype: np.dtype[Any]) -> np.dtype[Any]:
    """Return a floating output dtype compatible with the input precision."""
    input_dtype = np.dtype(dtype)
    if input_dtype == np.float32:
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def _complex_output_dtype(dtype: np.dtype[Any]) -> np.dtype[Any]:
    """Return a complex output dtype compatible with the input precision."""
    input_dtype = np.dtype(dtype)
    if input_dtype in (np.float32, np.complex64):
        return np.dtype(np.complex64)
    return np.dtype(np.complex128)


class Cepstrum(AudioOperation[NDArrayReal, NDArrayReal]):
    """Real cepstrum analysis."""

    name = "cepstrum"

    def __init__(
        self,
        sampling_rate: float,
        n_fft: int | None = None,
        window: str = "hann",
        floor: float = MIN_LOG_MAGNITUDE,
    ):
        if n_fft is not None and n_fft <= 0:
            raise ValueError(
                f"Invalid FFT size for cepstrum\n"
                f"  Got: {n_fft}\n"
                f"  Expected: Positive integer > 0\n"
                f"Cepstrum analysis requires a positive FFT size.\n"
                f"Use n_fft=None to match the input length automatically."
            )
        if floor <= 0:
            raise ValueError(
                f"Invalid log floor for cepstrum\n"
                f"  Got: {floor}\n"
                f"  Expected: Positive float > 0\n"
                f"The log-magnitude floor prevents log(0) during cepstrum analysis.\n"
                f"Use a small positive value such as {MIN_LOG_MAGNITUDE}."
            )

        self.n_fft = n_fft
        self.window = window
        self.floor = floor
        super().__init__(sampling_rate, n_fft=n_fft, window=window, floor=floor)

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (*input_shape[:-1], _resolve_n_fft(self.n_fft, int(input_shape[-1])))

    def get_display_name(self) -> str:
        return "ceps"

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        logger.debug(f"Applying cepstrum to array with shape: {x.shape}")

        target_length = _resolve_n_fft(self.n_fft, int(x.shape[-1]))
        if self.n_fft is not None and x.shape[-1] > self.n_fft:
            x = x[..., : self.n_fft]

        work = np.asarray(x, dtype=np.float64)
        win = get_window(self.window, work.shape[-1])
        spectrum = np.fft.rfft(work * win, n=target_length, axis=-1)
        log_magnitude = np.log(np.maximum(np.abs(spectrum), self.floor))
        result = np.fft.irfft(log_magnitude, n=target_length, axis=-1)
        return np.asarray(result, dtype=_real_output_dtype(np.dtype(x.dtype)))


class Lifter(AudioOperation[NDArrayReal, NDArrayReal]):
    """Apply low-pass or high-pass liftering in the quefrency domain."""

    name = "lifter"

    def __init__(self, sampling_rate: float, cutoff: float, mode: str = "low"):
        if cutoff <= 0:
            raise ValueError(
                f"Invalid lifter cutoff\n"
                f"  Got: {cutoff}\n"
                f"  Expected: Positive float > 0 seconds\n"
                f"Lifter cutoff must be a positive quefrency boundary.\n"
                f"Choose a small value such as 0.002 to separate envelope and pitch."
            )
        if mode not in {"low", "high"}:
            raise ValueError(
                f"Invalid lifter mode\n"
                f"  Got: {mode}\n"
                f"  Expected: 'low' or 'high'\n"
                f"Use 'low' to keep the spectral envelope or 'high' to emphasize fine structure."
            )

        self.cutoff = cutoff
        self.mode = mode
        super().__init__(sampling_rate, cutoff=cutoff, mode=mode)

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def get_display_name(self) -> str:
        return "lifter"

    def _process_array(self, x: NDArrayReal) -> NDArrayReal:
        logger.debug(f"Applying {self.mode} lifter to array with shape: {x.shape}")

        n_quefrency = int(x.shape[-1])
        cutoff_samples = int(np.floor(self.cutoff * self.sampling_rate))
        if cutoff_samples <= 0:
            raise ValueError(
                f"Invalid lifter cutoff\n"
                f"  Got: {self.cutoff}\n"
                f"  Expected: At least one quefrency sample ({1 / self.sampling_rate:.6g} s)\n"
                f"The cutoff is too small for the current sampling rate.\n"
                f"Increase the cutoff or use a higher sampling rate."
            )
        if cutoff_samples >= n_quefrency // 2:
            raise ValueError(
                f"Invalid lifter cutoff\n"
                f"  Got: {self.cutoff}\n"
                f"  Expected: cutoff < {n_quefrency / (2 * self.sampling_rate):.6g} s\n"
                f"The cutoff would overlap the mirrored negative quefrency region.\n"
                f"Choose a smaller cutoff for this cepstrum length."
            )

        mask = np.zeros(n_quefrency, dtype=bool)
        mask[: cutoff_samples + 1] = True
        if cutoff_samples > 0:
            mask[-cutoff_samples:] = True

        if self.mode == "high":
            mask = ~mask

        result = np.asarray(x).copy()
        result[..., ~mask] = 0.0
        return np.asarray(result, dtype=_real_output_dtype(np.dtype(x.dtype)))


class SpectralEnvelope(AudioOperation[NDArrayReal, NDArrayComplex]):
    """Reconstruct a smooth spectral envelope from a liftered cepstrum."""

    name = "spectral_envelope"

    def __init__(self, sampling_rate: float, n_fft: int | None = None):
        if n_fft is not None and n_fft <= 0:
            raise ValueError(
                f"Invalid FFT size for spectral envelope\n"
                f"  Got: {n_fft}\n"
                f"  Expected: Positive integer > 0\n"
                f"Envelope reconstruction requires a positive FFT size.\n"
                f"Use n_fft=None to match the cepstrum length automatically."
            )
        self.n_fft = n_fft
        super().__init__(sampling_rate, n_fft=n_fft)

    def calculate_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        n_fft = _resolve_n_fft(self.n_fft, int(input_shape[-1]))
        return (*input_shape[:-1], n_fft // 2 + 1)

    def get_display_name(self) -> str:
        return "env"

    def _process_array(self, x: NDArrayReal) -> NDArrayComplex:
        logger.debug(f"Applying spectral envelope reconstruction to array with shape: {x.shape}")

        n_fft = _resolve_n_fft(self.n_fft, int(x.shape[-1]))
        log_envelope = np.fft.rfft(np.asarray(x, dtype=np.float64), n=n_fft, axis=-1)
        envelope = np.exp(np.real(log_envelope))
        complex_dtype = _complex_output_dtype(np.dtype(x.dtype))
        return np.asarray(envelope.astype(complex_dtype, copy=False))

    def process(self, data: DaArray) -> DaArray:
        logger.debug("Adding delayed spectral envelope operation to computation graph")
        delayed_result = delayed(self._create_named_wrapper(), pure=self.pure)(data)
        output_shape = self.calculate_output_shape(data.shape)
        return da.from_delayed(
            delayed_result,
            shape=output_shape,
            dtype=_complex_output_dtype(np.dtype(data.dtype)),
        )


for op_class in [Cepstrum, Lifter, SpectralEnvelope]:
    register_operation(op_class)
