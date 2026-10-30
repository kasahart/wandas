"""Represented one-sided frequency bins shared by spectral Frame families."""

from __future__ import annotations

import numbers
from collections.abc import Sequence

import numpy as np

from wandas.utils.types import NDArrayReal


def complete_bin_count(n_fft: int) -> int:
    """Return the bin count of the complete one-sided ``n_fft`` spectrum."""
    return int(n_fft) // 2 + 1


def normalize_frequency_indices(frequency_indices: Sequence[int] | None, n_fft: int) -> tuple[int, ...]:
    """Validate the canonical rfft bins a Frame represents.

    Args:
        frequency_indices: Ordered, unique bin indices on the ``n_fft`` rfft grid.
            ``None`` means the complete one-sided spectrum.
        n_fft: FFT size defining the grid.

    Returns:
        The bins as a tuple of ints.

    Raises:
        TypeError: If the value is not a sequence of integers.
        ValueError: If it is empty, holds a bin outside the grid, or repeats a bin.
    """
    complete_count = complete_bin_count(n_fft)
    if frequency_indices is None:
        return tuple(range(complete_count))
    if isinstance(frequency_indices, (str, bytes)):
        raise TypeError("frequency_indices must be an ordered sequence of integers")
    normalized = tuple(frequency_indices)
    if not normalized:
        raise ValueError("frequency_indices must contain at least one frequency bin")
    if any(isinstance(index, bool) or not isinstance(index, numbers.Integral) for index in normalized):
        raise TypeError("frequency_indices must contain only integer bin indices")
    normalized = tuple(int(index) for index in normalized)
    if any(index < 0 or index >= complete_count for index in normalized):
        raise ValueError("frequency_indices contains a bin outside the n_fft frequency axis")
    if len(set(normalized)) != len(normalized):
        raise ValueError("frequency_indices must not contain duplicate bins")
    return normalized


def select_frequency_indices(frequency_indices: tuple[int, ...], selector: slice) -> tuple[int, ...]:
    """Map a public frequency-axis slice onto the represented canonical bins."""
    start, stop, step = selector.indices(len(frequency_indices))
    return tuple(frequency_indices[index] for index in range(start, stop, step))


def represented_freqs(frequency_indices: tuple[int, ...], n_fft: int, sampling_rate: float) -> NDArrayReal:
    """Return the frequencies in Hz of the represented bins."""
    complete = np.fft.rfftfreq(n_fft, 1.0 / sampling_rate)
    if is_complete_spectrum(frequency_indices, n_fft):
        return complete
    return complete[list(frequency_indices)]


def is_complete_spectrum(frequency_indices: tuple[int, ...], n_fft: int) -> bool:
    """Return whether the bins are the complete one-sided spectrum in order."""
    return frequency_indices == tuple(range(complete_bin_count(n_fft)))


def require_complete_spectrum(
    frequency_indices: tuple[int, ...], n_fft: int, *, frame_type: str, operation: str
) -> None:
    """Raise when an operation needs every bin but the Frame holds a subset."""
    if is_complete_spectrum(frequency_indices, n_fft):
        return
    raise ValueError(
        f"{operation} needs the complete spectrum\n"
        f"  Got: a {frame_type} with {len(frequency_indices)} of {complete_bin_count(n_fft)} frequency bins\n"
        "  Expected: every one-sided bin of n_fft\n"
        f"Apply {operation} before selecting frequency bins."
    )


def require_matching_frequency_bins(
    left: tuple[int, ...], left_n_fft: int, right: tuple[int, ...], right_n_fft: int, *, symbol: str
) -> None:
    """Raise when two spectral operands hold different frequency bins.

    Equal bin indices name equal frequencies only on the same ``n_fft`` grid, so
    both are compared.
    """
    if left == right and left_n_fft == right_n_fft:
        return
    raise ValueError(
        "Spectral frequency bins mismatch\n"
        f"  Left operand: {len(left)} bins from bin {left[0]} of n_fft={left_n_fft}\n"
        f"  Right operand: {len(right)} bins from bin {right[0]} of n_fft={right_n_fft}\n"
        f"Use the same n_fft and select the same frequency bins on both spectra before performing {symbol} operation."
    )
