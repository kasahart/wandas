"""Validation of optional, authoritative time-frequency frame origins."""

from __future__ import annotations

import numbers

import numpy as np

from wandas.utils.types import NDArrayReal


def _normalize_frame_time_origin(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError("frame_time_origin must be a finite number of seconds or None (unknown)")
    origin = float(value)
    if not np.isfinite(origin):
        raise ValueError("frame_time_origin must be finite or None (unknown)")
    return origin


def _physical_frame_center_times(
    origin: float | None, n_frames: int, hop_length: int, sampling_rate: int | float, source_time_offset: NDArrayReal
) -> NDArrayReal:
    """Reconstruct the SciPy physical clock with its exact floating-point order."""
    if origin is None:
        raise ValueError("Physical frame center times are unknown; recompute STFT or supply frame_time_origin")
    step = hop_length * (1.0 / sampling_rate)
    centers = (np.arange(n_frames) + origin / step) * step
    return source_time_offset[:, None] + centers[None, :]
