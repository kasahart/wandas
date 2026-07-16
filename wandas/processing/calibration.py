"""Lazy numerical helpers for per-channel calibration."""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence

import numpy as np
from dask.array.core import Array as DaArray


def apply_channel_factors(data: DaArray, factors: Sequence[float]) -> DaArray:
    """Multiply channel-first data by one factor per channel without computing it."""
    if not isinstance(data, DaArray):
        raise TypeError("Calibration data must be a Dask array")
    if data.ndim < 1:
        raise ValueError("Calibration data must have a channel axis")
    values = tuple(factors)
    if len(values) != int(data.shape[0]):
        raise ValueError(
            "Calibration factor length mismatch\n"
            f"  Got: {len(values)} factors\n"
            f"  Expected: {data.shape[0]} factors for the channel axis\n"
            "Align calibration metadata with the current channel order."
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(float(value))
        or float(value) <= 0
        for value in values
    ):
        raise ValueError(
            "Invalid calibration factors\n"
            f"  Got: {values!r}\n"
            "  Expected: one positive finite number per channel\n"
            "Validate each ChannelCalibration before applying it."
        )
    broadcast_shape = (len(values),) + (1,) * (data.ndim - 1)
    return data * np.asarray(values, dtype=float).reshape(broadcast_shape)


__all__ = ["apply_channel_factors"]
