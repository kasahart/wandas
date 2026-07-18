"""Lazy numerical helpers for per-channel calibration."""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence

import numpy as np
from dask.array.core import Array as DaArray

from wandas.utils.types import NDArrayReal


def _derive_absolute_calibration_factors(
    measured_rms: Sequence[float] | NDArrayReal,
    current_factors: Sequence[float],
    *,
    target_rms: float | None = None,
    target_level: float | None = None,
    ref: float,
) -> tuple[float, ...]:
    """Return absolute factors for one scalar-target calibration event."""
    if (target_rms is None) == (target_level is None):
        raise ValueError("Exactly one of target_rms or target_level is required")

    def scalar(value: object, *, name: str, positive: bool) -> float:
        if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Real):
            raise TypeError(f"{name} must be a real scalar")
        normalized = float(value)
        if not math.isfinite(normalized) or (positive and normalized <= 0.0):
            requirement = "positive and finite" if positive else "finite"
            raise ValueError(f"{name} must be {requirement}")
        return normalized

    reference = scalar(ref, name="Calibration reference", positive=True)
    if target_rms is not None:
        physical_target = scalar(target_rms, name="target_rms", positive=True)
    else:
        level = scalar(target_level, name="target_level", positive=False)
        with np.errstate(over="ignore", invalid="ignore"):
            physical_target = float(reference * np.power(10.0, level / 20.0))
        if not math.isfinite(physical_target) or physical_target <= 0.0:
            raise ValueError("target_level must produce a positive finite RMS")

    measured = tuple(measured_rms)
    current = tuple(current_factors)
    if not measured or len(measured) != len(current):
        raise ValueError("RMS values and current factors must have the same non-zero length")
    results: list[float] = []
    for rms, factor in zip(measured, current, strict=True):
        normalized_rms = scalar(rms, name="Reference RMS", positive=True)
        normalized_factor = scalar(factor, name="Current calibration factor", positive=True)
        absolute = normalized_factor * physical_target / normalized_rms
        if not math.isfinite(absolute) or absolute <= 0.0:
            raise ValueError("Derived calibration factor must be positive and finite")
        results.append(absolute)
    return tuple(results)


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
