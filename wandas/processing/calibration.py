"""Lazy numerical helpers for per-channel calibration."""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence

import numpy as np
from dask.array.core import Array as DaArray

from wandas.utils.types import NDArrayReal

CalibrationTarget = float | Sequence[float] | NDArrayReal


def _numeric_vector(value: object, *, heading: str, positive: bool) -> np.ndarray:
    """Normalize a scalar or one-dimensional real sequence."""
    if np.ma.is_masked(value):
        raise ValueError(f"{heading} cannot contain masked values")
    objects = np.asarray(value, dtype=object)
    if any(isinstance(item, bool | np.bool_) for item in objects.flat):
        raise TypeError(f"{heading} cannot contain boolean values")
    if any(isinstance(item, numbers.Complex) and not isinstance(item, numbers.Real) for item in objects.flat):
        raise TypeError(f"{heading} cannot contain complex values")

    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{heading} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{heading} must contain only finite values")
    if positive and not np.all(values > 0.0):
        raise ValueError(f"{heading} must contain only positive values")
    return values


def derive_calibration_factors(
    measured_rms: Sequence[float] | NDArrayReal,
    *,
    target_rms: CalibrationTarget | None = None,
    target_level: CalibrationTarget | None = None,
    ref: float,
) -> tuple[float, ...]:
    """Derive one raw-to-physical factor per measured calibration channel.

    Exactly one target is required. ``target_level`` uses the amplitude-level
    relationship ``target_rms = ref * 10 ** (target_level / 20)``. A scalar
    target broadcasts to every measured channel; a sequence must align exactly.
    """
    if (target_rms is None) == (target_level is None):
        raise ValueError("Exactly one of target_rms or target_level is required")
    if isinstance(ref, bool | np.bool_):
        raise TypeError("Calibration reference cannot be boolean")
    reference_objects = np.asarray(ref, dtype=object)
    if any(isinstance(item, numbers.Complex) and not isinstance(item, numbers.Real) for item in reference_objects.flat):
        raise TypeError("Calibration reference cannot be complex")
    reference = float(np.asarray(ref, dtype=float).item())
    if not math.isfinite(reference) or reference <= 0.0:
        raise ValueError("Calibration reference must be positive and finite")

    measured = _numeric_vector(measured_rms, heading="Invalid measured calibration RMS", positive=True)
    if target_rms is not None:
        target = _numeric_vector(target_rms, heading="Invalid calibration target RMS", positive=True)
    else:
        levels = _numeric_vector(target_level, heading="Invalid calibration target level", positive=False)
        with np.errstate(over="ignore", invalid="ignore"):
            target = reference * np.power(10.0, levels / 20.0)
        target = _numeric_vector(target, heading="Invalid calibration target level", positive=True)

    if target.size == 1:
        target = np.repeat(target, measured.size)
    elif target.size != measured.size:
        raise ValueError("Calibration target must have length one or match measured_rms")

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        factors = target / measured
    if not np.all(np.isfinite(factors)) or not np.all(factors > 0.0):
        raise ValueError("Derived calibration factors must be positive and finite")
    return tuple(float(item) for item in factors)


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


__all__ = ["CalibrationTarget", "apply_channel_factors", "derive_calibration_factors"]
