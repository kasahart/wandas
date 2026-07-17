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
    if isinstance(value, str | bytes | bool):
        raise TypeError(
            f"{heading}\n"
            f"  Got: {type(value).__name__} ({value!r})\n"
            "  Expected: a real number or one-dimensional numeric sequence\n"
            "Pass one value for broadcast or one value per channel."
        )
    try:
        values = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{heading}\n"
            f"  Got: {type(value).__name__} ({value!r})\n"
            "  Expected: a real number or one-dimensional numeric sequence\n"
            "Pass one value for broadcast or one value per channel."
        ) from exc
    if values.dtype.kind not in {"f", "i", "u"}:
        raise TypeError(
            f"{heading}\n"
            f"  Got: {values.dtype} values\n"
            "  Expected: real numeric values\n"
            "Replace booleans, strings, complex values, or objects with real numbers."
        )
    values = values.astype(np.float64, copy=False)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            f"{heading}\n"
            f"  Got shape: {values.shape}\n"
            "  Expected: one or more values in one dimension\n"
            "Pass one value for broadcast or one value per channel."
        )
    if not np.all(np.isfinite(values)) or (positive and not np.all(values > 0.0)):
        expectation = "finite values greater than zero" if positive else "finite values"
        raise ValueError(
            f"{heading}\n"
            f"  Got: {tuple(float(item) for item in values)}\n"
            f"  Expected: {expectation}\n"
            "Check the calibration recording and its known physical target."
        )
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
        raise ValueError(
            "Exactly one calibration target is required\n"
            f"  Got target_rms={target_rms!r}, target_level={target_level!r}\n"
            "  Expected: one of target_rms or target_level\n"
            "Use target_level for a level calibrator or target_rms for a known physical RMS."
        )
    if isinstance(ref, bool) or not isinstance(ref, numbers.Real):
        raise TypeError(
            "Invalid calibration reference\n"
            f"  Got: {type(ref).__name__} ({ref!r})\n"
            "  Expected: a positive finite number\n"
            "Use 2e-5 for sound pressure levels in Pa."
        )
    reference = float(ref)
    if not math.isfinite(reference) or reference <= 0.0:
        raise ValueError(
            "Invalid calibration reference\n"
            f"  Got: {reference!r}\n"
            "  Expected: a positive finite number\n"
            "Use 2e-5 for sound pressure levels in Pa."
        )

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
        raise ValueError(
            "Calibration target length mismatch\n"
            f"  Got: {target.size} targets for {measured.size} measured channels\n"
            "  Expected: one target or one target per measured channel\n"
            "Align target values with the calibration signal channels."
        )

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        factors = target / measured
    if not np.all(np.isfinite(factors)) or not np.all(factors > 0.0):
        raise ValueError(
            "Invalid derived calibration factors\n"
            f"  Got: {tuple(float(item) for item in factors)}\n"
            "  Expected: one positive finite factor per channel\n"
            "Check the calibration recording and its known physical target."
        )
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
