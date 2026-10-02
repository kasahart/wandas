"""Validation of optional, authoritative time-frequency frame origins."""

from __future__ import annotations

import numbers

import numpy as np


def _normalize_frame_time_origin(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError("frame_time_origin must be a finite number of seconds or None (unknown)")
    origin = float(value)
    if not np.isfinite(origin):
        raise ValueError("frame_time_origin must be finite or None (unknown)")
    return origin
