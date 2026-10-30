"""Frame time origins and selection of time-axis Frames by seconds."""

from __future__ import annotations

import math
import numbers
import re
from typing import Any, Literal, TypeAlias

import numpy as np

from wandas.utils.types import NDArrayReal

TimeValue: TypeAlias = float | str
"""Seconds as a number, or a string with units such as ``"1min30s"``."""

TimeReference: TypeAlias = Literal["local", "source"]

_TIME_UNIT_SECONDS = {"h": 3600.0, "min": 60.0, "s": 1.0, "ms": 1e-3}
_TIME_TOKEN = re.compile(r"(\d+(?:\.\d*)?|\.\d+)(h|min|ms|s)")


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


def _seconds(value: TimeValue, name: str) -> float:
    """Convert a number of seconds or a unit string such as ``"1h2min"`` to seconds."""
    if isinstance(value, str):
        text = value.replace(" ", "")
        tokens = _TIME_TOKEN.findall(text)
        units = [unit for _, unit in tokens]
        if not tokens or "".join(amount + unit for amount, unit in tokens) != text or len(set(units)) != len(units):
            raise TypeError(
                f"Invalid {name} time string\n"
                f"  Got: {value!r}\n"
                "  Expected: numbers with the units h, min, s or ms, each unit once, such as '1min30s'\n"
                "Pass a number of seconds or add units to the string."
            )
        return sum(float(amount) * _TIME_UNIT_SECONDS[unit] for amount, unit in tokens)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a number of seconds or a string with units, such as '1min30s'")
    seconds = float(value)
    if not math.isfinite(seconds):
        raise ValueError(f"{name} must be a finite number of seconds")
    return seconds


def _local_time_range(
    frame: Any, start: TimeValue, end: TimeValue | None, reference: TimeReference
) -> tuple[float, float | None]:
    """Return a requested ``[start, end)`` range in seconds from the frame start.

    ``reference="source"`` measures the range on the original source timeline,
    which needs one source offset shared by every channel.
    """
    start_seconds = _seconds(start, "start")
    end_seconds = None if end is None else _seconds(end, "end")
    if start_seconds < 0 or (end_seconds is not None and end_seconds < 0):
        raise ValueError("Trim times must be non-negative")
    if end_seconds is not None and start_seconds > end_seconds:
        raise ValueError("start must be less than or equal to end")
    offset = _shared_source_offset(frame, reference)
    return start_seconds - offset, None if end_seconds is None else end_seconds - offset


def _shared_source_offset(frame: Any, reference: TimeReference) -> float:
    """Return the shift from the requested time reference to the frame start."""
    if reference == "local":
        return 0.0
    if reference != "source":
        raise ValueError(f"reference must be 'local' or 'source', got {reference!r}")
    offsets = np.asarray(frame.source_time_offset, dtype=float)
    if not np.all(offsets == offsets[0]):
        raise ValueError(
            "Channels start at different source times\n"
            f"  Got: source_time_offset {offsets.tolist()}\n"
            "  Expected: one source offset shared by every channel\n"
            "Select the channels separately, or use reference='local'."
        )
    return float(offsets[0])


def _nearest_sample(seconds: float, sampling_rate: float, n_samples: int) -> int:
    """Return the sample nearest ``seconds``, clipped to ``[0, n_samples]``."""
    return min(max(round(seconds * sampling_rate), 0), n_samples)


def _local_frame_centers(frame: Any) -> NDArrayReal:
    """Return frame centers in seconds from the start of the frame's source signal."""
    return _physical_frame_center_times(
        frame.frame_time_origin, frame.n_frames, frame.hop_length, frame.sampling_rate, np.zeros(1)
    )[0]


def _frame_center_slice(frame: Any, start: TimeValue, end: TimeValue | None, reference: TimeReference) -> slice:
    """Return the frames whose centers lie in the half-open ``[start, end)`` range."""
    local_start, local_end = _local_time_range(frame, start, end, reference)
    centers = _local_frame_centers(frame)
    # Frame centers carry floating-point error, so a center a hair below a bound still counts as on it.
    tolerance = frame.hop_length / frame.sampling_rate * 1e-6
    first = int(np.searchsorted(centers, local_start - tolerance, side="left"))
    stop = len(centers) if local_end is None else int(np.searchsorted(centers, local_end - tolerance, side="left"))
    if first >= stop:
        raise ValueError(
            "Time range selects no frame\n"
            f"  Got: [{start!r}, {end!r}) with reference={reference!r}\n"
            f"  Expected: a range containing a frame center between {centers[0]:.6g} s and {centers[-1]:.6g} s\n"
            "Widen the range or check the time reference."
        )
    return slice(first, stop)


def _nearest_frame_index(frame: Any, time: TimeValue, reference: TimeReference, tolerance: TimeValue | None) -> int:
    """Return the frame whose center is nearest ``time``; ties go to the earlier frame."""
    seconds = _seconds(time, "time")
    local_time = seconds - _shared_source_offset(frame, reference)
    centers = _local_frame_centers(frame)
    half_hop = frame.hop_length / frame.sampling_rate / 2
    if local_time < centers[0] - half_hop or local_time > centers[-1] + half_hop:
        raise ValueError(
            "Time is outside the spectrogram\n"
            f"  Got: {time!r} with reference={reference!r}\n"
            f"  Expected: a time within half a hop of the frame centers {centers[0]:.6g} s to {centers[-1]:.6g} s\n"
            "Pick a time inside the recording."
        )
    distances = np.abs(centers - local_time)
    # The earliest center within rounding error of the smallest distance wins a tie.
    index = int(np.flatnonzero(distances <= distances.min() + half_hop * 1e-6)[0])
    if tolerance is not None:
        limit = _seconds(tolerance, "tolerance")
        if limit < 0:
            raise ValueError("tolerance must be non-negative")
        distance = abs(float(centers[index]) - local_time)
        if distance > limit:
            raise ValueError(
                "No frame center within the tolerance\n"
                f"  Got: nearest center {float(centers[index]):.6g} s, {distance:.6g} s from {time!r}\n"
                f"  Expected: at most {limit:.6g} s away\n"
                "Increase tolerance or use a smaller hop_length."
            )
    return index
