"""Level-trigger detection on one sampled signal."""

from typing import Literal

import numpy as np
import numpy.typing as npt

from wandas.utils.types import NDArrayReal

TriggerSlope = Literal["rising", "falling"]


def find_level_crossings(
    x: NDArrayReal,
    threshold: float,
    slope: TriggerSlope = "rising",
    holdoff: int = 0,
) -> npt.NDArray[np.int64]:
    """Return the sample indices where a 1-D signal crosses a level.

    A rising crossing is the first sample ``n`` with ``x[n - 1] < threshold <=
    x[n]``; a falling crossing is the first sample with ``x[n - 1] > threshold
    >= x[n]``. The first sample is never a crossing because it has no
    predecessor. After a crossing at ``n``, crossings before ``n + holdoff`` are
    ignored so that ringing or a bounce of the same event does not re-trigger.

    Args:
        x: 1-D signal.
        threshold: Trigger level in the unit of ``x``.
        slope: ``"rising"`` or ``"falling"``.
        holdoff: Minimum distance in samples between accepted crossings.

    Returns:
        Sorted sample indices of the accepted crossings.

    Raises:
        ValueError: If ``x`` is not 1-D, ``slope`` is unknown or ``holdoff`` is
            negative.
    """
    if x.ndim != 1:
        raise ValueError(f"Trigger detection needs a 1-D signal\n  Got: shape {x.shape}\n  Expected: (n_samples,)")
    if slope not in ("rising", "falling"):
        raise ValueError(f"Unknown trigger slope\n  Got: {slope!r}\n  Expected: 'rising' or 'falling'")
    if holdoff < 0:
        raise ValueError(f"Trigger holdoff must be non-negative\n  Got: {holdoff} samples")

    previous, current = x[:-1], x[1:]
    if slope == "rising":
        crossed = (previous < threshold) & (current >= threshold)
    else:
        crossed = (previous > threshold) & (current <= threshold)
    candidates = np.flatnonzero(crossed) + 1

    accepted: list[int] = []
    next_allowed = 0
    for index in candidates:
        if index >= next_allowed:
            accepted.append(int(index))
            next_allowed = int(index) + holdoff
    return np.asarray(accepted, dtype=np.int64)
