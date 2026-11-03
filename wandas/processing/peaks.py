"""Peak picking on one spectrum level curve."""

from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks

from wandas.utils.types import NDArrayReal

PeakInterpolation = Literal["none", "parabolic"]


def find_spectral_peaks(
    level_db: NDArrayReal,
    freqs: NDArrayReal,
    min_prominence_db: float = 10.0,
    fmin: float | None = None,
    fmax: float | None = None,
    max_peaks: int | None = None,
    interpolation: PeakInterpolation = "none",
) -> dict[str, npt.NDArray[np.float64]]:
    """Return the peaks of one spectrum level curve.

    Peaks are found with ``scipy.signal.find_peaks(level_db,
    prominence=min_prominence_db)`` on the whole curve, so the prominence of a
    peak near ``fmin`` or ``fmax`` still sees the bins outside that range. Peaks
    outside ``[fmin, fmax]`` are then dropped, and ``max_peaks`` keeps the
    highest levels. As in SciPy, the first and last bins (0 Hz and Nyquist for
    a one-sided spectrum) are never peaks, because they have only one
    neighbour.

    With ``interpolation="parabolic"``, a parabola through the level of the
    peak bin and its two neighbours refines the frequency and the level. A
    peak on a flat top keeps its bin values.

    Args:
        level_db: 1-D level curve in dB.
        freqs: Frequency of each bin in Hz, evenly spaced, same length as
            ``level_db``.
        min_prominence_db: Minimum prominence of a peak in dB.
        fmin: Lowest peak frequency to keep in Hz. ``None`` keeps all.
        fmax: Highest peak frequency to keep in Hz. ``None`` keeps all.
        max_peaks: Number of highest peaks to keep; peaks of equal level keep
            the lower frequency first. ``None`` keeps all.
        interpolation: ``"none"`` or ``"parabolic"``.

    Returns:
        ``frequency``, ``level_db`` and ``prominence_db`` arrays of the kept
        peaks, sorted by frequency.

    Raises:
        ValueError: If the inputs are not matching 1-D arrays,
            ``interpolation`` is unknown, or ``max_peaks`` is below 1.
    """
    if level_db.ndim != 1 or freqs.shape != level_db.shape:
        raise ValueError(
            "Peak picking needs a 1-D level curve and one frequency per bin\n"
            f"  Got: level shape {level_db.shape}, frequency shape {freqs.shape}"
        )
    if interpolation not in ("none", "parabolic"):
        raise ValueError(f"Unknown peak interpolation\n  Got: {interpolation!r}\n  Expected: 'none' or 'parabolic'")
    if max_peaks is not None and (
        isinstance(max_peaks, bool) or not isinstance(max_peaks, int | np.integer) or max_peaks < 1
    ):
        raise ValueError(f"max_peaks must be an integer of at least 1, or None\n  Got: {max_peaks!r}")

    indices, properties = find_peaks(level_db, prominence=min_prominence_db)
    prominences = np.asarray(properties["prominences"], dtype=np.float64)
    frequency = freqs[indices].astype(np.float64)
    level = level_db[indices].astype(np.float64)
    if interpolation == "parabolic" and indices.size:
        bin_width = float(freqs[1] - freqs[0])
        for k, index in enumerate(indices):
            # find_peaks never returns the first or last bin, so both neighbours exist.
            left, centre, right = (float(v) for v in level_db[index - 1 : index + 2])
            if left == centre or right == centre:
                continue  # flat top: no single vertex to refine
            offset = 0.5 * (left - right) / (left - 2.0 * centre + right)
            frequency[k] += offset * bin_width
            level[k] = centre - 0.25 * (left - right) * offset

    # The range and max_peaks apply to the returned (possibly refined) values.
    keep = np.ones(indices.shape, dtype=bool)
    if fmin is not None:
        keep &= frequency >= fmin
    if fmax is not None:
        keep &= frequency <= fmax
    frequency, level, prominences = frequency[keep], level[keep], prominences[keep]
    if max_peaks is not None:
        highest = np.argsort(-level, kind="stable")[:max_peaks]
        frequency, level, prominences = frequency[highest], level[highest], prominences[highest]

    order = np.argsort(frequency, kind="stable")
    return {
        "frequency": frequency[order],
        "level_db": level[order],
        "prominence_db": prominences[order],
    }
