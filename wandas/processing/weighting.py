"""
A-weighting filter design and application.

Vendored from waveform-analysis (https://github.com/endolith/waveform-analysis)
Commit: baece1e4db3fa2324090086efe1d74cce314e65b

Original license:

    The MIT License (MIT)

    Copyright (c) 2016 endolith@gmail.com

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import math
from typing import Any

import numpy as np
from numpy import pi
from numpy.typing import ArrayLike
from scipy.signal import bilinear_zpk, freqs, sosfilt, zpk2sos, zpk2tf

from wandas.utils.types import NDArrayReal
from wandas.utils.util import DB_FLOOR


def _reference_level_db(data: ArrayLike, reference: ArrayLike) -> NDArrayReal:
    """Compute the canonical amplitude level for a broadcastable reference.

    Magnitude is evaluated in float64 (or complex128 before taking absolute
    value), then converted in the log domain. The log-domain subtraction
    avoids avoidable overflow and underflow from forming ``magnitude / ref``
    directly. Ratios at or below ``DB_FLOOR`` return -240 dB.
    """
    reference_array = np.asarray(reference, dtype=float)
    if np.any(~np.isfinite(reference_array)) or np.any(reference_array <= 0.0):
        raise ValueError("Amplitude level references must be positive and finite")
    source = np.asarray(data)
    calculation_dtype = np.result_type(source.dtype, reference_array.dtype, np.float64)
    data_array = np.asarray(source, dtype=calculation_dtype)
    magnitude = np.abs(data_array)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_ratio = np.log10(magnitude) - np.log10(reference_array)
    minimum_log_ratio = math.log10(DB_FLOOR)
    result: NDArrayReal = np.asarray(20.0 * np.maximum(log_ratio, minimum_log_ratio), dtype=np.float64)
    return result


def a_weighting_db(frequencies: NDArrayReal, min_db: float | None = -45.0) -> NDArrayReal:
    """Evaluate the implemented analog A-weighting formula in dB.

    The curve is normalized near 0 dB at 1 kHz. This helper evaluates a
    frequency-response formula; it does not validate a digital implementation
    or certify IEC/JIS instrument conformance.

    Args:
        frequencies: Frequencies in Hz. The returned array has the same shape.
        min_db: Lower bound for finite curve values. ``None`` disables the
            bound. Defaults to ``-45.0`` dB.

    Returns:
        A-weighting values in dB with the same shape as ``frequencies``.
    """
    f = np.asarray(frequencies, dtype=float)
    f2 = f**2
    ra = (12194.0**2 * f2**2) / ((f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194.0**2))
    with np.errstate(divide="ignore", invalid="ignore"):
        weights: NDArrayReal = 20.0 * np.log10(ra) + 2.0
    if min_db is not None:
        weights = np.where(np.isfinite(weights), np.maximum(weights, min_db), min_db)
    return weights


def ABC_weighting(curve: str = "A") -> tuple[NDArrayReal, NDArrayReal, float]:
    """Design an analog A, B, or C frequency-weighting filter.

    Args:
        curve: Weighting curve type: ``"A"``, ``"B"``, or ``"C"``.

    Returns:
        tuple[NDArrayReal, NDArrayReal, float]: A tuple ``(z, p, k)`` containing
            zero locations, pole locations, and gain normalized to 0 dB at 1 kHz.

    Raises:
        ValueError: If ``curve`` is not ``"A"``, ``"B"``, or ``"C"``.
    """
    allowed_curves = {"A", "B", "C"}
    if curve not in allowed_curves:
        raise ValueError(f"Curve type not understood: {curve!r}. Expected one of {sorted(allowed_curves)}.")

    # ANSI S1.4-1983 C weighting
    #    2 poles on the real axis at "20.6 Hz" HPF
    #    2 poles on the real axis at "12.2 kHz" LPF
    #    -3 dB down points at "10^1.5 (or 31.62) Hz"
    #                         "10^3.9 (or 7943) Hz"
    #
    # IEC 61672 specifies "10^1.5 Hz" and "10^3.9 Hz" points and formulas for
    # derivation.

    z: list[float] = [0, 0]
    p: list[float] = [
        -2 * pi * 20.598997057568145,
        -2 * pi * 20.598997057568145,
        -2 * pi * 12194.21714799801,
        -2 * pi * 12194.21714799801,
    ]
    k: float = 1

    if curve == "A":
        # ANSI S1.4-1983 A weighting =
        #    Same as C weighting +
        #    2 poles on real axis at "107.7 and 737.9 Hz"
        p.append(-2 * pi * 107.65264864304628)
        p.append(-2 * pi * 737.8622307362899)
        z.append(0)
        z.append(0)

    elif curve == "B":
        # ANSI S1.4-1983 B weighting
        #    Same as C weighting +
        #    1 pole on real axis at "10^2.2 (or 158.5) Hz"
        p.append(-2 * pi * 10**2.2)  # exact
        z.append(0)

    # Normalize to 0 dB at 1 kHz for all curves
    b, a = zpk2tf(z, p, k)
    k /= abs(freqs(b, a, [2 * pi * 1000])[1][0])

    return np.array(z), np.array(p), k


def A_weighting(fs: float, output: str = "ba") -> Any:
    """Design the digital A-frequency-weighting filter.

    The bilinear transform introduces sampling-rate-dependent response error;
    no sampling rate by itself establishes complete instrument conformance.

    Args:
        fs: Sampling frequency in Hz.
        output: Filter representation: ``"ba"`` for numerator/denominator,
            ``"zpk"`` for zero/pole/gain, or ``"sos"`` for second-order
            sections. Defaults to ``"ba"``.

    Returns:
        The filter representation selected by ``output``.

    Raises:
        ValueError: If ``output`` is not a supported representation.
    """
    return frequency_weighting(fs, curve="A", output=output)


def frequency_weighting(fs: float, curve: str = "A", output: str = "ba") -> Any:
    """Design a digital A, B, or C frequency-weighting filter.

    Args:
        fs: Sampling frequency in Hz.
        curve: Frequency-weighting curve: ``"A"``, ``"B"``, or ``"C"``.
        output: Filter representation: ``"ba"``, ``"zpk"``, or ``"sos"``.

    Returns:
        The filter representation selected by ``output``.

    Raises:
        ValueError: If ``curve`` or ``output`` is not supported.
    """
    normalized_curve = str(curve).upper()
    allowed_curves = {"A", "B", "C"}
    if normalized_curve not in allowed_curves:
        raise ValueError(f"Curve type not understood: {curve!r}. Expected one of {sorted(allowed_curves)}.")

    z, p, k = ABC_weighting(normalized_curve)

    # Use the bilinear transformation to get the digital filter.
    z_d, p_d, k_d = bilinear_zpk(z, p, k, fs)

    if output == "zpk":
        return z_d, p_d, k_d
    if output in {"ba", "tf"}:
        return zpk2tf(z_d, p_d, k_d)
    if output == "sos":
        return zpk2sos(z_d, p_d, k_d)
    raise ValueError(f"'{output}' is not a valid output form.")


def A_weight(signal: NDArrayReal, fs: float) -> NDArrayReal:
    """Apply the digital A-weighting filter to a signal.

    Args:
        signal: Input samples with time on the last dimension.
        fs: Sampling frequency in Hz.

    Returns:
        A-weighted samples with the same shape as ``signal``.
    """
    return frequency_weight(signal, fs, curve="A")


def frequency_weight(signal: NDArrayReal, fs: float, curve: str = "A") -> NDArrayReal:
    """Apply a digital frequency-weighting filter to a signal.

    Args:
        signal: Input samples with time on the last dimension.
        fs: Sampling frequency in Hz.
        curve: Frequency-weighting curve: ``"A"``, ``"B"``, or ``"C"``.

    Returns:
        Frequency-weighted samples with the same shape as ``signal``.

    Raises:
        ValueError: If ``curve`` is not supported.
    """
    sos = frequency_weighting(fs, curve=curve, output="sos")
    return np.asarray(sosfilt(sos, signal))
