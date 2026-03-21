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

from typing import Any

import numpy as np
from numpy import pi
from scipy.signal import bilinear_zpk, freqs, sosfilt, zpk2sos, zpk2tf

from wandas.utils.types import NDArrayReal


def ABC_weighting(curve: str = "A") -> tuple[NDArrayReal, NDArrayReal, float]:  # noqa: N802
    """
    Design of an analog weighting filter with A, B, or C curve.

    Returns zeros, poles, gain of the filter.

    Parameters
    ----------
    curve : str
        Weighting curve type: 'A', 'B', or 'C'.

    Returns
    -------
    z : ndarray
        Zeros of the analog filter.
    p : ndarray
        Poles of the analog filter.
    k : float
        Gain of the analog filter, normalized to 0 dB at 1 kHz.
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


def A_weighting(fs: float, output: str = "ba") -> Any:  # noqa: N802
    """
    Design of a digital A-weighting filter.

    Designs a digital A-weighting filter for sampling frequency `fs`.

    Warning: fs should normally be higher than 20 kHz. For example,
    fs = 48000 yields a class 1-compliant filter.

    Parameters
    ----------
    fs : float
        Sampling frequency
    output : {'ba', 'zpk', 'sos'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        second-order sections ('sos'). Default is 'ba'.

    Returns
    -------
    Depending on `output`:
        - 'ba': tuple of (b, a) numerator/denominator arrays
        - 'zpk': tuple of (z, p, k) zeros/poles/gain
        - 'sos': second-order sections array
    """
    return frequency_weighting(fs, curve="A", output=output)


def frequency_weighting(fs: float, curve: str = "A", output: str = "ba") -> Any:
    """
    Design a digital frequency-weighting filter.

    Parameters
    ----------
    fs : float
        Sampling frequency
    curve : {'A', 'B', 'C'}, optional
        Frequency weighting curve.
    output : {'ba', 'zpk', 'sos'}, optional
        Type of output: numerator/denominator, pole-zero-gain, or SOS.

    Returns
    -------
    Any
        Filter representation requested by ``output``.
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
    elif output in {"ba", "tf"}:
        return zpk2tf(z_d, p_d, k_d)
    elif output == "sos":
        return zpk2sos(z_d, p_d, k_d)
    else:
        raise ValueError(f"'{output}' is not a valid output form.")


def A_weight(signal: NDArrayReal, fs: float) -> NDArrayReal:  # noqa: N802
    """
    Return the given signal after passing through a digital A-weighting filter.

    Parameters
    ----------
    signal : array_like
        Input signal, with time as dimension
    fs : float
        Sampling frequency

    Returns
    -------
    NDArrayReal
        A-weighted signal
    """
    return frequency_weight(signal, fs, curve="A")


def frequency_weight(signal: NDArrayReal, fs: float, curve: str = "A") -> NDArrayReal:
    """
    Apply a digital frequency-weighting filter to a signal.

    Parameters
    ----------
    signal : array_like
        Input signal, with time as dimension.
    fs : float
        Sampling frequency.
    curve : {'A', 'B', 'C'}, optional
        Frequency weighting curve.

    Returns
    -------
    NDArrayReal
        Frequency-weighted signal.
    """
    sos = frequency_weighting(fs, curve=curve, output="sos")
    return np.asarray(sosfilt(sos, signal))
