"""
Harmonic/Percussive Source Separation (HPSS).

The ``_softmask`` and ``_hpss`` functions are vendored from librosa
(https://github.com/librosa/librosa) and adapted for use without the
librosa package so that wandas works in Pyodide/browser environments.

Original authors: librosa development team
License: ISC License
Copyright (c) 2013--2023, librosa development team.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import get_window

from wandas.utils.types import NDArrayReal


def _softmask(
    X: np.ndarray,
    X_ref: np.ndarray,
    *,
    power: float = 1,
    split_zeros: bool = False,
) -> np.ndarray:
    """Compute a soft-mask: ``M = X**power / (X**power + X_ref**power)``.

    Vendored from ``librosa.util.softmask`` (ISC License).
    """
    if X.shape != X_ref.shape:
        raise ValueError(f"Shape mismatch: {X.shape}!={X_ref.shape}")
    if np.any(X < 0) or np.any(X_ref < 0):
        raise ValueError("X and X_ref must be non-negative")
    if power <= 0:
        raise ValueError("power must be strictly positive")

    dtype = X.dtype
    if not np.issubdtype(dtype, np.floating):
        dtype = np.float32

    Z = np.maximum(X, X_ref).astype(dtype)
    bad_idx = Z < np.finfo(dtype).tiny
    Z[bad_idx] = 1

    if np.isfinite(power):
        mask = (X / Z) ** power
        ref_mask = (X_ref / Z) ** power
        good_idx = ~bad_idx
        mask[good_idx] /= mask[good_idx] + ref_mask[good_idx]
        mask[bad_idx] = 0.5 if split_zeros else 0.0
    else:
        mask = (X > X_ref).astype(dtype)

    return mask


def _hpss(
    S: np.ndarray,
    *,
    kernel_size: int | tuple[int, int] | list[int] = 31,
    power: float = 2.0,
    mask: bool = False,
    margin: float | tuple[float, float] | list[float] = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Median-filtering Harmonic/Percussive Source Separation on a spectrogram.

    Vendored from ``librosa.decompose.hpss`` (ISC License).
    """
    if np.iscomplexobj(S):
        magnitude: np.ndarray = np.abs(S)
        phase: np.ndarray | float = np.exp(1j * np.angle(S))
    else:
        magnitude = S
        phase = 1

    if isinstance(kernel_size, (tuple, list)):
        win_harm, win_perc = int(kernel_size[0]), int(kernel_size[1])
    else:
        win_harm = win_perc = int(kernel_size)

    if isinstance(margin, (tuple, list)):
        margin_harm, margin_perc = float(margin[0]), float(margin[1])
    else:
        margin_harm = margin_perc = float(margin)

    if margin_harm < 1 or margin_perc < 1:
        raise ValueError("Margins must be >= 1.0. A typical range is between 1 and 10.")

    # Harmonic: filter along time axis (last axis)
    harm_shape = [1] * magnitude.ndim
    harm_shape[-1] = win_harm

    # Percussive: filter along frequency axis (second-to-last axis)
    perc_shape = [1] * magnitude.ndim
    perc_shape[-2] = win_perc

    harm = np.empty_like(magnitude)
    harm[:] = median_filter(magnitude, size=harm_shape, mode="reflect")

    perc = np.empty_like(magnitude)
    perc[:] = median_filter(magnitude, size=perc_shape, mode="reflect")

    split_zeros = margin_harm == 1 and margin_perc == 1

    mask_harm = _softmask(harm, perc * margin_harm, power=power, split_zeros=split_zeros)
    mask_perc = _softmask(perc, harm * margin_perc, power=power, split_zeros=split_zeros)

    if mask:
        return mask_harm, mask_perc

    return (magnitude * mask_harm) * phase, (magnitude * mask_perc) * phase


def _stft(
    y: NDArrayReal,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: str,
    center: bool,
    pad_mode: str,
) -> np.ndarray:
    """Compute STFT. Compatible with librosa default conventions (center padding).

    Input shape: (..., n_samples)
    Output shape: (..., n_freqs, n_frames)  where n_freqs = n_fft // 2 + 1
    """
    win: NDArrayReal = np.asarray(get_window(window, win_length, fftbins=True))
    if win_length < n_fft:
        left = (n_fft - win_length) // 2
        right = n_fft - win_length - left
        win = np.pad(win, (left, right))

    if center:
        pad_width = [(0, 0)] * (y.ndim - 1) + [(n_fft // 2, n_fft // 2)]
        y = np.pad(y, pad_width, mode=pad_mode)  # ty: ignore[no-matching-overload]

    n_frames = 1 + (y.shape[-1] - n_fft) // hop_length
    # Build frame view: shape (..., n_fft, n_frames)
    shape = y.shape[:-1] + (n_fft, n_frames)
    strides = y.strides[:-1] + (y.strides[-1], y.strides[-1] * hop_length)
    frames = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)

    # Apply window and FFT along the n_fft axis (second-to-last)
    windowed = frames * win[:, np.newaxis]
    stft_matrix: np.ndarray = np.fft.rfft(windowed, axis=-2)
    return stft_matrix


def _istft(
    stft_matrix: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: str,
    center: bool,
    length: int,
) -> NDArrayReal:
    """Compute inverse STFT via overlap-add. Inverts ``_stft``.

    Input shape: (..., n_freqs, n_frames)
    Output shape: (..., n_samples)
    """
    win: NDArrayReal = np.asarray(get_window(window, win_length, fftbins=True))
    if win_length < n_fft:
        left = (n_fft - win_length) // 2
        right = n_fft - win_length - left
        win = np.pad(win, (left, right))

    # IFFT: (..., n_fft, n_frames)
    frames = np.fft.irfft(stft_matrix, n=n_fft, axis=-2)
    n_frames = frames.shape[-1]

    out_len = n_fft + hop_length * (n_frames - 1)
    batch_shape = stft_matrix.shape[:-2]
    out = np.zeros(batch_shape + (out_len,), dtype=frames.real.dtype)
    win_sum = np.zeros(out_len, dtype=frames.real.dtype)

    for i in range(n_frames):
        start = i * hop_length
        out[..., start : start + n_fft] += frames[..., :, i].real * win
        win_sum[start : start + n_fft] += win * win

    # Normalize by sum-of-squared window (OLA normalisation)
    tiny = np.finfo(out.dtype).tiny
    win_sum = np.where(win_sum < tiny, 1.0, win_sum)
    out /= win_sum

    if center:
        out = out[..., n_fft // 2 : n_fft // 2 + length]
    else:
        out = out[..., :length]

    result: NDArrayReal = np.asarray(out)
    return result


def harmonic(
    y: NDArrayReal,
    *,
    kernel_size: int | tuple[int, int] | list[int] = 31,
    power: float = 2.0,
    mask: bool = False,
    margin: float | tuple[float, float] | list[float] = 1.0,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "constant",
) -> NDArrayReal:
    """Extract harmonic component from an audio signal.

    Equivalent to ``librosa.effects.harmonic``.
    Uses scipy for STFT/ISTFT; HPSS logic vendored from librosa (ISC License).

    Parameters
    ----------
    y : NDArrayReal
        Audio time series, shape (..., n_samples).
    kernel_size : int or tuple[int, int], default=31
        Kernel size for median filtering. If tuple, (harmonic_size, percussive_size).
    power : float, default=2.0
        Exponent for soft-masking.
    mask : bool, default=False
        If True, return soft masks instead of separated signals.
    margin : float or tuple[float, float], default=1.0
        Separation margin. Values > 1 increase separation at the cost of artifacts.
    n_fft : int, default=2048
        FFT size.
    hop_length : int or None, default=None
        Hop length. Defaults to n_fft // 4.
    win_length : int or None, default=None
        Window length. Defaults to n_fft.
    window : str, default="hann"
        Window type.
    center : bool, default=True
        If True, pad signal to center frames.
    pad_mode : str, default="constant"
        Padding mode for centering.

    Returns
    -------
    NDArrayReal
        Harmonic component of the audio signal.
    """
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    stft_m = _stft(y, n_fft, hop_length, win_length, window, center, pad_mode)
    harm_stft, _ = _hpss(stft_m, kernel_size=kernel_size, power=power, mask=mask, margin=margin)
    result = _istft(harm_stft, n_fft, hop_length, win_length, window, center, length=y.shape[-1])
    return result.astype(y.dtype)


def percussive(
    y: NDArrayReal,
    *,
    kernel_size: int | tuple[int, int] | list[int] = 31,
    power: float = 2.0,
    mask: bool = False,
    margin: float | tuple[float, float] | list[float] = 1.0,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "constant",
) -> NDArrayReal:
    """Extract percussive component from an audio signal.

    Equivalent to ``librosa.effects.percussive``.
    Uses scipy for STFT/ISTFT; HPSS logic vendored from librosa (ISC License).

    Parameters
    ----------
    y : NDArrayReal
        Audio time series, shape (..., n_samples).
    kernel_size : int or tuple[int, int], default=31
        Kernel size for median filtering. If tuple, (harmonic_size, percussive_size).
    power : float, default=2.0
        Exponent for soft-masking.
    mask : bool, default=False
        If True, return soft masks instead of separated signals.
    margin : float or tuple[float, float], default=1.0
        Separation margin. Values > 1 increase separation at the cost of artifacts.
    n_fft : int, default=2048
        FFT size.
    hop_length : int or None, default=None
        Hop length. Defaults to n_fft // 4.
    win_length : int or None, default=None
        Window length. Defaults to n_fft.
    window : str, default="hann"
        Window type.
    center : bool, default=True
        If True, pad signal to center frames.
    pad_mode : str, default="constant"
        Padding mode for centering.

    Returns
    -------
    NDArrayReal
        Percussive component of the audio signal.
    """
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    stft_m = _stft(y, n_fft, hop_length, win_length, window, center, pad_mode)
    _, perc_stft = _hpss(stft_m, kernel_size=kernel_size, power=power, mask=mask, margin=margin)
    result = _istft(perc_stft, n_fft, hop_length, win_length, window, center, length=y.shape[-1])
    return result.astype(y.dtype)
