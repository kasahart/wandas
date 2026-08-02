"""Independent SciPy-domain fixtures shared by ISTFT contract tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import get_window

_SAMPLING_RATE = 8_000.0


@dataclass(frozen=True)
class _OracleCase:
    """Small, invertible ShortTimeFFT configuration used by the oracle."""

    n_fft: int
    win_length: int
    hop_length: int
    window: str
    n_frames: int = 5
    channels: int = 1


_CASES = (
    _OracleCase(n_fft=16, win_length=16, hop_length=8, window="boxcar"),
    _OracleCase(n_fft=16, win_length=12, hop_length=5, window="hann"),
    _OracleCase(n_fft=15, win_length=15, hop_length=7, window="hann"),
    _OracleCase(n_fft=15, win_length=11, hop_length=4, window="boxcar"),
)


def _make_independent_oracle(
    case: _OracleCase,
) -> tuple[ShortTimeFFT, np.ndarray, np.ndarray]:
    """Build a SciPy-domain spectrum and its independent Wandas input form.

    ``domain_spectrogram`` is the direct one-sided complex value consumed by
    ``ShortTimeFFT.istft``.  It deliberately contains DC, interior bins, and an
    endpoint bin.  Channel-dependent gain and phase make channel mixing visible.

    The final array is converted to Wandas' stored peak-amplitude convention by
    the explicit one-sided rule: for an even FFT only ``1:-1`` is doubled, while
    for an odd FFT every positive-frequency bin ``1:`` is doubled.
    """

    scipy_sft = ShortTimeFFT(
        win=np.asarray(get_window(case.window, case.win_length), dtype=np.float64),
        hop=case.hop_length,
        fs=_SAMPLING_RATE,
        fft_mode="onesided",
        mfft=case.n_fft,
        scale_to="magnitude",
    )
    if not scipy_sft.invertible:
        raise AssertionError(f"Oracle configuration is not invertible: {case!r}")

    n_frequency_bins = case.n_fft // 2 + 1
    channel_index = np.arange(case.channels, dtype=np.float64)[:, None, None]
    frequency_index = np.arange(n_frequency_bins, dtype=np.float64)[None, :, None]
    time_index = np.arange(case.n_frames, dtype=np.float64)[None, None, :]

    channel_gain = 1.0 + 0.31 * channel_index
    amplitude = channel_gain * (0.35 + 0.11 * frequency_index) * (1.0 + 0.07 * time_index)
    phase = 0.17 * frequency_index + 0.31 * time_index + 0.43 * channel_index
    domain_spectrogram = np.asarray(amplitude * np.exp(1j * phase), dtype=np.complex128)

    # Real endpoint values are valid for a real-valued inverse and ensure that
    # the even Nyquist endpoint is distinguishable from an odd final bin.
    frame_values = np.arange(case.n_frames, dtype=np.float64)
    domain_spectrogram[:, 0, :] = channel_gain[:, 0, 0, None] * (0.95 + 0.13 * frame_values)[None, :]
    if case.n_fft % 2 == 0:
        domain_spectrogram[:, -1, :] = channel_gain[:, 0, 0, None] * (0.65 + 0.09 * frame_values)[None, :]

    normalized_wandas = domain_spectrogram.copy()
    if case.n_fft % 2 == 0:
        normalized_wandas[:, 1:-1, :] *= 2.0
    else:
        normalized_wandas[:, 1:, :] *= 2.0

    return scipy_sft, domain_spectrogram, normalized_wandas
