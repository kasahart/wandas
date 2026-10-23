"""Band levels from noct_synthesis agree with time-domain noct_spectrum."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame
from wandas.frames.spectral import SpectralFrame

pytest.importorskip("mosqito")

_FS = 48000
_FMIN = 100.0
_FMAX = 8000.0
_TOLERANCE_DB = 0.3


def _level_difference_db(frame: ChannelFrame, to_spectrum: Callable[[ChannelFrame], SpectralFrame]) -> np.ndarray:
    synthesized = to_spectrum(frame).noct_synthesis(fmin=_FMIN, fmax=_FMAX).data
    filtered = frame.noct_spectrum(fmin=_FMIN, fmax=_FMAX).data
    return 20 * np.log10(np.asarray(synthesized) / np.asarray(filtered))


@pytest.mark.parametrize(
    "to_spectrum",
    [
        pytest.param(lambda frame: frame.fft(), id="fft-hann"),
        pytest.param(lambda frame: frame.fft(window="blackman"), id="fft-blackman"),
        pytest.param(lambda frame: frame.fft(window="boxcar"), id="fft-boxcar"),
        pytest.param(lambda frame: frame.welch(n_fft=4096), id="welch-hann"),
        pytest.param(lambda frame: frame.welch(n_fft=4096, window="blackman"), id="welch-blackman"),
    ],
)
def test_sine_band_level_matches_noct_spectrum(to_spectrum: Callable[[ChannelFrame], SpectralFrame]) -> None:
    """A 1 kHz tone lands in the 1 kHz band at the level the time-domain filters give."""
    t = np.arange(_FS) / _FS
    frame = ChannelFrame.from_numpy(np.sin(2 * np.pi * 1000.0 * t)[np.newaxis, :], _FS)
    freqs = frame.noct_spectrum(fmin=_FMIN, fmax=_FMAX).freqs

    difference = _level_difference_db(frame, to_spectrum)

    assert abs(difference[int(np.argmin(np.abs(freqs - 1000.0)))]) < _TOLERANCE_DB


def test_white_noise_band_levels_match_noct_spectrum_with_welch_input() -> None:
    """Averaged white-noise band levels agree in the median band."""
    noise = np.random.default_rng(0).standard_normal(_FS)
    frame = ChannelFrame.from_numpy(noise[np.newaxis, :], _FS)

    difference = _level_difference_db(frame, lambda source: source.welch(n_fft=4096))

    assert abs(np.median(difference)) < _TOLERANCE_DB


def _pink_noise(duration_s: float, level_db: float) -> np.ndarray:
    n_samples = int(duration_s * _FS)
    spectrum = np.fft.rfft(np.random.default_rng(0).standard_normal(n_samples))
    freqs = np.fft.rfftfreq(n_samples, 1 / _FS)
    spectrum[0] = 0
    spectrum[1:] /= np.sqrt(freqs[1:])
    noise = np.fft.irfft(spectrum, n_samples)
    return noise * 2e-5 * 10 ** (level_db / 20) / np.sqrt(np.mean(noise**2))


@pytest.mark.parametrize("n", [1, 3])
def test_pink_noise_band_levels_match_mosqito_reference_case(n: int) -> None:
    """Mirror MoSQITo's own ``test_noct_synthesis_technical`` with Wandas frames.

    MoSQITo checks a 10 s, 48 kHz, 60 dB pink noise: an unwindowed full-length FFT passed to
    ``noct_synthesis`` must agree with ``noct_spectrum`` within 0.3 dB in every band from
    24 Hz to 12.6 kHz, for octave and third-octave bands.
    """
    noise = _pink_noise(duration_s=10.0, level_db=60.0)
    frame = ChannelFrame.from_numpy(noise[np.newaxis, :], _FS)

    synthesized = frame.fft(n_fft=noise.size, window="boxcar").noct_synthesis(fmin=24, fmax=12600, n=n).data
    filtered = frame.noct_spectrum(fmin=24, fmax=12600, n=n).data

    np.testing.assert_allclose(
        20 * np.log10(np.asarray(synthesized) / 2e-5),
        20 * np.log10(np.asarray(filtered) / 2e-5),
        atol=_TOLERANCE_DB,
    )
