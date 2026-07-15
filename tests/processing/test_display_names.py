"""Display-name contracts shared by built-in audio operations."""

from typing import Any

import pytest

from wandas.processing.base import AudioOperation
from wandas.processing.effects import (
    AddWithSNR,
    Fade,
    HpssHarmonic,
    HpssPercussive,
    Normalize,
    RemoveDC,
)
from wandas.processing.filters import AWeighting, BandPassFilter, HighPassFilter, LowPassFilter
from wandas.processing.spectral import (
    CSD,
    FFT,
    IFFT,
    ISTFT,
    STFT,
    Coherence,
    NOctSpectrum,
    NOctSynthesis,
    TransferFunction,
    Welch,
)
from wandas.processing.stats import ABS, ChannelDifference, Mean, Power, Sum
from wandas.processing.temporal import FixLength, ReSampling, RmsTrend, SoundLevel, Trim

_SAMPLE_RATE = 44_100
_PAIRWISE_SPECTRAL_PARAMS: dict[str, Any] = {
    "n_fft": 2_048,
    "hop_length": 512,
    "win_length": 2_048,
    "window": "hann",
    "detrend": "constant",
}

_DISPLAY_NAME_CASES = (
    pytest.param(LowPassFilter(_SAMPLE_RATE, cutoff=1_000), "lpf", id="low-pass"),
    pytest.param(HighPassFilter(_SAMPLE_RATE, cutoff=1_000), "hpf", id="high-pass"),
    pytest.param(BandPassFilter(_SAMPLE_RATE, low_cutoff=500, high_cutoff=2_000), "bpf", id="band-pass"),
    pytest.param(AWeighting(_SAMPLE_RATE), "Aw", id="a-weighting"),
    pytest.param(FFT(_SAMPLE_RATE), "FFT", id="fft"),
    pytest.param(IFFT(_SAMPLE_RATE), "iFFT", id="ifft"),
    pytest.param(STFT(_SAMPLE_RATE), "STFT", id="stft"),
    pytest.param(ISTFT(_SAMPLE_RATE), "iSTFT", id="istft"),
    pytest.param(Welch(_SAMPLE_RATE), "Welch", id="welch"),
    pytest.param(NOctSpectrum(_SAMPLE_RATE, fmin=20, fmax=20_000), "Oct", id="n-octave-spectrum"),
    pytest.param(NOctSynthesis(_SAMPLE_RATE, fmin=20, fmax=20_000), "Octs", id="n-octave-synthesis"),
    pytest.param(Coherence(_SAMPLE_RATE, **_PAIRWISE_SPECTRAL_PARAMS), "Coh", id="coherence"),
    pytest.param(
        CSD(_SAMPLE_RATE, **_PAIRWISE_SPECTRAL_PARAMS, scaling="spectrum", average="mean"),
        "CSD",
        id="cross-spectral-density",
    ),
    pytest.param(TransferFunction(_SAMPLE_RATE, **_PAIRWISE_SPECTRAL_PARAMS), "H", id="transfer-function"),
    pytest.param(HpssHarmonic(_SAMPLE_RATE), "Hrm", id="hpss-harmonic"),
    pytest.param(HpssPercussive(_SAMPLE_RATE), "Prc", id="hpss-percussive"),
    pytest.param(Normalize(_SAMPLE_RATE), "norm", id="normalize"),
    pytest.param(RemoveDC(_SAMPLE_RATE), "dcRM", id="remove-dc"),
    pytest.param(AddWithSNR(_SAMPLE_RATE, snr=10.0), "+SNR", id="add-with-snr"),
    pytest.param(Fade(_SAMPLE_RATE, fade_ms=50), "fade", id="fade"),
    pytest.param(ReSampling(_SAMPLE_RATE, target_sr=16_000), "rs", id="resampling"),
    pytest.param(Trim(_SAMPLE_RATE, start=0.0, end=1.0), "trim", id="trim"),
    pytest.param(FixLength(_SAMPLE_RATE, length=_SAMPLE_RATE), "fix", id="fix-length"),
    pytest.param(RmsTrend(_SAMPLE_RATE), "RMS", id="rms-trend"),
    pytest.param(
        SoundLevel(_SAMPLE_RATE, freq_weighting="A", time_weighting="Fast", dB=True),
        "LAF",
        id="sound-level-db",
    ),
    pytest.param(
        SoundLevel(_SAMPLE_RATE, freq_weighting="A", time_weighting="Fast", dB=False),
        "AFRMS",
        id="sound-level-linear",
    ),
    pytest.param(ABS(_SAMPLE_RATE), "abs", id="absolute"),
    pytest.param(Power(_SAMPLE_RATE, exponent=2.0), "pow", id="power"),
    pytest.param(Sum(_SAMPLE_RATE), "sum", id="sum"),
    pytest.param(Mean(_SAMPLE_RATE), "mean", id="mean"),
    pytest.param(ChannelDifference(_SAMPLE_RATE, other_channel=0), "diff", id="channel-difference"),
)


@pytest.mark.parametrize(("operation", "expected"), _DISPLAY_NAME_CASES)
def test_operation_display_name(operation: AudioOperation[Any, Any], expected: str) -> None:
    """Every built-in operation exposes its stable user-facing label."""
    assert operation.get_display_name() == expected
