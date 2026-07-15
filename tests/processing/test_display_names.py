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
    pytest.param(LowPassFilter, {"cutoff": 1_000}, "lpf", id="low-pass"),
    pytest.param(HighPassFilter, {"cutoff": 1_000}, "hpf", id="high-pass"),
    pytest.param(
        BandPassFilter,
        {"low_cutoff": 500, "high_cutoff": 2_000},
        "bpf",
        id="band-pass",
    ),
    pytest.param(AWeighting, {}, "Aw", id="a-weighting"),
    pytest.param(FFT, {}, "FFT", id="fft"),
    pytest.param(IFFT, {}, "iFFT", id="ifft"),
    pytest.param(STFT, {}, "STFT", id="stft"),
    pytest.param(ISTFT, {}, "iSTFT", id="istft"),
    pytest.param(Welch, {}, "Welch", id="welch"),
    pytest.param(NOctSpectrum, {"fmin": 20, "fmax": 20_000}, "Oct", id="n-octave-spectrum"),
    pytest.param(NOctSynthesis, {"fmin": 20, "fmax": 20_000}, "Octs", id="n-octave-synthesis"),
    pytest.param(Coherence, _PAIRWISE_SPECTRAL_PARAMS, "Coh", id="coherence"),
    pytest.param(
        CSD,
        {**_PAIRWISE_SPECTRAL_PARAMS, "scaling": "spectrum", "average": "mean"},
        "CSD",
        id="cross-spectral-density",
    ),
    pytest.param(TransferFunction, _PAIRWISE_SPECTRAL_PARAMS, "H", id="transfer-function"),
    pytest.param(HpssHarmonic, {}, "Hrm", id="hpss-harmonic"),
    pytest.param(HpssPercussive, {}, "Prc", id="hpss-percussive"),
    pytest.param(Normalize, {}, "norm", id="normalize"),
    pytest.param(RemoveDC, {}, "dcRM", id="remove-dc"),
    pytest.param(AddWithSNR, {"snr": 10.0}, "+SNR", id="add-with-snr"),
    pytest.param(Fade, {"fade_ms": 50}, "fade", id="fade"),
    pytest.param(ReSampling, {"target_sr": 16_000}, "rs", id="resampling"),
    pytest.param(Trim, {"start": 0.0, "end": 1.0}, "trim", id="trim"),
    pytest.param(FixLength, {"length": _SAMPLE_RATE}, "fix", id="fix-length"),
    pytest.param(RmsTrend, {}, "RMS", id="rms-trend"),
    pytest.param(
        SoundLevel,
        {"freq_weighting": "A", "time_weighting": "Fast", "dB": True},
        "LAF",
        id="sound-level-db",
    ),
    pytest.param(
        SoundLevel,
        {"freq_weighting": "A", "time_weighting": "Fast", "dB": False},
        "AFRMS",
        id="sound-level-linear",
    ),
    pytest.param(ABS, {}, "abs", id="absolute"),
    pytest.param(Power, {"exponent": 2.0}, "pow", id="power"),
    pytest.param(Sum, {}, "sum", id="sum"),
    pytest.param(Mean, {}, "mean", id="mean"),
    pytest.param(ChannelDifference, {"other_channel": 0}, "diff", id="channel-difference"),
)


@pytest.mark.parametrize(("operation_class", "parameters", "expected"), _DISPLAY_NAME_CASES)
def test_operation_display_name(
    operation_class: type[AudioOperation[Any, Any]],
    parameters: dict[str, Any],
    expected: str,
) -> None:
    """Every built-in operation exposes its stable user-facing label."""
    operation = operation_class(_SAMPLE_RATE, **parameters)

    assert operation.get_display_name() == expected
