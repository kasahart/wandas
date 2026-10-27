"""float32 frames stay float32 through operations that compute in float64."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame

_SR = 48_000


def _frame(dtype: type) -> ChannelFrame:
    time = np.arange(_SR // 2) / _SR
    data = np.stack([np.sin(2 * np.pi * 1_000 * time), 0.5 * np.sin(2 * np.pi * 250 * time)])
    return ChannelFrame.from_numpy(data.astype(dtype), _SR)


_CASES: dict[str, Callable[[ChannelFrame], Any]] = {
    "high_pass_filter": lambda frame: frame.high_pass_filter(cutoff=100),
    "low_pass_filter": lambda frame: frame.low_pass_filter(cutoff=5_000),
    "band_pass_filter": lambda frame: frame.band_pass_filter(low_cutoff=100, high_cutoff=5_000),
    "a_weighting": lambda frame: frame.a_weighting(),
    "rms_trend": lambda frame: frame.rms_trend(),
    "rms_trend_db": lambda frame: frame.rms_trend(dB=True),
    "fade": lambda frame: frame.fade(fade_ms=10),
    "fft": lambda frame: frame.fft(),
    "ifft": lambda frame: frame.fft().ifft(),
    "stft": lambda frame: frame.stft(),
    "istft": lambda frame: frame.stft().istft(),
    "noct_spectrum": lambda frame: frame.noct_spectrum(),
    "noct_synthesis": lambda frame: frame.fft().noct_synthesis(fmin=125, fmax=4_000),
    "cepstrum": lambda frame: frame.cepstrum(),
    "spectrogram_cepstrum": lambda frame: frame.stft().cepstrum(),
    "spectral_envelope": lambda frame: frame.cepstrum().lifter(0.002).to_spectral_envelope(),
    "loudness_zwtv": lambda frame: frame.loudness_zwtv(),
    "sharpness_din": lambda frame: frame.sharpness_din(),
    "roughness_dw": lambda frame: frame.roughness_dw(),
    "roughness_dw_spec": lambda frame: frame.roughness_dw_spec(),
}


def _single_precision(dtype: np.dtype[Any]) -> bool:
    return dtype in (np.dtype(np.float32), np.dtype(np.complex64))


@pytest.mark.parametrize("case", list(_CASES))
def test_float32_input_keeps_single_precision(case: str) -> None:
    result = _CASES[case](_frame(np.float32))

    assert _single_precision(result._data.dtype)
    assert result.data.dtype.itemsize == result._data.dtype.itemsize
    assert result.data.dtype.kind == result._data.dtype.kind


@pytest.mark.parametrize("case", ["high_pass_filter", "a_weighting", "rms_trend_db", "fft", "noct_spectrum"])
def test_float64_input_keeps_double_precision(case: str) -> None:
    result = _CASES[case](_frame(np.float64))

    assert result.data.dtype in (np.dtype(np.float64), np.dtype(np.complex128))


@pytest.mark.parametrize("case", ["high_pass_filter", "a_weighting", "rms_trend_db", "fft", "noct_spectrum"])
def test_float32_results_match_float64_to_single_precision(case: str) -> None:
    single = _CASES[case](_frame(np.float32)).data
    double = _CASES[case](_frame(np.float64)).data

    scale = float(np.max(np.abs(double)))
    np.testing.assert_allclose(single, double, rtol=1e-4, atol=1e-4 * scale)


def test_float32_spectral_levels_are_float32() -> None:
    spectrum = _frame(np.float32).fft()

    assert spectrum.dB.dtype == np.float32
    assert spectrum.dBA.dtype == np.float32
