from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from dask.array.core import Array as DaArray

import wandas.processing.custom  # noqa: F401
import wandas.processing.effects  # noqa: F401
import wandas.processing.filters  # noqa: F401
import wandas.processing.psychoacoustic as psychoacoustic_module
import wandas.processing.spectral  # noqa: F401
import wandas.processing.stats  # noqa: F401
import wandas.processing.temporal  # noqa: F401
from wandas.processing.base import _OPERATION_REGISTRY, AudioOperation
from wandas.processing.custom import CustomOperation
from wandas.processing.effects import AddWithSNR, Fade, HpssHarmonic, HpssPercussive, Normalize, RemoveDC
from wandas.processing.filters import AWeighting, BandPassFilter, HighPassFilter, LowPassFilter
from wandas.processing.psychoacoustic import (
    LoudnessZwst,
    LoudnessZwtv,
    RoughnessDw,
    RoughnessDwSpec,
    SharpnessDin,
    SharpnessDinSt,
)
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
from wandas.utils.dask_helpers import da_from_array

SR = 16000


@dataclass(frozen=True)
class OperationCase:
    name: str
    operation_factory: Callable[[], AudioOperation[Any, Any]]
    data: np.ndarray
    extra_inputs: tuple[np.ndarray, ...] = ()


def _as_dask(data: np.ndarray) -> DaArray:
    return da_from_array(data, chunks=(1, *(-1,) * (data.ndim - 1)))


def _wave(samples: int, *, channels: int = 2, dtype: np.dtype[Any] | type[Any] = np.float64) -> np.ndarray:
    t = np.arange(samples, dtype=np.float64) / SR
    rows = [np.sin(2 * np.pi * 440 * t)]
    if channels > 1:
        rows.append(0.5 * np.sin(2 * np.pi * 880 * t))
    return np.vstack(rows).astype(dtype)


def _complex_spectrum(freqs: int, *, frames: int | None = None) -> np.ndarray:
    if frames is None:
        return np.ones((2, freqs), dtype=np.complex128)
    return np.ones((2, freqs, frames), dtype=np.complex128)


def _custom_halve(x: np.ndarray) -> np.ndarray:
    return x[..., ::2]


def _halve_shape(input_shape: tuple[int, ...]) -> tuple[int, ...]:
    return (*input_shape[:-1], input_shape[-1] // 2)


def _psycho_signal(samples: int = 1000) -> np.ndarray:
    t = np.arange(samples, dtype=np.float64) / 1000.0
    return np.vstack(
        [
            0.05 * np.sin(2 * np.pi * 40 * t),
            0.04 * np.sin(2 * np.pi * 80 * t),
        ]
    )


def _patch_psychoacoustic_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        psychoacoustic_module._PsychoacousticOperation,
        "ensure_dependencies",
        lambda self: None,
    )

    def time_samples(ch: np.ndarray, sampling_rate: float) -> int:
        return int(ch.shape[-1] / (sampling_rate * 0.002))

    def roughness_samples(ch: np.ndarray, sampling_rate: float, overlap: float) -> int:
        window_samples = int(0.2 * sampling_rate)
        hop_samples = int(window_samples * (1 - overlap))
        if hop_samples > 0:
            return max(1, (ch.shape[-1] - window_samples) // hop_samples + 1)
        return 1

    def loudness_zwtv(ch: np.ndarray, sampling_rate: float, **_: Any) -> tuple[np.ndarray, None, None, None]:
        return np.linspace(0.0, 1.0, time_samples(ch, sampling_rate)), None, None, None

    def loudness_zwst(ch: np.ndarray, sampling_rate: float, **_: Any) -> tuple[float, None, None]:
        return float(np.mean(np.abs(ch)) + sampling_rate * 0.0), None, None

    def roughness_dw(
        ch: np.ndarray, sampling_rate: float, overlap: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, None]:
        n_times = roughness_samples(ch, sampling_rate, overlap)
        bark_axis = np.arange(47, dtype=np.float64)
        roughness = np.linspace(0.0, 1.0, n_times)
        specific = np.tile(roughness, (bark_axis.size, 1))
        return roughness, specific, bark_axis, None

    def sharpness_din_tv(ch: np.ndarray, sampling_rate: float, **_: Any) -> tuple[np.ndarray, None]:
        return np.linspace(0.0, 1.0, time_samples(ch, sampling_rate)), None

    def sharpness_din_st(ch: np.ndarray, sampling_rate: float, **_: Any) -> float:
        return float(np.mean(np.abs(ch)) + sampling_rate * 0.0)

    monkeypatch.setattr(psychoacoustic_module, "loudness_zwtv_mosqito", loudness_zwtv)
    monkeypatch.setattr(psychoacoustic_module, "loudness_zwst_mosqito", loudness_zwst)
    monkeypatch.setattr(psychoacoustic_module, "roughness_dw_mosqito", roughness_dw)
    monkeypatch.setattr(psychoacoustic_module, "sharpness_din_tv_mosqito", sharpness_din_tv)
    monkeypatch.setattr(psychoacoustic_module, "sharpness_din_st_mosqito", sharpness_din_st)


def _operation_cases() -> list[OperationCase]:
    real = _wave(256, dtype=np.float32)
    real64 = _wave(256, dtype=np.float64)
    stereo_long = _wave(4096, dtype=np.float64)
    psycho = _psycho_signal()

    return [
        OperationCase(
            "custom",
            lambda: CustomOperation(SR, func=_custom_halve, output_shape_func=_halve_shape),
            real64,
        ),
        OperationCase("hpss_harmonic", lambda: HpssHarmonic(SR), stereo_long),
        OperationCase("hpss_percussive", lambda: HpssPercussive(SR), stereo_long),
        OperationCase("normalize", lambda: Normalize(SR, norm=2, axis=-1), real),
        OperationCase("remove_dc", lambda: RemoveDC(SR), real.astype(np.int16)),
        OperationCase("add_with_snr", lambda: AddWithSNR(SR, snr=10), real.astype(np.int16), (real,)),
        OperationCase("fade", lambda: Fade(SR, fade_ms=0), real),
        OperationCase("highpass_filter", lambda: HighPassFilter(SR, cutoff=200.0), real),
        OperationCase("lowpass_filter", lambda: LowPassFilter(SR, cutoff=2000.0), real),
        OperationCase("bandpass_filter", lambda: BandPassFilter(SR, low_cutoff=200.0, high_cutoff=2000.0), real),
        OperationCase("a_weighting", lambda: AWeighting(SR), real),
        OperationCase("loudness_zwtv", lambda: LoudnessZwtv(1000, field_type="free"), psycho),
        OperationCase("loudness_zwst", lambda: LoudnessZwst(1000, field_type="free"), psycho),
        OperationCase("roughness_dw", lambda: RoughnessDw(1000, overlap=0.5), psycho),
        OperationCase("roughness_dw_spec", lambda: RoughnessDwSpec(1000, overlap=0.5), psycho),
        OperationCase("sharpness_din", lambda: SharpnessDin(1000, weighting="din", field_type="free"), psycho),
        OperationCase("sharpness_din_st", lambda: SharpnessDinSt(1000, weighting="din", field_type="free"), psycho),
        OperationCase("fft", lambda: FFT(SR, n_fft=64), real64),
        OperationCase("ifft", lambda: IFFT(SR, n_fft=64), _complex_spectrum(33)),
        OperationCase("stft", lambda: STFT(SR, n_fft=64, win_length=64, hop_length=16), real64),
        OperationCase(
            "istft",
            lambda: ISTFT(SR, n_fft=64, win_length=64, hop_length=16),
            _complex_spectrum(33, frames=10),
        ),
        OperationCase("welch", lambda: Welch(SR, n_fft=64, win_length=64, hop_length=32), real),
        OperationCase(
            "noct_spectrum",
            lambda: NOctSpectrum(48000, fmin=100, fmax=1000),
            np.ones((2, 48000), dtype=np.float64),
        ),
        OperationCase(
            "noct_synthesis",
            lambda: NOctSynthesis(48000, fmin=100, fmax=1000),
            np.ones((2, 513), dtype=np.float64),
        ),
        OperationCase("coherence", lambda: Coherence(SR, n_fft=64, win_length=64, hop_length=32), real),
        OperationCase("csd", lambda: CSD(SR, n_fft=64, win_length=64, hop_length=32), real),
        OperationCase("transfer_function", lambda: TransferFunction(SR, n_fft=64, win_length=64, hop_length=32), real),
        OperationCase("abs", lambda: ABS(SR), real),
        OperationCase("power", lambda: Power(SR, exponent=2.0), real),
        OperationCase("sum", lambda: Sum(SR), real),
        OperationCase("mean", lambda: Mean(SR), real),
        OperationCase("channel_difference", lambda: ChannelDifference(SR, other_channel=0), real),
        OperationCase("resampling", lambda: ReSampling(1000, target_sr=500), _wave(100, dtype=np.int16)),
        OperationCase("trim", lambda: Trim(1000, start=0.01, end=0.06), real64[:, :100]),
        OperationCase("fix_length", lambda: FixLength(SR, length=64), real64[:, :100]),
        OperationCase("rms_trend", lambda: RmsTrend(SR, frame_length=32, hop_length=16), real),
        OperationCase("sound_level", lambda: SoundLevel(SR, ref=1.0, freq_weighting="Z", time_weighting="Fast"), real),
    ]


OPERATION_CASES = _operation_cases()


@pytest.mark.parametrize("case", OPERATION_CASES, ids=[case.name for case in OPERATION_CASES])
def test_operation_lazy_metadata_matches_computed_result(case: OperationCase, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_psychoacoustic_backends(monkeypatch)
    operation = case.operation_factory()

    result_da = operation.process(_as_dask(case.data), *(_as_dask(input_data) for input_data in case.extra_inputs))
    result = result_da.compute()

    assert result_da.shape == result.shape
    assert result_da.dtype == result.dtype


def test_operation_lazy_metadata_contract_covers_every_registered_operation() -> None:
    covered_names = {case.name for case in OPERATION_CASES}
    concrete_registered_names = {
        name for name, operation_class in _OPERATION_REGISTRY.items() if not operation_class.__name__.startswith("_")
    }

    assert covered_names == concrete_registered_names
