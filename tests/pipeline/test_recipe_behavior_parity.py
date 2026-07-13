from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipePlan


def _frame(*, sampling_rate: int = 16000, seconds: float = 0.05) -> ChannelFrame:
    samples = int(sampling_rate * seconds)
    time = np.arange(samples) / sampling_rate
    data = (0.25 + np.sin(2 * np.pi * 1000 * time)).reshape(1, -1)
    frame = ChannelFrame.from_numpy(data, sampling_rate=sampling_rate, label="source")
    frame.metadata["parity"] = True
    frame.source_time_offset = [0.25]
    return frame


def _assert_replay(source: ChannelFrame, processed: Any) -> Any:
    replayed = RecipePlan.from_frame(processed, input_names=("signal",)).apply({"signal": source})
    assert type(replayed) is type(processed)
    assert replayed.shape == processed.shape
    assert replayed.sampling_rate == processed.sampling_rate
    assert replayed.labels == processed.labels
    np.testing.assert_allclose(replayed.compute(), processed.compute())
    np.testing.assert_allclose(replayed.source_time_offset, processed.source_time_offset)
    return replayed


@pytest.mark.parametrize(
    "build",
    [
        lambda frame: frame.abs(),
        lambda frame: frame.power(exponent=3.0),
        lambda frame: frame.a_weighting(),
        lambda frame: frame.fade(fade_ms=5.0),
        lambda frame: frame.high_pass_filter(cutoff=100.0, order=2),
        lambda frame: frame.low_pass_filter(cutoff=2000.0, order=2),
        lambda frame: frame.band_pass_filter(low_cutoff=100.0, high_cutoff=2000.0, order=2),
    ],
)
def test_supported_unary_audio_operations_replay(build: Callable[[ChannelFrame], ChannelFrame]) -> None:
    source = _frame()
    processed = build(source)

    replayed = _assert_replay(source, processed)

    assert replayed.metadata == processed.metadata


@pytest.mark.parametrize("method", ["hpss_harmonic", "hpss_percussive"])
def test_hpss_operations_replay(monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    import wandas.processing.effects as effects

    monkeypatch.setattr(
        effects,
        "require_librosa_effects",
        lambda _feature: SimpleNamespace(harmonic=lambda data, **_: data, percussive=lambda data, **_: data),
    )
    source = _frame()
    processed = getattr(source, method)(kernel_size=(7, 7), margin=(1.0, 2.0), n_fft=64)

    _assert_replay(source, processed)


def _patch_psychoacoustic(monkeypatch: pytest.MonkeyPatch) -> None:
    import wandas.processing.psychoacoustic as psycho

    def loudness(signal: np.ndarray, _rate: float, *, field_type: str) -> tuple[np.ndarray, None, None, None]:
        del field_type
        return np.full(max(1, signal.shape[-1] // 96), 1.5), None, None, None

    def roughness(
        signal: np.ndarray, _rate: float, *, overlap: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, None]:
        del overlap
        count = max(1, signal.shape[-1] // 7200)
        return np.full(count, 0.2), np.ones((47, count)), np.linspace(0.5, 23.5, 47), None

    def sharpness(
        signal: np.ndarray,
        _rate: float,
        *,
        weighting: str,
        field_type: str,
        skip: int,
    ) -> tuple[np.ndarray, None]:
        del weighting, field_type, skip
        return np.full(max(1, signal.shape[-1] // 96), 0.4), None

    monkeypatch.setattr(psycho, "require_mosqito_sq_metric", lambda _feature, _name: object())
    monkeypatch.setattr(psycho, "loudness_zwtv_mosqito", loudness)
    monkeypatch.setattr(psycho, "roughness_dw_mosqito", roughness)
    monkeypatch.setattr(psycho, "sharpness_din_tv_mosqito", sharpness)
    psycho.RoughnessDwSpec._bark_axis_cache.clear()


@pytest.mark.parametrize(
    "build",
    [
        lambda frame: frame.loudness_zwtv(field_type="diffuse"),
        lambda frame: frame.roughness_dw(overlap=0.25),
        lambda frame: frame.sharpness_din(weighting="din", field_type="diffuse"),
    ],
)
def test_psychoacoustic_operations_replay(
    monkeypatch: pytest.MonkeyPatch, build: Callable[[ChannelFrame], ChannelFrame]
) -> None:
    _patch_psychoacoustic(monkeypatch)
    source = _frame(sampling_rate=48000, seconds=0.2)

    _assert_replay(source, build(source))


def test_roughness_typed_transition_replays(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_psychoacoustic(monkeypatch)
    source = _frame(sampling_rate=48000, seconds=0.2)
    processed = source.roughness_dw_spec(overlap=0.25)

    replayed = _assert_replay(source, processed)

    np.testing.assert_allclose(replayed.bark_axis, processed.bark_axis)
    assert replayed.overlap == processed.overlap


@pytest.mark.parametrize(
    "build",
    [
        lambda frame: frame.stft(n_fft=128, hop_length=32, win_length=128).istft(),
        lambda frame: frame.welch(n_fft=128, hop_length=32, win_length=128, average="mean"),
    ],
)
def test_stft_istft_and_welch_transitions_replay(build: Callable[[ChannelFrame], Any]) -> None:
    source = _frame()

    _assert_replay(source, build(source))


def _patch_noct(monkeypatch: pytest.MonkeyPatch) -> None:
    import wandas.frames.noct as noct_frame
    import wandas.processing.spectral as spectral

    def center(*, fmin: float, fmax: float, n: int, **_: Any) -> tuple[np.ndarray, np.ndarray]:
        count = max(1, int(np.ceil(np.log2(fmax / fmin) * n)))
        indices = np.arange(count, dtype=float)
        return indices, fmin * 2.0 ** (indices / n)

    def spectrum(*, sig: np.ndarray, fmin: float, fmax: float, n: int, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        _, frequencies = center(fmin=fmin, fmax=fmax, n=n, **kwargs)
        channels = 1 if sig.ndim == 1 else sig.shape[-1]
        values = np.ones((len(frequencies), channels))
        return (values[:, 0] if channels == 1 else values), frequencies

    def synthesis(
        *, spectrum: np.ndarray, fmin: float, fmax: float, n: int, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        _, frequencies = center(fmin=fmin, fmax=fmax, n=n, **kwargs)
        channels = 1 if spectrum.ndim == 1 else spectrum.shape[-1]
        return np.ones((len(frequencies), channels)), frequencies

    monkeypatch.setattr(spectral, "require_mosqito_center_freq", lambda _feature: center)
    monkeypatch.setattr(spectral, "_center_freq", center)
    monkeypatch.setattr(spectral, "noct_spectrum", spectrum)
    monkeypatch.setattr(spectral, "noct_synthesis", synthesis)
    monkeypatch.setattr(noct_frame, "_center_freq", center)


def test_noct_spectrum_and_synthesis_transitions_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_noct(monkeypatch)
    source = _frame(sampling_rate=48000)

    spectrum = source.noct_spectrum(fmin=125, fmax=8000, n=3, G=10, fr=1000)
    _assert_replay(source, spectrum)

    synthesis = source.fft(n_fft=1024).noct_synthesis(fmin=125, fmax=8000, n=3, G=10, fr=1000)
    _assert_replay(source, synthesis)


def test_add_channel_preserves_metadata_and_source_time_contract() -> None:
    base = _frame()
    added = _frame()
    added.source_time_offset = [2.5]
    processed_frame = base.add_channel(added, label="frame-added")
    replayed_frame = RecipePlan.from_frame(processed_frame, input_names=("base", "added")).apply(
        {"base": base, "added": added}
    )

    assert replayed_frame.metadata == processed_frame.metadata
    np.testing.assert_allclose(replayed_frame.source_time_offset, processed_frame.source_time_offset)

    raw = np.ones((1, base.n_samples))
    processed_raw = base.add_channel(raw, label="raw-added", source_time_offset=3.5)
    replayed_raw = RecipePlan.from_frame(processed_raw, input_names=("base", "raw")).apply({"base": base, "raw": raw})

    assert replayed_raw.metadata == processed_raw.metadata
    np.testing.assert_allclose(replayed_raw.source_time_offset, processed_raw.source_time_offset)
