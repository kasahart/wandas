"""Graph construction contracts for channel-independent operations."""

from __future__ import annotations

from typing import Any

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.frames.channel import ChannelFrame
from wandas.processing.base import AudioOperation, _try_build_channelwise_graph, create_operation

_RNG = np.random.default_rng(505)
_SAMPLES = _RNG.normal(size=(3, 4_096))
_FS = 8_000


def _frame(channels: int = 3, *, chunks: Any = (1, -1)) -> ChannelFrame:
    values = np.tile(_SAMPLES, (channels // 3 + 1, 1))[:channels]
    return ChannelFrame(da.from_array(values, chunks=chunks), sampling_rate=_FS)


def _key_count(frame: ChannelFrame) -> int:
    return len(dict(frame._data.__dask_graph__()))


def _graphs(operation: AudioOperation[Any, Any], data: DaArray) -> tuple[DaArray, DaArray]:
    output_shape = operation.calculate_output_shape(data.shape)
    output_dtype = operation.calculate_output_dtype(data.dtype)
    channelwise = _try_build_channelwise_graph(
        operation, data, (), output_shape=output_shape, output_dtype=output_dtype
    )
    assert channelwise is not None
    whole = operation._build_whole_frame_graph(data, (), output_shape=output_shape, output_dtype=output_dtype)
    return channelwise, whole


@pytest.mark.parametrize("channels", [2, 8])
def test_chained_stages_grow_the_graph_linearly(channels: int) -> None:
    frame = _frame(channels)
    counts = [_key_count(frame)]
    for stage in range(6):
        frame = frame.high_pass_filter(cutoff=50 + stage)
        counts.append(_key_count(frame))

    increments = np.diff(counts)
    assert set(increments[1:]) == {increments[1]}
    assert increments[1] == channels


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda cf: cf.high_pass_filter(cutoff=200), id="high-pass"),
        pytest.param(lambda cf: cf.a_weighting(), id="a-weighting"),
        pytest.param(lambda cf: cf.resampling(target_sr=4_000), id="resampling"),
        pytest.param(lambda cf: cf.noct_spectrum(fmin=100, fmax=2_000, n=3), id="noct-spectrum"),
        pytest.param(lambda cf: cf.hpss_harmonic(), id="hpss-harmonic"),
    ],
)
def test_frame_methods_match_whole_frame_execution(apply: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    channelwise = apply(_frame()).data

    monkeypatch.setattr("wandas.processing.base._try_build_channelwise_graph", lambda *args, **kwargs: None)
    monkeypatch.setattr("wandas.processing.effects._try_build_channelwise_graph", lambda *args, **kwargs: None)
    whole = apply(_frame()).data

    np.testing.assert_array_equal(channelwise, whole)


def test_normalize_on_integer_input_matches_whole_frame_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    values = (_SAMPLES * 1_000).astype(np.int16)

    def normalized() -> np.ndarray:
        return ChannelFrame(da.from_array(values, chunks=(1, -1)), sampling_rate=_FS).normalize().data

    channelwise = normalized()
    monkeypatch.setattr("wandas.processing.effects._try_build_channelwise_graph", lambda *args, **kwargs: None)

    np.testing.assert_array_equal(channelwise, normalized())


@pytest.mark.parametrize(
    ("name", "params"),
    [
        pytest.param("fft", {"n_fft": 1_024, "window": "hann"}, id="fft"),
        pytest.param("stft", {"n_fft": 512, "hop_length": 128}, id="stft"),
        pytest.param("rms_trend", {"frame_length": 512, "hop_length": 256}, id="rms-trend"),
    ],
)
def test_shape_changing_operations_match_whole_frame_execution(name: str, params: dict[str, Any]) -> None:
    operation = create_operation(name, _FS, **params)
    channelwise, whole = _graphs(operation, da.from_array(_SAMPLES, chunks=(1, -1)))

    assert channelwise.shape == whole.shape
    assert channelwise.chunks[0] == (1, 1, 1)
    np.testing.assert_array_equal(channelwise.compute(), whole.compute())


def test_graph_construction_does_not_call_the_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, ...]] = []
    operation = create_operation("highpass_filter", _FS, cutoff=200)
    kernel = operation._process

    def recording_kernel(x: np.ndarray) -> np.ndarray:
        calls.append(x.shape)
        return kernel(x)

    monkeypatch.setattr(operation, "_process", recording_kernel)
    result = operation.process(da.from_array(_SAMPLES, chunks=(1, -1)))

    assert calls == []
    result.compute()
    assert sorted(calls) == [(1, 4_096)] * 3


def test_time_chunked_input_reaches_the_kernel_as_complete_channels(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, ...]] = []
    operation = create_operation("highpass_filter", _FS, cutoff=200)
    kernel = operation._process

    def recording_kernel(x: np.ndarray) -> np.ndarray:
        calls.append(x.shape)
        return kernel(x)

    monkeypatch.setattr(operation, "_process", recording_kernel)
    result = operation.process(da.from_array(_SAMPLES, chunks=(3, 1_000)))

    expected = create_operation("highpass_filter", _FS, cutoff=200).process(da.from_array(_SAMPLES))
    np.testing.assert_array_equal(result.compute(), expected.compute())
    assert sorted(calls) == [(1, 4_096)] * 3


def test_task_names_follow_operation_purity() -> None:
    data = da.from_array(_SAMPLES, chunks=(1, -1))
    pure = create_operation("highpass_filter", _FS, cutoff=200)
    impure = create_operation("highpass_filter", _FS, cutoff=200)
    impure.pure = False

    assert pure.process(data).name == create_operation("highpass_filter", _FS, cutoff=200).process(data).name
    assert impure.process(data).name != impure.process(data).name
