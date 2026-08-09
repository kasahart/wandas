"""Public contracts for the synchronous Frame cache boundary."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import dask.array as da
import numpy as np
import pytest
from dask import delayed
from dask.callbacks import Callback

from tests.builtin_frame_cases import BUILTIN_FRAME_CASES
from wandas.core.base_frame import BaseFrame
from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipePlan


def _assert_state_value_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual == expected


@pytest.mark.parametrize(
    "frame_factory",
    [pytest.param(case.factory, id=case.id) for case in BUILTIN_FRAME_CASES],
)
def test_cache_preserves_every_builtin_frame_family(
    frame_factory: Callable[[], BaseFrame[Any]],
) -> None:
    frame = frame_factory().with_source_time_offset(0.25)
    expected_raw = frame._data.compute()
    expected_public = frame.data
    expected_state = frame._get_additional_init_kwargs()

    cached = frame.cache()

    assert cached is not frame
    assert type(cached) is type(frame)
    assert cached.previous is frame
    assert isinstance(cached._data, da.Array)
    assert cached._data is not frame._data
    assert cached._data.shape == frame._data.shape
    assert cached._data.dtype == frame._data.dtype
    np.testing.assert_array_equal(cached._data.compute(), expected_raw)
    np.testing.assert_array_equal(cached.data, expected_public)
    assert cached.label == frame.label
    assert cached.metadata == frame.metadata
    assert cached.channels.to_list() == frame.channels.to_list()
    assert cached._channel_ids == frame._channel_ids
    np.testing.assert_array_equal(cached.source_time_offset, frame.source_time_offset)
    assert cached._xr.dims == frame._xr.dims
    assert set(cached._xr.coords) == set(frame._xr.coords)
    for coordinate_name in frame._xr.coords:
        np.testing.assert_array_equal(
            cached._xr.coords[coordinate_name].values,
            frame._xr.coords[coordinate_name].values,
        )
    actual_state = cached._get_additional_init_kwargs()
    assert actual_state.keys() == expected_state.keys()
    for name, expected in expected_state.items():
        _assert_state_value_equal(actual_state[name], expected)
    assert cached.lineage is frame.lineage
    assert cached.operation_history == frame.operation_history


def test_cache_computes_source_once_and_reuses_it_for_materialization_and_operations() -> None:
    calls: list[str] = []
    values = np.linspace(0.25, 1.0, 64, dtype=np.float64).reshape(1, -1)

    @delayed
    def source() -> np.ndarray[Any, Any]:
        calls.append("source")
        return values

    lazy = da.from_delayed(source(), shape=values.shape, dtype=values.dtype)
    spectrogram = ChannelFrame(lazy, sampling_rate=64.0).stft(
        n_fft=16,
        hop_length=8,
        win_length=16,
        window="boxcar",
    )

    cached = spectrogram.cache()
    assert calls == ["source"]

    np.testing.assert_array_equal(cached.data, cached.data)
    np.testing.assert_array_equal(cached.dB, cached.dB)
    _ = cached.abs().data

    assert calls == ["source"]


def test_audio_operation_after_cache_is_lazy_and_does_not_rerun_source() -> None:
    calls: list[str] = []
    executed_tasks: list[Any] = []
    values = np.array([[1.0, 2.0, 4.0, 8.0]])

    @delayed
    def source() -> np.ndarray[Any, Any]:
        calls.append("source")
        return values

    lazy = da.from_delayed(source(), shape=values.shape, dtype=values.dtype)
    frame = ChannelFrame(lazy, sampling_rate=4.0).with_calibration([3.0])
    cached = frame.cache()
    cached_history = cached.operation_history

    with Callback(pretask=lambda key, _graph, _state: executed_tasks.append(key)):
        normalized = cached.normalize(norm=2.0)

    assert executed_tasks == []
    assert calls == ["source"]
    assert isinstance(normalized._data, da.Array)
    assert normalized.previous is cached
    assert cached.operation_history == cached_history
    assert len(normalized.operation_history) == len(cached_history) + 1
    assert normalized.operation_history[-1]["operation"] == "wandas.audio.normalize"
    assert normalized.channels[0].calibration.factor == 1.0

    expected = values[0] / np.linalg.norm(values[0], ord=2)
    np.testing.assert_allclose(normalized.data, expected)
    assert calls == ["source"]


def test_cache_keeps_raw_samples_so_calibration_is_not_applied_twice() -> None:
    raw = np.array([[1.0, 2.0, 4.0]])
    frame = ChannelFrame.from_numpy(raw, sampling_rate=8.0).with_calibration([3.0])

    cached = frame.cache()

    np.testing.assert_array_equal(cached._data.compute(), raw)
    np.testing.assert_array_equal(cached.data, np.array([3.0, 6.0, 12.0]))
    np.testing.assert_array_equal(cached.data, frame.data)
    assert cached.channels[0].calibration == frame.channels[0].calibration


def test_cache_owns_samples_and_materializations_cannot_mutate_it() -> None:
    caller_owned = np.array([[1.0, 2.0, 4.0]])

    @delayed
    def source() -> np.ndarray[Any, Any]:
        return caller_owned

    frame = ChannelFrame(
        da.from_delayed(source(), shape=caller_owned.shape, dtype=caller_owned.dtype),
        sampling_rate=8.0,
    )
    cached = frame.cache()

    caller_owned[:] = -1.0
    materialized = cached.data
    materialized[:] = 99.0

    np.testing.assert_array_equal(cached.data, np.array([1.0, 2.0, 4.0]))


def test_cache_rejects_a_non_numpy_compute_result(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = ChannelFrame.from_numpy(np.arange(4.0), sampling_rate=4.0)

    def return_list(_array: da.Array) -> list[float]:
        return [0.0, 1.0, 2.0, 3.0]

    monkeypatch.setattr(da.Array, "compute", return_list)

    with pytest.raises(ValueError, match="Computed result is not an np.ndarray"):
        frame.cache()


@pytest.mark.parametrize("error_type", [RuntimeError, MemoryError])
def test_cache_propagates_compute_failures_without_changing_source(
    error_type: type[Exception],
) -> None:
    @delayed
    def fail() -> np.ndarray[Any, Any]:
        raise error_type("cache failed")

    lazy = da.from_delayed(fail(), shape=(1, 4), dtype=np.float64)
    frame = ChannelFrame(lazy, sampling_rate=4.0, metadata={"nested": {"value": 1}})
    original_data = frame._data
    original_lineage = frame.lineage
    original_history = frame.operation_history
    original_metadata = frame.metadata

    with pytest.raises(error_type, match="cache failed"):
        frame.cache()

    assert frame._data is original_data
    assert frame.lineage is original_lineage
    assert frame.operation_history == original_history
    assert frame.metadata == original_metadata


def test_cache_does_not_add_lineage_history_or_recipe_nodes() -> None:
    source = ChannelFrame.from_numpy(np.arange(16.0), sampling_rate=16.0)
    processed = source.normalize(norm=2.0)
    before = RecipePlan.from_frame(processed, input_names=("signal",)).to_dict()

    cached = processed.cache()
    after = RecipePlan.from_frame(cached, input_names=("signal",)).to_dict()

    assert cached.lineage is processed.lineage
    assert cached.operation_history == processed.operation_history
    assert after == before
