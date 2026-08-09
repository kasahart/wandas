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
from wandas.core.metadata import ChannelCalibration
from wandas.frames.channel import ChannelFrame


def _assert_state_value_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual == expected


@pytest.mark.parametrize(
    "frame_factory",
    [pytest.param(case.factory, id=case.id) for case in BUILTIN_FRAME_CASES],
)
def test_astype_preserves_every_builtin_frame_family_and_state(
    frame_factory: Callable[[], BaseFrame[Any]],
) -> None:
    frame = frame_factory().with_source_time_offset(0.25)
    target = "complex64" if frame._data.dtype.kind == "c" else "float32"
    expected_raw = frame._data.compute().astype(target)
    expected_state = frame._get_additional_init_kwargs()

    converted = frame.astype(target)

    assert converted is not frame
    assert type(converted) is type(frame)
    assert converted.previous is frame
    assert isinstance(converted._data, da.Array)
    assert converted._data.dtype == np.dtype(target)
    assert converted._data.shape == frame._data.shape
    np.testing.assert_array_equal(converted._data.compute(), expected_raw)
    assert converted.label == frame.label
    assert converted.metadata == frame.metadata
    assert converted.channels.to_list() == frame.channels.to_list()
    assert converted._channel_ids == frame._channel_ids
    np.testing.assert_array_equal(converted.source_time_offset, frame.source_time_offset)
    assert converted._xr.dims == frame._xr.dims
    assert set(converted._xr.coords) == set(frame._xr.coords)
    for coordinate_name in frame._xr.coords:
        np.testing.assert_array_equal(
            converted._xr.coords[coordinate_name].values,
            frame._xr.coords[coordinate_name].values,
        )
    actual_state = converted._get_additional_init_kwargs()
    assert actual_state.keys() == expected_state.keys()
    for name, expected in expected_state.items():
        _assert_state_value_equal(actual_state[name], expected)


def test_astype_is_lazy_immutable_and_records_one_normalized_lineage_entry() -> None:
    calls: list[str] = []
    values = np.array([[1.125, 2.25, 4.5]], dtype=np.float64)

    @delayed
    def source() -> np.ndarray[Any, Any]:
        calls.append("source")
        return values

    lazy = da.from_delayed(source(), shape=values.shape, dtype=values.dtype)
    frame = ChannelFrame(
        lazy,
        sampling_rate=8.0,
        label="source",
        metadata={"nested": {"value": 1}},
        channel_metadata=[
            {
                "label": "mic",
                "calibration": ChannelCalibration(factor=3.0, unit="Pa"),
            }
        ],
        source_time_offset=0.5,
    )
    original_history = frame.operation_history
    executed: list[Any] = []

    with Callback(pretask=lambda key, _graph, _state: executed.append(key)):
        converted = frame.astype(np.float32)

    assert calls == []
    assert executed == []
    assert converted._data.dtype == np.dtype(np.float32)
    assert frame._data.dtype == np.dtype(np.float64)
    assert frame.operation_history == original_history
    assert len(converted.operation_history) == len(original_history) + 1
    assert converted.operation_history[-1] == {
        "operation": "wandas.frame.astype",
        "version": 1,
        "params": {"dtype": "float32"},
    }
    assert converted.channels[0].calibration == frame.channels[0].calibration
    np.testing.assert_allclose(converted._data.compute(), values.astype(np.float32))
    public_values = converted.data
    assert public_values.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(public_values, values[0].astype(np.float32) * 3.0)
    assert calls == ["source", "source"]


@pytest.mark.parametrize(
    ("values", "target", "expected"),
    [
        (
            np.array([[1.123456789, -2.987654321]], dtype=np.float64),
            "float32",
            np.array([[1.123456789, -2.987654321]], dtype=np.float32),
        ),
        (
            np.array([[1.123456789 + 2.987654321j]], dtype=np.complex128),
            "complex64",
            np.array([[1.123456789 + 2.987654321j]], dtype=np.complex64),
        ),
    ],
)
def test_astype_converts_values_without_changing_input(
    values: np.ndarray[Any, Any],
    target: str,
    expected: np.ndarray[Any, Any],
) -> None:
    frame = ChannelFrame.from_numpy(values.copy(), sampling_rate=8.0)

    converted = frame.astype(target)

    np.testing.assert_array_equal(converted._data.compute(), expected)
    np.testing.assert_array_equal(frame._data.compute(), values)


@pytest.mark.parametrize(
    ("values", "target", "message"),
    [
        (np.ones((1, 4), dtype=np.float64), "complex64", "real or integer input"),
        (np.ones((1, 4), dtype=np.complex128), "float32", "Use 'complex64'"),
        (np.ones((1, 4), dtype=np.float64), "float16", "float32, float64"),
        (np.ones((1, 4), dtype=np.float64), "int32", "float32, float64"),
    ],
)
def test_astype_rejects_invalid_target_synchronously(
    values: np.ndarray[Any, Any],
    target: str,
    message: str,
) -> None:
    frame = ChannelFrame.from_numpy(values, sampling_rate=8.0)

    with pytest.raises(ValueError, match=message):
        frame.astype(target)


def test_astype_then_cache_computes_once_and_halves_cached_raw_tensor_bytes() -> None:
    calls: list[str] = []
    values = np.linspace(-1.0, 1.0, 4096, dtype=np.float64).reshape(1, -1)

    @delayed
    def source() -> np.ndarray[Any, Any]:
        calls.append("source")
        return values

    lazy = da.from_delayed(source(), shape=values.shape, dtype=values.dtype)
    frame = ChannelFrame(lazy, sampling_rate=4096.0)

    cached = frame.astype("float32").cache()

    assert calls == ["source"]
    assert cached._data.dtype == np.dtype(np.float32)
    cached_raw = cached._data.compute()
    assert calls == ["source"]
    assert cached_raw.nbytes == values.nbytes // 2
    np.testing.assert_allclose(cached_raw, values.astype(np.float32))
    np.testing.assert_array_equal(cached.data, cached.data)
    assert calls == ["source"]
