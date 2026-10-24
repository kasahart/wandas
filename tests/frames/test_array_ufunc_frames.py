"""Element-wise NumPy ufuncs on a frame return a lazy frame with lineage (issue #514)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.frames.channel import ChannelFrame
from wandas.frames.spectral import SpectralFrame
from wandas.pipeline import RecipePlan
from wandas.pipeline.compiler import RecipeExtractionError

_VALUES = np.array([[1.0, -4.0, 9.0], [-16.0, 25.0, -36.0]])
_ONES = np.ones((2, 3))


def _frame() -> ChannelFrame:
    return ChannelFrame.from_numpy(_VALUES, sampling_rate=3)


def _replay(result: ChannelFrame, *inputs: Any) -> ChannelFrame:
    plan = RecipePlan.from_frame(result, input_names=tuple(f"input_{i}" for i in range(len(inputs))))
    restored = RecipePlan.from_dict(json.loads(json.dumps(plan.to_dict())))
    return restored.apply({f"input_{i}": value for i, value in enumerate(inputs)})


@pytest.mark.filterwarnings("error::DeprecationWarning")
@pytest.mark.parametrize(
    ("apply", "expected", "operation_id"),
    [
        pytest.param(np.abs, np.abs(_VALUES), "wandas.numpy.ufunc", id="abs"),
        pytest.param(lambda cf: np.sqrt(np.abs(cf)), np.sqrt(np.abs(_VALUES)), "wandas.numpy.ufunc", id="sqrt"),
        pytest.param(lambda cf: np.maximum(cf, 0.0), np.maximum(_VALUES, 0.0), "wandas.numpy.ufunc", id="maximum"),
        pytest.param(lambda cf: np.maximum(0.0, cf), np.maximum(0.0, _VALUES), "wandas.numpy.ufunc", id="maximum-left"),
        pytest.param(lambda cf: np.hypot(cf, cf), np.hypot(_VALUES, _VALUES), "wandas.numpy.ufunc", id="frame-frame"),
        pytest.param(lambda cf: np.add(cf, 2.0), _VALUES + 2.0, "wandas.operator.add", id="add"),
        pytest.param(
            lambda cf: np.float64(2) - cf, 2.0 - _VALUES, "wandas.operator.reverse_subtract", id="scalar-left"
        ),
    ],
)
def test_numpy_ufunc_returns_a_lazy_frame_that_replays(
    apply: Callable[[Any], ChannelFrame], expected: np.ndarray, operation_id: str
) -> None:
    source = _frame()

    result = apply(source)

    assert type(result) is ChannelFrame
    assert isinstance(result._data, DaArray)
    np.testing.assert_array_equal(result.data, expected)
    assert result.lineage.operation is not None
    assert result.lineage.operation.operation_id == operation_id
    np.testing.assert_array_equal(_replay(result, source).data, expected)


@pytest.mark.parametrize(
    ("apply", "expected", "operation_id"),
    [
        pytest.param(lambda cf: _ONES + cf, _ONES + _VALUES, "wandas.operator.reverse_add", id="add"),
        pytest.param(lambda cf: _ONES - cf, _ONES - _VALUES, "wandas.operator.reverse_subtract", id="subtract"),
        pytest.param(lambda cf: _ONES / cf, _ONES / _VALUES, "wandas.operator.reverse_divide", id="divide"),
        pytest.param(lambda cf: np.arctan2(_ONES, cf), np.arctan2(_ONES, _VALUES), "wandas.numpy.ufunc", id="arctan2"),
        pytest.param(lambda cf: np.arctan2(cf, _ONES), np.arctan2(_VALUES, _ONES), "wandas.numpy.ufunc", id="right"),
    ],
)
def test_array_operands_keep_their_position_on_replay(
    apply: Callable[[ChannelFrame], ChannelFrame], expected: np.ndarray, operation_id: str
) -> None:
    source = _frame()

    result = apply(source)

    assert type(result) is ChannelFrame
    np.testing.assert_array_equal(result.data, expected)
    plan = RecipePlan.from_frame(result)
    assert plan.nodes[-1].operation == operation_id
    array_first = isinstance(result, ChannelFrame) and plan.inputs[0].kind == "array"
    inputs = (_ONES, source) if array_first else (source, _ONES)
    np.testing.assert_array_equal(_replay(result, *inputs).data, expected)


def test_ufunc_keeps_frame_type_and_channel_state() -> None:
    frame = _frame()
    spectrum = ChannelFrame.from_numpy(np.ones((2, 16)), sampling_rate=16).fft()

    magnitude = np.abs(spectrum)
    result = np.abs(frame)

    assert type(magnitude) is SpectralFrame
    np.testing.assert_allclose(magnitude.data, np.abs(spectrum.data))
    assert result.labels == ["absolute(ch0)", "absolute(ch1)"]
    assert [ch.unit for ch in result.channels] == [ch.unit for ch in frame.channels]
    assert result.sampling_rate == frame.sampling_rate


def test_ufunc_outside_numpy_returns_a_frame_but_cannot_be_saved() -> None:
    special = pytest.importorskip("scipy.special")

    result = special.expit(_frame())

    assert type(result) is ChannelFrame
    np.testing.assert_allclose(result.data, special.expit(_VALUES))
    with pytest.raises(RecipeExtractionError, match="not a NumPy ufunc"):
        RecipePlan.from_frame(result)


def test_reductions_comparisons_and_keyword_calls_keep_arrays() -> None:
    frame: Any = _frame()
    out = np.empty((2, 3))

    assert np.max(frame) == _VALUES.max()
    assert np.add.reduce(frame, axis=None) == _VALUES.sum()
    np.testing.assert_array_equal(np.greater(frame, 0.0), _VALUES > 0.0)
    np.testing.assert_array_equal(_ONES == frame, _VALUES == 1.0)
    np.testing.assert_array_equal(np.isfinite(frame), np.isfinite(_VALUES))
    np.testing.assert_array_equal(np.matmul(frame, np.ones((3, 1))), _VALUES @ np.ones((3, 1)))
    assert type(np.modf(frame)[0]) is np.ndarray
    assert np.add(frame, 1.0, out=out) is out
    np.testing.assert_array_equal(out, _VALUES + 1.0)
    assert type(np.sqrt(np.abs(frame), where=_VALUES > 0)) is np.ndarray


def test_a_frame_cannot_be_an_out_target() -> None:
    frame = _frame()
    add = cast(Any, np.add)

    with pytest.raises(TypeError, match="cannot be the out= target"):
        add(_VALUES, 1.0, out=(frame,))


def test_frames_without_arithmetic_reject_two_input_ufuncs() -> None:
    signal = ChannelFrame.from_numpy(np.random.default_rng(514).standard_normal((1, 4800)), sampling_rate=48000)
    bands = signal.noct_spectrum(fmin=100, fmax=1000)

    assert type(np.log10(bands)) is type(bands)
    with pytest.raises(TypeError):
        np.maximum(bands, 0.0)


def test_unsupported_operands_are_left_to_numpy() -> None:
    with pytest.raises(TypeError):
        np.maximum(_frame(), object())


def test_a_frame_subclass_on_the_right_keeps_operand_order() -> None:
    class _SubFrame(ChannelFrame):
        pass

    left = _frame()
    right = _SubFrame.from_numpy(_ONES * 2.0, sampling_rate=3)

    result = np.arctan2(left, right)

    assert type(result) is ChannelFrame
    np.testing.assert_array_equal(result.data, np.arctan2(_VALUES, _ONES * 2.0))


@pytest.mark.parametrize(
    ("params", "message"),
    [
        pytest.param([["ufunc", "no_such_ufunc"]], "Unknown NumPy ufunc", id="unknown"),
        pytest.param([["ufunc", "modf"]], "return one output", id="two-outputs"),
        pytest.param([["operand", 0.0], ["reverse", False], ["ufunc", "sqrt"]], "takes one input", id="operand"),
        pytest.param([["operand", 0.0], ["reverse", 1], ["ufunc", "maximum"]], "boolean", id="reverse"),
    ],
)
def test_recipe_rejects_invalid_ufunc_params(params: list[list[Any]], message: str) -> None:
    payload = RecipePlan.from_frame(np.sqrt(_frame())).to_dict()
    payload["nodes"][-1]["params"] = {"$type": "map", "entries": params}

    with pytest.raises(ValueError, match=message):
        RecipePlan.from_dict(payload)
