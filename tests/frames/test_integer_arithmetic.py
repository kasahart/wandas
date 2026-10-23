"""Integer ChannelFrame arithmetic must not wrap around."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.frames.channel import ChannelFrame

_SAMPLES = np.array([[29500, -30000, 100, -32768]], dtype=np.int16)
_EXACT = _SAMPLES.astype(np.float64)


def _frame(values: np.ndarray = _SAMPLES) -> ChannelFrame:
    return ChannelFrame.from_numpy(values, sampling_rate=8)


@pytest.mark.parametrize(
    ("apply", "expected"),
    [
        pytest.param(lambda cf: cf.power(2), _EXACT**2, id="power"),
        pytest.param(lambda cf: cf.abs(), np.abs(_EXACT), id="abs"),
        pytest.param(lambda cf: cf * cf, _EXACT * _EXACT, id="frame-mul"),
        pytest.param(lambda cf: cf + cf, _EXACT + _EXACT, id="frame-add"),
        pytest.param(lambda cf: cf - cf * -1, _EXACT + _EXACT, id="frame-sub"),
        pytest.param(lambda cf: cf**2, _EXACT**2, id="scalar-pow"),
        pytest.param(lambda cf: cf * 3, _EXACT * 3, id="scalar-mul"),
        pytest.param(lambda cf: 3 * cf, 3 * _EXACT, id="reverse-scalar-mul"),
        pytest.param(lambda cf: np.int16(3) * cf, 3 * _EXACT, id="numpy-scalar-mul"),
        pytest.param(lambda cf: 2 - cf, 2 - _EXACT, id="reverse-scalar-sub"),
        pytest.param(lambda cf: cf + np.full((1, 4), 30000, dtype=np.int16), _EXACT + 30000, id="array-add"),
    ],
)
def test_integer_frame_arithmetic_is_exact_in_float64(
    apply: Callable[[ChannelFrame], ChannelFrame], expected: np.ndarray
) -> None:
    result = apply(_frame())

    assert isinstance(result._data, DaArray)
    assert result._data.dtype == np.float64
    np.testing.assert_array_equal(result.data, expected[0])


def test_integer_channel_difference_is_exact_in_float64() -> None:
    frame = _frame(np.vstack([_SAMPLES, np.array([[1, 2, 3, 4]], dtype=np.int16)]))

    result = frame.channel_difference(1)

    assert isinstance(result._data, DaArray)
    assert result._data.dtype == np.float64
    np.testing.assert_array_equal(result.data, np.vstack([_EXACT - [1, 2, 3, 4], np.zeros((1, 4))]))


@pytest.mark.filterwarnings("ignore:NumPy ufunc:DeprecationWarning")
@pytest.mark.parametrize(
    ("apply", "expected"),
    [
        pytest.param(lambda cf: np.full((1, 4), 30000, dtype=np.int16) + cf, _EXACT + 30000, id="array-left-add"),
        pytest.param(np.square, _EXACT**2, id="square"),
        pytest.param(np.abs, np.abs(_EXACT), id="numpy-abs"),
        pytest.param(lambda cf: np.square(cf, dtype=None), _EXACT**2, id="square-default-dtype"),
    ],
)
def test_numpy_calls_on_integer_frame_compute_in_float64(
    apply: Callable[[ChannelFrame], np.ndarray], expected: np.ndarray
) -> None:
    result = apply(_frame())

    assert result.dtype == np.float64
    np.testing.assert_array_equal(np.asarray(result).reshape(expected.shape), expected)


@pytest.mark.filterwarnings("ignore:NumPy ufunc:DeprecationWarning")
def test_numpy_calls_that_cannot_wrap_keep_the_integer_dtype() -> None:
    frame = _frame()
    large = _frame(np.array([[2**62 + 1, 3]], dtype=np.int64))
    out = np.empty(4, dtype=np.int16)

    np.testing.assert_array_equal(np.bitwise_and(frame, 1), _SAMPLES[0] & 1)
    np.testing.assert_array_equal(np.right_shift(frame, 1), _SAMPLES[0] >> 1)
    np.testing.assert_array_equal(np.equal(large, 2**62), [False, False])
    np.testing.assert_array_equal(np.greater(large, 2**62), [True, False])
    np.add(frame, 1, out=out)
    np.testing.assert_array_equal(out, _SAMPLES[0] + np.int16(1))
    assert np.add(frame, 1, dtype=np.int16).dtype == np.int16
    assert np.add(frame, 1, signature=(np.int32, np.int32, np.int32)).dtype == np.int32


def test_list_and_unsupported_operands_follow_numpy() -> None:
    # Lists and strings are outside the annotated operand types.
    frame = cast(Any, _frame())

    result = frame + [30000, 30000, 30000, 30000]

    assert result._data.dtype == np.float64
    np.testing.assert_array_equal(result.data, _EXACT[0] + 30000)
    with pytest.raises(TypeError):
        _ = frame + "a"
    with pytest.raises(ValueError, match="inhomogeneous"):
        _ = frame + [[1], [1, 2]]


@pytest.mark.filterwarnings("ignore:NumPy ufunc:DeprecationWarning")
def test_floating_operands_keep_numpy_promotion() -> None:
    frame = _frame(_SAMPLES.astype(np.float32))
    integer_frame = _frame()

    assert frame.power(2)._data.dtype == np.float32
    assert (frame * frame)._data.dtype == np.float32
    assert (frame * 3)._data.dtype == np.float32
    assert (frame + integer_frame)._data.dtype == np.float32
    assert (integer_frame + frame)._data.dtype == np.float32
    assert (frame + _SAMPLES)._data.dtype == np.float32
    assert (integer_frame + _SAMPLES.astype(np.float32))._data.dtype == np.float32
    assert (_SAMPLES.astype(np.float32) + integer_frame).dtype == np.float32
