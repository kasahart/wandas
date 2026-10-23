"""Integer ChannelFrame arithmetic must not wrap around."""

from __future__ import annotations

from collections.abc import Callable

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


def test_floating_frame_arithmetic_keeps_its_dtype() -> None:
    frame = _frame(_SAMPLES.astype(np.float32))

    assert frame.power(2)._data.dtype == np.float32
    assert (frame * frame)._data.dtype == np.float32
    assert (frame * 3)._data.dtype == np.float32
