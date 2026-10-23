"""NumPy ufunc calls that compute a frame eagerly must warn before their return type changes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame

_VALUES = np.array([[1.0, -4.0, 9.0], [-16.0, 25.0, -36.0]])


def _frame() -> ChannelFrame:
    return ChannelFrame.from_numpy(_VALUES, sampling_rate=3)


@pytest.mark.parametrize(
    ("apply", "expected"),
    [
        pytest.param(np.abs, np.abs(_VALUES), id="abs"),
        pytest.param(np.sqrt, np.sqrt(np.abs(_VALUES)), id="sqrt"),
        pytest.param(lambda cf: np.ones((2, 3)) + cf, _VALUES + 1, id="ndarray-left-add"),
        pytest.param(lambda cf: np.maximum(cf, 0.0), np.maximum(_VALUES, 0.0), id="maximum"),
        pytest.param(np.fix, np.fix(_VALUES), id="numpy-python-function"),
    ],
)
def test_eager_numpy_ufunc_warns_and_keeps_ndarray_result(
    apply: Callable[[Any], np.ndarray], expected: np.ndarray
) -> None:
    frame: Any = _frame().abs() if apply is np.sqrt else _frame()

    with pytest.warns(DeprecationWarning, match="A future release will return a frame") as record:
        result = apply(frame)

    assert type(result) is np.ndarray
    np.testing.assert_array_equal(result, expected)
    assert all(warning.filename == __file__ for warning in record)


@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_lazy_paths_do_not_warn() -> None:
    frame = _frame()

    assert isinstance(np.float64(2) * frame, ChannelFrame)
    assert isinstance(frame + np.ones((2, 3)), ChannelFrame)
    assert isinstance(frame.abs(), ChannelFrame)
    np.testing.assert_array_equal(np.abs(np.asarray(frame)), np.abs(_VALUES))


@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_reductions_and_comparisons_keep_arrays_without_warning() -> None:
    frame: Any = _frame()

    assert np.max(frame) == _VALUES.max()
    assert np.add.reduce(frame, axis=None) == _VALUES.sum()
    np.testing.assert_array_equal(np.greater(frame, 0.0), _VALUES > 0.0)
    np.testing.assert_array_equal(np.ones((2, 3)) == frame, _VALUES == 1.0)
    np.testing.assert_array_equal(np.isfinite(frame), np.isfinite(_VALUES))
    np.testing.assert_array_equal(np.matmul(frame, np.ones((3, 1))), _VALUES @ np.ones((3, 1)))
    np.testing.assert_array_equal(frame @ np.ones((3, 1)), _VALUES @ np.ones((3, 1)))
