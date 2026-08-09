from __future__ import annotations

from typing import Any

import dask.array as da
import numpy as np
import pytest
from dask.callbacks import Callback

from wandas.processing import Astype, create_operation, get_operation


@pytest.mark.parametrize(
    ("input_dtype", "target", "expected_dtype"),
    [
        (np.dtype(np.int16), "float32", np.dtype(np.float32)),
        (np.dtype(np.float64), np.float32, np.dtype(np.float32)),
        (np.dtype(np.float32), np.dtype("float64"), np.dtype(np.float64)),
        (np.dtype(np.complex128), "complex64", np.dtype(np.complex64)),
        (np.dtype(np.complex64), np.complex128, np.dtype(np.complex128)),
    ],
)
def test_astype_builds_lazy_graph_with_exact_output_dtype(
    input_dtype: np.dtype[Any],
    target: Any,
    expected_dtype: np.dtype[Any],
) -> None:
    values = np.arange(16).reshape(2, 8).astype(input_dtype)
    source = da.from_array(values, chunks=(1, -1))
    executed: list[Any] = []

    with Callback(pretask=lambda key, _graph, _state: executed.append(key)):
        result = Astype(8.0, dtype=target).process(source)

    assert executed == []
    assert result.shape == source.shape
    assert result.dtype == expected_dtype
    np.testing.assert_array_equal(result.compute(), values.astype(expected_dtype))
    np.testing.assert_array_equal(source.compute(), values)


def test_astype_preserves_existing_channel_and_time_axis_chunks() -> None:
    values = np.arange(40, dtype=np.float64).reshape(2, 20)
    source = da.from_array(values, chunks=(1, 6))

    result = Astype(20.0, dtype="float32").process(source)

    assert result.chunks == source.chunks
    np.testing.assert_array_equal(result.compute(), values.astype(np.float32))


def test_astype_eager_kernel_converts_without_mutating_input() -> None:
    values = np.array([[1.125, -2.25]], dtype=np.float64)
    original = values.copy()

    result = Astype(2.0, dtype="float32")._process(values)

    assert result.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(result, original.astype(np.float32))
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize(
    ("input_dtype", "target", "message"),
    [
        (np.dtype(np.float64), "complex64", "real or integer input"),
        (np.dtype(np.complex128), "float32", "Use 'complex64'"),
        (np.dtype(np.complex128), "float64", "complex input"),
        (np.dtype(np.float64), "float16", "float32, float64"),
        (np.dtype(np.float64), "int16", "float32, float64"),
        (np.dtype(np.float64), "bool", "float32, float64"),
        (np.dtype(np.float64), "object", "float32, float64"),
    ],
)
def test_astype_rejects_unsupported_or_cross_domain_dtype_before_graph_build(
    input_dtype: np.dtype[Any],
    target: Any,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = da.ones((2, 8), chunks=(1, -1), dtype=input_dtype)

    def unexpected_graph(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("graph construction must not start")

    monkeypatch.setattr(Astype, "_build_execution_graph", unexpected_graph)

    with pytest.raises(ValueError, match=message):
        operation = Astype(8.0, dtype=target)
        operation.process(source)


@pytest.mark.parametrize("dtype", [None, "not-a-dtype"])
def test_astype_rejects_values_that_are_not_explicit_numpy_dtypes(dtype: Any) -> None:
    with pytest.raises(TypeError, match="explicit NumPy dtype|valid NumPy dtype"):
        Astype(8.0, dtype=dtype)


@pytest.mark.parametrize("input_dtype", [np.dtype(np.bool_), np.dtype(object)])
def test_astype_rejects_non_numeric_input_dtype(input_dtype: np.dtype[Any]) -> None:
    values = np.ones((2, 8), dtype=input_dtype)
    source = da.from_array(values, chunks=(1, -1))

    with pytest.raises(ValueError, match="Unsupported input dtype"):
        Astype(8.0, dtype="float32").process(source)


def test_astype_has_stable_registry_key_and_public_export() -> None:
    operation = create_operation("astype", 8.0, dtype="float32")

    assert type(operation) is Astype
    assert get_operation("astype") is Astype
    assert operation.name == "astype"
    assert operation.params == {"dtype": "float32"}
