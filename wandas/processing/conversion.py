"""Explicit numerical representation conversions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from wandas.processing.base import ChannelIndependentAudioOperation, register_operation

_REAL_TARGET_DTYPES = frozenset({"float32", "float64"})
_COMPLEX_TARGET_DTYPES = frozenset({"complex64", "complex128"})
_SUPPORTED_TARGET_DTYPES = _REAL_TARGET_DTYPES | _COMPLEX_TARGET_DTYPES


def _normalize_target_dtype(dtype: npt.DTypeLike) -> str:
    """Return the canonical name for a target NumPy dtype."""
    if dtype is None:
        raise TypeError(
            "Invalid dtype for astype\n"
            "  Got: None\n"
            "  Expected: an explicit NumPy dtype\n"
            "Use 'float32', 'float64', 'complex64', or 'complex128'."
        )
    try:
        target = np.dtype(dtype)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Invalid dtype for astype\n"
            f"  Got: {dtype!r}\n"
            "  Expected: a valid NumPy dtype\n"
            "Use 'float32', 'float64', 'complex64', or 'complex128'."
        ) from exc
    return target.name


def _normalize_astype_dtype(input_dtype: npt.DTypeLike, dtype: npt.DTypeLike) -> str:
    """Validate an astype conversion and return its canonical target name."""
    source = np.dtype(input_dtype)
    target = _normalize_target_dtype(dtype)

    if source.kind == "c":
        if target == "float32":
            raise ValueError(
                "Invalid dtype conversion for complex input\n"
                "  Got: float32\n"
                "  Expected: complex64 or complex128\n"
                "Use 'complex64' to reduce a complex Frame to 32-bit components."
            )
        if target not in _COMPLEX_TARGET_DTYPES:
            raise ValueError(
                "Invalid dtype conversion for complex input\n"
                f"  Got: {target}\n"
                "  Expected: complex64 or complex128\n"
                "Keep complex Frames in the complex numerical domain."
            )
        return target

    if source.kind in {"f", "i", "u"}:
        if target not in _REAL_TARGET_DTYPES:
            raise ValueError(
                "Invalid dtype conversion for real or integer input\n"
                f"  Got: {target}\n"
                "  Expected: float32, float64\n"
                "Keep real and integer Frames in the real numerical domain."
            )
        return target

    raise ValueError(
        "Unsupported input dtype for astype\n"
        f"  Got: {source.name}\n"
        "  Expected: a real, integer, or complex numeric dtype\n"
        "Convert non-numeric data before constructing a Frame."
    )


def _validate_astype_recipe_params(params: Mapping[str, Any]) -> None:
    """Validate the canonical serialized parameter contract for astype."""
    if set(params) != {"dtype"}:
        raise ValueError("astype Recipe params must contain exactly 'dtype'")
    dtype = params["dtype"]
    if not isinstance(dtype, str) or dtype not in _SUPPORTED_TARGET_DTYPES:
        raise ValueError("astype Recipe dtype must be one of float32, float64, complex64, or complex128")


class Astype(ChannelIndependentAudioOperation[Any, Any]):
    """Convert a raw Frame tensor to a supported real or complex floating dtype.

    The eager kernel is channel-independent, preserves shape, and never mutates
    its input. :meth:`process` builds a lazy Dask graph whose dtype metadata is
    the exact selected target before computation. Real or integer inputs can
    produce float32/float64; complex inputs can produce complex64/complex128.

    Args:
        sampling_rate: Sampling rate in Hz. It is preserved and does not affect
            the numerical cast.
        dtype: Supported target NumPy dtype or equivalent dtype-like value.

    Raises:
        TypeError: If *dtype* is not understood by NumPy.
        ValueError: If the target is unsupported or :meth:`process` receives an
            input whose real/complex domain does not match it.
    """

    name = "astype"
    _display = "astype"

    def __init__(self, sampling_rate: float, dtype: npt.DTypeLike) -> None:
        """Initialize a dtype conversion with a canonical target dtype."""
        super().__init__(sampling_rate, dtype=_normalize_target_dtype(dtype))

    @property
    def dtype(self) -> str:
        """Return the canonical target dtype name."""
        return self._config_value("dtype")

    def validate_params(self) -> None:
        """Reject unsupported target representations at construction time."""
        if self.dtype not in _SUPPORTED_TARGET_DTYPES:
            raise ValueError(
                "Unsupported dtype for astype\n"
                f"  Got: {self.dtype}\n"
                "  Expected: float32, float64, complex64, or complex128\n"
                "Choose a supported floating representation."
            )

    def calculate_output_dtype(
        self,
        input_dtype: np.dtype[Any],
        *input_dtypes: np.dtype[Any],
    ) -> np.dtype[Any]:
        """Return exact output metadata after validating the source domain."""
        del input_dtypes
        return np.dtype(_normalize_astype_dtype(input_dtype, self.dtype))

    def _process(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Convert one eager channel-first tensor without mutating the input."""
        target = _normalize_astype_dtype(data.dtype, self.dtype)
        return data.astype(target, copy=False)


register_operation(Astype)


__all__ = ["Astype"]
