"""Process-wide default floating dtype for data that Wandas creates."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["default_dtype", "get_default_dtype", "resolve_dtype", "set_default_dtype"]

_SUPPORTED = (np.dtype(np.float32), np.dtype(np.float64))
_default: np.dtype[Any] = np.dtype(np.float64)


def _normalized(dtype: npt.DTypeLike) -> np.dtype[Any]:
    # np.dtype(None) is float64, so None is rejected before converting.
    try:
        normalized = None if dtype is None else np.dtype(dtype)
    except TypeError:
        normalized = None
    if normalized is None or not any(normalized == supported for supported in _SUPPORTED):
        raise ValueError(f"Wandas default dtype must be float32 or float64\n  Got: {dtype!r}")
    return normalized


def get_default_dtype() -> np.dtype[Any]:
    """Return the dtype Wandas uses for data it reads or generates.

    Returns:
        ``float64`` unless changed with :func:`set_default_dtype` or
        :func:`default_dtype`.
    """
    return _default


def set_default_dtype(dtype: npt.DTypeLike) -> None:
    """Set the dtype Wandas uses for data it reads or generates.

    The setting applies to :func:`wandas.read`, :meth:`ChannelFrame.from_file`,
    the dataset loaders and :func:`wandas.generate_sin` when they are called
    without ``dtype``. It is process-wide, not stored in Recipes or WDF files, and
    not passed to worker processes.

    Args:
        dtype: ``"float32"`` or ``"float64"`` (or the matching NumPy dtype).

    Raises:
        ValueError: If ``dtype`` is not float32 or float64.
    """
    global _default
    _default = _normalized(dtype)


@contextmanager
def default_dtype(dtype: npt.DTypeLike) -> Iterator[None]:
    """Use ``dtype`` as the default inside a ``with`` block, then restore the previous one.

    Args:
        dtype: ``"float32"`` or ``"float64"`` (or the matching NumPy dtype).

    Raises:
        ValueError: If ``dtype`` is not float32 or float64.
    """
    global _default
    previous = _default
    _default = _normalized(dtype)
    try:
        yield
    finally:
        _default = previous


def resolve_dtype(dtype: npt.DTypeLike | None) -> np.dtype[Any]:
    """Return ``dtype`` validated, or the current default when it is ``None``."""
    return _default if dtype is None else _normalized(dtype)
