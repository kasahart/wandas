"""Frame-owned orchestration for explicit raw dtype conversion."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy.typing as npt

from wandas.pipeline.decorators import OperationCapture
from wandas.processing.conversion import (
    _SUPPORTED_TARGET_DTYPES,
    Astype,
    _normalize_astype_dtype,
)
from wandas.processing.semantic import InputBinding


def capture_astype(args: tuple[Any, ...], params: Mapping[str, Any]) -> OperationCapture:
    """Capture Frame astype intent with a canonical dtype string."""
    receiver = args[0]
    dtype = _normalize_astype_dtype(receiver._data.dtype, params["dtype"])
    return OperationCapture(
        (InputBinding("frame", "frame"),),
        (receiver.lineage,),
        {"dtype": dtype},
    )


def validate_astype_recipe_params(params: Mapping[str, Any]) -> None:
    """Validate the canonical serialized parameter contract for Frame astype."""
    if set(params) != {"dtype"}:
        raise ValueError("astype Recipe params must contain exactly 'dtype'")
    dtype = params["dtype"]
    if not isinstance(dtype, str) or dtype not in _SUPPORTED_TARGET_DTYPES:
        raise ValueError("astype Recipe dtype must be one of float32, float64, complex64, or complex128")


def astype_frame(frame: Any, dtype: npt.DTypeLike) -> Any:
    """Apply the processing cast while preserving Frame-owned state and lineage."""
    target = _normalize_astype_dtype(frame._data.dtype, dtype)
    operation = Astype(frame.sampling_rate, dtype=target)
    processed_data = operation.process(frame._data)
    return frame._create_new_instance(
        processed_data,
        lineage=frame._required_semantic_lineage(),
    )


__all__: list[str] = []
