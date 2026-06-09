"""Signal-safe chunk policy validation for processing operations."""

from __future__ import annotations

from typing import Any

STRICT_CORE_DIMS_BY_OPERATION: dict[str, tuple[str, ...]] = {
    "highpass_filter": ("time",),
    "lowpass_filter": ("time",),
    "bandpass_filter": ("time",),
    "a_weighting": ("time",),
    "normalize": ("time",),
    "resampling": ("time",),
    "rms_trend": ("time",),
    "sound_level": ("time",),
    "fft": ("time",),
    "stft": ("time",),
    "welch": ("time",),
    "coherence": ("time",),
    "csd": ("time",),
    "transfer_function": ("time",),
    "loudness_zwtv": ("time",),
    "loudness_zwst": ("time",),
    "roughness_dw": ("time",),
    "roughness_dw_spec": ("time",),
    "sharpness_din": ("time",),
    "sharpness_din_st": ("time",),
    "ifft": ("frequency",),
    "istft": ("frequency", "time"),
    "noct_spectrum": ("time",),
    "noct_synthesis": ("frequency",),
}


def validate_strict_chunks(frame: Any, operation_name: str, params: dict[str, Any] | None = None) -> None:
    """Reject unsafe split chunks for operations with signal core dimensions."""
    core_dims = STRICT_CORE_DIMS_BY_OPERATION.get(operation_name)
    if not core_dims:
        return

    data_array = frame.to_xarray(copy=False)
    if operation_name == "normalize":
        axis = params.get("axis", -1) if params is not None else -1
        if axis is None or not _axis_targets_dim(axis, data_array.dims, "time"):
            return

    if data_array.chunks is None:
        return

    chunks_by_dim = dict(zip(data_array.dims, data_array.chunks, strict=True))
    for dim in core_dims:
        chunks = chunks_by_dim.get(dim)
        if chunks is not None and len(chunks) > 1:
            raise ValueError(
                f"Operation '{operation_name}' requires contiguous chunks along {dim}\n"
                f"  Got chunks for {dim}: {chunks}\n"
                f"  Rechunk the frame so {dim} is a single chunk before running this operation."
            )


def _axis_targets_dim(axis: Any, dims: tuple[str, ...], dim: str) -> bool:
    if axis is None:
        return True
    if axis == dim:
        return True
    if isinstance(axis, int):
        normalized = axis if axis >= 0 else len(dims) + axis
        return 0 <= normalized < len(dims) and dims[normalized] == dim
    return False
