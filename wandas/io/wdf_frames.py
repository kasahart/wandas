"""Typed Frame codecs for the WDF persistence boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

import numpy as np
from dask.array.core import Array as DaArray

from wandas.core.base_frame import BaseFrame

FrameEncoder = Callable[[BaseFrame[Any]], dict[str, Any]]
FrameDecoder = Callable[[dict[str, Any], Mapping[str, Any]], BaseFrame[Any]]


@dataclass(frozen=True)
class FrameCodec:
    """One exact Frame type's constructor-state persistence contract."""

    frame_type: type[BaseFrame[Any]]
    encode: FrameEncoder
    decode: FrameDecoder


def _require_fields(state: Mapping[str, Any], expected: set[str], frame_type: str) -> None:
    if set(state) != expected:
        raise ValueError(
            "Invalid WDF Frame constructor state\n"
            f"  Frame type: {frame_type}\n"
            f"  Got fields: {sorted(state)}\n"
            f"  Expected fields: {sorted(expected)}\n"
            "Resave the file with a compatible Wandas version."
        )


def _channel_state(frame: BaseFrame[Any]) -> dict[str, Any]:
    del frame
    return {}


def _channel_decode(common: dict[str, Any], state: Mapping[str, Any]) -> BaseFrame[Any]:
    from wandas.frames.channel import ChannelFrame

    _require_fields(state, set(), "ChannelFrame")
    return ChannelFrame(**common)


def _spectral_state(frame: BaseFrame[Any]) -> dict[str, Any]:
    typed = cast(Any, frame)
    return {"n_fft": int(typed.n_fft), "window": str(typed.window)}


def _spectral_decode(common: dict[str, Any], state: Mapping[str, Any]) -> BaseFrame[Any]:
    from wandas.frames.spectral import SpectralFrame

    _require_fields(state, {"n_fft", "window"}, "SpectralFrame")
    return SpectralFrame(**common, n_fft=state["n_fft"], window=state["window"])


def _spectrogram_state(frame: BaseFrame[Any]) -> dict[str, Any]:
    typed = cast(Any, frame)
    return {
        "n_fft": int(typed.n_fft),
        "hop_length": int(typed.hop_length),
        "win_length": int(typed.win_length),
        "window": str(typed.window),
    }


def _spectrogram_decode(common: dict[str, Any], state: Mapping[str, Any]) -> BaseFrame[Any]:
    from wandas.frames.spectrogram import SpectrogramFrame

    expected = {"n_fft", "hop_length", "win_length", "window"}
    _require_fields(state, expected, "SpectrogramFrame")
    return SpectrogramFrame(
        **common,
        n_fft=state["n_fft"],
        hop_length=state["hop_length"],
        win_length=state["win_length"],
        window=state["window"],
    )


def _cepstral_state(frame: BaseFrame[Any]) -> dict[str, Any]:
    typed = cast(Any, frame)
    return {"n_fft": int(typed.n_fft), "window": str(typed.window)}


def _cepstral_decode(common: dict[str, Any], state: Mapping[str, Any]) -> BaseFrame[Any]:
    from wandas.frames.cepstral import CepstralFrame

    _require_fields(state, {"n_fft", "window"}, "CepstralFrame")
    return CepstralFrame(**common, n_fft=state["n_fft"], window=state["window"])


def _cepstrogram_state(frame: BaseFrame[Any]) -> dict[str, Any]:
    typed = cast(Any, frame)
    return {
        "n_fft": int(typed.n_fft),
        "hop_length": int(typed.hop_length),
        "win_length": int(typed.win_length),
        "window": str(typed.window),
    }


def _cepstrogram_decode(common: dict[str, Any], state: Mapping[str, Any]) -> BaseFrame[Any]:
    from wandas.frames.cepstrogram import CepstrogramFrame

    expected = {"n_fft", "hop_length", "win_length", "window"}
    _require_fields(state, expected, "CepstrogramFrame")
    return CepstrogramFrame(
        **common,
        n_fft=state["n_fft"],
        hop_length=state["hop_length"],
        win_length=state["win_length"],
        window=state["window"],
    )


def _noct_state(frame: BaseFrame[Any]) -> dict[str, Any]:
    typed = cast(Any, frame)
    return {
        "fmin": float(typed.fmin),
        "fmax": float(typed.fmax),
        "n": int(typed.n),
        "G": int(typed.G),
        "fr": int(typed.fr),
    }


def _noct_decode(common: dict[str, Any], state: Mapping[str, Any]) -> BaseFrame[Any]:
    from wandas.frames.noct import NOctFrame

    expected = {"fmin", "fmax", "n", "G", "fr"}
    _require_fields(state, expected, "NOctFrame")
    return NOctFrame(
        **common,
        fmin=state["fmin"],
        fmax=state["fmax"],
        n=state["n"],
        G=state["G"],
        fr=state["fr"],
    )


def _roughness_state(frame: BaseFrame[Any]) -> dict[str, Any]:
    typed = cast(Any, frame)
    return {"bark_axis": typed.bark_axis.tolist(), "overlap": float(typed.overlap)}


def _roughness_decode(common: dict[str, Any], state: Mapping[str, Any]) -> BaseFrame[Any]:
    from wandas.frames.roughness import RoughnessFrame

    _require_fields(state, {"bark_axis", "overlap"}, "RoughnessFrame")
    return RoughnessFrame(
        **common,
        bark_axis=np.asarray(state["bark_axis"], dtype=float),
        overlap=state["overlap"],
    )


@lru_cache(maxsize=1)
def _codecs() -> tuple[FrameCodec, ...]:
    """Build the registry lazily to keep Frame imports cycle-free."""
    from wandas.frames.cepstral import CepstralFrame
    from wandas.frames.cepstrogram import CepstrogramFrame
    from wandas.frames.channel import ChannelFrame
    from wandas.frames.noct import NOctFrame
    from wandas.frames.roughness import RoughnessFrame
    from wandas.frames.spectral import SpectralFrame
    from wandas.frames.spectrogram import SpectrogramFrame

    return (
        FrameCodec(ChannelFrame, _channel_state, _channel_decode),
        FrameCodec(SpectralFrame, _spectral_state, _spectral_decode),
        FrameCodec(SpectrogramFrame, _spectrogram_state, _spectrogram_decode),
        FrameCodec(CepstralFrame, _cepstral_state, _cepstral_decode),
        FrameCodec(CepstrogramFrame, _cepstrogram_state, _cepstrogram_decode),
        FrameCodec(NOctFrame, _noct_state, _noct_decode),
        FrameCodec(RoughnessFrame, _roughness_state, _roughness_decode),
    )


def _codecs_by_type() -> dict[type[BaseFrame[Any]], FrameCodec]:
    return {codec.frame_type: codec for codec in _codecs()}


def _codecs_by_name() -> dict[str, FrameCodec]:
    return {codec.frame_type.__name__: codec for codec in _codecs()}


def encode_frame_state(frame: BaseFrame[Any]) -> dict[str, Any]:
    """Encode exact Frame type, semantic dimensions, and constructor state."""
    codecs_by_name = _codecs_by_name()
    codec = _codecs_by_type().get(type(frame))
    if codec is None:
        raise TypeError(
            "Unsupported Frame type for WDF save\n"
            f"  Got: {type(frame).__name__}\n"
            f"  Supported: {', '.join(codecs_by_name)}\n"
            "Convert the result to a supported built-in Frame before saving."
        )
    return {
        "frame_type": codec.frame_type.__name__,
        "dims": list(frame._xr.dims),
        "constructor": codec.encode(frame),
    }


def decode_frame_state(
    state: Mapping[str, Any],
    *,
    data: DaArray,
    common: dict[str, Any],
) -> BaseFrame[Any]:
    """Reconstruct and validate one registered typed Frame."""
    if set(state) != {"frame_type", "dims", "constructor"}:
        raise ValueError("Invalid WDF Frame state fields")
    frame_type = state["frame_type"]
    dims = state["dims"]
    constructor = state["constructor"]
    if not isinstance(frame_type, str) or not isinstance(dims, list) or not all(isinstance(dim, str) for dim in dims):
        raise ValueError("Invalid WDF Frame type or semantic dimensions")
    if not isinstance(constructor, Mapping):
        raise ValueError("Invalid WDF Frame constructor state")
    codecs_by_name = _codecs_by_name()
    codec = codecs_by_name.get(frame_type)
    if codec is None:
        raise ValueError(
            "Unsupported WDF frame type\n"
            f"  Got: {frame_type!r}\n"
            f"  Supported: {', '.join(codecs_by_name)}\n"
            "Load the file with a compatible Wandas version."
        )
    try:
        frame = codec.decode({**common, "data": data}, constructor)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Invalid typed WDF Frame state\n"
            f"  Frame type: {frame_type}\n"
            f"  Cause: {exc}\n"
            "Resave the file with a compatible Wandas version."
        ) from exc
    if list(frame._xr.dims) != dims:
        raise ValueError(
            "WDF semantic dimensions do not match reconstructed Frame\n"
            f"  Stored: {dims}\n"
            f"  Reconstructed: {list(frame._xr.dims)}\n"
            "The file is malformed or uses an incompatible Frame contract."
        )
    return frame


def frame_dimension_coordinates(frame: BaseFrame[Any]) -> dict[str, np.ndarray[Any, Any]]:
    """Return non-channel dimension coordinates that carry represented axes."""
    coordinates: dict[str, np.ndarray[Any, Any]] = {}
    for dim in frame._xr.dims:
        if dim == "channel" or dim not in frame._xr.coords:
            continue
        coordinate = frame._xr.coords[dim]
        if coordinate.dims == (dim,):
            coordinates[str(dim)] = np.asarray(coordinate.values).copy()
    return coordinates


def restore_frame_coordinates(
    frame: BaseFrame[Any],
    coordinates: Mapping[str, np.ndarray[Any, Any]],
) -> None:
    """Restore validated represented-axis coordinates on a new typed Frame."""
    for name, values in coordinates.items():
        if name == "channel" or name not in frame._xr.dims:
            raise ValueError(f"Invalid WDF coordinate dimension: {name!r}")
        axis = frame._xr.dims.index(name)
        if values.ndim != 1 or len(values) != int(frame._data.shape[axis]):
            raise ValueError(
                "WDF coordinate length does not match Frame data\n"
                f"  Coordinate: {name!r}\n"
                f"  Got: {values.shape}\n"
                f"  Expected length: {frame._data.shape[axis]}"
            )
        frame._xr = frame._xr.assign_coords({name: (name, values.copy())})


__all__ = [
    "decode_frame_state",
    "encode_frame_state",
    "frame_dimension_coordinates",
    "restore_frame_coordinates",
]
