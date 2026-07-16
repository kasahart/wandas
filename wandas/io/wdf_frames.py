"""Typed Frame codecs for the WDF persistence boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, cast

import numpy as np
from dask.array.core import Array as DaArray

from wandas.core.base_frame import BaseFrame

FrameEncoder = Callable[[BaseFrame[Any]], dict[str, Any]]
FrameDecoder = Callable[[dict[str, Any], Mapping[str, Any]], BaseFrame[Any]]
DataDomain = Literal["real", "complex", "numeric"]


@dataclass(frozen=True)
class FrameCodec:
    """One exact Frame type's constructor-state persistence contract."""

    frame_type: type[BaseFrame[Any]]
    encode: FrameEncoder
    decode: FrameDecoder
    data_domain: DataDomain


def _invalid_constructor_value(
    frame_type: str,
    field: str,
    value: object,
    expected: str,
) -> ValueError:
    """Build one actionable typed-state validation error."""
    return ValueError(
        "Invalid WDF Frame constructor value\n"
        f"  Frame type: {frame_type}\n"
        f"  Field: {field}\n"
        f"  Got: {value!r}\n"
        f"  Expected: {expected}\n"
        "Resave the file with a compatible Wandas version."
    )


def _positive_integer(state: Mapping[str, Any], field: str, frame_type: str) -> int:
    """Return one strict positive JSON integer, excluding booleans."""
    value = state[field]
    if type(value) is not int or value <= 0:
        raise _invalid_constructor_value(frame_type, field, value, "a positive JSON integer")
    return value


def _nonblank_string(state: Mapping[str, Any], field: str, frame_type: str) -> str:
    """Return one strict non-blank JSON string."""
    value = state[field]
    if not isinstance(value, str) or not value.strip():
        raise _invalid_constructor_value(frame_type, field, value, "a non-blank JSON string")
    return value


def _finite_number(state: Mapping[str, Any], field: str, frame_type: str) -> float:
    """Return one finite JSON number, excluding booleans."""
    value = state[field]
    if type(value) not in {int, float} or not np.isfinite(value):
        raise _invalid_constructor_value(frame_type, field, value, "a finite JSON number")
    return float(value)


def _require_rank(data: DaArray, expected: set[int], frame_type: str) -> None:
    """Validate the stored tensor rank before a constructor can normalize it."""
    if data.ndim not in expected:
        ranks = ", ".join(f"{rank}D" for rank in sorted(expected))
        raise ValueError(
            "Invalid WDF Frame tensor rank\n"
            f"  Frame type: {frame_type}\n"
            f"  Got: {data.ndim}D with shape {data.shape}\n"
            f"  Expected: {ranks}\n"
            "The file is malformed; resave it with a compatible Wandas version."
        )


def _validate_codec_dtype(codec: FrameCodec, dtype: np.dtype[Any]) -> None:
    """Enforce one Frame codec's numeric data-domain contract."""
    is_integer = np.issubdtype(dtype, np.integer)
    is_real = np.issubdtype(dtype, np.floating) or is_integer
    is_complex = np.issubdtype(dtype, np.complexfloating)
    valid = {
        "real": is_real,
        "complex": is_complex,
        "numeric": is_real or is_complex,
    }[codec.data_domain]
    if not valid:
        expected = {
            "real": "a real numeric dtype",
            "complex": "a complex numeric dtype",
            "numeric": "a real or complex numeric dtype",
        }[codec.data_domain]
        raise ValueError(
            "Invalid WDF Frame tensor dtype\n"
            f"  Frame type: {codec.frame_type.__name__}\n"
            f"  Got: {dtype}\n"
            f"  Expected: {expected}\n"
            "Choose a dtype that preserves the Frame's analysis domain."
        )


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
    n_fft = _positive_integer(state, "n_fft", "SpectralFrame")
    window = _nonblank_string(state, "window", "SpectralFrame")
    data = common["data"]
    _require_rank(data, {2}, "SpectralFrame")
    expected_bins = n_fft // 2 + 1
    if int(data.shape[-1]) > expected_bins:
        raise _invalid_constructor_value(
            "SpectralFrame",
            "n_fft",
            n_fft,
            f"an FFT size whose one-sided spectrum has at most {expected_bins} represented bins",
        )
    return SpectralFrame(**common, n_fft=n_fft, window=window)


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
    n_fft = _positive_integer(state, "n_fft", "SpectrogramFrame")
    hop_length = _positive_integer(state, "hop_length", "SpectrogramFrame")
    win_length = _positive_integer(state, "win_length", "SpectrogramFrame")
    window = _nonblank_string(state, "window", "SpectrogramFrame")
    if win_length > n_fft:
        raise _invalid_constructor_value(
            "SpectrogramFrame", "win_length", win_length, f"a value no greater than n_fft ({n_fft})"
        )
    if hop_length > win_length:
        raise _invalid_constructor_value(
            "SpectrogramFrame", "hop_length", hop_length, f"a value no greater than win_length ({win_length})"
        )
    data = common["data"]
    _require_rank(data, {3}, "SpectrogramFrame")
    return SpectrogramFrame(
        **common,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    )


def _cepstral_state(frame: BaseFrame[Any]) -> dict[str, Any]:
    typed = cast(Any, frame)
    return {"n_fft": int(typed.n_fft), "window": str(typed.window)}


def _cepstral_decode(common: dict[str, Any], state: Mapping[str, Any]) -> BaseFrame[Any]:
    from wandas.frames.cepstral import CepstralFrame

    _require_fields(state, {"n_fft", "window"}, "CepstralFrame")
    n_fft = _positive_integer(state, "n_fft", "CepstralFrame")
    window = _nonblank_string(state, "window", "CepstralFrame")
    _require_rank(common["data"], {2}, "CepstralFrame")
    return CepstralFrame(**common, n_fft=n_fft, window=window)


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
    n_fft = _positive_integer(state, "n_fft", "CepstrogramFrame")
    hop_length = _positive_integer(state, "hop_length", "CepstrogramFrame")
    win_length = _positive_integer(state, "win_length", "CepstrogramFrame")
    window = _nonblank_string(state, "window", "CepstrogramFrame")
    if win_length > n_fft:
        raise _invalid_constructor_value(
            "CepstrogramFrame", "win_length", win_length, f"a value no greater than n_fft ({n_fft})"
        )
    if hop_length > win_length:
        raise _invalid_constructor_value(
            "CepstrogramFrame", "hop_length", hop_length, f"a value no greater than win_length ({win_length})"
        )
    _require_rank(common["data"], {3}, "CepstrogramFrame")
    return CepstrogramFrame(
        **common,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
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
    fmin = _finite_number(state, "fmin", "NOctFrame")
    fmax = _finite_number(state, "fmax", "NOctFrame")
    n = _positive_integer(state, "n", "NOctFrame")
    reference_band = _positive_integer(state, "G", "NOctFrame")
    reference_frequency = _positive_integer(state, "fr", "NOctFrame")
    if fmin < 0:
        raise _invalid_constructor_value("NOctFrame", "fmin", fmin, "a non-negative frequency")
    if fmax < fmin:
        raise _invalid_constructor_value("NOctFrame", "fmax", fmax, f"a frequency no lower than fmin ({fmin})")
    _require_rank(common["data"], {2}, "NOctFrame")
    return NOctFrame(
        **common,
        fmin=fmin,
        fmax=fmax,
        n=n,
        G=reference_band,
        fr=reference_frequency,
    )


def _roughness_state(frame: BaseFrame[Any]) -> dict[str, Any]:
    typed = cast(Any, frame)
    return {"bark_axis": typed.bark_axis.tolist(), "overlap": float(typed.overlap)}


def _roughness_decode(common: dict[str, Any], state: Mapping[str, Any]) -> BaseFrame[Any]:
    from wandas.frames.roughness import RoughnessFrame

    _require_fields(state, {"bark_axis", "overlap"}, "RoughnessFrame")
    raw_bark_axis = state["bark_axis"]
    if not isinstance(raw_bark_axis, list) or len(raw_bark_axis) != 47:
        raise _invalid_constructor_value("RoughnessFrame", "bark_axis", raw_bark_axis, "47 finite JSON numbers")
    if any(type(value) not in {int, float} or not np.isfinite(value) for value in raw_bark_axis):
        raise _invalid_constructor_value("RoughnessFrame", "bark_axis", raw_bark_axis, "47 finite JSON numbers")
    bark_axis = np.asarray(raw_bark_axis, dtype=float)
    overlap = _finite_number(state, "overlap", "RoughnessFrame")
    if not 0.0 <= overlap <= 1.0:
        raise _invalid_constructor_value("RoughnessFrame", "overlap", overlap, "a value between 0.0 and 1.0")
    data = common["data"]
    _require_rank(data, {2, 3}, "RoughnessFrame")
    if int(data.shape[-2]) != len(bark_axis):
        raise _invalid_constructor_value(
            "RoughnessFrame",
            "bark_axis",
            raw_bark_axis,
            f"one value for each of the {data.shape[-2]} stored Bark bins",
        )
    return RoughnessFrame(
        **common,
        bark_axis=bark_axis,
        overlap=overlap,
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
        FrameCodec(ChannelFrame, _channel_state, _channel_decode, "real"),
        FrameCodec(SpectralFrame, _spectral_state, _spectral_decode, "numeric"),
        FrameCodec(SpectrogramFrame, _spectrogram_state, _spectrogram_decode, "complex"),
        FrameCodec(CepstralFrame, _cepstral_state, _cepstral_decode, "real"),
        FrameCodec(CepstrogramFrame, _cepstrogram_state, _cepstrogram_decode, "real"),
        FrameCodec(NOctFrame, _noct_state, _noct_decode, "real"),
        FrameCodec(RoughnessFrame, _roughness_state, _roughness_decode, "real"),
    )


def _codecs_by_type() -> dict[type[BaseFrame[Any]], FrameCodec]:
    return {codec.frame_type: codec for codec in _codecs()}


def _codecs_by_name() -> dict[str, FrameCodec]:
    return {codec.frame_type.__name__: codec for codec in _codecs()}


def _codec_for_frame(frame: BaseFrame[Any]) -> FrameCodec:
    """Return the exact built-in codec or reject extension subclasses."""
    codec = _codecs_by_type().get(type(frame))
    if codec is None:
        codecs_by_name = _codecs_by_name()
        raise TypeError(
            "Unsupported Frame type for WDF save\n"
            f"  Got: {type(frame).__name__}\n"
            f"  Supported: {', '.join(codecs_by_name)}\n"
            "Convert the result to a supported built-in Frame before saving."
        )
    return codec


def encode_frame_state(frame: BaseFrame[Any]) -> dict[str, Any]:
    """Encode exact Frame type, semantic dimensions, and constructor state."""
    codec = _codec_for_frame(frame)
    _validate_codec_dtype(codec, frame._data.dtype)
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
        _validate_codec_dtype(codec, data.dtype)
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


def validate_frame_save_dtype(frame: BaseFrame[Any], target_dtype: np.dtype[Any]) -> None:
    """Validate an explicit save dtype against source data and Frame domain."""
    codec = _codec_for_frame(frame)
    _validate_codec_dtype(codec, frame._data.dtype)
    _validate_codec_dtype(codec, target_dtype)
    if np.issubdtype(frame._data.dtype, np.complexfloating) and not np.issubdtype(target_dtype, np.complexfloating):
        raise ValueError(
            "WDF dtype conversion would discard complex data\n"
            f"  Frame type: {type(frame).__name__}\n"
            f"  Source dtype: {frame._data.dtype}\n"
            f"  Requested dtype: {target_dtype}\n"
            "Choose a complex dtype such as 'complex64' or omit dtype to preserve the analysis result."
        )


def _coordinate_grid(frame: BaseFrame[Any], name: str) -> tuple[float, float | None]:
    """Return represented-axis spacing and an optional upper domain bound."""
    typed = cast(Any, frame)
    if name == "frequency":
        spacing = float(frame.sampling_rate) / int(typed.n_fft)
        return spacing, (int(typed.n_fft) // 2) * spacing
    if name == "quefrency":
        spacing = 1.0 / float(frame.sampling_rate)
        return spacing, (int(typed.n_fft) - 1) * spacing
    if name == "time" and hasattr(typed, "hop_length"):
        return float(typed.hop_length) / float(frame.sampling_rate), None
    raise ValueError(
        "Invalid WDF coordinate dimension\n"
        f"  Frame type: {type(frame).__name__}\n"
        f"  Coordinate: {name!r}\n"
        "Persist only registered represented-axis coordinates for this Frame type."
    )


def _validate_coordinate_values(
    frame: BaseFrame[Any],
    name: str,
    values: np.ndarray[Any, Any],
    expected_length: int,
) -> np.ndarray[Any, Any]:
    """Validate one numeric, finite, ordered represented-axis coordinate."""
    if values.ndim != 1 or len(values) != expected_length:
        raise ValueError(
            "WDF coordinate length does not match Frame data\n"
            f"  Coordinate: {name!r}\n"
            f"  Got: {values.shape}\n"
            f"  Expected length: {expected_length}\n"
            "Resave the file with a compatible Wandas version."
        )
    is_real_numeric = np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.floating)
    if not is_real_numeric:
        raise ValueError(
            "Invalid WDF coordinate dtype\n"
            f"  Coordinate: {name!r}\n"
            f"  Got: {values.dtype}\n"
            "  Expected: a numeric finite real array\n"
            "Resave the file with numeric represented-axis coordinates."
        )
    normalized = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(normalized)):
        raise ValueError(
            "Invalid WDF coordinate values\n"
            f"  Coordinate: {name!r}\n"
            "  Expected: a numeric finite real array\n"
            "Replace NaN or infinite axis values and resave the file."
        )
    if len(normalized) > 1 and not np.all(np.diff(normalized) > 0):
        raise ValueError(
            "Invalid WDF coordinate ordering\n"
            f"  Coordinate: {name!r}\n"
            "  Expected: strictly increasing represented-axis values\n"
            "Resave an ordered forward slice of the Frame axis."
        )
    grid = _coordinate_grid(frame, name)
    if len(normalized):
        spacing, upper = grid
        scaled = normalized / spacing
        on_grid = np.allclose(scaled, np.rint(scaled), rtol=0.0, atol=1e-7)
        in_bounds = normalized[0] >= -spacing * 1e-7 and (upper is None or normalized[-1] <= upper + spacing * 1e-7)
        if not on_grid or not in_bounds:
            raise ValueError(
                "Invalid WDF coordinate sampling grid\n"
                f"  Coordinate: {name!r}\n"
                f"  Expected spacing: an integer multiple of {spacing}\n"
                f"  Expected upper bound: {upper}\n"
                "Resave a valid represented-axis slice for this Frame domain."
            )
    return normalized


def frame_dimension_coordinates(frame: BaseFrame[Any]) -> dict[str, np.ndarray[Any, Any]]:
    """Return non-channel dimension coordinates that carry represented axes."""
    coordinates: dict[str, np.ndarray[Any, Any]] = {}
    for dim in frame._xr.dims:
        if dim == "channel" or dim not in frame._xr.coords:
            continue
        coordinate = frame._xr.coords[dim]
        if coordinate.dims == (dim,):
            axis = frame._xr.dims.index(dim)
            values = np.asarray(coordinate.values)
            coordinates[str(dim)] = _validate_coordinate_values(
                frame,
                str(dim),
                values,
                int(frame._data.shape[axis]),
            ).copy()
    return coordinates


def restore_frame_coordinates(
    frame: BaseFrame[Any],
    coordinates: Mapping[str, np.ndarray[Any, Any]],
) -> None:
    """Restore validated represented-axis coordinates on a new typed Frame."""
    expected = set(frame_dimension_coordinates(frame))
    stored = set(coordinates)
    unexpected = stored - expected
    if unexpected:
        raise ValueError(
            "Invalid WDF coordinate dimension\n"
            f"  Frame type: {type(frame).__name__}\n"
            f"  Unexpected: {sorted(unexpected)}\n"
            f"  Expected: {sorted(expected)}\n"
            "Resave the file with a compatible Wandas version."
        )
    missing = expected - stored
    if missing:
        raise ValueError(
            "Incomplete WDF Frame coordinates\n"
            f"  Frame type: {type(frame).__name__}\n"
            f"  Missing: {sorted(missing)}\n"
            "Resave the file with a compatible Wandas version; represented axes cannot be reconstructed safely."
        )
    for name, values in coordinates.items():
        axis = frame._xr.dims.index(name)
        validated = _validate_coordinate_values(frame, name, values, int(frame._data.shape[axis]))
        frame._xr = frame._xr.assign_coords({name: (name, validated.copy())})


__all__ = [
    "decode_frame_state",
    "encode_frame_state",
    "frame_dimension_coordinates",
    "restore_frame_coordinates",
    "validate_frame_save_dtype",
]
