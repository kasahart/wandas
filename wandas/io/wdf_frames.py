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
FrameConstructorValidator = Callable[[Mapping[str, Any], DaArray], object]
DataDomain = Literal["real", "complex", "numeric"]


@dataclass(frozen=True)
class FrameCodec:
    """Persistence contract for one exact built-in Frame type.

    Attributes:
        frame_type: Concrete Frame class accepted by the codec. Subclasses do not
            match implicitly.
        encode: Extracts only the constructor state not shared by every Frame.
        validate_constructor: Validates that state identically at save and load.
        decode: Reconstructs the concrete Frame from common and type-specific state.
        data_domain: Numeric dtype family accepted at both save and load boundaries.
        data_ranks: Exact tensor ranks supported by the concrete Frame contract.
    """

    frame_type: type[BaseFrame[Any]]
    encode: FrameEncoder
    validate_constructor: FrameConstructorValidator
    decode: FrameDecoder
    data_domain: DataDomain
    data_ranks: frozenset[int]


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
    """Return one finite JSON number without coercion."""
    value = state[field]
    if type(value) not in {int, float} or not np.isfinite(value):
        raise _invalid_constructor_value(frame_type, field, value, "a finite JSON number")
    return value


def _require_rank(data: DaArray, expected: set[int], frame_type: str) -> None:
    """Validate the typed tensor rank before a constructor can normalize it."""
    if data.ndim not in expected:
        ranks = ", ".join(f"{rank}D" for rank in sorted(expected))
        raise ValueError(
            "Invalid WDF Frame tensor rank\n"
            f"  Frame type: {frame_type}\n"
            f"  Got: {data.ndim}D with shape {data.shape}\n"
            f"  Expected: {ranks}\n"
            "Use a tensor rank supported by this Frame codec before saving or loading."
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


def _validate_codec_tensor(codec: FrameCodec, data: DaArray) -> None:
    """Enforce one codec's symmetric save/load tensor contract."""
    _require_rank(data, set(codec.data_ranks), codec.frame_type.__name__)
    _validate_codec_dtype(codec, data.dtype)


def _require_fields(state: Mapping[str, Any], expected: set[str], frame_type: str) -> None:
    """Require an exact constructor-state field set without defaults."""
    if set(state) != expected:
        raise ValueError(
            "Invalid WDF Frame constructor state\n"
            f"  Frame type: {frame_type}\n"
            f"  Got fields: {sorted(state)}\n"
            f"  Expected fields: {sorted(expected)}\n"
            "Resave the file with a compatible Wandas version."
        )


# Each encoder below persists only constructor arguments unique to its Frame type.
# Common state such as sampling rate and channel metadata lives in the WDF container,
# so it is intentionally absent from these mappings.
def _channel_state(frame: BaseFrame[Any]) -> dict[str, Any]:
    del frame
    return {}


def _validate_channel_constructor_state(state: Mapping[str, Any], data: DaArray) -> None:
    del data
    _require_fields(state, set(), "ChannelFrame")


def _channel_decode(common: dict[str, Any], state: Mapping[str, Any]) -> BaseFrame[Any]:
    from wandas.frames.channel import ChannelFrame

    _validate_channel_constructor_state(state, common["data"])
    return ChannelFrame(**common)


def _spectral_state(frame: BaseFrame[Any]) -> dict[str, Any]:
    typed = cast(Any, frame)
    return {"n_fft": typed.n_fft, "window": typed.window}


def _validate_spectral_constructor_state(state: Mapping[str, Any], data: DaArray) -> tuple[int, str]:
    _require_fields(state, {"n_fft", "window"}, "SpectralFrame")
    n_fft = _positive_integer(state, "n_fft", "SpectralFrame")
    expected_bins = n_fft // 2 + 1
    if int(data.shape[-1]) != expected_bins:
        raise _invalid_constructor_value(
            "SpectralFrame",
            "n_fft",
            n_fft,
            f"a value producing the {data.shape[-1]} stored frequency bins",
        )
    return n_fft, _nonblank_string(state, "window", "SpectralFrame")


def _spectral_decode(common: dict[str, Any], state: Mapping[str, Any]) -> BaseFrame[Any]:
    from wandas.frames.spectral import SpectralFrame

    n_fft, window = _validate_spectral_constructor_state(state, common["data"])
    return SpectralFrame(**common, n_fft=n_fft, window=window)


def _spectrogram_state(frame: BaseFrame[Any]) -> dict[str, Any]:
    typed = cast(Any, frame)
    return {
        "n_fft": typed.n_fft,
        "hop_length": typed.hop_length,
        "win_length": typed.win_length,
        "window": typed.window,
    }


def _validate_spectrogram_constructor_state(state: Mapping[str, Any], data: DaArray) -> tuple[int, int, int, str]:
    del data
    expected = {"n_fft", "hop_length", "win_length", "window"}
    _require_fields(state, expected, "SpectrogramFrame")
    return (
        _positive_integer(state, "n_fft", "SpectrogramFrame"),
        _positive_integer(state, "hop_length", "SpectrogramFrame"),
        _positive_integer(state, "win_length", "SpectrogramFrame"),
        _nonblank_string(state, "window", "SpectrogramFrame"),
    )


def _spectrogram_decode(common: dict[str, Any], state: Mapping[str, Any]) -> BaseFrame[Any]:
    from wandas.frames.spectrogram import SpectrogramFrame

    n_fft, hop_length, win_length, window = _validate_spectrogram_constructor_state(state, common["data"])
    return SpectrogramFrame(
        **common,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    )


def _cepstral_state(frame: BaseFrame[Any]) -> dict[str, Any]:
    typed = cast(Any, frame)
    return {"n_fft": typed.n_fft, "window": typed.window}


def _validate_cepstral_constructor_state(state: Mapping[str, Any], data: DaArray) -> tuple[int, str]:
    del data
    _require_fields(state, {"n_fft", "window"}, "CepstralFrame")
    return (
        _positive_integer(state, "n_fft", "CepstralFrame"),
        _nonblank_string(state, "window", "CepstralFrame"),
    )


def _cepstral_decode(common: dict[str, Any], state: Mapping[str, Any]) -> BaseFrame[Any]:
    from wandas.frames.cepstral import CepstralFrame

    n_fft, window = _validate_cepstral_constructor_state(state, common["data"])
    return CepstralFrame(**common, n_fft=n_fft, window=window)


def _cepstrogram_state(frame: BaseFrame[Any]) -> dict[str, Any]:
    typed = cast(Any, frame)
    return {
        "n_fft": typed.n_fft,
        "hop_length": typed.hop_length,
        "win_length": typed.win_length,
        "window": typed.window,
    }


def _validate_cepstrogram_constructor_state(state: Mapping[str, Any], data: DaArray) -> tuple[int, int, int, str]:
    del data
    expected = {"n_fft", "hop_length", "win_length", "window"}
    _require_fields(state, expected, "CepstrogramFrame")
    return (
        _positive_integer(state, "n_fft", "CepstrogramFrame"),
        _positive_integer(state, "hop_length", "CepstrogramFrame"),
        _positive_integer(state, "win_length", "CepstrogramFrame"),
        _nonblank_string(state, "window", "CepstrogramFrame"),
    )


def _cepstrogram_decode(common: dict[str, Any], state: Mapping[str, Any]) -> BaseFrame[Any]:
    from wandas.frames.cepstrogram import CepstrogramFrame

    n_fft, hop_length, win_length, window = _validate_cepstrogram_constructor_state(state, common["data"])
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
        "fmin": typed.fmin,
        "fmax": typed.fmax,
        "n": typed.n,
        "G": typed.G,
        "fr": typed.fr,
    }


def _validated_noct_constructor_state(state: Mapping[str, Any]) -> tuple[float, float, int, int, int]:
    """Validate the exact NOct constructor state loaded from WDF."""
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
    return fmin, fmax, n, reference_band, reference_frequency


def _validate_noct_constructor_state(state: Mapping[str, Any], data: DaArray) -> tuple[float, float, int, int, int]:
    del data
    return _validated_noct_constructor_state(state)


def _noct_decode(common: dict[str, Any], state: Mapping[str, Any]) -> BaseFrame[Any]:
    from wandas.frames.noct import NOctFrame

    fmin, fmax, n, reference_band, reference_frequency = _validate_noct_constructor_state(state, common["data"])
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
    return {"bark_axis": typed.bark_axis.tolist(), "overlap": typed.overlap}


def _validated_roughness_constructor_state(state: Mapping[str, Any]) -> tuple[np.ndarray[Any, Any], float]:
    """Validate the exact Roughness constructor state loaded from WDF."""
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
    return bark_axis, overlap


def _validate_roughness_constructor_state(
    state: Mapping[str, Any], data: DaArray
) -> tuple[np.ndarray[Any, Any], float]:
    bark_axis, overlap = _validated_roughness_constructor_state(state)
    if int(data.shape[-2]) != len(bark_axis):
        raise _invalid_constructor_value(
            "RoughnessFrame",
            "bark_axis",
            bark_axis.tolist(),
            f"one value for each of the {data.shape[-2]} stored Bark bins",
        )
    return bark_axis, overlap


def _roughness_decode(common: dict[str, Any], state: Mapping[str, Any]) -> BaseFrame[Any]:
    from wandas.frames.roughness import RoughnessFrame

    bark_axis, overlap = _validate_roughness_constructor_state(state, common["data"])
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
        FrameCodec(
            ChannelFrame,
            _channel_state,
            _validate_channel_constructor_state,
            _channel_decode,
            "real",
            frozenset({2}),
        ),
        FrameCodec(
            SpectralFrame,
            _spectral_state,
            _validate_spectral_constructor_state,
            _spectral_decode,
            "numeric",
            frozenset({2}),
        ),
        FrameCodec(
            SpectrogramFrame,
            _spectrogram_state,
            _validate_spectrogram_constructor_state,
            _spectrogram_decode,
            "numeric",
            frozenset({3}),
        ),
        FrameCodec(
            CepstralFrame,
            _cepstral_state,
            _validate_cepstral_constructor_state,
            _cepstral_decode,
            "real",
            frozenset({2}),
        ),
        FrameCodec(
            CepstrogramFrame,
            _cepstrogram_state,
            _validate_cepstrogram_constructor_state,
            _cepstrogram_decode,
            "real",
            frozenset({3}),
        ),
        FrameCodec(
            NOctFrame,
            _noct_state,
            _validate_noct_constructor_state,
            _noct_decode,
            "real",
            frozenset({2, 3}),
        ),
        FrameCodec(
            RoughnessFrame,
            _roughness_state,
            _validate_roughness_constructor_state,
            _roughness_decode,
            "real",
            frozenset({2, 3}),
        ),
    )


def _codecs_by_type() -> dict[type[BaseFrame[Any]], FrameCodec]:
    """Index codecs by exact Python type for the save boundary."""
    return {codec.frame_type: codec for codec in _codecs()}


def _codecs_by_name() -> dict[str, FrameCodec]:
    """Index codecs by stable schema name for the load boundary."""
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


def encode_frame(frame: BaseFrame[Any]) -> tuple[str, dict[str, Any]]:
    """Return the exact built-in type name and validated constructor state."""
    codec = _codec_for_frame(frame)
    _validate_codec_tensor(codec, frame._data)
    constructor = codec.encode(frame)
    codec.validate_constructor(constructor, frame._data)
    return codec.frame_type.__name__, constructor


def decode_frame(
    frame_type: str,
    constructor: Mapping[str, Any],
    *,
    data: DaArray,
    common: dict[str, Any],
    stored_dims: tuple[str, ...],
) -> BaseFrame[Any]:
    """Reconstruct and validate one registered typed Frame.

    Constructor validation happens before the stored semantic xarray dimensions are
    compared with those generated by the concrete Frame. This prevents a valid tensor
    from being interpreted with a different axis meaning.
    """
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
        _validate_codec_tensor(codec, data)
        frame = codec.decode({**common, "data": data}, constructor)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Invalid typed WDF Frame state\n"
            f"  Frame type: {frame_type}\n"
            f"  Cause: {exc}\n"
            "Resave the file with a compatible Wandas version."
        ) from exc
    if frame._xr.dims != stored_dims:
        raise ValueError(
            "WDF semantic dimensions do not match reconstructed Frame\n"
            f"  Stored: {list(stored_dims)}\n"
            f"  Reconstructed: {list(frame._xr.dims)}\n"
            "The file is malformed or uses an incompatible Frame contract."
        )
    return frame


def _coordinate_spacing(frame: BaseFrame[Any], name: str) -> float | None:
    """Infer a dimension coordinate's canonical spacing from the Frame."""
    coordinate = np.asarray(frame._xr.coords[name].values, dtype=float)
    if len(coordinate) < 2:
        return None
    return float(coordinate[1] - coordinate[0])


def _validate_coordinate_values(
    frame: BaseFrame[Any],
    name: str,
    values: np.ndarray[Any, Any],
    expected_length: int,
) -> np.ndarray[Any, Any]:
    """Validate one external coordinate against its Frame-domain sampling grid.

    WDF coordinates originate outside Python's type system, so their rank, dtype,
    length, ordering, and domain grid are checked before they reach xarray.
    """
    if values.ndim != 1 or len(values) != expected_length:
        raise ValueError(
            "WDF coordinate length does not match Frame data\n"
            f"  Coordinate: {name!r}\n"
            f"  Got: {values.shape}\n"
            f"  Expected length: {expected_length}\n"
            "Resave the file with a compatible Wandas version."
        )
    if not np.issubdtype(values.dtype, np.floating):
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
    if len(normalized) > 1:
        differences = np.diff(normalized)
        if not (np.all(differences > 0) or np.all(differences < 0)):
            raise ValueError(
                "Invalid WDF coordinate ordering\n"
                f"  Coordinate: {name!r}\n"
                "  Expected: strictly monotonic represented-axis values\n"
                "Resave an ordered forward or reversed slice of the Frame axis."
            )
    spacing = _coordinate_spacing(frame, name)
    if len(normalized) and spacing is not None:
        scaled = normalized / spacing
        on_grid = np.allclose(scaled, np.rint(scaled), rtol=0.0, atol=1e-7)
        steps = np.diff(normalized) / spacing
        consecutive = np.allclose(np.abs(steps), 1.0, rtol=0.0, atol=1e-7)
        if not on_grid or not consecutive:
            raise ValueError(
                "Invalid WDF coordinate sampling grid\n"
                f"  Coordinate: {name!r}\n"
                f"  Expected spacing: consecutive values on the {spacing} grid\n"
                "Resave a valid represented-axis slice for this Frame domain."
            )
    return normalized


def frame_dimension_coordinates(frame: BaseFrame[Any]) -> dict[str, np.ndarray[Any, Any]]:
    """Extract persisted represented-axis coordinates from the internal DataArray.

    xarray also supports scalar, auxiliary, and multidimensional coordinates. WDF
    persists only a one-dimensional coordinate whose sole dimension has the same
    name, because only that form maps unambiguously to one tensor axis. Channel
    coordinates use the dedicated channel metadata schema instead.
    """
    coordinates: dict[str, np.ndarray[Any, Any]] = {}
    for dim in frame._xr.dims:
        if dim in {"channel", "frequency", "time"} or dim not in frame._xr.coords:
            continue
        coordinate = frame._xr.coords[dim]

        # ``coordinate.dims == (dim,)`` distinguishes a true dimension coordinate
        # from xarray auxiliary coordinates that merely depend on this dimension.
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
    """Restore represented axes after the typed Frame constructor has run.

    The constructor first establishes canonical dimensions and coordinate defaults.
    Stored values are then validated against that concrete Frame before replacing its
    represented-axis coordinates.
    """
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

        # xarray's ``assign_coords`` returns a new DataArray instead of mutating the
        # existing one, so the internal container must be rebound explicitly.
        frame._xr = frame._xr.assign_coords({name: (name, validated.copy())})


__all__ = [
    "decode_frame",
    "encode_frame",
    "frame_dimension_coordinates",
    "restore_frame_coordinates",
]
