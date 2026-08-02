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
    if not isinstance(state, Mapping):
        raise ValueError(
            "Invalid WDF Frame constructor state\n"
            f"  Frame type: {frame_type}\n"
            f"  Got: {type(state).__name__}\n"
            "  Expected: a JSON object\n"
            "Resave the file with a compatible Wandas version."
        )
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
    expected = {"n_fft", "hop_length", "win_length", "window"}
    _require_fields(state, expected, "SpectrogramFrame")
    n_fft = _positive_integer(state, "n_fft", "SpectrogramFrame")
    hop_length = _positive_integer(state, "hop_length", "SpectrogramFrame")
    win_length = _positive_integer(state, "win_length", "SpectrogramFrame")
    expected_bins = n_fft // 2 + 1
    if int(data.shape[-2]) != expected_bins:
        raise _invalid_constructor_value(
            "SpectrogramFrame",
            "n_fft",
            n_fft,
            f"a value producing the {data.shape[-2]} stored frequency bins",
        )
    if win_length > n_fft:
        raise _invalid_constructor_value("SpectrogramFrame", "win_length", win_length, f"a value <= n_fft ({n_fft})")
    if hop_length > win_length:
        raise _invalid_constructor_value(
            "SpectrogramFrame",
            "hop_length",
            hop_length,
            f"a value <= win_length ({win_length})",
        )
    return n_fft, hop_length, win_length, _nonblank_string(state, "window", "SpectrogramFrame")


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


_PAIRWISE_STATE_FIELDS = frozenset(
    {"n_fft", "window", "frequency_indices", "source_channel_count", "source_channel_ids", "pairs"}
)
_PAIRWISE_SCALING_FIELDS = frozenset({"scaling"})
_PAIRWISE_TRANSFER_FIELDS = frozenset({"scaling", "denominator_role", "definition"})
_PAIR_STATE_FIELDS = frozenset({"row_id", "pair_index", "output", "input", "domain", "display_label"})
_PAIR_ROLE_FIELDS = frozenset({"index", "source_id", "label", "unit", "reference"})
_PAIR_DOMAIN_FIELDS = frozenset({"unit", "reference"})
_VALID_SCALINGS = frozenset({"spectrum", "density"})
_VALID_TRANSFER_DEFINITIONS = {
    "input": "canonical_input_denominator",
    "output": "legacy_output_denominator",
}


def _nonnegative_integer_value(value: object, field: str, frame_type: str) -> int:
    """Return one strict non-negative JSON integer."""
    if type(value) is not int or value < 0:
        raise _invalid_constructor_value(frame_type, field, value, "a non-negative JSON integer")
    return value


def _positive_reference_value(value: object, field: str, frame_type: str) -> float:
    """Return one strict positive finite reference value."""
    if type(value) not in {int, float}:
        raise _invalid_constructor_value(frame_type, field, value, "a positive finite JSON number")
    normalized = float(cast(int | float, value))
    if not np.isfinite(normalized) or normalized <= 0:
        raise _invalid_constructor_value(frame_type, field, value, "a positive finite JSON number")
    return normalized


def _string_value(
    value: object,
    field: str,
    frame_type: str,
    *,
    nonblank: bool = False,
    canonical: bool = False,
) -> str:
    """Validate one JSON string without using it as pair identity implicitly."""
    if not isinstance(value, str):
        raise _invalid_constructor_value(frame_type, field, value, "a JSON string")
    if nonblank and not value.strip():
        raise _invalid_constructor_value(frame_type, field, value, "a non-blank JSON string")
    if canonical and value != value.strip():
        raise _invalid_constructor_value(
            frame_type,
            field,
            value,
            "a JSON string without surrounding whitespace",
        )
    return value


def _string_vector(
    value: object,
    field: str,
    frame_type: str,
    expected_length: int,
    *,
    unique: bool = False,
    nonblank: bool = False,
    canonical: bool = False,
) -> tuple[str, ...]:
    """Validate one exact JSON string vector."""
    if not isinstance(value, list) or len(value) != expected_length:
        raise _invalid_constructor_value(
            frame_type,
            field,
            value,
            f"a JSON string list of length {expected_length}",
        )
    result = tuple(
        _string_value(
            item,
            f"{field}[{index}]",
            frame_type,
            nonblank=nonblank,
            canonical=canonical,
        )
        for index, item in enumerate(value)
    )
    if unique and len(set(result)) != len(result):
        raise _invalid_constructor_value(frame_type, field, value, "a list of unique JSON strings")
    return result


def _pair_role_state(role: Any) -> dict[str, Any]:
    """Encode one typed source role using JSON scalar values only."""
    return {
        "index": int(role.index),
        "source_id": str(role.channel_id),
        "label": str(role.label),
        "unit": str(role.unit),
        "reference": float(role.reference),
    }


def _pair_record_state(record: Any) -> dict[str, Any]:
    """Encode one immutable selected pair row without history or label inference."""
    pair = record.pair
    return {
        "row_id": str(record.row_id),
        "pair_index": int(pair.pair_index),
        "output": _pair_role_state(pair.output),
        "input": _pair_role_state(pair.input),
        "domain": {
            "unit": str(record.domain.unit),
            "reference": float(record.domain.reference),
        },
        "display_label": str(record.display_label),
    }


def _pairwise_state(frame: BaseFrame[Any], *, fields: set[str]) -> dict[str, Any]:
    """Encode typed pair state shared by the three dedicated Frame codecs."""
    typed = cast(Any, frame)
    state: dict[str, Any] = {
        "n_fft": int(typed.n_fft),
        "window": str(typed.window),
        "frequency_indices": [int(value) for value in typed._frequency_indices],
        "source_channel_count": int(typed.n_source_channels),
        "source_channel_ids": [str(value) for value in typed.source_channel_ids],
        "pairs": [_pair_record_state(record) for record in typed.pair_state],
    }
    if "scaling" in fields:
        state["scaling"] = str(typed.scaling)
    if "denominator_role" in fields:
        state["denominator_role"] = str(typed.denominator_role)
        state["definition"] = str(typed.definition)
    return state


def _pair_role_from_state(
    value: object,
    *,
    field: str,
    frame_type: str,
    source_channel_count: int,
    source_channel_ids: tuple[str, ...],
) -> Any:
    """Validate and reconstruct one immutable typed source role."""
    if not isinstance(value, Mapping):
        raise _invalid_constructor_value(frame_type, field, value, "a JSON object")
    value = cast(Mapping[str, Any], value)
    _require_fields(value, set(_PAIR_ROLE_FIELDS), frame_type)
    index = _nonnegative_integer_value(value["index"], f"{field}.index", frame_type)
    if index >= source_channel_count:
        raise _invalid_constructor_value(
            frame_type,
            f"{field}.index",
            index,
            f"an index in [0, {source_channel_count})",
        )
    source_id = _string_value(
        value["source_id"],
        f"{field}.source_id",
        frame_type,
        nonblank=True,
        canonical=False,
    )
    if source_id != source_channel_ids[index]:
        raise _invalid_constructor_value(
            frame_type,
            f"{field}.source_id",
            source_id,
            f"source_channel_ids[{index}] ({source_channel_ids[index]!r})",
        )
    label = _string_value(value["label"], f"{field}.label", frame_type)
    unit = _string_value(value["unit"], f"{field}.unit", frame_type, canonical=True)
    reference = _positive_reference_value(value["reference"], f"{field}.reference", frame_type)

    from wandas.processing.spectral_contracts import SpectralChannelRole

    return SpectralChannelRole(
        index=index,
        label=label,
        unit=unit,
        reference=reference,
        channel_id=source_id,
    )


def _pair_domain_from_state(value: object, *, field: str, frame_type: str) -> Any:
    """Validate and reconstruct one derived unit/reference domain."""
    if not isinstance(value, Mapping):
        raise _invalid_constructor_value(frame_type, field, value, "a JSON object")
    value = cast(Mapping[str, Any], value)
    _require_fields(value, set(_PAIR_DOMAIN_FIELDS), frame_type)
    unit = _string_value(value["unit"], f"{field}.unit", frame_type, canonical=True)
    reference = _positive_reference_value(value["reference"], f"{field}.reference", frame_type)

    from wandas.processing.spectral_contracts import DerivedSpectralDomain

    return DerivedSpectralDomain(unit=unit, reference=reference)


def _validated_pairwise_constructor_state(
    state: Mapping[str, Any],
    data: DaArray,
    *,
    frame_type: str,
    quantity: Literal["coherence", "csd", "transfer"],
) -> tuple[int, str, tuple[int, ...], tuple[str, ...], tuple[Any, ...], str | None, str | None]:
    """Validate and decode a dedicated pairwise Frame constructor state."""
    if quantity == "coherence":
        expected_fields = set(_PAIRWISE_STATE_FIELDS)
    elif quantity == "csd":
        expected_fields = set(_PAIRWISE_STATE_FIELDS | _PAIRWISE_SCALING_FIELDS)
    else:
        expected_fields = set(_PAIRWISE_STATE_FIELDS | _PAIRWISE_TRANSFER_FIELDS)
    _require_fields(state, expected_fields, frame_type)

    n_fft = _positive_integer(state, "n_fft", frame_type)
    window = _nonblank_string(state, "window", frame_type)
    if window != window.strip():
        raise _invalid_constructor_value(
            frame_type,
            "window",
            window,
            "a non-blank JSON string without surrounding whitespace",
        )
    complete_frequency_count = n_fft // 2 + 1
    frequency_indices = state["frequency_indices"]
    if not isinstance(frequency_indices, list) or not frequency_indices:
        raise _invalid_constructor_value(
            frame_type,
            "frequency_indices",
            frequency_indices,
            "a non-empty JSON list of canonical rfft bin indices",
        )
    normalized_frequency_indices: list[int] = []
    seen_frequency_indices: set[int] = set()
    for index, value in enumerate(frequency_indices):
        bin_index = _nonnegative_integer_value(value, f"frequency_indices[{index}]", frame_type)
        if bin_index >= complete_frequency_count:
            raise _invalid_constructor_value(
                frame_type,
                f"frequency_indices[{index}]",
                bin_index,
                f"an index in [0, {complete_frequency_count})",
            )
        if bin_index in seen_frequency_indices:
            raise _invalid_constructor_value(
                frame_type,
                f"frequency_indices[{index}]",
                bin_index,
                "a unique canonical rfft bin index",
            )
        seen_frequency_indices.add(bin_index)
        normalized_frequency_indices.append(bin_index)
    if len(normalized_frequency_indices) != int(data.shape[-1]):
        raise _invalid_constructor_value(
            frame_type,
            "frequency_indices",
            frequency_indices,
            f"a list with exactly {data.shape[-1]} represented frequency bins",
        )
    source_count = _positive_integer(state, "source_channel_count", frame_type)
    source_ids = _string_vector(
        state["source_channel_ids"],
        "source_channel_ids",
        frame_type,
        source_count,
        unique=True,
        nonblank=True,
        canonical=False,
    )

    raw_pairs = state["pairs"]
    data_rows = int(data.shape[0])
    if not isinstance(raw_pairs, list) or len(raw_pairs) != data_rows or not raw_pairs:
        raise _invalid_constructor_value(
            frame_type,
            "pairs",
            raw_pairs,
            f"a non-empty JSON list with exactly {data_rows} selected pair rows",
        )
    if len(raw_pairs) > source_count * source_count:
        raise _invalid_constructor_value(
            frame_type,
            "pairs",
            len(raw_pairs),
            f"at most {source_count * source_count} unique pair rows",
        )

    scaling: str | None = None
    denominator_role: str | None = None
    if quantity in {"csd", "transfer"}:
        scaling_value = state["scaling"]
        if not isinstance(scaling_value, str) or scaling_value not in _VALID_SCALINGS:
            raise _invalid_constructor_value(
                frame_type,
                "scaling",
                scaling_value,
                "one of the JSON strings 'spectrum' or 'density'",
            )
        scaling = scaling_value
    if quantity == "transfer":
        denominator_value = state["denominator_role"]
        if not isinstance(denominator_value, str) or denominator_value not in _VALID_TRANSFER_DEFINITIONS:
            raise _invalid_constructor_value(
                frame_type,
                "denominator_role",
                denominator_value,
                "one of the JSON strings 'input' or 'output'",
            )
        denominator_role = cast(str, denominator_value)
        definition = state["definition"]
        expected_definition = _VALID_TRANSFER_DEFINITIONS[denominator_role]
        if definition != expected_definition:
            raise _invalid_constructor_value(
                frame_type,
                "definition",
                definition,
                f"the exact JSON string {expected_definition!r} for denominator_role={denominator_role!r}",
            )

    from wandas.frames.pairwise import SpectralPairState
    from wandas.processing.spectral_contracts import (
        OrderedSpectralPair,
        derive_coherence_domain,
        derive_csd_domain,
        derive_transfer_domain,
    )

    records: list[Any] = []
    seen_pair_indices: set[int] = set()
    seen_row_ids: set[str] = set()
    for row, raw_pair in enumerate(raw_pairs):
        field = f"pairs[{row}]"
        if not isinstance(raw_pair, Mapping):
            raise _invalid_constructor_value(frame_type, field, raw_pair, "a JSON object")
        raw_pair = cast(Mapping[str, Any], raw_pair)
        _require_fields(raw_pair, set(_PAIR_STATE_FIELDS), frame_type)
        row_id = _string_value(raw_pair["row_id"], f"{field}.row_id", frame_type, nonblank=True, canonical=True)
        if row_id in seen_row_ids:
            raise _invalid_constructor_value(
                frame_type,
                f"{field}.row_id",
                row_id,
                "a unique pair row ID; duplicate IDs are invalid",
            )
        seen_row_ids.add(row_id)
        pair_index = _nonnegative_integer_value(raw_pair["pair_index"], f"{field}.pair_index", frame_type)
        if pair_index >= source_count * source_count:
            raise _invalid_constructor_value(
                frame_type,
                f"{field}.pair_index",
                pair_index,
                f"an index in [0, {source_count * source_count})",
            )
        if pair_index in seen_pair_indices:
            raise _invalid_constructor_value(
                frame_type,
                f"{field}.pair_index",
                pair_index,
                "a unique pair index; duplicate pairs are invalid",
            )
        seen_pair_indices.add(pair_index)

        output = _pair_role_from_state(
            raw_pair["output"],
            field=f"{field}.output",
            frame_type=frame_type,
            source_channel_count=source_count,
            source_channel_ids=source_ids,
        )
        input_ = _pair_role_from_state(
            raw_pair["input"],
            field=f"{field}.input",
            frame_type=frame_type,
            source_channel_count=source_count,
            source_channel_ids=source_ids,
        )
        pair = OrderedSpectralPair(output=output, input=input_, n_channels=source_count)
        if pair.pair_index != pair_index:
            raise _invalid_constructor_value(
                frame_type,
                f"{field}.pair_index",
                pair_index,
                f"output.index * source_channel_count + input.index ({pair.pair_index})",
            )
        domain = _pair_domain_from_state(raw_pair["domain"], field=f"{field}.domain", frame_type=frame_type)
        display_label = _string_value(
            raw_pair["display_label"],
            f"{field}.display_label",
            frame_type,
            nonblank=True,
            canonical=True,
        )

        if quantity == "coherence":
            expected_domain = derive_coherence_domain()
        elif quantity == "csd":
            expected_domain = derive_csd_domain(pair, cast(str, scaling))
        else:
            expected_domain = derive_transfer_domain(pair, cast(Any, denominator_role))
        if domain != expected_domain:
            raise _invalid_constructor_value(
                frame_type,
                f"{field}.domain",
                {"unit": domain.unit, "reference": domain.reference},
                f"the domain derived from the typed pair ({expected_domain!r})",
            )
        records.append(
            SpectralPairState(
                pair=pair,
                domain=domain,
                row_id=row_id,
                display_label=display_label,
            )
        )

    return n_fft, window, tuple(normalized_frequency_indices), source_ids, tuple(records), scaling, denominator_role


def _validate_pairwise_common(
    common: Mapping[str, Any],
    records: tuple[Any, ...],
    *,
    frame_type: str,
) -> None:
    """Ensure common WDF row metadata agrees with typed pair state."""
    channel_ids = common.get("channel_ids")
    expected_ids = [record.row_id for record in records]
    if not isinstance(channel_ids, list) or channel_ids != expected_ids:
        raise ValueError(
            "Invalid WDF pair row identity\n"
            f"  Frame type: {frame_type}\n"
            f"  Stored channel IDs: {channel_ids!r}\n"
            f"  Expected typed pair row IDs: {expected_ids!r}\n"
            "Pair row order and identity must be preserved explicitly."
        )
    channel_metadata = common.get("channel_metadata")
    if not isinstance(channel_metadata, list) or len(channel_metadata) != len(records):
        raise ValueError(
            "Invalid WDF pair channel metadata\n"
            f"  Frame type: {frame_type}\n"
            f"  Got: {channel_metadata!r}\n"
            f"  Expected one metadata record per selected pair ({len(records)})"
        )
    from wandas.core.metadata import ChannelMetadata

    for row, (metadata, record) in enumerate(zip(channel_metadata, records, strict=True)):
        if not isinstance(metadata, ChannelMetadata):
            raise ValueError(
                "Invalid WDF pair channel metadata\n"
                f"  Frame type: {frame_type}\n"
                f"  Row: {row}\n"
                f"  Got: {type(metadata).__name__}\n"
                "  Expected: ChannelMetadata"
            )
        if (
            metadata.unit != record.domain.unit
            or float(metadata.ref) != float(record.domain.reference)
            or float(metadata.calibration.factor) != 1.0
        ):
            raise ValueError(
                "Invalid WDF pair channel metadata\n"
                f"  Frame type: {frame_type}\n"
                f"  Row: {row}\n"
                "  Stored unit/reference metadata does not agree with the typed pair domain\n"
                "  Pair domain and consumed calibration must be reconstructed from constructor state."
            )


def _validate_coherence_constructor_state(state: Mapping[str, Any], data: DaArray) -> object:
    return _validated_pairwise_constructor_state(state, data, frame_type="CoherenceFrame", quantity="coherence")


def _coherence_state(frame: BaseFrame[Any]) -> dict[str, Any]:
    return _pairwise_state(frame, fields=set())


def _coherence_decode(common: dict[str, Any], state: Mapping[str, Any]) -> BaseFrame[Any]:
    from wandas.frames.pairwise import CoherenceFrame

    n_fft, window, frequency_indices, source_ids, records, _, _ = _validated_pairwise_constructor_state(
        state,
        common["data"],
        frame_type="CoherenceFrame",
        quantity="coherence",
    )
    _validate_pairwise_common(common, records, frame_type="CoherenceFrame")
    return CoherenceFrame(
        **common,
        n_fft=n_fft,
        window=window,
        frequency_indices=frequency_indices,
        pair_state=records,
        source_channel_ids=source_ids,
    )


def _validate_csd_constructor_state(state: Mapping[str, Any], data: DaArray) -> object:
    return _validated_pairwise_constructor_state(state, data, frame_type="CrossSpectralFrame", quantity="csd")


def _csd_state(frame: BaseFrame[Any]) -> dict[str, Any]:
    return _pairwise_state(frame, fields=set(_PAIRWISE_SCALING_FIELDS))


def _csd_decode(common: dict[str, Any], state: Mapping[str, Any]) -> BaseFrame[Any]:
    from wandas.frames.pairwise import CrossSpectralFrame

    n_fft, window, frequency_indices, source_ids, records, scaling, _ = _validated_pairwise_constructor_state(
        state,
        common["data"],
        frame_type="CrossSpectralFrame",
        quantity="csd",
    )
    _validate_pairwise_common(common, records, frame_type="CrossSpectralFrame")
    return CrossSpectralFrame(
        **common,
        n_fft=n_fft,
        window=window,
        frequency_indices=frequency_indices,
        pair_state=records,
        source_channel_ids=source_ids,
        scaling=cast(Any, scaling),
    )


def _validate_transfer_constructor_state(state: Mapping[str, Any], data: DaArray) -> object:
    return _validated_pairwise_constructor_state(state, data, frame_type="TransferFunctionFrame", quantity="transfer")


def _transfer_state(frame: BaseFrame[Any]) -> dict[str, Any]:
    return _pairwise_state(frame, fields=set(_PAIRWISE_TRANSFER_FIELDS))


def _transfer_decode(common: dict[str, Any], state: Mapping[str, Any]) -> BaseFrame[Any]:
    from wandas.frames.pairwise import TransferFunctionFrame

    n_fft, window, frequency_indices, source_ids, records, scaling, denominator_role = (
        _validated_pairwise_constructor_state(
            state,
            common["data"],
            frame_type="TransferFunctionFrame",
            quantity="transfer",
        )
    )
    _validate_pairwise_common(common, records, frame_type="TransferFunctionFrame")
    return TransferFunctionFrame(
        **common,
        n_fft=n_fft,
        window=window,
        frequency_indices=frequency_indices,
        pair_state=records,
        source_channel_ids=source_ids,
        scaling=cast(Any, scaling),
        denominator_role=cast(Any, denominator_role),
    )


@lru_cache(maxsize=1)
def _codecs() -> tuple[FrameCodec, ...]:
    """Build the registry lazily to keep Frame imports cycle-free."""
    from wandas.frames.cepstral import CepstralFrame
    from wandas.frames.cepstrogram import CepstrogramFrame
    from wandas.frames.channel import ChannelFrame
    from wandas.frames.noct import NOctFrame
    from wandas.frames.pairwise import CoherenceFrame, CrossSpectralFrame, TransferFunctionFrame
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
            CoherenceFrame,
            _coherence_state,
            _validate_coherence_constructor_state,
            _coherence_decode,
            "real",
            frozenset({2}),
        ),
        FrameCodec(
            CrossSpectralFrame,
            _csd_state,
            _validate_csd_constructor_state,
            _csd_decode,
            "complex",
            frozenset({2}),
        ),
        FrameCodec(
            TransferFunctionFrame,
            _transfer_state,
            _validate_transfer_constructor_state,
            _transfer_decode,
            "complex",
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
