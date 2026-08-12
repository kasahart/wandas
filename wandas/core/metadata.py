import copy
import json
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast, overload

import numpy as np
from numpy.typing import ArrayLike

from wandas.utils.types import NDArrayReal
from wandas.utils.util import DB_FLOOR, PA_REFERENCE, unit_to_ref


class _RefUnset:
    pass


class _ExtraUnset:
    pass


_REF_UNSET = _RefUnset()
_EXTRA_UNSET = _ExtraUnset()
_REFERENCE_MATCH_ABS_TOL = 1e-15


def _is_twenty_micro_reference(value: float) -> bool:
    """Return whether a reference is numerically equivalent to 20 micro-units."""
    return math.isclose(value, PA_REFERENCE, rel_tol=0.0, abs_tol=_REFERENCE_MATCH_ABS_TOL)


def _format_reference_label(value: float, unit: str) -> str:
    """Format one stable human-readable linear reference."""
    if unit and _is_twenty_micro_reference(value):
        return f"20 µ{unit}"
    formatted_value = "1" if value == 1.0 else f"{value:.6g}"
    return f"{formatted_value} {unit or 'input unit'}"


def _is_level_domain_unit(unit: str) -> bool:
    """Return whether *unit* describes values already expressed as levels."""
    return unit == "dB" or unit == "dBFS" or unit.startswith("dB ")


def _normalize_channel_label(value: object) -> str:
    """Return one validated channel label without coercing runtime input."""
    if not isinstance(value, str):
        raise TypeError("Channel label must be a string")
    return str(value)


def _validate_channel_extra(value: object) -> Mapping[str, Any]:
    """Validate channel extra without taking ownership."""
    if not isinstance(value, Mapping):
        raise TypeError("Channel extra must be a mapping")
    return cast(Mapping[str, Any], value)


def _snapshot_channel_extra(
    value: object,
    *,
    preserve_mapping_type: bool = False,
) -> dict[str, Any]:
    """Return an owned channel-extra mapping without retaining caller state."""
    normalized = _validate_channel_extra(value)
    if preserve_mapping_type and isinstance(value, dict):
        return cast(dict[str, Any], copy.deepcopy(value))
    return copy.deepcopy(dict(normalized))


@dataclass(frozen=True, slots=True, init=False)
class ChannelCalibration:
    """Immutable calibration applied to one raw signal channel.

    ``factor`` converts raw samples to physical values. ``unit`` and ``ref``
    describe that linear measurement domain. A missing reference is inferred
    from the unit (for example, ``Pa`` uses ``2e-5``). Canonically decoded
    audio uses the explicit ``FS`` unit with reference 1; an empty unit remains
    a generic dimensionless domain and is never inferred to be full scale.
    """

    factor: float = 1.0
    unit: str = ""
    ref: float = 1.0

    def __init__(
        self,
        factor: float = 1.0,
        unit: str = "",
        ref: float | _RefUnset = _REF_UNSET,
    ) -> None:
        """Normalize values and enforce the calibration contract."""
        if isinstance(factor, bool) or not isinstance(factor, numbers.Real):
            raise TypeError(
                "Invalid channel calibration factor\n"
                f"  Got: {type(factor).__name__} ({factor!r})\n"
                "  Expected: a positive finite number\n"
                "Pass the raw-to-physical scale supplied by the sensor certificate."
            )
        normalized_factor = float(factor)
        if not math.isfinite(normalized_factor) or normalized_factor <= 0:
            raise ValueError(
                "Invalid channel calibration factor\n"
                f"  Got: {normalized_factor!r}\n"
                "  Expected: a positive finite number\n"
                "Check the certificate or calibration table before configuring the channel."
            )
        if not isinstance(unit, str):
            raise TypeError(
                "Invalid channel calibration unit\n"
                f"  Got: {type(unit).__name__} ({unit!r})\n"
                "  Expected: a string\n"
                "Pass a physical unit such as 'Pa' or 'm/s^2', or use an empty string."
            )
        normalized_unit = unit.strip()
        if unit and not normalized_unit:
            raise ValueError(
                "Invalid channel calibration unit\n"
                "  Got: a whitespace-only string\n"
                "  Expected: a physical unit or an empty string\n"
                "Remove surrounding whitespace or use ''."
            )

        if isinstance(ref, _RefUnset):
            normalized_ref = float(unit_to_ref(normalized_unit)) if normalized_unit else 1.0
        else:
            if isinstance(ref, bool) or not isinstance(ref, numbers.Real):
                raise TypeError(
                    "Invalid channel calibration reference\n"
                    f"  Got: {type(ref).__name__} ({ref!r})\n"
                    "  Expected: a positive finite number\n"
                    "Pass the reference value used for level conversion."
                )
            normalized_ref = float(ref)
            if not math.isfinite(normalized_ref) or normalized_ref <= 0:
                raise ValueError(
                    "Invalid channel calibration reference\n"
                    f"  Got: {normalized_ref!r}\n"
                    "  Expected: a positive finite number\n"
                    "Check the physical reference for this channel."
                )

        object.__setattr__(self, "factor", normalized_factor)
        object.__setattr__(self, "unit", normalized_unit)
        object.__setattr__(self, "ref", normalized_ref)

    def with_factor(self, factor: float) -> "ChannelCalibration":
        """Return this physical domain with a replacement factor."""
        return ChannelCalibration(factor=factor, unit=self.unit, ref=self.ref)

    def with_unit(self, unit: str) -> "ChannelCalibration":
        """Replace the unit and reset ``ref`` to that unit's default.

        The calibration factor is preserved. Chain ``with_ref()`` after this
        method when the new unit needs a non-default reference.
        """
        return ChannelCalibration(factor=self.factor, unit=unit)

    def with_ref(self, ref: float) -> "ChannelCalibration":
        """Return a replacement reference preserving factor and unit."""
        return self._with_ref(ref)

    @property
    def level_reference(self) -> "LevelReference":
        """Return the reference-relative amplitude-level descriptor.

        Frame users normally obtain this value through
        ``frame.channels[index].level_reference`` so the descriptor stays tied
        to the channel whose already-calibrated amplitudes it interprets.

        Raises:
            ValueError: If this calibration already describes level-domain
                values rather than linear amplitudes.
        """
        if _is_level_domain_unit(self.unit):
            raise ValueError(
                "Level reference is unavailable for an already-level channel\n"
                f"  Got channel unit: {self.unit!r}\n"
                "  Expected: a linear amplitude domain\n"
                "Use the channel unit as display metadata, or retain the linear "
                "source Frame for further level conversion."
            )
        return LevelReference(
            reference_value=self.ref,
            reference_unit=self.unit,
        )

    def _with_unit(self, unit: str) -> "ChannelCalibration":
        """Return a private domain replacement using the legacy unit/ref rule."""
        return self.with_unit(unit)

    def _with_ref(self, ref: float) -> "ChannelCalibration":
        """Return a private reference replacement preserving factor and unit."""
        return ChannelCalibration(factor=self.factor, unit=self.unit, ref=ref)

    def to_dict(self) -> dict[str, float | str]:
        """Return a JSON-safe snapshot."""
        return {"factor": self.factor, "unit": self.unit, "ref": self.ref}

    @classmethod
    def from_dict(cls, value: object) -> "ChannelCalibration":
        """Decode an exact calibration snapshot."""
        if not isinstance(value, Mapping) or set(value) != {"factor", "unit", "ref"}:
            raise ValueError(
                "Invalid channel calibration snapshot\n"
                f"  Got: {value!r}\n"
                "  Expected: factor, unit, and ref fields\n"
                "Use ChannelCalibration.to_dict() when serializing calibration values."
            )
        return cls(
            factor=value["factor"],  # ty: ignore[invalid-argument-type]
            unit=value["unit"],  # ty: ignore[invalid-argument-type]
            ref=value["ref"],  # ty: ignore[invalid-argument-type]
        )


@dataclass(frozen=True, slots=True)
class LevelReference:
    """Immutable amplitude-level reference derived from channel metadata.

    Obtain this descriptor from ``frame.channels[index].level_reference``.
    ``reference_value`` and ``reference_unit`` describe the linear amplitude
    domain. ``unit`` and ``label`` provide canonical level text for UI, CSV,
    and report output. Labels use readable engineering-prefix text for a
    20-micro-unit reference and stable significant digits otherwise; the
    structured ``reference_value`` itself is never rounded. ``to_level()``
    accepts amplitudes that are already in this linear domain; it never applies
    a calibration factor.

    Zero and sub-floor amplitudes return ``minimum_level`` (-240 dB). Scalar
    input returns ``float`` and array-like input returns a NumPy array with the
    same broadcast shape. Pa relative to 20 µPa is labeled dB SPL. Only an
    explicit ``FS`` channel with reference 1 is labeled dBFS; an identity
    calibration with an empty unit remains generic relative dB.

    Args:
        reference_value: Positive finite linear amplitude reference.
        reference_unit: Linear unit, such as ``"Pa"``, ``"m/s^2"``, or the
            explicit canonical-audio marker ``"FS"``.
    """

    reference_value: float
    reference_unit: str

    def __post_init__(self) -> None:
        if isinstance(self.reference_value, bool) or not isinstance(self.reference_value, numbers.Real):
            raise TypeError("Level reference value must be a positive finite number")
        normalized_reference = float(self.reference_value)
        if not math.isfinite(normalized_reference) or normalized_reference <= 0.0:
            raise ValueError("Level reference value must be a positive finite number")
        if not isinstance(self.reference_unit, str):
            raise TypeError("Level reference unit must be a string")
        normalized_unit = self.reference_unit.strip()
        if self.reference_unit and not normalized_unit:
            raise ValueError("Level reference unit must not contain only whitespace")
        object.__setattr__(self, "reference_value", normalized_reference)
        object.__setattr__(self, "reference_unit", normalized_unit)

    @property
    def unit(self) -> str:
        """Canonical level unit: ``dBFS``, ``dB SPL``, or ``dB``."""
        if self.reference_unit == "FS" and self.reference_value == 1.0:
            return "dBFS"
        if self.reference_unit == "Pa" and _is_twenty_micro_reference(self.reference_value):
            return "dB SPL"
        return "dB"

    @property
    def label(self) -> str:
        """Canonical human-readable label including the linear reference."""
        if self.unit == "dBFS":
            return "dBFS"
        return f"{self.unit} re {_format_reference_label(self.reference_value, self.reference_unit)}"

    @property
    def minimum_level(self) -> float:
        """Finite level returned at or below the amplitude-ratio floor."""
        return 20.0 * math.log10(DB_FLOOR)

    @overload
    def to_level(self, amplitude: complex | float) -> float: ...

    @overload
    def to_level(self, amplitude: ArrayLike) -> NDArrayReal: ...

    def to_level(self, amplitude: ArrayLike | complex | float) -> NDArrayReal | float:
        """Convert already-calibrated amplitude to reference-relative level.

        Args:
            amplitude: Real or complex scalar or array-like amplitude. The
                magnitude is used, so signs and complex phases do not affect
                the result.

        Returns:
            A float for scalar input or a float64 NumPy array for array-like
            input. Array broadcasting follows NumPy rules.
        """
        from wandas.processing.weighting import _reference_level_db

        result = _reference_level_db(amplitude, self.reference_value)
        if np.isscalar(amplitude):
            return float(cast(float, np.asarray(result, dtype=np.float64).item()))
        return np.asarray(result, dtype=np.float64)


def _format_level_unit(calibration: ChannelCalibration) -> str:
    """Serialize a level unit with an exact, round-trippable reference."""
    reference = calibration.level_reference
    if reference.unit == "dBFS":
        return "dBFS"
    linear_unit = reference.reference_unit or "input unit"
    return f"{reference.unit} re {reference.reference_value!r} {linear_unit}"


def _format_level_unit_for_display(unit: str) -> str:
    """Format a serialized level-domain unit for a human-facing boundary."""
    if not isinstance(unit, str):
        return ""
    if unit == "dBFS":
        return unit
    if unit.startswith("dB SPL re "):
        level_unit = "dB SPL"
    elif unit.startswith("dB re "):
        level_unit = "dB"
    else:
        return unit
    reference_text = unit.removeprefix(f"{level_unit} re ")
    value_text, separator, reference_unit = reference_text.partition(" ")
    if not separator:
        return unit
    normalized_unit = "" if reference_unit == "input unit" else reference_unit
    try:
        reference = LevelReference(float(value_text), normalized_unit)
    except (TypeError, ValueError):
        return unit
    if reference.unit != level_unit:
        return unit
    return reference.label


@dataclass(init=False)
class ChannelMetadata:
    """Metadata for a single channel."""

    label: str = ""
    calibration: ChannelCalibration = field(default_factory=ChannelCalibration)
    extra: dict[str, Any] = field(default_factory=dict)
    _initialized: bool = field(default=False, init=False, repr=False)

    _MODEL_FIELDS = frozenset({"label", "calibration", "unit", "ref", "extra"})

    def __init__(
        self,
        label: str = "",
        unit: str = "",
        ref: float | _RefUnset = _REF_UNSET,
        extra: dict[str, Any] | _ExtraUnset = _EXTRA_UNSET,
        calibration: ChannelCalibration | None = None,
    ) -> None:
        if extra is _EXTRA_UNSET:
            extra_value = {}
        else:
            extra_value = _snapshot_channel_extra(extra, preserve_mapping_type=True)

        if calibration is not None:
            if not isinstance(calibration, ChannelCalibration):
                raise TypeError("ChannelMetadata calibration must be a ChannelCalibration")
            if unit or not isinstance(ref, _RefUnset):
                raise ValueError(
                    "Conflicting channel calibration metadata\n"
                    "  Got: calibration together with unit or ref\n"
                    "  Expected: one authoritative physical-domain definition\n"
                    "Pass calibration alone, or use the legacy unit/ref arguments."
                )
            calibration_value = calibration
        else:
            calibration_value = ChannelCalibration(factor=1.0, unit=unit, ref=ref)

        object.__setattr__(self, "label", _normalize_channel_label(label))
        object.__setattr__(self, "calibration", calibration_value)
        object.__setattr__(self, "extra", extra_value)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not isinstance(self.calibration, ChannelCalibration):
            raise TypeError("ChannelMetadata calibration must be a ChannelCalibration")
        object.__setattr__(self, "_initialized", True)

    @property
    def unit(self) -> str:
        """Linear measurement unit owned by :attr:`calibration`."""
        return self.calibration.unit

    @unit.setter
    def unit(self, value: str) -> None:
        try:
            current = object.__getattribute__(self, "calibration")
        except AttributeError:
            current = ChannelCalibration()
        self.calibration = current.with_unit(value)

    @property
    def ref(self) -> float:
        """Level reference owned by :attr:`calibration`."""
        return float(self.calibration.ref)

    @ref.setter
    def ref(self, value: float) -> None:
        try:
            current = object.__getattribute__(self, "calibration")
        except AttributeError:
            current = ChannelCalibration()
        self.calibration = current.with_ref(value)

    @property
    def level_reference(self) -> LevelReference:
        """Structured amplitude-level context for this channel.

        Values passed to :meth:`LevelReference.to_level` are interpreted in
        this channel's already-calibrated linear unit.

        Raises:
            ValueError: If this channel already contains level-domain values.
        """
        return self.calibration.level_reference

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "label":
            value = _normalize_channel_label(value)
        if name == "extra":
            value = _snapshot_channel_extra(value)
        if name == "calibration" and not isinstance(value, ChannelCalibration):
            raise TypeError("ChannelMetadata calibration must be a ChannelCalibration")
        object.__setattr__(self, name, value)

    def __getitem__(self, key: str) -> Any:
        """Provide dictionary-like behavior."""
        if key in self._MODEL_FIELDS:
            return getattr(self, key)
        return self.extra.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Provide dictionary-like behavior."""
        if key in self._MODEL_FIELDS:
            setattr(self, key, value)
        else:
            self.extra[key] = value

    def matches_query(self, query: dict[str, Any]) -> bool:
        """Check whether this channel matches all key-value pairs in query."""
        for key, expected in query.items():
            actual = getattr(self, key, None)
            if actual is None:
                actual = self.extra.get(key)
                if actual is None:
                    return False

            if hasattr(expected, "search") and callable(expected.search):
                if not (isinstance(actual, str) and expected.search(actual)):
                    return False
            elif actual != expected:
                return False
        return True

    def to_json(self) -> str:
        """Convert to JSON with one authoritative calibration object."""
        return json.dumps(
            {
                "label": self.label,
                "calibration": self.calibration.to_dict(),
                "extra": self.extra,
            },
            indent=4,
        )

    @classmethod
    def from_json(cls, json_data: str) -> "ChannelMetadata":
        """Convert from current or legacy JSON format."""
        try:
            data = json.loads(json_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid ChannelMetadata JSON: {e}") from e
        if not isinstance(data, dict):
            raise ValueError("ChannelMetadata JSON must decode to an object")
        try:
            if "calibration" in data:
                calibration_data = data.pop("calibration")
                calibration = ChannelCalibration.from_dict(calibration_data)
                if "unit" in data or "ref" in data:
                    raise ValueError("ChannelMetadata JSON must not combine calibration with legacy unit/ref fields")
                data["calibration"] = calibration
            return cls(**data)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid ChannelMetadata JSON object: {e}") from e
