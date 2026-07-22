import copy
import json
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from wandas.utils.util import unit_to_ref


class _RefUnset:
    pass


class _ExtraUnset:
    pass


_REF_UNSET = _RefUnset()
_EXTRA_UNSET = _ExtraUnset()


@dataclass(frozen=True, slots=True, init=False)
class ChannelCalibration:
    """Immutable calibration applied to one raw signal channel.

    ``factor`` converts raw samples to physical values. ``unit`` and ``ref``
    describe that physical domain. A missing reference is inferred from the
    unit (for example, ``Pa`` uses ``2e-5``).
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
        """Return a replacement physical unit using the legacy unit/ref rule."""
        return self._with_unit(unit)

    def with_ref(self, ref: float) -> "ChannelCalibration":
        """Return a replacement reference preserving factor and unit."""
        return self._with_ref(ref)

    def _with_unit(self, unit: str) -> "ChannelCalibration":
        """Return a private domain replacement using the legacy unit/ref rule."""
        if unit:
            return ChannelCalibration(factor=self.factor, unit=unit)
        return ChannelCalibration(factor=self.factor, unit="", ref=self.ref)

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
        if not isinstance(unit, str):
            raise TypeError("ChannelMetadata unit must be a string")
        if not isinstance(ref, _RefUnset) and (isinstance(ref, bool) or not isinstance(ref, numbers.Real)):
            raise TypeError("ChannelMetadata ref must be a number")
        if extra is _EXTRA_UNSET:
            extra_value = {}
        elif isinstance(extra, dict):
            extra_value = copy.deepcopy(extra)
        else:
            extra_value = extra

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

        object.__setattr__(self, "label", label)
        object.__setattr__(self, "calibration", calibration_value)
        object.__setattr__(self, "extra", extra_value)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not isinstance(self.label, str):
            raise TypeError("ChannelMetadata label must be a string")
        if not isinstance(self.calibration, ChannelCalibration):
            raise TypeError("ChannelMetadata calibration must be a ChannelCalibration")
        if not isinstance(self.extra, dict):
            raise TypeError("ChannelMetadata extra must be a dictionary")
        object.__setattr__(self, "_initialized", True)

    @property
    def unit(self) -> str:
        """Physical unit owned by :attr:`calibration`."""
        return self.calibration.unit

    @unit.setter
    def unit(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("ChannelMetadata unit must be a string")
        try:
            current = object.__getattribute__(self, "calibration")
        except AttributeError:
            current = ChannelCalibration()
        self.calibration = current._with_unit(value)

    @property
    def ref(self) -> float:
        """Level reference owned by :attr:`calibration`."""
        return float(self.calibration.ref)

    @ref.setter
    def ref(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError("ChannelMetadata ref must be a number")
        try:
            current = object.__getattribute__(self, "calibration")
        except AttributeError:
            current = ChannelCalibration()
        self.calibration = current._with_ref(value)

    def __setattr__(self, name: str, value: Any) -> None:
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
