"""Calibration values and numerical gain application."""

from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import numpy as np

from wandas.processing.base import AudioOperation, register_operation
from wandas.utils.types import NDArrayReal
from wandas.utils.util import unit_to_ref

CalibrationInput: TypeAlias = float | Sequence[float] | NDArrayReal


def _numeric_values(value: Any, *, error_heading: str) -> tuple[float, ...]:
    """Return one-dimensional finite numeric values with a stable error contract."""
    if isinstance(value, str | bytes | bool):
        raise TypeError(
            f"{error_heading}\n"
            f"  Got: {type(value).__name__} ({value!r})\n"
            "  Expected: a number or one-dimensional numeric sequence\n"
            "Pass one value for broadcast or one value per channel."
        )
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{error_heading}\n"
            f"  Got: {type(value).__name__} ({value!r})\n"
            "  Expected: a number or one-dimensional numeric sequence\n"
            "Pass one value for broadcast or one value per channel."
        ) from exc
    if array.dtype.kind not in {"f", "i", "u"}:
        raise TypeError(
            f"{error_heading}\n"
            f"  Got: {array.dtype} values\n"
            "  Expected: real numeric values\n"
            "Replace booleans, strings, complex values, or objects with real numbers."
        )
    array = array.astype(np.float64, copy=False)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(
            f"{error_heading}\n"
            f"  Got shape: {array.shape}\n"
            "  Expected: one or more values in a one-dimensional sequence\n"
            "Pass one value for broadcast or one value per channel."
        )
    return tuple(float(item) for item in array)


def _positive_values(value: Any, *, error_heading: str) -> tuple[float, ...]:
    """Return finite values greater than zero."""
    values = _numeric_values(value, error_heading=error_heading)
    if not all(np.isfinite(item) and item > 0.0 for item in values):
        raise ValueError(
            f"{error_heading}\n"
            f"  Got: {values}\n"
            "  Expected: finite values greater than zero\n"
            "Use a non-silent calibration measurement and a positive physical target."
        )
    return values


def _finite_values(value: Any, *, error_heading: str) -> tuple[float, ...]:
    """Return finite values, including zero and negative levels."""
    values = _numeric_values(value, error_heading=error_heading)
    if not all(np.isfinite(item) for item in values):
        raise ValueError(
            f"{error_heading}\n"
            f"  Got: {values}\n"
            "  Expected: finite numeric values\n"
            "Replace NaN or infinite level values before deriving calibration."
        )
    return values


def _broadcast_values(
    values: tuple[float, ...],
    channel_count: int,
    *,
    error_heading: str,
) -> tuple[float, ...]:
    """Broadcast one value or require an exact channel count."""
    if len(values) == 1:
        return values * channel_count
    if len(values) != channel_count:
        raise ValueError(
            f"{error_heading}\n"
            f"  Got: {len(values)} target values for {channel_count} measured channels\n"
            "  Expected: one target value or one value per measured channel\n"
            "Pass a scalar to broadcast or align the target sequence with the calibration channels."
        )
    return values


def _domain_reference(unit: str, ref: float | None) -> float:
    """Validate physical unit metadata and resolve its reference value."""
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError(
            "Invalid calibration unit\n"
            f"  Got: {unit!r}\n"
            "  Expected: a non-empty physical unit string\n"
            "Use 'Pa' for sound pressure or another unit matching the target RMS values."
        )
    normalized_unit = unit.strip()
    resolved = unit_to_ref(normalized_unit) if ref is None else ref
    if isinstance(resolved, bool) or not isinstance(resolved, numbers.Real):
        raise TypeError(
            "Invalid calibration reference\n"
            f"  Got: {type(resolved).__name__} ({resolved!r})\n"
            "  Expected: a finite positive number\n"
            "Use 20e-6 for sound pressure levels in Pa."
        )
    result = float(resolved)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(
            "Invalid calibration reference\n"
            f"  Got: {result}\n"
            "  Expected: a finite value greater than zero\n"
            "Use 20e-6 for sound pressure levels in Pa."
        )
    return result


@dataclass(frozen=True, slots=True)
class Calibration:
    """Immutable, auditable mapping from measured RMS to physical RMS.

    A calibration stores the RMS measured from one or more calibration-signal
    channels and the known physical RMS those channels represent. The gain is
    derived as ``target_rms / measured_rms`` and is therefore never duplicated
    as mutable state.

    Parameters
    ----------
    measured_rms : tuple of float
        Positive RMS values measured from the calibration signal.
    target_rms : tuple of float
        Positive physical RMS values represented by the calibration signal.
    unit : str
        Physical output unit, for example ``"Pa"``.
    ref : float
        Positive reference value used for level conversions. Sound pressure in
        Pa conventionally uses ``20e-6``.
    """

    measured_rms: tuple[float, ...]
    target_rms: tuple[float, ...]
    unit: str
    ref: float

    def __post_init__(self) -> None:
        """Normalize immutable fields and enforce the calibration invariant."""
        measured = _positive_values(self.measured_rms, error_heading="Invalid measured calibration RMS")
        target = _positive_values(self.target_rms, error_heading="Invalid calibration target RMS")
        target = _broadcast_values(
            target,
            len(measured),
            error_heading="Calibration target length mismatch",
        )
        reference = _domain_reference(self.unit, self.ref)
        object.__setattr__(self, "measured_rms", measured)
        object.__setattr__(self, "target_rms", target)
        object.__setattr__(self, "unit", self.unit.strip())
        object.__setattr__(self, "ref", reference)

    @classmethod
    def from_rms(
        cls,
        measured_rms: CalibrationInput,
        *,
        target_rms: CalibrationInput | None = None,
        target_level: CalibrationInput | None = None,
        unit: str,
        ref: float | None = None,
    ) -> Calibration:
        """Derive calibration from measured RMS and one known physical target.

        Exactly one of ``target_rms`` or ``target_level`` is required.
        ``target_level`` uses the amplitude-level relationship
        ``target_rms = ref * 10 ** (target_level / 20)``. Scalars broadcast to
        all measured channels; sequences must have one value per channel.
        """
        if (target_rms is None) == (target_level is None):
            raise ValueError(
                "Exactly one calibration target is required\n"
                f"  Got target_rms={target_rms!r}, target_level={target_level!r}\n"
                "  Expected: one of target_rms or target_level\n"
                "Use target_level=94.0 for a 94 dB acoustic calibrator or pass its known RMS value."
            )

        measured = _positive_values(measured_rms, error_heading="Invalid measured calibration RMS")
        reference = _domain_reference(unit, ref)
        if target_rms is not None:
            target = _positive_values(target_rms, error_heading="Invalid calibration target RMS")
        else:
            levels = _finite_values(target_level, error_heading="Invalid calibration target level")
            with np.errstate(over="ignore", invalid="ignore"):
                target_array = reference * np.power(10.0, np.asarray(levels, dtype=np.float64) / 20.0)
            target = _positive_values(target_array, error_heading="Invalid calibration target level")
        target = _broadcast_values(
            target,
            len(measured),
            error_heading="Calibration target length mismatch",
        )
        return cls(measured, target, unit, reference)

    @property
    def factors(self) -> tuple[float, ...]:
        """Return per-channel physical-unit multipliers."""
        return tuple(target / measured for measured, target in zip(self.measured_rms, self.target_rms, strict=True))

    @property
    def target_levels(self) -> tuple[float, ...]:
        """Return target amplitude levels in dB relative to :attr:`ref`."""
        return tuple(20.0 * float(np.log10(target / self.ref)) for target in self.target_rms)

    def factors_for_channels(self, channel_count: int) -> tuple[float, ...]:
        """Return factors aligned to ``channel_count``, broadcasting mono calibration."""
        if channel_count <= 0:
            raise ValueError("Calibration channel count must be positive")
        if len(self.factors) == 1:
            return self.factors * channel_count
        if len(self.factors) != channel_count:
            raise ValueError(
                "Calibration channel mismatch\n"
                f"  Got: {len(self.factors)} calibration channels for {channel_count} signal channels\n"
                "  Expected: one calibration channel or one per signal channel\n"
                "Use a mono calibration for broadcast or derive aligned per-channel factors."
            )
        return self.factors

    def _recipe_params(self) -> dict[str, Any]:
        """Return canonical public intent used by semantic Recipe capture."""
        return {
            "measured_rms": self.measured_rms,
            "target_rms": self.target_rms,
            "unit": self.unit,
            "ref": self.ref,
        }

    @classmethod
    def _from_recipe_params(cls, params: Mapping[str, Any]) -> Calibration:
        """Decode and validate calibration parameters at the Recipe boundary."""
        expected = {"measured_rms", "target_rms", "unit", "ref"}
        if set(params) != expected:
            raise ValueError(
                "Invalid calibration Recipe parameters\n"
                f"  Got keys: {sorted(params)}\n"
                f"  Expected keys: {sorted(expected)}\n"
                "Recreate the Recipe from a current calibrate() call."
            )
        try:
            return cls(
                measured_rms=cast(tuple[float, ...], tuple(params["measured_rms"])),
                target_rms=cast(tuple[float, ...], tuple(params["target_rms"])),
                unit=cast(str, params["unit"]),
                ref=cast(float, params["ref"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid calibration Recipe parameters\n"
                f"  Got: {dict(params)!r}\n"
                "  Expected: valid measured RMS, target RMS, unit, and reference values\n"
                "Recreate the Recipe from a current calibrate() call."
            ) from exc


class ApplyCalibration(AudioOperation[NDArrayReal, NDArrayReal]):
    """Multiply each channel by a positive physical calibration factor."""

    name = "apply_calibration"
    _display = "cal"

    def __init__(self, sampling_rate: float, factors: CalibrationInput) -> None:
        normalized = _positive_values(factors, error_heading="Invalid calibration factors")
        super().__init__(sampling_rate, factors=normalized)

    @property
    def factors(self) -> tuple[float, ...]:
        """Calibration factors captured at operation construction time."""
        return cast(tuple[float, ...], self._config_value("factors"))

    def validate_params(self) -> None:
        _positive_values(self._config_value("factors"), error_heading="Invalid calibration factors")

    def calculate_output_dtype(
        self,
        input_dtype: np.dtype[Any],
        *input_dtypes: np.dtype[Any],
    ) -> np.dtype[Any]:
        """Calibration always promotes the signal to float64."""
        del input_dtypes
        return np.result_type(input_dtype, np.float64)

    def _process(self, data: NDArrayReal) -> NDArrayReal:
        """Apply scalar or channel-wise factors to an eager array block."""
        channel_count = 1 if data.ndim == 1 else int(data.shape[0])
        factors = self.factors
        if len(factors) not in {1, channel_count}:
            raise ValueError(
                "Calibration channel mismatch\n"
                f"  Got: {len(factors)} calibration factors for {channel_count} signal channels\n"
                "  Expected: one factor or one factor per signal channel\n"
                "Use a scalar factor for broadcast or align factors with the channel axis."
            )
        scale_shape = (len(factors),) + (1,) * (data.ndim - 1)
        scale = np.asarray(factors, dtype=np.float64).reshape(scale_shape)
        result: NDArrayReal = np.asarray(data * scale)
        return result


register_operation(ApplyCalibration)


__all__ = ["ApplyCalibration", "Calibration", "CalibrationInput"]
