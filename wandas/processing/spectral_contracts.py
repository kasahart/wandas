"""Architecture-neutral contracts for ordered pairwise spectra.

The concrete Frame types for pairwise quantities consume these contracts.
This module owns only immutable channel roles, pair ordering,
derived linear-domain metadata, level formulas, and transfer-ratio edge-case
handling so those rules can be shared without making ``SpectralFrame`` infer a
quantity from lineage or display labels.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from wandas.utils.types import NDArrayComplex, NDArrayReal

SpectralScaling = Literal["spectrum", "density"]
TransferDenominator = Literal["input", "output"]


def _normalize_index(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer index")
    return int(value)


def _normalize_reference(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a positive finite number")
    reference = float(value)
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return reference


@dataclass(frozen=True, slots=True)
class SpectralChannelRole:
    """Immutable channel identity used by a pairwise spectral quantity."""

    index: int
    label: str
    unit: str
    reference: float
    channel_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _normalize_index(self.index, name="Channel index"))
        if not isinstance(self.label, str):
            raise TypeError("Channel label must be a string")
        if not isinstance(self.unit, str):
            raise TypeError("Channel unit must be a string")
        object.__setattr__(self, "unit", self.unit.strip())
        object.__setattr__(
            self,
            "reference",
            _normalize_reference(self.reference, name="Channel reference"),
        )
        if not isinstance(self.channel_id, str):
            raise TypeError("Channel id must be a string")
        object.__setattr__(self, "channel_id", self.channel_id.strip() or f"c{self.index}")

    @property
    def source_id(self) -> str:
        """Return the opaque source-channel identity used for pair selection."""
        return self.channel_id


@dataclass(frozen=True, slots=True)
class OrderedSpectralPair:
    """Output/input channel roles with the canonical flattened index."""

    output: SpectralChannelRole
    input: SpectralChannelRole
    n_channels: int

    def __post_init__(self) -> None:
        n_channels = _normalize_index(self.n_channels, name="Channel count")
        if n_channels <= 0:
            raise ValueError("Channel count must be positive")
        if not isinstance(self.output, SpectralChannelRole) or not isinstance(self.input, SpectralChannelRole):
            raise TypeError("OrderedSpectralPair requires SpectralChannelRole values")
        if not 0 <= self.output.index < n_channels:
            raise ValueError("Output channel index is outside the channel count")
        if not 0 <= self.input.index < n_channels:
            raise ValueError("Input channel index is outside the channel count")
        object.__setattr__(self, "n_channels", n_channels)

    @property
    def pair_index(self) -> int:
        """Return ``output_index * n_channels + input_index``."""
        return flatten_pair_index(self.output.index, self.input.index, self.n_channels)


@dataclass(frozen=True, slots=True)
class DerivedSpectralDomain:
    """Linear unit and numeric reference for a derived spectral quantity."""

    unit: str
    reference: float

    def __post_init__(self) -> None:
        if not isinstance(self.unit, str):
            raise TypeError("Derived spectral unit must be a string")
        object.__setattr__(self, "unit", self.unit.strip())
        object.__setattr__(
            self,
            "reference",
            _normalize_reference(self.reference, name="Derived spectral reference"),
        )


def flatten_pair_index(output_index: int, input_index: int, n_channels: int) -> int:
    """Return the canonical output-major/input-minor flattened pair index."""
    n = _normalize_index(n_channels, name="Channel count")
    output = _normalize_index(output_index, name="Output channel index")
    input_ = _normalize_index(input_index, name="Input channel index")
    if n <= 0:
        raise ValueError("Channel count must be positive")
    if not 0 <= output < n:
        raise ValueError("Output channel index is outside the channel count")
    if not 0 <= input_ < n:
        raise ValueError("Input channel index is outside the channel count")
    return output * n + input_


def _product_unit(input_unit: str, output_unit: str) -> str:
    factors = [unit for unit in (input_unit, output_unit) if unit]
    return "*".join(factors) or "1"


def _validate_scaling(scaling: str) -> SpectralScaling:
    if scaling == "spectrum" or scaling == "density":
        return cast(SpectralScaling, scaling)
    raise ValueError("CSD scaling must be 'spectrum' or 'density'")


def derive_csd_domain(pair: OrderedSpectralPair, scaling: str) -> DerivedSpectralDomain:
    """Derive CSD unit/reference from ``input * output`` and scaling."""
    mode = _validate_scaling(scaling)
    unit = _product_unit(pair.input.unit, pair.output.unit)
    if mode == "density":
        unit = f"{unit}/Hz"
    return DerivedSpectralDomain(
        unit=unit,
        reference=pair.input.reference * pair.output.reference,
    )


def derive_coherence_domain() -> DerivedSpectralDomain:
    """Return the dimensionless domain of magnitude-squared coherence."""
    return DerivedSpectralDomain(unit="1", reference=1.0)


def derive_transfer_domain(
    pair: OrderedSpectralPair,
    denominator_role: TransferDenominator = "input",
) -> DerivedSpectralDomain:
    """Derive transfer unit/reference for the selected denominator role.

    The default is the canonical ``H[output, input] = P[output, input] /
    P[input, input]`` contract. ``denominator_role="output"`` is retained for
    truthful Recipe v1 replay, whose released numerical definition divided by
    the output auto-spectrum.
    """
    if denominator_role not in {"input", "output"}:
        raise ValueError("Transfer denominator role must be 'input' or 'output'")
    numerator_role = pair.output if denominator_role == "input" else pair.input
    denominator = pair.input if denominator_role == "input" else pair.output
    output_unit = numerator_role.unit
    input_unit = denominator.unit
    if not output_unit and not input_unit or output_unit == input_unit:
        unit = "1"
    elif not input_unit:
        unit = output_unit
    elif not output_unit:
        unit = f"1/{input_unit}"
    else:
        unit = f"{output_unit}/{input_unit}"
    return DerivedSpectralDomain(
        unit=unit,
        reference=numerator_role.reference / denominator.reference,
    )


def csd_level(value: NDArrayComplex, reference: float) -> NDArrayReal:
    """Return CSD level as ``10 * log10(abs(value) / reference)``."""
    normalized_reference = _normalize_reference(reference, name="CSD level reference")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(10.0 * np.log10(np.abs(value) / normalized_reference))


def transfer_level(value: NDArrayComplex, reference: float) -> NDArrayReal:
    """Return transfer level as ``20 * log10(abs(value) / reference)``."""
    normalized_reference = _normalize_reference(reference, name="Transfer level reference")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(20.0 * np.log10(np.abs(value) / normalized_reference))


def reject_pairwise_a_weighting(enabled: bool) -> None:
    """Reject A-weighting for dedicated pairwise quantities explicitly."""
    if enabled:
        raise ValueError(
            "A-weighting is unsupported for coherence, CSD, and transfer-function quantities. "
            "Use the dedicated quantity-aware projection instead."
        )


def transfer_function_ratio(
    cross_spectrum: NDArrayComplex,
    input_power: NDArrayReal,
) -> NDArrayComplex:
    """Divide ``H[output, input]`` by the matching input auto-spectrum.

    The cross-spectrum must have shape ``(output, input, ...)`` and the power
    array must have shape ``(input, ...)``. Exact zero denominator bins remain
    complex NaN; nonzero near-zero bins are not floored or clipped.
    """
    cross = np.asarray(cross_spectrum)
    power = np.asarray(input_power)
    if cross.ndim < 2:
        raise ValueError("Cross-spectrum must have output and input axes")
    if power.ndim != cross.ndim - 1:
        raise ValueError("Input power must omit only the output axis")
    if cross.shape[1] != power.shape[0] or cross.shape[2:] != power.shape[1:]:
        raise ValueError("Transfer denominator must have one value per input channel and matching trailing axes")

    denominator = power[np.newaxis, ...]
    result = np.full(
        cross.shape,
        np.nan + 1j * np.nan,
        dtype=np.result_type(cross.dtype, np.complex64),
    )
    # Non-finite denominator values are undefined rather than a reason to let
    # IEEE division silently turn ``cross / inf`` into a plausible zero.
    valid = np.broadcast_to((denominator != 0) & np.isfinite(denominator), cross.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(cross, denominator, out=result, where=valid)
    return result


def as_output_input_pairs(input_output_values: np.ndarray) -> np.ndarray:
    """Transpose a square ``(input, output, ...)`` matrix to ``(output, input, ...)``."""
    values = np.asarray(input_output_values)
    if values.ndim < 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Pairwise matrix must have equal input and output axes")
    return np.swapaxes(values, 0, 1)


def flatten_output_input_pairs(output_input_values: np.ndarray) -> np.ndarray:
    """Flatten an ``(output, input, ...)`` matrix in canonical pair order."""
    values = np.asarray(output_input_values)
    if values.ndim < 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Pairwise matrix must have equal output and input axes")
    n_channels = values.shape[0]
    return values.reshape((n_channels * n_channels, *values.shape[2:]))


__all__ = [
    "DerivedSpectralDomain",
    "OrderedSpectralPair",
    "SpectralChannelRole",
    "SpectralScaling",
    "TransferDenominator",
    "as_output_input_pairs",
    "csd_level",
    "derive_coherence_domain",
    "derive_csd_domain",
    "derive_transfer_domain",
    "flatten_output_input_pairs",
    "flatten_pair_index",
    "reject_pairwise_a_weighting",
    "transfer_function_ratio",
    "transfer_level",
]
