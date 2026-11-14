"""Units, references and level kinds of element-wise arithmetic results.

One table decides, for every operator and NumPy ufunc a Frame accepts, what
the result channel describes. For a channel ``a`` and another operand ``b``
(a Frame), ``k`` (a scalar or array) or ``n`` (a real scalar exponent):

- ``a ± b``: ``a``'s unit and reference, with a warning when the units or
  kinds differ; differing kinds give no level, in either order. ``maximum``,
  ``hypot``, ``fmod`` and similar follow it.
- ``a ± k``, ``a * k``, ``a / k``: unchanged.
- ``a * b``: product of the units and of the references. Amplitude times
  amplitude is a power quantity; a dimensionless amplitude channel (unit
  ``1``, such as a window) is a gain and keeps the other kind; otherwise the
  result has no level.
- ``a / b``: quotient of the units (``1`` for equal units) and references.
  Equal kinds keep the kind (a ratio of amplitudes is an amplitude ratio); a
  dimensionless amplitude divisor keeps ``a``'s kind; otherwise no level.
- ``k / a`` and ``reciprocal``: reciprocal unit and reference, same kind.
- ``a ** n``, ``power(n)``, ``square``, ``sqrt``, ``cbrt``: unit and reference
  to the power ``n``. Amplitude squared is power, power to the 1/2 is
  amplitude, ``n == 1`` keeps the kind; otherwise no level.
- ``a ** b``, ``a ** array``: unknown unit ``""``, reference 1, no level.
  ``k ** a``: ``1``, no level.
- ``abs``, ``negative``, ``conjugate`` and rounding: unchanged.
- ``exp``, ``log``, trigonometric, ``sign``, ``arctan2``, bitwise and other
  ufuncs: dimensionless ``1``, reference 1, no level.

The unknown unit ``""`` (an uncalibrated channel) stays unknown when combined.
"""

from __future__ import annotations

import math
import numbers
import os
import sys
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import numpy as np

from wandas.core._units import is_dimensionless, same_unit, unit_power, unit_product, unit_quotient
from wandas.core.metadata import ChannelCalibration, LevelKind

# Kinds of operation. Operators use their symbol, ufuncs their NumPy name.
ADDITIVE = "additive"
MULTIPLY = "multiply"
DIVIDE = "divide"
POWER = "power"
KEEP = "keep"
LEFT = "left"
DIMENSIONLESS = "dimensionless"
SQUARE = "square"
SQRT = "sqrt"
CBRT = "cbrt"
RECIPROCAL = "reciprocal"

_BINARY_RULES: dict[str, str] = {
    "+": ADDITIVE,
    "-": ADDITIVE,
    "*": MULTIPLY,
    "/": DIVIDE,
    "**": POWER,
    "add": ADDITIVE,
    "subtract": ADDITIVE,
    "maximum": ADDITIVE,
    "minimum": ADDITIVE,
    "fmax": ADDITIVE,
    "fmin": ADDITIVE,
    "hypot": ADDITIVE,
    "fmod": ADDITIVE,
    "remainder": ADDITIVE,
    "nextafter": ADDITIVE,
    "multiply": MULTIPLY,
    "divide": DIVIDE,
    "true_divide": DIVIDE,
    "floor_divide": DIVIDE,
    "power": POWER,
    "float_power": POWER,
    "copysign": LEFT,
    "ldexp": LEFT,
    "arctan2": DIMENSIONLESS,
    "logaddexp": DIMENSIONLESS,
    "logaddexp2": DIMENSIONLESS,
    "heaviside": DIMENSIONLESS,
    "gcd": DIMENSIONLESS,
    "lcm": DIMENSIONLESS,
    "bitwise_and": DIMENSIONLESS,
    "bitwise_or": DIMENSIONLESS,
    "bitwise_xor": DIMENSIONLESS,
    "bitwise_left_shift": DIMENSIONLESS,
    "bitwise_right_shift": DIMENSIONLESS,
    "left_shift": DIMENSIONLESS,
    "right_shift": DIMENSIONLESS,
}

_UNARY_RULES: dict[str, str] = {
    "negative": KEEP,
    "positive": KEEP,
    "absolute": KEEP,
    "fabs": KEEP,
    "conjugate": KEEP,
    "rint": KEEP,
    "floor": KEEP,
    "ceil": KEEP,
    "trunc": KEEP,
    "spacing": KEEP,
    "square": SQUARE,
    "sqrt": SQRT,
    "cbrt": CBRT,
    "reciprocal": RECIPROCAL,
    **{
        name: DIMENSIONLESS
        for name in (
            "exp",
            "exp2",
            "expm1",
            "log",
            "log2",
            "log10",
            "log1p",
            "sin",
            "cos",
            "tan",
            "arcsin",
            "arccos",
            "arctan",
            "sinh",
            "cosh",
            "tanh",
            "arcsinh",
            "arccosh",
            "arctanh",
            "deg2rad",
            "rad2deg",
            "degrees",
            "radians",
            "sign",
            "invert",
            "bitwise_count",
        )
    },
}


def binary_rule(name: str) -> str | None:
    """Return the rule of a two-input operator or ufunc, ``None`` when unlisted."""
    return _BINARY_RULES.get(name)


def unary_rule(name: str) -> str | None:
    """Return the rule of a one-input ufunc, ``None`` when unlisted."""
    return _UNARY_RULES.get(name)


def _calibration(unit: str, ref: float, level: LevelKind) -> ChannelCalibration:
    if not math.isfinite(ref) or ref <= 0.0:
        # A reference that over- or underflows cannot describe a level.
        return ChannelCalibration(factor=1.0, unit=unit, ref=1.0, level="none")
    return ChannelCalibration(factor=1.0, unit=unit, ref=ref, level=level)


_UNKNOWN = ChannelCalibration(factor=1.0, unit="", ref=1.0, level="none")
_ONE = ChannelCalibration(factor=1.0, unit="1", ref=1.0, level="none")


def _is_gain(calibration: ChannelCalibration) -> bool:
    """Return whether a channel is a dimensionless amplitude factor such as a window."""
    return calibration.level == "amplitude" and is_dimensionless(calibration.unit)


def _product_level(left: ChannelCalibration, right: ChannelCalibration) -> LevelKind:
    if _is_gain(left):
        return right.level
    if _is_gain(right):
        return left.level
    if left.level == right.level == "amplitude":
        return "power"
    return "none"


def _quotient_level(left: ChannelCalibration, right: ChannelCalibration) -> LevelKind:
    if _is_gain(right):
        return left.level
    if left.level == right.level:
        return left.level
    return "none"


def _power_level(level: LevelKind, exponent: float) -> LevelKind:
    if exponent == 1:
        return level
    if level == "amplitude" and exponent == 2:
        return "power"
    if level == "power" and exponent == 0.5:
        return "amplitude"
    return "none"


def _real_exponent(operand: Any) -> float | None:
    """Return a real scalar exponent, or ``None`` for arrays and complex values."""
    if isinstance(operand, np.ndarray) and operand.ndim == 0:
        operand = operand.item()
    if isinstance(operand, numbers.Real) and not isinstance(operand, bool):
        value = float(operand)
        return value if math.isfinite(value) else None
    return None


def raised(calibration: ChannelCalibration, exponent: float | None) -> ChannelCalibration:
    """Return the calibration of ``values ** exponent``; ``None`` means not a real scalar."""
    if exponent is None:
        return _UNKNOWN
    try:
        ref = calibration.ref**exponent
    except OverflowError:
        ref = math.inf
    return _calibration(
        unit_power(calibration.unit, exponent),
        float(ref),
        _power_level(calibration.level, exponent),
    )


def reciprocal(calibration: ChannelCalibration) -> ChannelCalibration:
    """Return the calibration of ``1 / values``: the level kind is unchanged."""
    return _calibration(unit_power(calibration.unit, -1.0), 1.0 / calibration.ref, calibration.level)


def derive_binary(
    name: str,
    left: ChannelCalibration,
    right: ChannelCalibration | None,
    operand: Any,
    *,
    reverse: bool,
    warn: Callable[[str], None] | None = None,
) -> ChannelCalibration:
    """Return the result calibration of one channel of a two-input operation.

    Args:
        name: Operator symbol (``+``, ``*``, ...) or NumPy ufunc name.
        left: Calibration of the Frame's channel, after its factor was applied.
        right: Calibration of the other Frame's matching channel, or ``None``
            when the other operand is a scalar or array.
        operand: The other operand itself.
        reverse: Whether the other operand is the first input.
        warn: Called with a message when the operation mixes units.
    """
    rule = binary_rule(name)
    if right is None:
        if rule == POWER:
            if reverse:
                return _ONE
            return raised(left, _real_exponent(operand))
        if rule == DIVIDE and reverse:
            return reciprocal(left)
        if rule in (ADDITIVE, MULTIPLY, DIVIDE, LEFT):
            return left
        if rule == DIMENSIONLESS:
            return _ONE
        return _UNKNOWN
    if rule == ADDITIVE:
        if (not same_unit(left.unit, right.unit) or left.level != right.level) and warn is not None:
            warn(
                f"'{name}' combines channels in different units\n"
                f"  Left: {left.unit or 'unknown unit'!r} ({left.level}), "
                f"right: {right.unit or 'unknown unit'!r} ({right.level})\n"
                "  The result keeps the left unit.\n"
                "Convert or recalibrate one operand first, or set the result unit with with_calibration()."
            )
        return left if left.level == right.level else left.with_level("none")
    if rule == MULTIPLY:
        return _calibration(unit_product(left.unit, right.unit), left.ref * right.ref, _product_level(left, right))
    if rule == DIVIDE:
        return _calibration(unit_quotient(left.unit, right.unit), left.ref / right.ref, _quotient_level(left, right))
    if rule == LEFT:
        return left
    if rule == DIMENSIONLESS:
        return _ONE
    # Frame ** Frame and unlisted ufuncs: the result unit is not known.
    return _UNKNOWN


def derive_unary(name: str, calibration: ChannelCalibration) -> ChannelCalibration:
    """Return the result calibration of one channel of a one-input ufunc."""
    rule = unary_rule(name)
    if rule == KEEP:
        return calibration
    if rule == SQUARE:
        return raised(calibration, 2.0)
    if rule == SQRT:
        return raised(calibration, 0.5)
    if rule == CBRT:
        return raised(calibration, 1.0 / 3.0)
    if rule == RECIPROCAL:
        return reciprocal(calibration)
    if rule == DIMENSIONLESS:
        return _ONE
    return _UNKNOWN


_WANDAS_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def warn_mixed_units(message: str) -> None:
    """Emit the mixed-unit warning of additive arithmetic at the caller's line."""
    level = 1
    frame = sys._getframe(0)
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(_WANDAS_PACKAGE_DIR + os.sep):
        frame = frame.f_back
        level += 1
    warnings.warn(message, UserWarning, stacklevel=level)


# Recipe operation version 1 of the operators recorded "units are kept";
# its replay sets this flag so the result keeps the left operand's channels.
_OPERAND_UNITS_KEPT: ContextVar[bool] = ContextVar("wandas_operand_units_kept", default=False)


@contextmanager
def operand_units_kept() -> Iterator[None]:
    """Replay the released contract in which arithmetic kept the left operand's units."""
    token = _OPERAND_UNITS_KEPT.set(True)
    try:
        yield
    finally:
        _OPERAND_UNITS_KEPT.reset(token)


def units_are_kept() -> bool:
    """Return whether arithmetic replays the released "units are kept" contract."""
    return _OPERAND_UNITS_KEPT.get()
