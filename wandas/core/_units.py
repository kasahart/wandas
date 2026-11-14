"""Small unit algebra for channel units derived by element-wise arithmetic.

Channel units are free strings. This module reads the strings that wandas and
its users commonly write (``Pa``, ``m/s``, ``m/s^2``, ``Pa*m/s``, ``Pa^2``,
``1/Pa``, ``Pa/(m/s)``) as products of named factors with rational exponents,
combines them, and prints one canonical form, so that ``Pa*Pa`` and ``Pa^2``
are the same unit. Anything it cannot read, such as ``dB SPL re 2e-05 Pa``,
stays one opaque factor and is printed in parentheses when combined.

Conventions:

- ``""`` is an unknown (uncalibrated) unit. Combining it with anything gives
  ``""`` again: the result unit is unknown too.
- ``"1"`` is dimensionless and is the printed form of a unit whose factors
  all cancel.
"""

from __future__ import annotations

import re
from fractions import Fraction

# A unit is an ordered tuple of (factor, exponent) with no zero exponents.
UnitFactors = tuple[tuple[str, Fraction], ...]

_UNKNOWN = ""
_DIMENSIONLESS = "1"
_NAME = re.compile(r"[^\s*/^()]+")
_EXPONENT = re.compile(r"-?\d+(?:\.\d+)?|\(-?\d+/\d+\)")
_MAX_DENOMINATOR = 1_000_000


class _UnreadableUnitError(ValueError):
    """Raised internally when a unit string is not a product of factors."""


def _parse_exponent(text: str) -> Fraction:
    if text.startswith("("):
        return Fraction(text[1:-1])
    return Fraction(text)


def _combine(factors: list[tuple[str, Fraction]]) -> UnitFactors:
    """Merge equal factors, keeping first-appearance order and dropping zeros."""
    merged: dict[str, Fraction] = {}
    for name, exponent in factors:
        merged[name] = merged.get(name, Fraction(0)) + exponent
    return tuple((name, exponent) for name, exponent in merged.items() if exponent != 0)


class _Reader:
    """Recursive-descent reader for ``factor (('*' | '/') factor)*``."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.position = 0

    def read(self) -> list[tuple[str, Fraction]]:
        factors = self._product()
        if self.position != len(self.text):
            raise _UnreadableUnitError(self.text)
        return factors

    def _product(self) -> list[tuple[str, Fraction]]:
        factors = self._power()
        while self.position < len(self.text) and self.text[self.position] in "*/":
            sign = 1 if self.text[self.position] == "*" else -1
            self.position += 1
            factors.extend((name, sign * exponent) for name, exponent in self._power())
        return factors

    def _power(self) -> list[tuple[str, Fraction]]:
        factors = self._atom()
        if self.position < len(self.text) and self.text[self.position] == "^":
            self.position += 1
            match = _EXPONENT.match(self.text, self.position)
            if match is None:
                raise _UnreadableUnitError(self.text)
            self.position = match.end()
            exponent = _parse_exponent(match.group())
            factors = [(name, value * exponent) for name, value in factors]
        return factors

    def _atom(self) -> list[tuple[str, Fraction]]:
        if self.position < len(self.text) and self.text[self.position] == "(":
            start = self.position
            self.position += 1
            try:
                factors = self._product()
                if self.position >= len(self.text) or self.text[self.position] != ")":
                    raise _UnreadableUnitError(self.text)
            except (_UnreadableUnitError, ZeroDivisionError):
                # A parenthesised opaque factor, as format_unit prints one.
                self.position = start
                return [(self._opaque_name(), Fraction(1))]
            self.position += 1
            return factors
        match = _NAME.match(self.text, self.position)
        if match is None:
            raise _UnreadableUnitError(self.text)
        self.position = match.end()
        name = match.group()
        if name == _DIMENSIONLESS:
            return []
        return [(name, Fraction(1))]

    def _opaque_name(self) -> str:
        """Read ``(text)`` up to its matching parenthesis and return ``text``."""
        depth = 0
        for index in range(self.position, len(self.text)):
            if self.text[index] == "(":
                depth += 1
            elif self.text[index] == ")":
                depth -= 1
                if depth == 0:
                    name = self.text[self.position + 1 : index].strip()
                    if not name:
                        break
                    self.position = index + 1
                    return name
        raise _UnreadableUnitError(self.text)


def parse_unit(unit: str) -> UnitFactors | None:
    """Return the factors of a unit, ``None`` for the unknown unit ``""``."""
    text = unit.strip()
    if not text:
        return None
    try:
        return _combine(_Reader(text).read())
    except (_UnreadableUnitError, ZeroDivisionError):
        return ((text, Fraction(1)),)


def _format_exponent(exponent: Fraction) -> str:
    if exponent.denominator == 1:
        return str(exponent.numerator)
    as_float = float(exponent)
    if Fraction(repr(as_float)) == exponent and len(repr(as_float)) <= 6:
        return repr(as_float)
    return f"({exponent.numerator}/{exponent.denominator})"


def _format_name(name: str) -> str:
    return name if _NAME.fullmatch(name) else f"({name})"


def _format_factor(name: str, exponent: Fraction) -> str:
    text = _format_name(name)
    return text if exponent == 1 else f"{text}^{_format_exponent(exponent)}"


def format_unit(factors: UnitFactors | None) -> str:
    """Print factors canonically: ``Pa*m/s``, ``Pa^2``, ``1/Pa``, ``Pa/(m*s)``."""
    if factors is None:
        return _UNKNOWN
    numerator = [_format_factor(name, exponent) for name, exponent in factors if exponent > 0]
    denominator = [_format_factor(name, -exponent) for name, exponent in factors if exponent < 0]
    if not numerator and not denominator:
        return _DIMENSIONLESS
    head = "*".join(numerator) or _DIMENSIONLESS
    if not denominator:
        return head
    tail = denominator[0] if len(denominator) == 1 else f"({'*'.join(denominator)})"
    return f"{head}/{tail}"


def unit_product(left: str, right: str) -> str:
    """Return the unit of ``left * right``."""
    a, b = parse_unit(left), parse_unit(right)
    if a is None or b is None:
        return _UNKNOWN
    return format_unit(_combine([*a, *b]))


def unit_quotient(left: str, right: str) -> str:
    """Return the unit of ``left / right``."""
    a, b = parse_unit(left), parse_unit(right)
    if a is None or b is None:
        return _UNKNOWN
    return format_unit(_combine([*a, *((name, -exponent) for name, exponent in b)]))


def unit_power(unit: str, exponent: float | Fraction) -> str:
    """Return the unit of ``unit ** exponent``."""
    factors = parse_unit(unit)
    if factors is None:
        return _UNKNOWN
    value = Fraction(exponent).limit_denominator(_MAX_DENOMINATOR)
    return format_unit(_combine([(name, power * value) for name, power in factors]))


def same_unit(left: str, right: str) -> bool:
    """Return whether two unit strings name the same unit."""
    return format_unit(parse_unit(left)) == format_unit(parse_unit(right))


def is_dimensionless(unit: str) -> bool:
    """Return whether a unit is ``1`` (not the unknown unit ``""``)."""
    return parse_unit(unit) == ()
