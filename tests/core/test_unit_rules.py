"""Units, references and level kinds derived by element-wise arithmetic (#651)."""

from __future__ import annotations

import warnings
from typing import Any

import dask.array as da
import numpy as np
import pytest

from wandas.core._unit_rules import binary_rule, derive_binary, derive_unary, operand_units_kept, raised, unary_rule
from wandas.core._units import format_unit, parse_unit, same_unit, unit_power, unit_product, unit_quotient
from wandas.core.metadata import ChannelCalibration, ChannelMetadata
from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipePlan

_SR = 8000.0


def _frame(unit: str, level: Any = "amplitude", values: np.ndarray | None = None) -> ChannelFrame:
    data = values if values is not None else 0.5 + 0.25 * np.sin(np.arange(1024) / 7.0)
    return ChannelFrame(
        da.from_array(np.atleast_2d(data), chunks=(1, -1)),
        sampling_rate=_SR,
        channel_metadata=[ChannelMetadata(label=unit or "raw", calibration=ChannelCalibration(1.0, unit, level=level))],
    )


def _domain(frame: Any) -> tuple[str, float, str]:
    calibration = frame.channels[0].calibration
    return calibration.unit, calibration.ref, calibration.level


class TestUnitAlgebra:
    @pytest.mark.parametrize(
        ("text", "canonical"),
        [
            ("Pa", "Pa"),
            ("m/s", "m/s"),
            ("m/s^2", "m/s^2"),
            ("Pa*Pa", "Pa^2"),
            ("Pa*m/s", "Pa*m/s"),
            ("Pa/(m/s)", "Pa*s/m"),
            ("1/Pa", "1/Pa"),
            ("Pa/Pa", "1"),
            ("1", "1"),
            ("Pa^0.5", "Pa^0.5"),
            ("dB SPL re 2e-05 Pa", "(dB SPL re 2e-05 Pa)"),
            ("Pa/(", "(Pa/()"),
        ],
    )
    def test_reading_and_printing(self, text: str, canonical: str) -> None:
        assert format_unit(parse_unit(text)) == canonical

    def test_combinations(self) -> None:
        assert unit_product("Pa", "Pa") == "Pa^2"
        assert unit_product("Pa", "m/s") == "Pa*m/s"
        assert unit_quotient("Pa", "m/s") == "Pa*s/m"
        assert unit_quotient("m/s^2", "m/s") == "1/s"
        assert unit_quotient("Pa", "m*s") == "Pa/(m*s)"
        assert unit_power("Pa", 0.5) == "Pa^0.5"
        assert unit_power("m/s^2", 1 / 3) == "m^(1/3)/s^(2/3)"
        assert unit_power("dB SPL re 2e-05 Pa", 2) == "(dB SPL re 2e-05 Pa)^2"
        assert same_unit("Pa*Pa", "Pa^2")
        # The unknown unit of an uncalibrated channel stays unknown.
        assert unit_product("", "Pa") == ""
        assert unit_power("", 2) == ""
        assert parse_unit("") is None

    @pytest.mark.parametrize(
        "unit",
        ["Pa", "Pa*s/m", "m^(1/3)/s^(2/3)", "kg m", "dB SPL re 2e-05 Pa", "dB (A)", "N*m re 1"],
    )
    def test_printed_units_read_back_to_the_same_factors(self, unit: str) -> None:
        # Frame arithmetic stores the printed unit, which the next operation reads.
        for derived in (unit, unit_product(unit, "s"), unit_power(unit, 2), unit_quotient("1", unit)):
            assert parse_unit(format_unit(parse_unit(derived))) == parse_unit(derived)

    def test_opaque_factors_cancel_across_operations(self) -> None:
        assert unit_quotient(unit_product("kg m", "s"), "s") == "(kg m)"
        assert unit_power(unit_power("kg m", 2), 0.5) == "(kg m)"
        assert same_unit(unit_product("kg m", "kg m"), "(kg m)^2")
        assert unit_quotient(unit_product("dB (A)", "s"), "dB (A)") == "s"


_PA2 = ("Pa^2", pytest.approx(4e-10), "power")


class TestOperators:
    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            pytest.param(lambda p, q, v, g: p + q, ("Pa", 2e-5, "amplitude"), id="p+p2"),
            pytest.param(lambda p, q, v, g: p - q, ("Pa", 2e-5, "amplitude"), id="p-p2"),
            pytest.param(lambda p, q, v, g: p * q, _PA2, id="p*p2"),
            pytest.param(lambda p, q, v, g: p * v, ("Pa*m/s", pytest.approx(2e-5), "power"), id="p*v"),
            pytest.param(lambda p, q, v, g: p / q, ("1", 1.0, "amplitude"), id="p/p2"),
            pytest.param(lambda p, q, v, g: p / v, ("Pa*s/m", pytest.approx(2e-5), "amplitude"), id="p/v"),
            pytest.param(lambda p, q, v, g: p * g, ("Pa", 2e-5, "amplitude"), id="p*gain"),
            pytest.param(lambda p, q, v, g: g * p, ("Pa", 2e-5, "amplitude"), id="gain*p"),
            pytest.param(lambda p, q, v, g: p * 2, ("Pa", 2e-5, "amplitude"), id="p*2"),
            pytest.param(lambda p, q, v, g: 2 * p, ("Pa", 2e-5, "amplitude"), id="2*p"),
            pytest.param(lambda p, q, v, g: p / 2, ("Pa", 2e-5, "amplitude"), id="p/2"),
            pytest.param(lambda p, q, v, g: p + 1, ("Pa", 2e-5, "amplitude"), id="p+1"),
            pytest.param(lambda p, q, v, g: 1 - p, ("Pa", 2e-5, "amplitude"), id="1-p"),
            pytest.param(lambda p, q, v, g: 2 / p, ("1/Pa", pytest.approx(5e4), "amplitude"), id="2/p"),
            pytest.param(lambda p, q, v, g: p**2, _PA2, id="p**2"),
            pytest.param(lambda p, q, v, g: p.power(2), _PA2, id="power(2)"),
            pytest.param(lambda p, q, v, g: np.square(p), _PA2, id="square"),
            pytest.param(lambda p, q, v, g: p**0.5, ("Pa^0.5", pytest.approx(2e-5**0.5), "none"), id="p**0.5"),
            pytest.param(lambda p, q, v, g: np.sqrt(p * p), ("Pa", pytest.approx(2e-5), "amplitude"), id="sqrt(p*p)"),
            pytest.param(lambda p, q, v, g: p**3, ("Pa^3", pytest.approx(8e-15), "none"), id="p**3"),
            pytest.param(lambda p, q, v, g: p**1, ("Pa", 2e-5, "amplitude"), id="p**1"),
            pytest.param(lambda p, q, v, g: p**-1, ("1/Pa", pytest.approx(5e4), "amplitude"), id="p**-1"),
            pytest.param(
                lambda p, q, v, g: np.reciprocal(p), ("1/Pa", pytest.approx(5e4), "amplitude"), id="reciprocal"
            ),
            pytest.param(lambda p, q, v, g: np.copysign(p, -1.0), ("Pa", 2e-5, "amplitude"), id="copysign(p,k)"),
            pytest.param(lambda p, q, v, g: np.copysign(1.0, p), ("1", 1.0, "none"), id="copysign(k,p)"),
            pytest.param(lambda p, q, v, g: p**q, ("", 1.0, "none"), id="p**p2"),
            pytest.param(lambda p, q, v, g: 2**p, ("1", 1.0, "none"), id="2**p"),
            pytest.param(lambda p, q, v, g: np.abs(np.negative(p)), ("Pa", 2e-5, "amplitude"), id="abs(negative)"),
            pytest.param(lambda p, q, v, g: np.log10(p), ("1", 1.0, "none"), id="log10"),
            pytest.param(lambda p, q, v, g: np.exp(g), ("1", 1.0, "none"), id="exp"),
            pytest.param(lambda p, q, v, g: np.maximum(p, q), ("Pa", 2e-5, "amplitude"), id="maximum"),
            pytest.param(lambda p, q, v, g: np.arctan2(p, q), ("1", 1.0, "none"), id="arctan2"),
            pytest.param(lambda p, q, v, g: p.fft() * q.fft(), _PA2, id="spectrum*spectrum"),
            pytest.param(lambda p, q, v, g: p.fft() ** 2, _PA2, id="spectrum**2"),
        ],
    )
    def test_result_domain(self, expression: Any, expected: tuple[Any, Any, str]) -> None:
        frames = (_frame("Pa"), _frame("Pa"), _frame("m/s"), _frame("1"))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _domain(expression(*frames)) == expected

    def test_mixed_units_warn_and_keep_the_left_unit(self) -> None:
        with pytest.warns(UserWarning, match="combines channels in different units") as record:
            result = _frame("Pa") + _frame("m/s")
        assert _domain(result) == ("Pa", 2e-5, "amplitude")
        assert record[0].filename == __file__

    def test_uncalibrated_channels_stay_unknown(self) -> None:
        raw = _frame("")
        assert _domain(raw * raw) == ("", 1.0, "power")
        assert _domain(raw * _frame("Pa")) == ("", pytest.approx(2e-5), "power")

    def test_squared_pressure_has_the_same_level_as_the_pressure(self) -> None:
        pressure = _frame("Pa")
        squared = pressure**2
        level = pressure.channels[0].level_reference.to_level(0.02)
        assert level == pytest.approx(60.0)
        assert squared.channels[0].level_reference.to_level(0.02**2) == pytest.approx(level)
        assert squared.channels[0].level_reference.unit == "dB"
        np.testing.assert_allclose((pressure.fft() ** 2).dB, pressure.fft().dB, atol=1e-9)

    def test_values_are_unchanged(self) -> None:
        pressure = _frame("Pa")
        np.testing.assert_allclose((pressure * pressure).data, pressure.data**2)


_OPERATORS = [
    pytest.param(lambda p, q: p + q, 2, "wandas.operator.add", id="add"),
    pytest.param(lambda p, q: p * q, 2, "wandas.operator.multiply", id="multiply"),
    pytest.param(lambda p, q: p / q, 2, "wandas.operator.divide", id="divide"),
    pytest.param(lambda p, q: p**2, 1, "wandas.operator.power", id="power"),
    pytest.param(lambda p, q: 2 / p, 1, "wandas.operator.reverse_divide", id="reverse_divide"),
    pytest.param(lambda p, q: np.sqrt(p), 1, "wandas.numpy.ufunc", id="ufunc-sqrt"),
    pytest.param(lambda p, q: np.maximum(p, q), 2, "wandas.numpy.ufunc", id="ufunc-maximum"),
    pytest.param(lambda p, q: p.power(2), 1, "wandas.audio.power", id="audio-power"),
]


@pytest.mark.parametrize(("expression", "n_inputs", "operation"), _OPERATORS)
def test_released_recipe_version_keeps_the_operand_units(expression: Any, n_inputs: int, operation: str) -> None:
    """Version 2 derives the unit; Recipes saved with version 1 replay the kept units."""
    left, right = _frame("Pa"), _frame("Pa")
    current = expression(left, right)
    assert current.operation_history[-1]["operation"] == operation
    assert current.operation_history[-1]["version"] == 2

    names = ("left", "right")[:n_inputs]
    inputs = dict(zip(names, (left, right), strict=False))
    plan = RecipePlan.from_frame(current, input_names=names)
    assert _domain(plan.apply(inputs)) == _domain(current)

    payload = plan.to_dict()
    for node in payload["nodes"]:
        if node["operation"] == operation:
            node["version"] = 1
    replayed = RecipePlan.from_dict(payload).apply(inputs)
    assert _domain(replayed) == ("Pa", 2e-5, "amplitude")
    assert replayed.operation_history[-1]["version"] == 1
    np.testing.assert_array_equal(replayed.data, current.data)


def _frame_ufuncs() -> list[str]:
    """NumPy ufuncs that a Frame evaluates lazily (numeric, one output, one or two inputs)."""
    names = []
    for name in dir(np):
        ufunc = getattr(np, name)
        if not isinstance(ufunc, np.ufunc) or ufunc.__name__ != name:
            continue
        boolean_output = all(signature.endswith("?") for signature in ufunc.types if "O" not in signature)
        if ufunc.nout == 1 and ufunc.nin in (1, 2) and ufunc.signature is None and not boolean_output:
            names.append(name)
    return names


@pytest.mark.parametrize("name", _frame_ufuncs())
def test_every_lazy_ufunc_has_a_unit_rule(name: str) -> None:
    """A new NumPy ufunc needs a row in wandas/core/_unit_rules.py before Frames derive its unit."""
    ufunc = getattr(np, name)
    rule = unary_rule(name) if ufunc.nin == 1 else binary_rule(name)
    assert rule is not None, f"Add {name!r} to the unit rules in wandas/core/_unit_rules.py"


def test_level_kinds_combine_by_the_operation() -> None:
    window = _frame("1")
    power = _frame("Pa^2", level="power")
    amplitude = _frame("Pa")
    assert _domain(window * power)[2] == "power"
    assert _domain(power / window)[2] == "power"
    assert _domain(power / power) == ("1", 1.0, "power")
    with pytest.warns(UserWarning, match="different units"):
        assert _domain(amplitude + power)[2] == "none"
    with pytest.warns(UserWarning, match="different units"):
        assert _domain(power + amplitude)[2] == "none"


_PA_CAL = ChannelCalibration(factor=1.0, unit="Pa", ref=2e-5)
_PA2_CAL = ChannelCalibration(factor=1.0, unit="Pa^2", ref=4e-10, level="power")


class TestRuleEdges:
    def test_product_and_quotient_of_unrelated_kinds_have_no_level(self) -> None:
        product = derive_binary("*", _PA2_CAL, _PA_CAL, None, reverse=False)
        assert (product.unit, product.level) == ("Pa^3", "none")
        quotient = derive_binary("/", _PA2_CAL, _PA_CAL, None, reverse=False)
        assert (quotient.unit, quotient.level) == ("Pa", "none")

    def test_a_reference_that_overflows_has_no_level(self) -> None:
        huge = ChannelCalibration(factor=1.0, unit="V", ref=1e300)
        assert raised(huge, 2.0) == ChannelCalibration(factor=1.0, unit="V^2", ref=1.0, level="none")

    def test_zero_dimensional_exponent_and_cbrt(self) -> None:
        assert raised(_PA_CAL, None).unit == ""
        assert derive_binary("**", _PA_CAL, None, np.array(2.0), reverse=False).level == "power"
        assert derive_unary("cbrt", _PA_CAL).unit == "Pa^(1/3)"

    def test_frame_operands_of_left_and_unlisted_rules(self) -> None:
        assert derive_binary("copysign", _PA_CAL, _PA2_CAL, None, reverse=False) == _PA_CAL
        assert derive_binary("not_a_ufunc", _PA_CAL, None, 2.0, reverse=False).unit == ""
        assert derive_binary("arctan2", _PA_CAL, None, 2.0, reverse=False).unit == "1"
        assert derive_binary("**", _PA_CAL, None, 2j, reverse=False).unit == ""
        assert derive_unary("not_a_ufunc", _PA_CAL).unit == ""
        assert format_unit(None) == ""

    def test_unreadable_units_stay_one_factor(self) -> None:
        assert parse_unit("Pa^x") == (("Pa^x", 1),)
        assert parse_unit("Pa*()") == (("Pa*()", 1),)
        assert unit_quotient("", "Pa") == ""


def test_released_recipe_version_gives_mixed_kinds_no_level() -> None:
    with operand_units_kept():
        result = _frame("Pa") * _frame("Pa^2", level="power")
    assert _domain(result) == ("Pa", 2e-5, "none")
