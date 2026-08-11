"""Public measurement-level metadata contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import wandas as wd
from wandas.core import ChannelMetadata, LevelReference


def test_level_reference_is_public_structured_channel_context() -> None:
    channel = ChannelMetadata(
        label="microphone",
        calibration=wd.ChannelCalibration(factor=0.42, unit="Pa"),
    )

    reference = channel.level_reference

    assert isinstance(reference, LevelReference)
    assert isinstance(reference, wd.LevelReference)
    assert reference.unit == "dB SPL"
    assert reference.reference_value == 2e-5
    assert reference.reference_unit == "Pa"
    assert reference.label == "dB SPL re 20 µPa"
    assert reference.minimum_level == -240.0
    with pytest.raises(FrozenInstanceError):
        reference.reference_value = 1.0  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize("reference_value", [True, "1"])
def test_level_reference_rejects_non_numeric_reference_values(reference_value: object) -> None:
    with pytest.raises(TypeError, match="positive finite number"):
        LevelReference(reference_value=reference_value, reference_unit="Pa")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("reference_value", [0.0, -1.0, np.inf, np.nan])
def test_level_reference_rejects_non_positive_or_non_finite_reference_values(reference_value: float) -> None:
    with pytest.raises(ValueError, match="positive finite number"):
        LevelReference(reference_value=reference_value, reference_unit="Pa")


def test_level_reference_rejects_non_string_reference_unit() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        LevelReference(reference_value=1.0, reference_unit=None)  # ty: ignore[invalid-argument-type]


def test_level_reference_rejects_whitespace_only_reference_unit() -> None:
    with pytest.raises(ValueError, match="must not contain only whitespace"):
        LevelReference(reference_value=1.0, reference_unit=" \t")


def test_level_reference_distinguishes_explicit_full_scale_from_identity() -> None:
    identity = ChannelMetadata().level_reference
    full_scale = ChannelMetadata(unit="FS").level_reference

    assert identity.unit == "dB"
    assert identity.reference_value == 1.0
    assert identity.reference_unit == ""
    assert identity.label == "dB re 1 input unit"
    assert full_scale.unit == "dBFS"
    assert full_scale.reference_value == 1.0
    assert full_scale.reference_unit == "FS"
    assert full_scale.label == "dBFS"


@pytest.mark.parametrize("reference_value", [2e-5, 20e-6, 20 * 1e-6, 20 * 10**-6])
def test_level_reference_recognizes_equivalent_twenty_micropascal_values(reference_value: float) -> None:
    reference = LevelReference(reference_value=reference_value, reference_unit="Pa")

    assert reference.unit == "dB SPL"
    assert reference.label == "dB SPL re 20 µPa"
    assert reference.reference_value == reference_value


def test_level_reference_formats_micro_reference_and_stable_significant_digits() -> None:
    voltage = LevelReference(reference_value=2e-5, reference_unit="V")
    noisy_decimal = 0.1 + 0.2
    generic = LevelReference(reference_value=noisy_decimal, reference_unit="V")

    assert voltage.label == "dB re 20 µV"
    assert generic.label == "dB re 0.3 V"
    assert generic.reference_value == noisy_decimal


def test_noncanonical_pa_and_full_scale_references_remain_generic_db() -> None:
    pascal = LevelReference(reference_value=1.0, reference_unit="Pa")
    full_scale = LevelReference(reference_value=2.0, reference_unit="FS")

    assert pascal.unit == "dB"
    assert pascal.label == "dB re 1 Pa"
    assert full_scale.unit == "dB"
    assert full_scale.label == "dB re 2 FS"


def test_level_reference_formats_non_acoustic_physical_domain() -> None:
    reference = ChannelMetadata(calibration=wd.ChannelCalibration(factor=9.81, unit="m/s^2", ref=1.0)).level_reference

    assert reference.unit == "dB"
    assert reference.reference_value == 1.0
    assert reference.reference_unit == "m/s^2"
    assert reference.label == "dB re 1 m/s^2"


def test_level_reference_converts_scalar_and_array_amplitudes_with_one_floor() -> None:
    reference = ChannelMetadata(unit="Pa").level_reference
    values = np.array([[2e-5, -2e-4], [0.0, 2e-17]])

    result = reference.to_level(values)

    assert isinstance(result, np.ndarray)
    assert result.shape == values.shape
    np.testing.assert_allclose(result, [[0.0, 20.0], [-240.0, -240.0]])
    assert reference.to_level(2e-5) == pytest.approx(0.0)
    assert isinstance(reference.to_level(2e-5), float)
    assert reference.to_level(2e-5j) == pytest.approx(0.0)

    zero_dimensional = reference.to_level(np.array(2e-5))
    assert isinstance(zero_dimensional, np.ndarray)
    assert zero_dimensional.shape == ()
    assert zero_dimensional.item() == pytest.approx(0.0)
