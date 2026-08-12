"""Public measurement-level contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import wandas as wd
from wandas.core import ChannelMetadata, LevelReference


def test_level_reference_is_public_immutable_channel_context() -> None:
    channel = ChannelMetadata(
        label="microphone",
        calibration=wd.ChannelCalibration(factor=0.42, unit="Pa"),
    )

    reference = channel.level_reference

    assert isinstance(reference, LevelReference)
    assert isinstance(reference, wd.LevelReference)
    assert reference.reference_value == 2e-5
    assert reference.reference_unit == "Pa"
    assert reference.unit == "dB SPL"
    assert reference.label == "dB SPL re 20 µPa"
    with pytest.raises(FrozenInstanceError):
        reference.reference_value = 1.0  # ty: ignore[invalid-assignment]


def test_explicit_full_scale_is_distinct_from_empty_identity_unit() -> None:
    full_scale = LevelReference(reference_value=1.0, reference_unit="FS")
    identity = ChannelMetadata().level_reference

    assert (full_scale.unit, full_scale.label) == ("dBFS", "dBFS")
    assert identity.reference_value == 1.0
    assert identity.reference_unit == ""
    assert identity.unit == "dB"
    assert identity.label == "dB re 1 input unit"


@pytest.mark.parametrize(
    "reference_value",
    [2e-5, 20e-6, np.nextafter(2e-5, 0.0), np.nextafter(2e-5, np.inf)],
)
def test_twenty_micropascal_tolerance_uses_spl_label(reference_value: float) -> None:
    reference = LevelReference(reference_value=reference_value, reference_unit="Pa")

    assert reference.unit == "dB SPL"
    assert reference.label == "dB SPL re 20 µPa"
    assert reference.reference_value == reference_value


def test_noncanonical_pa_and_non_pa_references_are_generic() -> None:
    pascal = LevelReference(reference_value=1.0, reference_unit="Pa")
    voltage = LevelReference(reference_value=0.5, reference_unit="V")

    assert (pascal.unit, pascal.label) == ("dB", "dB re 1 Pa")
    assert (voltage.unit, voltage.label) == ("dB", "dB re 0.5 V")


def test_to_level_handles_scalar_array_zero_signed_and_complex_amplitudes() -> None:
    reference = LevelReference(reference_value=2e-5, reference_unit="Pa")
    amplitudes = np.array([[2e-5, -2e-4], [0.0, 2e-5j]])

    result = reference.to_level(amplitudes)

    assert isinstance(result, np.ndarray)
    assert result.shape == amplitudes.shape
    np.testing.assert_allclose(result, [[0.0, 20.0], [-240.0, 0.0]])
    assert reference.to_level(-2e-5) == pytest.approx(0.0)
    assert isinstance(reference.to_level(-2e-5), float)
    assert reference.to_level(2e-5j) == pytest.approx(0.0)


def test_array_like_input_preserves_numpy_shape() -> None:
    reference = LevelReference(reference_value=1.0, reference_unit="V")

    nested = reference.to_level([[1.0], [0.1]])
    zero_dimensional = reference.to_level(np.array(1.0))

    assert isinstance(nested, np.ndarray)
    assert nested.shape == (2, 1)
    assert isinstance(zero_dimensional, np.ndarray)
    assert zero_dimensional.shape == ()


def test_to_level_does_not_reapply_channel_calibration_factor() -> None:
    channel = ChannelMetadata(
        calibration=wd.ChannelCalibration(factor=0.42, unit="Pa"),
    )

    assert channel.level_reference.to_level(2e-5) == pytest.approx(0.0)


@pytest.mark.parametrize("reference_value", [True, "1"])
def test_level_reference_rejects_non_numeric_reference(reference_value: object) -> None:
    with pytest.raises(TypeError, match="positive finite number"):
        LevelReference(reference_value=reference_value, reference_unit="Pa")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("reference_value", [0.0, -1.0, np.inf, np.nan])
def test_level_reference_rejects_non_positive_or_non_finite_reference(reference_value: float) -> None:
    with pytest.raises(ValueError, match="positive finite number"):
        LevelReference(reference_value=reference_value, reference_unit="Pa")


def test_level_reference_validates_unit_text() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        LevelReference(reference_value=1.0, reference_unit=None)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="must not contain only whitespace"):
        LevelReference(reference_value=1.0, reference_unit="  ")
