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
    assert reference.label == "dB SPL re 2e-05 Pa"
    assert reference.minimum_level == -240.0
    with pytest.raises(FrozenInstanceError):
        reference.reference_value = 1.0  # ty: ignore[invalid-assignment]


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
