from unittest.mock import patch

import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.processing import ApplyCalibration, Calibration, create_operation
from wandas.utils.dask_helpers import da_from_array


def test_calibration_from_rms_computes_per_channel_factors() -> None:
    calibration = Calibration.from_rms(
        (0.5, 0.25),
        target_rms=(1.0, 2.0),
        unit="Pa",
    )

    assert calibration.measured_rms == (0.5, 0.25)
    assert calibration.target_rms == (1.0, 2.0)
    assert calibration.factors == (2.0, 8.0)
    assert calibration.unit == "Pa"
    assert calibration.ref == pytest.approx(20e-6)


def test_calibration_from_rms_converts_sound_pressure_level_to_rms() -> None:
    calibration = Calibration.from_rms(
        (0.5,),
        target_level=94.0,
        unit="Pa",
    )

    expected_rms = 20e-6 * 10 ** (94.0 / 20.0)
    np.testing.assert_allclose(calibration.target_rms, (expected_rms,), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(calibration.factors, (expected_rms / 0.5,), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(calibration.target_levels, (94.0,), rtol=1e-12, atol=1e-12)


def test_calibration_from_rms_normalizes_unit_before_default_reference_resolution() -> None:
    calibration = Calibration.from_rms(
        (0.5,),
        target_level=94.0,
        unit=" Pa ",
    )

    expected_rms = 20e-6 * 10 ** (94.0 / 20.0)
    assert calibration.unit == "Pa"
    assert calibration.ref == pytest.approx(20e-6)
    np.testing.assert_allclose(calibration.target_rms, (expected_rms,), rtol=1e-12, atol=0.0)


def test_calibration_from_rms_broadcasts_scalar_target() -> None:
    calibration = Calibration.from_rms(
        (0.25, 0.5),
        target_rms=1.0,
        unit="m/s^2",
    )

    assert calibration.target_rms == (1.0, 1.0)
    assert calibration.factors == (4.0, 2.0)
    assert calibration.ref == 1.0


@pytest.mark.parametrize(
    ("target_rms", "target_level"),
    [
        (None, None),
        (1.0, 94.0),
    ],
)
def test_calibration_from_rms_invalid_target_contract_raises_error(
    target_rms: float | None,
    target_level: float | None,
) -> None:
    with pytest.raises(ValueError, match="Exactly one calibration target is required"):
        Calibration.from_rms(
            (0.5,),
            target_rms=target_rms,
            target_level=target_level,
            unit="Pa",
        )


@pytest.mark.parametrize("measured_rms", [(0.0,), (-1.0,), (float("nan"),), (float("inf"),)])
def test_calibration_from_rms_invalid_measurement_raises_error(measured_rms: tuple[float]) -> None:
    with pytest.raises(ValueError, match="Invalid measured calibration RMS"):
        Calibration.from_rms(measured_rms, target_rms=1.0, unit="Pa")


@pytest.mark.parametrize(
    "measured_rms",
    [
        "not-numeric",
        object(),
        [True],
        ["1.0"],
        [[1.0], [2.0, 3.0]],
    ],
)
def test_calibration_from_rms_non_numeric_measurement_raises_error(measured_rms: object) -> None:
    with pytest.raises(TypeError, match="Invalid measured calibration RMS"):
        Calibration.from_rms(measured_rms, target_rms=1.0, unit="Pa")  # ty: ignore[invalid-argument-type]


def test_calibration_from_rms_channel_mismatch_raises_error() -> None:
    with pytest.raises(ValueError, match="Calibration target length mismatch"):
        Calibration.from_rms((0.5, 0.25), target_rms=(1.0, 2.0, 3.0), unit="Pa")


def test_calibration_from_rms_non_finite_target_level_raises_error() -> None:
    with pytest.raises(ValueError, match="Invalid calibration target level"):
        Calibration.from_rms((0.5,), target_level=float("nan"), unit="Pa")


@pytest.mark.parametrize(
    ("unit", "ref", "error"),
    [
        ("", None, "Invalid calibration unit"),
        ("Pa", 0.0, "Invalid calibration reference"),
        ("Pa", float("nan"), "Invalid calibration reference"),
    ],
)
def test_calibration_from_rms_invalid_domain_metadata_raises_error(
    unit: str,
    ref: float | None,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        Calibration.from_rms((0.5,), target_rms=1.0, unit=unit, ref=ref)


def test_calibration_from_rms_non_numeric_reference_raises_error() -> None:
    with pytest.raises(TypeError, match="Invalid calibration reference"):
        Calibration.from_rms((0.5,), target_rms=1.0, unit="Pa", ref="20e-6")  # ty: ignore[invalid-argument-type]


def test_calibration_rejects_non_positive_requested_channel_count() -> None:
    calibration = Calibration.from_rms((0.5,), target_rms=1.0, unit="Pa")

    with pytest.raises(ValueError, match="Calibration channel count must be positive"):
        calibration.factors_for_channels(0)


def test_calibration_recipe_params_reject_missing_fields() -> None:
    with pytest.raises(ValueError, match="Invalid calibration Recipe parameters"):
        Calibration._from_recipe_params({"unit": "Pa"})


def test_calibration_recipe_params_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="Invalid calibration Recipe parameters"):
        Calibration._from_recipe_params(
            {
                "measured_rms": None,
                "target_rms": (1.0,),
                "unit": "Pa",
                "ref": 20e-6,
            }
        )


def test_apply_calibration_process_is_lazy_and_matches_analytical_gain() -> None:
    source = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    lazy_source = da_from_array(source, chunks=(1, -1))
    operation = ApplyCalibration(48_000, factors=(2.0, 0.5))

    with patch.object(DaArray, "compute", autospec=True, side_effect=AssertionError("unexpected compute")):
        result = operation.process(lazy_source)

    assert isinstance(result, DaArray)
    assert result.shape == source.shape
    assert result.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(
        result.compute(),
        np.array([[2.0, -4.0], [1.5, -2.0]]),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(source, np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32))


def test_apply_calibration_single_factor_broadcasts_to_channels() -> None:
    operation = ApplyCalibration(48_000, factors=(4.0,))
    data = np.array([[0.25, -0.25], [0.5, -0.5]])

    np.testing.assert_allclose(operation._process(data), data * 4.0, rtol=0.0, atol=0.0)


def test_apply_calibration_channel_mismatch_raises_error() -> None:
    operation = ApplyCalibration(48_000, factors=(1.0, 2.0, 3.0))

    with pytest.raises(ValueError, match="Calibration channel mismatch"):
        operation._process(np.ones((2, 8)))


@pytest.mark.parametrize("factors", [(), (0.0,), (-1.0,), (float("nan"),), (float("inf"),)])
def test_apply_calibration_invalid_factors_raise_error(factors: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="Invalid calibration factors"):
        ApplyCalibration(48_000, factors=factors)


def test_apply_calibration_snapshots_caller_owned_factors() -> None:
    factors = [2.0, 4.0]
    operation = ApplyCalibration(48_000, factors=factors)

    factors[0] = 99.0

    assert operation.factors == (2.0, 4.0)
    assert operation.params == {"factors": (2.0, 4.0)}


def test_apply_calibration_is_registered_with_display_name() -> None:
    operation = create_operation("apply_calibration", 48_000, factors=(2.0,))

    assert isinstance(operation, ApplyCalibration)
    assert operation.get_display_name() == "cal"
