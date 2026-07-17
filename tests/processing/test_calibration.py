"""Contracts for lazy per-channel calibration broadcasting."""

import dask.array as da
import numpy as np
import pytest

from wandas.processing.calibration import apply_channel_factors, derive_calibration_factors


def test_derive_calibration_factors_uses_known_rms_per_channel() -> None:
    factors = derive_calibration_factors(
        (0.5, 0.25),
        target_rms=(1.0, 2.0),
        ref=1.0,
    )

    assert factors == (2.0, 8.0)


def test_derive_calibration_factors_converts_level_and_broadcasts_target() -> None:
    factors = derive_calibration_factors(
        (0.5, 0.25),
        target_level=94.0,
        ref=2e-5,
    )

    target_rms = 2e-5 * 10 ** (94.0 / 20.0)
    np.testing.assert_allclose(factors, (target_rms / 0.5, target_rms / 0.25), rtol=1e-12, atol=0.0)


@pytest.mark.parametrize(
    ("target_rms", "target_level"),
    [(None, None), (1.0, 94.0)],
)
def test_derive_calibration_factors_requires_exactly_one_target(
    target_rms: float | None,
    target_level: float | None,
) -> None:
    with pytest.raises(ValueError, match="Exactly one calibration target is required"):
        derive_calibration_factors(
            (0.5,),
            target_rms=target_rms,
            target_level=target_level,
            ref=2e-5,
        )


@pytest.mark.parametrize("measured_rms", [(0.0,), (-1.0,), (float("nan"),), (float("inf"),)])
def test_derive_calibration_factors_rejects_invalid_measurement(measured_rms: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="Invalid measured calibration RMS"):
        derive_calibration_factors(measured_rms, target_rms=1.0, ref=1.0)


@pytest.mark.parametrize("measured_rms", ["bad", True, (1 + 2j,), (object(),)])
def test_derive_calibration_factors_rejects_non_real_measurement(measured_rms: object) -> None:
    with pytest.raises(TypeError, match="Invalid measured calibration RMS"):
        derive_calibration_factors(measured_rms, target_rms=1.0, ref=1.0)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "measured_rms",
    [
        [[1.0], [2.0]],
        [[1.0], [2.0, 3.0]],
        [],
    ],
)
def test_derive_calibration_factors_rejects_invalid_measurement_shape(measured_rms: object) -> None:
    with pytest.raises((TypeError, ValueError), match="Invalid measured calibration RMS"):
        derive_calibration_factors(measured_rms, target_rms=1.0, ref=1.0)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("ref", ["2e-5", True])
def test_derive_calibration_factors_rejects_non_numeric_reference(ref: object) -> None:
    with pytest.raises(TypeError, match="Invalid calibration reference"):
        derive_calibration_factors((0.5,), target_rms=1.0, ref=ref)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("ref", [0.0, -1.0, float("nan"), float("inf")])
def test_derive_calibration_factors_rejects_invalid_reference(ref: float) -> None:
    with pytest.raises(ValueError, match="Invalid calibration reference"):
        derive_calibration_factors((0.5,), target_rms=1.0, ref=ref)


@pytest.mark.parametrize("target_rms", [0.0, -1.0, float("nan"), float("inf")])
def test_derive_calibration_factors_rejects_invalid_target_rms(target_rms: float) -> None:
    with pytest.raises(ValueError, match="Invalid calibration target RMS"):
        derive_calibration_factors((0.5,), target_rms=target_rms, ref=1.0)


@pytest.mark.parametrize("target_level", [float("nan"), float("inf")])
def test_derive_calibration_factors_rejects_invalid_target_level(target_level: float) -> None:
    with pytest.raises(ValueError, match="Invalid calibration target level"):
        derive_calibration_factors((0.5,), target_level=target_level, ref=1.0)


def test_derive_calibration_factors_rejects_overflow() -> None:
    with pytest.raises(ValueError, match="Invalid derived calibration factors"):
        derive_calibration_factors(
            (np.nextafter(0.0, 1.0),),
            target_rms=float(np.finfo(float).max),
            ref=1.0,
        )


def test_derive_calibration_factors_rejects_target_channel_mismatch() -> None:
    with pytest.raises(ValueError, match="Calibration target length mismatch"):
        derive_calibration_factors((0.5, 0.25), target_rms=(1.0, 2.0, 3.0), ref=1.0)


def test_apply_channel_factors_broadcasts_across_every_non_channel_axis() -> None:
    raw = da.from_array(np.ones((2, 3, 4)), chunks=(1, -1, -1))

    calibrated = apply_channel_factors(raw, (2.0, 5.0))

    assert isinstance(calibrated, da.Array)
    np.testing.assert_array_equal(calibrated.compute()[0], np.full((3, 4), 2.0))
    np.testing.assert_array_equal(calibrated.compute()[1], np.full((3, 4), 5.0))


def test_apply_channel_factors_rejects_channel_count_mismatch() -> None:
    raw = da.from_array(np.ones((2, 4)), chunks=(1, -1))

    with pytest.raises(ValueError, match="Calibration factor length mismatch"):
        apply_channel_factors(raw, (2.0,))


def test_apply_channel_factors_requires_dask_data_with_a_channel_axis() -> None:
    with pytest.raises(TypeError, match="must be a Dask array"):
        apply_channel_factors(np.ones((1, 2)), (1.0,))  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="must have a channel axis"):
        apply_channel_factors(da.from_array(np.asarray(1.0)), ())


@pytest.mark.parametrize("factors", [(0.0,), (-1.0,), (float("nan"),), (True,)])
def test_apply_channel_factors_rejects_invalid_values(factors: tuple[float, ...]) -> None:
    raw = da.from_array(np.ones((1, 2)), chunks=(1, -1))

    with pytest.raises(ValueError, match="Invalid calibration factors"):
        apply_channel_factors(raw, factors)
