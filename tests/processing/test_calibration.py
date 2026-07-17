"""Contracts for lazy per-channel calibration broadcasting."""

import dask.array as da
import numpy as np
import pytest

from wandas.processing.calibration import apply_channel_factors


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
