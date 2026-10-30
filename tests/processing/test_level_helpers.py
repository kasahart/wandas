"""Processing helpers behind ``ChannelFrame.leq`` and ``percentile_levels``."""

import numpy as np
import pytest

from wandas.processing.temporal import (
    equivalent_level_db,
    log10_weighted_rms,
    normalize_freq_weighting,
    percentile_levels,
)


def test_log10_weighted_rms_matches_the_direct_rms() -> None:
    x = np.random.default_rng(596).standard_normal((2, 1000))
    np.testing.assert_allclose(log10_weighted_rms(x, 1000.0, "Z"), np.log10(np.sqrt(np.mean(x**2, axis=-1))))
    assert log10_weighted_rms(np.zeros((1, 10)), 1000.0, "A")[0] == -np.inf


def test_equivalent_level_db_floors_at_minus_240() -> None:
    result = equivalent_level_db(np.array([-np.inf, 0.0]), np.array([1.0, 2.0]), np.array([1.0, 2e-5]))
    np.testing.assert_allclose(result, [-240.0, 20 * np.log10(2.0 / 2e-5)])


def test_percentile_levels_shapes() -> None:
    levels = np.arange(200, dtype=np.float64).reshape(2, 100)
    assert percentile_levels(levels, 50).shape == (2,)
    assert percentile_levels(levels, np.array([5.0, 95.0])).shape == (2, 2)


def test_normalize_freq_weighting() -> None:
    assert normalize_freq_weighting(None) == "Z"
    assert normalize_freq_weighting("a") == "A"
    with pytest.raises(ValueError, match="frequency weighting"):
        normalize_freq_weighting("B")
