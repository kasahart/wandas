import numpy as np
import pytest

from wandas.processing.trigger import find_level_crossings


class TestFindLevelCrossings:
    """Level-crossing detection against hand-computed indices."""

    def test_rising_crossings_fire_on_first_sample_at_or_above_threshold(self) -> None:
        x = np.array([0.0, 0.5, 1.0, 2.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_equal(find_level_crossings(x, 1.0), [2, 5])

    def test_falling_crossings_fire_on_first_sample_at_or_below_threshold(self) -> None:
        x = np.array([0.0, -2.0, 0.0, -1.0, -1.5, 0.0])
        np.testing.assert_array_equal(find_level_crossings(x, -1.0, slope="falling"), [1, 3])

    def test_first_sample_is_never_a_crossing(self) -> None:
        x = np.array([5.0, 5.0, 0.0, 5.0])
        np.testing.assert_array_equal(find_level_crossings(x, 1.0), [3])

    def test_holdoff_suppresses_retriggers_within_the_window(self) -> None:
        x = np.zeros(20)
        x[[2, 4, 8, 15]] = 1.0
        np.testing.assert_array_equal(find_level_crossings(x, 0.5, holdoff=5), [2, 8, 15])
        np.testing.assert_array_equal(find_level_crossings(x, 0.5, holdoff=6), [2, 8, 15])
        np.testing.assert_array_equal(find_level_crossings(x, 0.5, holdoff=7), [2, 15])

    def test_no_crossing_returns_empty_int_array(self) -> None:
        result = find_level_crossings(np.zeros(10), 1.0)
        assert result.shape == (0,)
        assert result.dtype == np.int64

    def test_input_is_not_modified(self) -> None:
        x = np.array([0.0, 2.0, 0.0])
        original = x.copy()
        find_level_crossings(x, 1.0)
        np.testing.assert_array_equal(x, original)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"x": np.zeros((2, 3)), "threshold": 0.0}, "1-D signal"),
            ({"x": np.zeros(3), "threshold": 0.0, "slope": "both"}, "Unknown trigger slope"),
            ({"x": np.zeros(3), "threshold": 0.0, "holdoff": -1}, "non-negative"),
        ],
    )
    def test_invalid_arguments_raise(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            find_level_crossings(**kwargs)
