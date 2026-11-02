import numpy as np
import pytest
from scipy.signal import find_peaks

from wandas.processing.peaks import find_spectral_peaks


def _comb(freqs: np.ndarray, centres: list[float], heights: list[float]) -> np.ndarray:
    """Level curve with narrow triangular peaks on a -60 dB floor."""
    level = np.full(freqs.shape, -60.0)
    for centre, height in zip(centres, heights):
        level = np.maximum(level, height - 20.0 * np.abs(freqs - centre))
    return level


class TestFindSpectralPeaks:
    """Peak picking against scipy.signal.find_peaks and closed-form curves."""

    def setup_method(self) -> None:
        self.freqs = np.arange(0.0, 101.0)
        self.level = _comb(self.freqs, [10.0, 40.0, 70.0], [0.0, -20.0, -10.0])

    def test_peaks_match_scipy_find_peaks_sorted_by_frequency(self) -> None:
        expected, properties = find_peaks(self.level, prominence=10.0)
        peaks = find_spectral_peaks(self.level, self.freqs, min_prominence_db=10.0)

        np.testing.assert_array_equal(peaks["frequency"], self.freqs[expected])
        np.testing.assert_array_equal(peaks["level_db"], self.level[expected])
        np.testing.assert_array_equal(peaks["prominence_db"], properties["prominences"])
        np.testing.assert_array_equal(peaks["frequency"], [10.0, 40.0, 70.0])

    def test_prominence_threshold_drops_small_peaks(self) -> None:
        # Prominences over the -60 dB floor are 60, 40 and 50 dB.
        peaks = find_spectral_peaks(self.level, self.freqs, min_prominence_db=45.0)
        np.testing.assert_array_equal(peaks["frequency"], [10.0, 70.0])

    def test_frequency_range_and_max_peaks(self) -> None:
        in_range = find_spectral_peaks(self.level, self.freqs, fmin=20.0, fmax=80.0)
        np.testing.assert_array_equal(in_range["frequency"], [40.0, 70.0])

        # max_peaks keeps the highest levels, and the result stays sorted by frequency.
        highest = find_spectral_peaks(self.level, self.freqs, max_peaks=2)
        np.testing.assert_array_equal(highest["frequency"], [10.0, 70.0])

    def test_parabolic_interpolation_recovers_vertex_of_a_parabola(self) -> None:
        freqs = np.arange(0.0, 50.0) * 0.5
        vertex, top = 12.3, -3.0
        level = top - 2.0 * (freqs - vertex) ** 2

        peaks = find_spectral_peaks(level, freqs, min_prominence_db=1.0, interpolation="parabolic")

        # A parabola through three samples of a parabola is that parabola.
        np.testing.assert_allclose(peaks["frequency"], [vertex], rtol=1e-12)
        np.testing.assert_allclose(peaks["level_db"], [top], rtol=1e-12)
        bin_values = find_spectral_peaks(level, freqs, min_prominence_db=1.0)
        np.testing.assert_array_equal(bin_values["frequency"], [12.5])

    def test_max_peaks_keeps_lower_frequency_among_equal_levels(self) -> None:
        level = np.array([0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0])
        peaks = find_spectral_peaks(level, np.arange(7.0), min_prominence_db=1.0, max_peaks=1)
        np.testing.assert_array_equal(peaks["frequency"], [1.0])

    def test_edge_bins_are_never_peaks(self) -> None:
        level = np.array([10.0, -40.0, -40.0, -10.0, -40.0, -40.0, 10.0])
        peaks = find_spectral_peaks(level, np.arange(7.0), min_prominence_db=1.0, interpolation="parabolic")
        np.testing.assert_array_equal(peaks["frequency"], [3.0])

    def test_no_peak_returns_empty_arrays(self) -> None:
        peaks = find_spectral_peaks(np.zeros(16), np.arange(16.0))
        for values in peaks.values():
            assert values.shape == (0,)

    def test_input_is_not_modified(self) -> None:
        original = self.level.copy()
        find_spectral_peaks(self.level, self.freqs, interpolation="parabolic")
        np.testing.assert_array_equal(self.level, original)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"interpolation": "cubic"}, "Unknown peak interpolation"),
            ({"max_peaks": 0}, "max_peaks must be an integer of at least 1"),
            ({"max_peaks": 1.5}, "max_peaks must be an integer of at least 1"),
            ({"max_peaks": True}, "max_peaks must be an integer of at least 1"),
        ],
    )
    def test_invalid_options_raise(self, kwargs: dict[str, object], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            find_spectral_peaks(self.level, self.freqs, **kwargs)  # ty: ignore[invalid-argument-type]

    def test_mismatched_shapes_raise(self) -> None:
        with pytest.raises(ValueError, match="1-D level curve"):
            find_spectral_peaks(self.level, self.freqs[:-1])
