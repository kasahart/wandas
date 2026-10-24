"""N-octave bands outside MoSQITo's nominal-frequency table (issue #491)."""

import numpy as np
import pytest

import wandas as wd
from wandas.processing.spectral import NOctSpectrum, noct_band_labels

pytest.importorskip("mosqito")

from mosqito.sound_level_meter import noct_spectrum  # noqa: E402


def _signal(sampling_rate: int) -> wd.ChannelFrame:
    rng = np.random.default_rng(491)
    data = rng.standard_normal((2, sampling_rate // 2))
    return wd.ChannelFrame.from_numpy(data, sampling_rate=sampling_rate)


@pytest.mark.parametrize(
    ("sampling_rate", "fmin", "fmax", "n", "n_bands", "first", "last"),
    [
        (16000, 20.0, 6300.0, 3, 26, 10**1.3, 6300.0),
        (96000, 25.0, 25000.0, 3, 31, 25.0, 10**4.4),
        (48000, 16.0, 8000.0, 1, 10, 10**1.2, 8000.0),
    ],
)
def test_noct_spectrum_returns_every_band_mosqito_computes(
    sampling_rate: int,
    fmin: float,
    fmax: float,
    n: int,
    n_bands: int,
    first: float,
    last: float,
) -> None:
    frame = _signal(sampling_rate)

    result = frame.noct_spectrum(fmin=fmin, fmax=fmax, n=n)

    expected, _ = noct_spectrum(frame.data.T, sampling_rate, fmin=fmin, fmax=fmax, n=n)
    assert result.shape == (2, n_bands)
    np.testing.assert_array_equal(result.data, np.asarray(expected).T)
    assert result.freqs.shape == (n_bands,)
    np.testing.assert_allclose(result.freqs[[0, -1]], [first, last])


def test_bands_inside_the_table_keep_their_nominal_labels() -> None:
    frame = _signal(16000)

    wide = frame.noct_spectrum(fmin=20, fmax=6300)
    inside = frame.noct_spectrum(fmin=25, fmax=6300)

    np.testing.assert_array_equal(wide.freqs[1:], inside.freqs)
    np.testing.assert_array_equal(wide.data[:, 1:], inside.data)
    assert inside.freqs[:4].tolist() == [25.0, 31.5, 40.0, 50.0]


def test_output_shape_matches_the_computed_bands() -> None:
    operation = NOctSpectrum(16000, fmin=20, fmax=6300, n=3)

    assert operation.calculate_output_shape((2, 8000)) == (2, 26)


def test_noct_band_labels_fill_only_the_bands_without_a_nominal_value() -> None:
    exact = 1000.0 * 10.0 ** (np.arange(-18, -14) / 10)
    nominal = np.array([25.0, 31.5])

    labels = noct_band_labels(exact, nominal)

    np.testing.assert_allclose(labels, [exact[0], exact[1], 25.0, 31.5])


def test_noct_band_labels_without_any_nominal_value_are_exact() -> None:
    exact = np.array([10.0, 12.589])

    np.testing.assert_array_equal(noct_band_labels(exact, np.array([])), exact)
