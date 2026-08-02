"""Independent numerical contracts for the public ISTFT operation.

The spectra in this module are authored in SciPy's magnitude-scaled domain.  No
Wandas forward transform or spectral normalization helper is involved in making
the expected values.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import get_window

from wandas.processing import ISTFT

_SAMPLING_RATE = 8_000.0


@dataclass(frozen=True)
class _OracleCase:
    """Small, invertible ShortTimeFFT configuration used by the oracle."""

    n_fft: int
    win_length: int
    hop_length: int
    window: str
    n_frames: int = 5
    channels: int = 1


_CASES = (
    _OracleCase(n_fft=16, win_length=16, hop_length=8, window="boxcar"),
    _OracleCase(n_fft=16, win_length=12, hop_length=5, window="hann"),
    _OracleCase(n_fft=15, win_length=15, hop_length=7, window="hann"),
    _OracleCase(n_fft=15, win_length=11, hop_length=4, window="boxcar"),
)


def _make_independent_oracle(
    case: _OracleCase,
) -> tuple[ShortTimeFFT, np.ndarray, np.ndarray]:
    """Build a SciPy-domain spectrum and its independent Wandas input form.

    ``domain_spectrogram`` is the direct one-sided complex value consumed by
    ``ShortTimeFFT.istft``.  It deliberately contains DC, interior bins, and an
    endpoint bin.  Channel-dependent gain and phase make channel mixing visible.

    The final array is converted to Wandas' stored peak-amplitude convention by
    the explicit one-sided rule: for an even FFT only ``1:-1`` is doubled, while
    for an odd FFT every positive-frequency bin ``1:`` is doubled.
    """

    scipy_sft = ShortTimeFFT(
        win=np.asarray(get_window(case.window, case.win_length), dtype=np.float64),
        hop=case.hop_length,
        fs=_SAMPLING_RATE,
        fft_mode="onesided",
        mfft=case.n_fft,
        scale_to="magnitude",
    )
    if not scipy_sft.invertible:
        raise AssertionError(f"Oracle configuration is not invertible: {case!r}")

    n_frequency_bins = case.n_fft // 2 + 1
    channel_index = np.arange(case.channels, dtype=np.float64)[:, None, None]
    frequency_index = np.arange(n_frequency_bins, dtype=np.float64)[None, :, None]
    time_index = np.arange(case.n_frames, dtype=np.float64)[None, None, :]

    channel_gain = 1.0 + 0.31 * channel_index
    amplitude = channel_gain * (0.35 + 0.11 * frequency_index) * (1.0 + 0.07 * time_index)
    phase = 0.17 * frequency_index + 0.31 * time_index + 0.43 * channel_index
    domain_spectrogram = np.asarray(amplitude * np.exp(1j * phase), dtype=np.complex128)

    # Real endpoint values are valid for a real-valued inverse and ensure that
    # the even Nyquist endpoint is distinguishable from an odd final bin.
    frame_values = np.arange(case.n_frames, dtype=np.float64)
    domain_spectrogram[:, 0, :] = channel_gain[:, 0, 0, None] * (0.95 + 0.13 * frame_values)[None, :]
    if case.n_fft % 2 == 0:
        domain_spectrogram[:, -1, :] = channel_gain[:, 0, 0, None] * (0.65 + 0.09 * frame_values)[None, :]

    normalized_wandas = domain_spectrogram.copy()
    if case.n_fft % 2 == 0:
        normalized_wandas[:, 1:-1, :] *= 2.0
    else:
        normalized_wandas[:, 1:, :] *= 2.0

    return scipy_sft, domain_spectrogram, normalized_wandas


def _as_dask_input(values: np.ndarray) -> DaArray:
    """Create the channel-first lazy input required by ``ISTFT.process``."""

    return da.from_array(
        values.copy(),
        chunks=(1, values.shape[1], values.shape[2]),
    )


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("channels", [1, 2])
def test_istft_process_matches_independent_scipy_oracle(
    case: _OracleCase,
    channels: int,
) -> None:
    """ISTFT reconstructs the complete SciPy-domain inverse for mono/multi data."""

    case = replace(case, channels=channels)
    scipy_sft, scipy_domain, normalized_wandas = _make_independent_oracle(case)
    expected = scipy_sft.istft(scipy_domain)
    input_snapshot = normalized_wandas.copy()
    lazy_input = _as_dask_input(normalized_wandas)

    operation = ISTFT(
        sampling_rate=_SAMPLING_RATE,
        n_fft=case.n_fft,
        hop_length=case.hop_length,
        win_length=case.win_length,
        window=case.window,
    )
    actual_lazy = operation.process(lazy_input)

    assert isinstance(actual_lazy, DaArray)
    assert actual_lazy.shape == expected.shape
    assert actual_lazy.shape == operation.calculate_output_shape(lazy_input.shape)
    assert actual_lazy.dtype == np.dtype(np.float64)
    assert operation.length is None
    assert operation.calculate_output_dtype(lazy_input.dtype) == np.dtype(np.float64)

    actual = actual_lazy.compute()
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(lazy_input.compute(), input_snapshot)
    np.testing.assert_array_equal(normalized_wandas, input_snapshot)


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("channels", [1, 2])
def test_istft_process_trims_only_at_the_operation_boundary(
    case: _OracleCase,
    channels: int,
) -> None:
    """An explicit operation length trims SciPy's full independent output."""

    case = replace(case, channels=channels)
    scipy_sft, scipy_domain, normalized_wandas = _make_independent_oracle(case)
    expected_full = scipy_sft.istft(scipy_domain)
    length = expected_full.shape[-1] - 3
    expected = expected_full[..., :length]

    operation = ISTFT(
        sampling_rate=_SAMPLING_RATE,
        n_fft=case.n_fft,
        hop_length=case.hop_length,
        win_length=case.win_length,
        window=case.window,
        length=length,
    )
    lazy_input = _as_dask_input(normalized_wandas)
    actual_lazy = operation.process(lazy_input)

    assert isinstance(actual_lazy, DaArray)
    assert actual_lazy.shape == expected.shape
    assert actual_lazy.shape == operation.calculate_output_shape(lazy_input.shape)
    assert actual_lazy.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(actual_lazy.compute(), expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("case", _CASES)
def test_independent_fixture_is_sensitive_to_istft_contract_mutations(case: _OracleCase) -> None:
    """The fixture rejects scaling, endpoint, placement, phase, and gain mutations."""

    case = replace(case, channels=2)
    scipy_sft, scipy_domain, normalized_wandas = _make_independent_oracle(case)
    expected = scipy_sft.istft(scipy_domain)
    positive_bins = (
        (slice(None), slice(1, -1), slice(None))
        if case.n_fft % 2 == 0
        else (
            slice(None),
            slice(1, None),
            slice(None),
        )
    )

    missing_peak_factor = scipy_domain.copy()
    double_peak_factor = normalized_wandas.copy()
    double_peak_factor[positive_bins] *= 2.0
    double_denormalization = normalized_wandas.copy()
    double_denormalization[positive_bins] /= 2.0

    wrong_endpoint_parity = normalized_wandas.copy()
    if case.n_fft % 2 == 0:
        wrong_endpoint_parity[:, -1, :] *= 2.0
    else:
        wrong_endpoint_parity[:, -1, :] /= 2.0

    mutations = {
        "missing positive-frequency factor": missing_peak_factor,
        "positive-frequency factor applied twice": double_peak_factor,
        "positive-frequency bin divided twice": double_denormalization,
        "wrong endpoint parity": wrong_endpoint_parity,
        "frequency-bin shift": np.roll(normalized_wandas, 1, axis=1),
        "discarded complex phase": np.abs(normalized_wandas).astype(np.complex128),
        "wrong overall gain": normalized_wandas * 1.25,
    }

    for description, mutated_input in mutations.items():
        actual = (
            ISTFT(
                sampling_rate=_SAMPLING_RATE,
                n_fft=case.n_fft,
                hop_length=case.hop_length,
                win_length=case.win_length,
                window=case.window,
            )
            .process(_as_dask_input(mutated_input))
            .compute()
        )
        assert not np.allclose(actual, expected, rtol=2e-12, atol=2e-12), description
