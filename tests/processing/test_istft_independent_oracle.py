"""Independent numerical contracts for the public ISTFT operation.

The spectra in this module are authored in SciPy's magnitude-scaled domain.  No
Wandas forward transform or spectral normalization helper is involved in making
the expected values.
"""

from __future__ import annotations

from dataclasses import replace

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

from tests.processing.istft_oracle_fixtures import (
    _CASES,
    _SAMPLING_RATE,
    _make_independent_oracle,
    _OracleCase,
)
from wandas.processing import ISTFT


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
    # Applying the inverse peak-amplitude factor twice changes 2*D to D/2.
    double_denormalization[positive_bins] /= 4.0

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
