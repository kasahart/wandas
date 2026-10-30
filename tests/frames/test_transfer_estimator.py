"""H1 and H2 transfer-function estimators (``transfer_function(estimator=...)``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
from dask.array.core import Array as DaArray
from scipy import signal as ss

import wandas as wd
from tests.frame_helpers import channel_first_values
from tests.pairwise_test_helpers import make_pairwise_source
from wandas.frames.pairwise import TransferFunctionFrame
from wandas.io import wdf_frames
from wandas.pipeline import RecipePlan
from wandas.processing import create_operation
from wandas.processing.spectral_contracts import transfer_function_estimate

_FS = 1024.0
_N = 1 << 15
_PARAMS: dict[str, Any] = {"n_fft": 256, "win_length": 256, "hop_length": 128, "window": "hann"}


def _measured_system(*, input_noise: float = 0.0, output_noise: float = 0.0) -> wd.ChannelFrame:
    """Return channels (measured input, measured output) of a 2-tap FIR system ``y = 0.5 x + 0.25 x[n-1]``."""
    rng = np.random.default_rng(587)
    x = rng.standard_normal(_N)
    y = ss.lfilter([0.5, 0.25], [1.0], x)
    measured_x = x + input_noise * rng.standard_normal(_N)
    measured_y = y + output_noise * rng.standard_normal(_N)
    return wd.from_numpy(np.stack([measured_x, measured_y]), sampling_rate=_FS, ch_labels=["in", "out"])


def _true_gain(freqs: np.ndarray) -> np.ndarray:
    _, response = ss.freqz([0.5, 0.25], [1.0], worN=freqs, fs=_FS)
    return np.abs(response)


def _out_in(frame: TransferFunctionFrame) -> np.ndarray:
    return np.asarray(frame.select_pair(1, 0).data)


def test_h1_and_h2_agree_without_noise() -> None:
    source = _measured_system()
    h1 = source.transfer_function(**_PARAMS)
    h2 = source.transfer_function(**_PARAMS, estimator="H2")

    assert h1.estimator == "H1"
    assert h2.estimator == "H2"
    # Only the window's small leakage bias keeps the coherence below 1.
    np.testing.assert_allclose(_out_in(h2), _out_in(h1), rtol=5e-3)
    np.testing.assert_allclose(np.abs(_out_in(h2)), _true_gain(h2.freqs), rtol=5e-3)


def test_input_noise_biases_h1_low_and_leaves_h2_near_the_true_gain() -> None:
    source = _measured_system(input_noise=0.5)
    h1 = np.abs(_out_in(source.transfer_function(**_PARAMS)))
    h2 = np.abs(_out_in(source.transfer_function(**_PARAMS, estimator="H2")))
    true_gain = _true_gain(source.transfer_function(**_PARAMS).freqs)

    # With input noise variance 0.25, H1 is scaled by 1 / (1 + 0.25).
    np.testing.assert_allclose(np.median(h1 / true_gain), 0.8, rtol=0.03)
    np.testing.assert_allclose(np.median(h2 / true_gain), 1.0, rtol=0.03)


def test_h1_over_h2_is_the_coherence() -> None:
    source = _measured_system(input_noise=0.3, output_noise=0.2)
    h1 = _out_in(source.transfer_function(**_PARAMS))
    h2 = _out_in(source.transfer_function(**_PARAMS, estimator="H2"))
    coherence = np.asarray(source.coherence(**_PARAMS).select_pair(1, 0).data)

    np.testing.assert_allclose(h1 / h2, coherence, rtol=1e-9, atol=1e-12)


def test_h2_matches_scipy_cross_and_auto_spectra() -> None:
    source = _measured_system(input_noise=0.3)
    data = channel_first_values(source)
    kwargs = {"fs": _FS, "nperseg": 256, "noverlap": 128, "nfft": 256, "window": "hann", "scaling": "spectrum"}
    _, p_out_out = ss.csd(data[1], data[1], **kwargs)
    _, p_in_out = ss.csd(data[1], data[0], **kwargs)  # SciPy csd(x, y) = conj(X) * Y

    actual = _out_in(source.transfer_function(**_PARAMS, estimator="H2"))
    np.testing.assert_allclose(actual, p_out_out / p_in_out, rtol=1e-9)


def test_h2_has_the_h1_unit_and_reference_and_stays_lazy() -> None:
    source = make_pairwise_source(n_channels=2)
    with patch.object(DaArray, "compute", autospec=True) as compute:
        h1 = source.transfer_function(n_fft=32, win_length=32, hop_length=16)
        h2 = source.transfer_function(n_fft=32, win_length=32, hop_length=16, estimator="H2")
        compute.assert_not_called()

    assert type(h2) is TransferFunctionFrame
    assert isinstance(h2._data, DaArray)
    assert h2.pair_state == h1.pair_state
    assert h2.denominator_role == "input"
    assert h2.definition == "h2_input_cross_denominator"
    assert h1.definition == "canonical_input_denominator"
    assert h2.operation_history[-1]["params"]["estimator"] == "H2"
    # Diagonal pairs are 1 for both estimators.
    np.testing.assert_allclose(np.asarray(h2.select_pair(0, 0).data), 1.0, rtol=1e-9)
    # Selection keeps the estimator.
    assert h2.select_pair(1, 0).estimator == "H2"


def test_csd_derivation_uses_the_same_estimator_without_recomputing() -> None:
    source = _measured_system(input_noise=0.3)
    csd = source.csd(**_PARAMS)
    with patch.object(DaArray, "compute", autospec=True) as compute:
        derived = csd.transfer_function(estimator="H2")
        compute.assert_not_called()
    direct = source.transfer_function(**_PARAMS, estimator="H2")

    assert derived.estimator == "H2"
    np.testing.assert_allclose(channel_first_values(derived), channel_first_values(direct), rtol=1e-10)
    assert derived.operation_history[-1]["params"] == {"estimator": "H2"}


def test_h2_survives_wdf_roundtrip(tmp_path: Path) -> None:
    frame = _measured_system(input_noise=0.3).transfer_function(**_PARAMS, estimator="H2")
    path = tmp_path / "h2.wdf"
    frame.save(path)
    loaded = cast(TransferFunctionFrame, wd.load(path))

    assert type(loaded) is TransferFunctionFrame
    assert loaded.estimator == "H2"
    assert loaded.definition == "h2_input_cross_denominator"
    np.testing.assert_allclose(channel_first_values(loaded), channel_first_values(frame), equal_nan=True)


def test_wdf_rejects_an_h2_definition_on_the_legacy_output_denominator() -> None:
    frame = make_pairwise_source(n_channels=2).transfer_function(n_fft=8, win_length=8, hop_length=4)
    state = wdf_frames._pairwise_state(frame, fields={"scaling", "denominator_role", "definition"})
    state.update(denominator_role="output", definition="h2_input_cross_denominator")
    with pytest.raises(ValueError, match="definition"):
        wdf_frames._validate_transfer_constructor_state(state, frame._data)


@pytest.mark.parametrize("from_csd", [False, True])
def test_h2_recipe_roundtrip_replays_the_estimator(from_csd: bool) -> None:
    source = _measured_system(input_noise=0.3)
    authored = (
        source.csd(**_PARAMS).transfer_function(estimator="H2")
        if from_csd
        else source.transfer_function(**_PARAMS, estimator="H2")
    )
    plan = RecipePlan.from_dict(RecipePlan.from_frame(authored, input_names=("signal",)).to_dict())
    replayed = cast(TransferFunctionFrame, plan.apply({"signal": source}))

    assert replayed.estimator == "H2"
    np.testing.assert_allclose(channel_first_values(replayed), channel_first_values(authored), rtol=1e-12)


def test_default_recipe_omits_the_estimator() -> None:
    frame = make_pairwise_source(n_channels=2).transfer_function(n_fft=8, win_length=8, hop_length=4)
    payload = RecipePlan.from_frame(frame, input_names=("signal",)).to_dict()
    assert "estimator" not in str(payload)


@pytest.mark.parametrize("estimator", ["h2", "H3", None])
def test_unknown_estimator_is_rejected(estimator: object) -> None:
    source = make_pairwise_source(n_channels=2)
    with pytest.raises(ValueError, match="estimator"):
        source.transfer_function(n_fft=8, win_length=8, hop_length=4, estimator=cast(Any, estimator))
    with pytest.raises(ValueError, match="estimator"):
        create_operation("transfer_function_from_csd", 256.0, estimator=estimator)
    with pytest.raises(ValueError, match="estimator"):
        transfer_function_estimate(np.ones((2, 2, 3), dtype=complex), cast(Any, estimator))


def test_h2_zero_cross_spectrum_is_complex_nan() -> None:
    cross = np.zeros((2, 2, 3), dtype=complex)
    cross[0, 0] = cross[1, 1] = 1.0
    result = transfer_function_estimate(cross, "H2")
    assert np.isnan(result[1, 0]).all()
    assert np.isnan(result[0, 1]).all()
    np.testing.assert_allclose(result[0, 0], 1.0)


def _transfer_constructor_kwargs(frame: TransferFunctionFrame) -> dict[str, Any]:
    return {
        "data": frame._data,
        "sampling_rate": frame.sampling_rate,
        "n_fft": frame.n_fft,
        "window": frame.window,
        "pair_state": frame.pair_state,
        "scaling": frame.scaling,
        "source_channel_ids": frame.source_channel_ids,
    }


def test_transfer_frame_rejects_an_unknown_or_legacy_h2_estimator() -> None:
    # Equal units and references give the same domain for either denominator role.
    frame = make_pairwise_source(n_channels=2, units=("V", "V"), references=(1.0, 1.0)).transfer_function(
        n_fft=8, win_length=8, hop_length=4
    )
    kwargs = _transfer_constructor_kwargs(frame)
    assert TransferFunctionFrame(**kwargs, denominator_role="output").definition == "legacy_output_denominator"
    with pytest.raises(ValueError, match="estimator must be 'H1' or 'H2'"):
        TransferFunctionFrame(**kwargs, estimator=cast(Any, "H3"))
    with pytest.raises(ValueError, match="H1 only"):
        TransferFunctionFrame(**kwargs, denominator_role="output", estimator="H2")
