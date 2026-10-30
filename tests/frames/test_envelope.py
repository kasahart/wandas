"""Hilbert amplitude envelope (``ChannelFrame.envelope``)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from dask.array.core import Array as DaArray
from scipy import signal as ss

import wandas as wd
from tests.frame_helpers import channel_first_values
from wandas.pipeline import RecipePlan
from wandas.processing import get_operation

_FS = 8000.0
_T = np.arange(8000) / _FS
_EDGE = 400  # samples left out near both ends, where the FFT-based analytic signal rings


def _am_signal() -> np.ndarray:
    modulation = 1.0 + 0.5 * np.cos(2 * np.pi * 10.0 * _T)
    return np.stack([2.0 * np.cos(2 * np.pi * 1000.0 * _T), modulation * np.cos(2 * np.pi * 1000.0 * _T)])


def test_envelope_recovers_tone_amplitude_and_am_modulation() -> None:
    frame = wd.from_numpy(_am_signal(), sampling_rate=_FS)
    env = channel_first_values(frame.envelope())

    inner = slice(_EDGE, -_EDGE)
    np.testing.assert_allclose(env[0, inner], 2.0, rtol=1e-3)
    np.testing.assert_allclose(env[1, inner], (1.0 + 0.5 * np.cos(2 * np.pi * 10.0 * _T))[inner], atol=2e-3)


def test_envelope_equals_scipy_hilbert_magnitude() -> None:
    data = np.random.default_rng(591).standard_normal((3, 1001))
    frame = wd.from_numpy(data, sampling_rate=_FS)
    np.testing.assert_allclose(channel_first_values(frame.envelope()), np.abs(ss.hilbert(data, axis=-1)), rtol=1e-12)


def test_envelope_is_lazy_keeps_frame_state_and_records_one_step() -> None:
    frame = wd.from_numpy(_am_signal(), sampling_rate=_FS, ch_labels=["tone", "am"]).with_calibration(
        {"tone": wd.ChannelCalibration(factor=0.5, unit="Pa")}
    )
    with patch.object(DaArray, "compute", autospec=True) as compute:
        env = frame.envelope()
        compute.assert_not_called()

    assert type(env) is wd.ChannelFrame
    assert isinstance(env._data, DaArray)
    assert env.sampling_rate == frame.sampling_rate
    assert env.n_samples == frame.n_samples
    assert env.labels == ["env(tone)", "env(am)"]
    np.testing.assert_array_equal(env.source_time_offset, frame.source_time_offset)
    assert env.channels[0].calibration.unit == "Pa"
    # The calibration factor is applied before the envelope is taken.
    np.testing.assert_allclose(channel_first_values(env)[0, _EDGE:-_EDGE], 1.0, rtol=1e-3)
    assert [entry["operation"] for entry in env.operation_history][-1] == "wandas.audio.envelope"
    assert len(env.operation_history) == len(frame.operation_history) + 1
    np.testing.assert_array_equal(np.asarray(frame._data), _am_signal())


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_envelope_keeps_input_precision(dtype: type[np.floating]) -> None:
    frame = wd.from_numpy(_am_signal().astype(dtype), sampling_rate=_FS)
    env = frame.envelope()
    assert env._data.dtype == dtype
    assert channel_first_values(env).dtype == dtype


def test_envelope_operation_is_registered() -> None:
    operation = get_operation("envelope")(_FS)
    assert operation.name == "envelope"
    assert operation.calculate_output_dtype(np.dtype(np.float32)) == np.float32


def test_envelope_recipe_roundtrip() -> None:
    source = wd.from_numpy(_am_signal(), sampling_rate=_FS)
    authored = source.band_pass_filter(500, 1500).envelope()
    plan = RecipePlan.from_dict(RecipePlan.from_frame(authored, input_names=("signal",)).to_dict())

    assert [node.operation for node in plan.nodes][-1] == "wandas.audio.envelope"
    replayed = plan.apply({"signal": source})
    np.testing.assert_allclose(channel_first_values(replayed), channel_first_values(authored), rtol=1e-12)
