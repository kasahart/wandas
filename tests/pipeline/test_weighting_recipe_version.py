"""Recipe compatibility for the versioned frequency-weighting filter."""

from __future__ import annotations

from typing import Any

import dask.array as da
import numpy as np
import pytest
from scipy import signal

from tests.frame_helpers import channel_first_values
from wandas.core.metadata import ChannelCalibration, ChannelMetadata
from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipePlan, default_recipe_registry
from wandas.processing.weighting import ABC_weighting

_SAMPLING_RATE = 16_000.0


def _values() -> np.ndarray:
    return np.random.default_rng(607).standard_normal((2, 4096))


def _source(values: np.ndarray) -> ChannelFrame:
    return ChannelFrame(
        da.from_array(values, chunks=(1, -1)),
        sampling_rate=_SAMPLING_RATE,
        channel_metadata=[
            ChannelMetadata(label=f"mic{index}", calibration=ChannelCalibration(factor=1.0, unit="Pa", ref=2e-5))
            for index in range(values.shape[0])
        ],
    )


def _released_bilinear(values: np.ndarray, curve: str) -> np.ndarray:
    """The released filter: the whole analog design through ``bilinear_zpk``."""
    sos = signal.zpk2sos(*signal.bilinear_zpk(*ABC_weighting(curve), _SAMPLING_RATE))
    return signal.sosfilt(sos, values, axis=-1)


def _payload(operation: str, version: int, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "wandas.recipe",
        "version": 2,
        "inputs": [{"id": "input-0", "name": "signal", "kind": "frame"}],
        "nodes": [
            {
                "id": "node-0",
                "operation": operation,
                "version": version,
                "inputs": ["input-0"],
                "params": {"$type": "map", "entries": [[key, value] for key, value in sorted(params.items())]},
            }
        ],
        "output": "node-0",
    }


# (operation, released version, Recipe params, curve, same call on pre-weighted data)
_CASES = [
    pytest.param("a_weighting", 1, {}, "A", lambda frame: frame, id="a_weighting"),
    pytest.param(
        "rms_trend",
        2,
        {"frame_length": 256, "hop_length": 128, "dB": True, "Aw": True},
        "A",
        lambda frame: frame.rms_trend(frame_length=256, hop_length=128, dB=True),
        id="rms_trend-dB",
    ),
    pytest.param(
        "rms_trend",
        2,
        {"frame_length": 256, "hop_length": 128, "dB": False, "Aw": True},
        "A",
        lambda frame: frame.rms_trend(frame_length=256, hop_length=128),
        id="rms_trend",
    ),
    pytest.param(
        "sound_level",
        2,
        {"freq_weighting": "A", "time_weighting": "Fast", "dB": True},
        "A",
        lambda frame: frame.sound_level(freq_weighting="Z", dB=True),
        id="sound_level-A-dB",
    ),
    pytest.param(
        "sound_level",
        2,
        {"freq_weighting": "C", "time_weighting": "Slow", "dB": False},
        "C",
        lambda frame: frame.sound_level(freq_weighting="Z", time_weighting="Slow"),
        id="sound_level-C",
    ),
]


@pytest.mark.parametrize(("operation", "version", "params", "curve", "unweighted_call"), _CASES)
def test_released_payload_replays_the_bilinear_filter(
    operation: str,
    version: int,
    params: dict[str, Any],
    curve: str,
    unweighted_call: Any,
) -> None:
    """A released payload keeps the bilinear-only filter; the current call records the next version."""
    values = _values()
    source = _source(values)
    operation_id = f"wandas.audio.{operation}"

    replayed = RecipePlan.from_dict(_payload(operation_id, version, params)).apply({"signal": source})

    expected = unweighted_call(_source(_released_bilinear(values, curve)))
    np.testing.assert_allclose(channel_first_values(replayed), channel_first_values(expected), rtol=1e-12, atol=1e-9)
    assert replayed.operation_history == [{"operation": operation_id, "version": version, "params": params}]
    assert replayed.labels == getattr(source, operation)(**params).labels

    current = getattr(source, operation)(**params)
    assert current.operation_history[-1]["version"] == version + 1
    assert default_recipe_registry().require(operation_id, version).version == version
    current_replayed = RecipePlan.from_frame(current, input_names=("signal",)).apply({"signal": source})
    np.testing.assert_allclose(channel_first_values(current_replayed), channel_first_values(current))
    # 16 kHz white noise has much of its power near Nyquist, where the two filters differ.
    assert not np.allclose(channel_first_values(current), channel_first_values(replayed), rtol=1e-3)
