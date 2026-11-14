"""Recipe compatibility for the psychoacoustic result units (#652)."""

from __future__ import annotations

from typing import Any

import dask.array as da
import numpy as np
import pytest

from tests.frame_helpers import channel_first_values
from wandas.core.metadata import ChannelCalibration, ChannelMetadata
from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipePlan, default_recipe_registry

pytest.importorskip("mosqito")

_SAMPLING_RATE = 48_000.0


def _source() -> ChannelFrame:
    times = np.arange(int(_SAMPLING_RATE * 0.5)) / _SAMPLING_RATE
    values = np.vstack([0.2 * np.sin(2 * np.pi * 1000.0 * times), 0.1 * np.sin(2 * np.pi * 500.0 * times)])
    return ChannelFrame(
        da.from_array(values, chunks=(1, -1)),
        sampling_rate=_SAMPLING_RATE,
        channel_metadata=[
            ChannelMetadata(label="mic0", calibration=ChannelCalibration(factor=2.0, unit="Pa", ref=2e-5)),
            ChannelMetadata(label="mic1", calibration=ChannelCalibration(factor=1.0, unit="Pa", ref=2e-5)),
        ],
    )


def _payload(operation: str, version: int, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "wandas.recipe",
        "version": 3,
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


# (method, Recipe params, unit of the result)
_CASES = [
    pytest.param("loudness_zwtv", {"field_type": "free"}, "sone", id="loudness_zwtv"),
    pytest.param("roughness_dw", {"overlap": 0.5}, "asper", id="roughness_dw"),
    pytest.param("roughness_dw_spec", {"overlap": 0.5}, "asper/Bark", id="roughness_dw_spec"),
    pytest.param("sharpness_din", {"weighting": "din", "field_type": "free"}, "acum", id="sharpness_din"),
]


@pytest.mark.parametrize(("method", "params", "unit"), _CASES)
def test_result_carries_the_metric_unit_and_released_payload_keeps_the_input_unit(
    method: str,
    params: dict[str, Any],
    unit: str,
) -> None:
    """The current call records version 2 with the metric unit; version 1 replays the input unit."""
    source = _source()
    operation_id = f"wandas.audio.{method}"

    current = getattr(source, method)(**params)
    assert [(channel.unit, channel.ref) for channel in current.channels] == [(unit, 1.0), (unit, 1.0)]
    assert [channel.calibration.level for channel in current.channels] == ["none", "none"]
    assert current.operation_history[-1]["version"] == 2
    current_replayed = RecipePlan.from_frame(current, input_names=("signal",)).apply({"signal": source})
    assert [channel.unit for channel in current_replayed.channels] == [unit, unit]
    np.testing.assert_allclose(channel_first_values(current_replayed), channel_first_values(current))

    assert default_recipe_registry().require(operation_id, 1).version == 1
    replayed = RecipePlan.from_dict(_payload(operation_id, 1, params)).apply({"signal": source})
    assert [(channel.unit, channel.ref) for channel in replayed.channels] == [("Pa", 2e-5), ("Pa", 2e-5)]
    assert [channel.calibration.level for channel in replayed.channels] == ["amplitude", "amplitude"]
    assert replayed.operation_history == [{"operation": operation_id, "version": 1, "params": params}]
    assert replayed.labels == current.labels
    # Only the unit metadata differs between the versions.
    np.testing.assert_array_equal(channel_first_values(replayed), channel_first_values(current))
