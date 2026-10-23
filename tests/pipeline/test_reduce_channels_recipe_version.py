"""Recipe compatibility for the reference and offset kept by channel sum and mean."""

from __future__ import annotations

from typing import Any

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

from tests.frame_helpers import channel_first_values
from wandas.core.metadata import ChannelCalibration, ChannelMetadata
from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipePlan, default_recipe_registry


def _source() -> ChannelFrame:
    return ChannelFrame(
        da.from_array(np.array([[1.0, 2.0, 3.0, 4.0], [3.0, 2.0, 1.0, 0.0]]), chunks=(1, -1)),
        sampling_rate=8,
        channel_metadata=[
            ChannelMetadata(label=label, calibration=ChannelCalibration(factor=2.0, unit="V", ref=1e-3))
            for label in ("a", "b")
        ],
        source_time_offset=np.array([2.0, 5.0]),
    )


def _released_payload(operation: str) -> dict[str, Any]:
    """Return the literal schema shape written for the released v1 operation."""
    return {
        "schema": "wandas.recipe",
        "version": 2,
        "inputs": [{"id": "input-0", "name": "signal", "kind": "frame"}],
        "nodes": [
            {
                "id": "node-0",
                "operation": operation,
                "version": 1,
                "inputs": ["input-0"],
                "params": {"$type": "map", "entries": []},
            }
        ],
        "output": "node-0",
    }


@pytest.mark.parametrize(
    ("operation", "expected"),
    [("sum", np.array([[8.0, 8.0, 8.0, 8.0]])), ("mean", np.array([[4.0, 4.0, 4.0, 4.0]]))],
)
def test_released_v1_payload_resets_reference_and_offset(operation: str, expected: np.ndarray) -> None:
    replayed = RecipePlan.from_dict(_released_payload(f"wandas.audio.{operation}")).apply({"signal": _source()})

    assert isinstance(replayed._data, DaArray)
    np.testing.assert_array_equal(channel_first_values(replayed), expected)
    assert replayed.channels[0].calibration == ChannelCalibration(factor=1.0, unit="V", ref=1.0)
    np.testing.assert_array_equal(replayed.source_time_offset, np.array([0.0]))
    assert replayed.operation_history == [{"operation": f"wandas.audio.{operation}", "version": 1, "params": {}}]


@pytest.mark.parametrize("operation", ["sum", "mean"])
def test_public_reduction_emits_and_roundtrips_version_2(operation: str) -> None:
    source = _source()
    expected = getattr(source, operation)()

    plan = RecipePlan.from_frame(expected, input_names=("signal",))
    replayed = RecipePlan.from_dict(plan.to_dict()).apply({"signal": source})

    operation_id = f"wandas.audio.{operation}"
    assert [(node.operation, node.version) for node in plan.nodes] == [(operation_id, 2)]
    assert default_recipe_registry().require(operation_id, 1).version == 1
    assert replayed.channels[0].calibration == ChannelCalibration(factor=1.0, unit="V", ref=1e-3)
    np.testing.assert_array_equal(replayed.source_time_offset, np.array([2.0]))
    assert replayed.operation_history == expected.operation_history
    np.testing.assert_array_equal(channel_first_values(replayed), channel_first_values(expected))
