"""Recipe compatibility for versioned RMS and sound-level contracts."""

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
from wandas.processing.temporal import RmsTrend, SoundLevel


def _calibrated_source() -> ChannelFrame:
    return ChannelFrame(
        da.from_array(np.array([[1.0, 2.0, 3.0, 4.0]]), chunks=(1, -1)),
        sampling_rate=8,
        channel_metadata=[
            ChannelMetadata(
                label="mic",
                calibration=ChannelCalibration(factor=2.0, unit="Pa", ref=2e-5),
            )
        ],
    )


def _released_payload(operation: str, params: list[list[Any]]) -> dict[str, Any]:
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
                "params": {"$type": "map", "entries": params},
            }
        ],
        "output": "node-0",
    }


@pytest.mark.parametrize(
    ("operation", "params", "expected", "expected_label", "current_class"),
    [
        pytest.param(
            "wandas.audio.rms_trend",
            [["dB", True], ["frame_length", 4], ["hop_length", 2]],
            np.array([[100.96910013008056, 108.750612633917, 107.95880017344075]]),
            "RMS(mic)",
            RmsTrend,
            id="rms-trend",
        ),
        pytest.param(
            "wandas.audio.sound_level",
            [["dB", True]],
            np.array([[98.00799915372218, 104.41070558174084, 108.26386467355925, 110.99697579355491]]),
            "LZF(mic)",
            SoundLevel,
            id="sound-level",
        ),
    ],
)
def test_released_v1_payload_replays_golden_numerics_metadata_and_history(
    operation: str,
    params: list[list[Any]],
    expected: np.ndarray,
    expected_label: str,
    current_class: type[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A v1 payload must use its private kernel, never the corrected v2 one."""

    def fail_v2(_self: Any, _data: Any) -> Any:
        raise AssertionError("released v1 payload invoked the v2 numerical kernel")

    monkeypatch.setattr(current_class, "_process", fail_v2)
    plan = RecipePlan.from_dict(_released_payload(operation, params))
    replayed = plan.apply({"signal": _calibrated_source()})

    assert isinstance(replayed._data, DaArray)
    np.testing.assert_allclose(channel_first_values(replayed), expected, rtol=0.0, atol=5e-14)
    assert replayed.channels[0].label == expected_label
    assert replayed.channels[0].calibration == ChannelCalibration(
        factor=1.0,
        unit="Pa",
        ref=2e-5,
    )
    assert replayed.operation_history == [
        {
            "operation": operation,
            "version": 1,
            "params": dict(params),
        }
    ]


@pytest.mark.parametrize(
    ("operation", "params", "expected_shape"),
    [
        pytest.param(
            "wandas.audio.rms_trend",
            [["dB", True], ["frame_length", 4], ["hop_length", 2]],
            (0, 5),
            id="rms-trend",
        ),
        pytest.param(
            "wandas.audio.sound_level",
            [["dB", True]],
            (0, 8),
            id="sound-level",
        ),
    ],
)
def test_released_v1_payload_preserves_zero_channel_behavior(
    operation: str,
    params: list[list[Any]],
    expected_shape: tuple[int, int],
) -> None:
    source = ChannelFrame(
        da.from_array(np.empty((0, 8)), chunks=(0, 8)),
        sampling_rate=8,
    )

    replayed = RecipePlan.from_dict(_released_payload(operation, params)).apply({"signal": source})

    assert isinstance(replayed._data, DaArray)
    assert replayed.shape == expected_shape
    np.testing.assert_array_equal(replayed.data, np.empty(expected_shape))


@pytest.mark.parametrize("operation", ["rms_trend", "sound_level"])
@pytest.mark.parametrize("db_output", [False, True])
def test_public_level_operations_emit_and_roundtrip_version_2(
    operation: str,
    db_output: bool,
) -> None:
    source = _calibrated_source()
    if operation == "rms_trend":
        expected = source.rms_trend(frame_length=4, hop_length=2, dB=db_output)
    else:
        expected = source.sound_level(freq_weighting="Z", time_weighting="Fast", dB=db_output)

    plan = RecipePlan.from_frame(expected, input_names=("signal",))
    loaded = RecipePlan.from_dict(plan.to_dict())
    replayed = loaded.apply({"signal": source})

    operation_id = f"wandas.audio.{operation}"
    assert [(node.operation, node.version) for node in plan.nodes] == [(operation_id, 2)]
    assert default_recipe_registry().require(operation_id, 1).version == 1
    assert default_recipe_registry().require(operation_id, 2).version == 2
    assert replayed.operation_history == expected.operation_history
    assert [channel.calibration for channel in replayed.channels] == [
        channel.calibration for channel in expected.channels
    ]
    np.testing.assert_allclose(channel_first_values(replayed), channel_first_values(expected))
