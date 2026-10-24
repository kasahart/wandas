"""Recipe compatibility for versioned RMS and sound-level contracts."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest

from tests.frame_helpers import channel_first_values
from wandas.core.metadata import ChannelCalibration, ChannelMetadata
from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipePlan, default_recipe_registry


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
    assert default_recipe_registry().require(operation_id, 2).version == 2
    assert replayed.operation_history == expected.operation_history
    assert [channel.calibration for channel in replayed.channels] == [
        channel.calibration for channel in expected.channels
    ]
    np.testing.assert_allclose(channel_first_values(replayed), channel_first_values(expected))
