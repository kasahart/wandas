"""Recipe contract for structural Frame trimming."""

import json

import dask.array as da
import numpy as np
from dask.array.core import Array as DaArray

from tests.frame_helpers import channel_first_values
from wandas.core.metadata import ChannelCalibration
from wandas.frames.channel import ChannelFrame, ChannelMetadata
from wandas.pipeline import RecipePlan


def test_trim_recipe_replay_preserves_structural_time_slice_contract() -> None:
    raw = np.arange(16, dtype=np.float32).reshape(2, 8)
    source = ChannelFrame(
        da.from_array(raw, chunks=(1, 4)),
        sampling_rate=8,
        label="recipe-source",
        metadata={"workflow": "trim"},
        channel_metadata=[
            ChannelMetadata(
                label="left",
                calibration=ChannelCalibration(factor=2.0, unit="Pa", ref=2e-5),
                extra={"sensor": "mic-1"},
            ),
            ChannelMetadata(
                label="right",
                calibration=ChannelCalibration(factor=0.5, unit="V", ref=1.0),
                extra={"sensor": "mic-2"},
            ),
        ],
        channel_ids=["left-id", "right-id"],
        source_time_offset=[0.25, 0.5],
    )

    processed = source.trim(start=0.25, end=0.75)
    plan = RecipePlan.from_frame(processed, input_names=("signal",))
    loaded = RecipePlan.from_dict(json.loads(json.dumps(plan.to_dict(), allow_nan=False)))
    replayed = loaded.apply({"signal": source})

    assert isinstance(replayed._data, DaArray)
    assert replayed.shape == (2, 4)
    assert replayed.labels == source.labels
    assert replayed.metadata == source.metadata
    assert [channel.id for channel in replayed.channels] == ["left-id", "right-id"]
    assert [channel.calibration for channel in replayed.channels] == [
        channel.calibration for channel in source.channels
    ]
    assert [channel.extra for channel in replayed.channels] == [
        {"sensor": "mic-1"},
        {"sensor": "mic-2"},
    ]
    np.testing.assert_array_equal(replayed.source_time_offset, np.array([0.5, 0.75]))
    np.testing.assert_array_equal(channel_first_values(replayed), channel_first_values(processed))
    assert replayed.operation_history == [
        {
            "operation": "wandas.audio.trim",
            "version": 1,
            "params": {"start": 0.25, "end": 0.75},
        }
    ]
