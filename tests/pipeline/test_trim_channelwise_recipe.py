"""Recipe round-trip contracts for channel-wise Trim execution."""

import json

import dask.array as da
import numpy as np
from dask.array.core import Array as DaArray

from tests.frame_helpers import channel_first_values
from wandas.core.metadata import ChannelCalibration
from wandas.frames.channel import ChannelFrame, ChannelMetadata
from wandas.pipeline import RecipePlan


def _source() -> ChannelFrame:
    return ChannelFrame(
        da.from_array(np.arange(16, dtype=np.float64).reshape(2, 8), chunks=(1, 4)),
        sampling_rate=8,
        label="recipe-source",
        metadata={"workflow": {"name": "trim"}},
        channel_metadata=[
            ChannelMetadata(
                label="first",
                calibration=ChannelCalibration(factor=2.0, unit="Pa", ref=2e-5),
                extra={"position": 1},
            ),
            ChannelMetadata(
                label="second",
                calibration=ChannelCalibration(factor=0.25, unit="V", ref=1.0),
                extra={"position": 2},
            ),
        ],
        channel_ids=["first-id", "second-id"],
        source_time_offset=[0.125, 0.375],
    )


def test_trim_recipe_extract_serialize_deserialize_and_replay_preserves_contract() -> None:
    source = _source()
    source_values = channel_first_values(source).copy()
    processed = source.trim(start=0.25, end=0.75)

    plan = RecipePlan.from_frame(processed, input_names=("signal",))
    payload = json.loads(json.dumps(plan.to_dict(), allow_nan=False))
    loaded = RecipePlan.from_dict(payload)
    replayed = loaded.apply({"signal": source})

    assert loaded.to_dict() == plan.to_dict()
    assert [node.operation for node in loaded.nodes] == ["wandas.audio.trim"]
    assert isinstance(replayed, ChannelFrame)
    assert isinstance(replayed._data, DaArray)
    assert replayed.previous is source
    assert replayed.shape == processed.shape
    assert replayed._data.dtype == processed._data.dtype
    assert replayed._data.chunks == processed._data.chunks
    assert replayed.sampling_rate == processed.sampling_rate
    assert replayed.label == processed.label
    assert replayed.metadata == processed.metadata
    assert [channel.id for channel in replayed.channels] == ["first-id", "second-id"]
    assert replayed.labels == processed.labels
    assert [channel.calibration for channel in replayed.channels] == [
        channel.calibration for channel in processed.channels
    ]
    assert [channel.extra for channel in replayed.channels] == [channel.extra for channel in processed.channels]
    np.testing.assert_array_equal(replayed.source_time_offset, processed.source_time_offset)
    assert replayed.operation_history == processed.operation_history
    np.testing.assert_array_equal(channel_first_values(replayed), channel_first_values(processed))

    np.testing.assert_array_equal(channel_first_values(source), source_values)
    assert source.operation_history == []


def test_trim_recipe_replays_slice_boundary_shape_and_metadata() -> None:
    source = _source()
    raw = np.arange(16, dtype=np.float64).reshape(2, 8)
    start = 1.25
    end = 1.75
    expected_slice = slice(10, 14)
    expected_offset = np.array([1.125, 1.375])
    expected = raw[:, expected_slice] * np.array([[2.0], [0.25]], dtype=np.float64)
    processed = source.trim(start=start, end=end)
    plan = RecipePlan.from_frame(processed, input_names=("signal",))
    loaded = RecipePlan.from_dict(json.loads(json.dumps(plan.to_dict(), allow_nan=False)))
    replayed = loaded.apply({"signal": source})

    assert isinstance(replayed._data, DaArray)
    assert replayed.shape == expected.shape
    assert replayed._data.chunks == ((1, 1), (expected.shape[-1],))
    assert replayed.metadata == source.metadata
    assert replayed.labels == ["trim(first)", "trim(second)"]
    assert replayed.operation_history[-1] == {
        "operation": "wandas.audio.trim",
        "version": 1,
        "params": {"start": start, "end": end},
    }
    np.testing.assert_array_equal(channel_first_values(replayed), expected)
    np.testing.assert_array_equal(replayed.source_time_offset, expected_offset)
    assert replayed.source_time.shape == expected.shape
    if expected.shape[-1]:
        np.testing.assert_array_equal(replayed.source_time[:, 0], expected_offset)
