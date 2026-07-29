"""Recipe round-trip contracts for channel-wise HPSS execution."""

import json

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

from tests.frame_helpers import channel_first_values
from wandas.core.metadata import ChannelCalibration
from wandas.frames.channel import ChannelFrame, ChannelMetadata
from wandas.pipeline import RecipePlan

_SAMPLING_RATE = 8_000
_SAMPLES = 256
_PARAMS = {
    "kernel_size": (7, 9),
    "power": 2.0,
    "margin": (1.0, 2.0),
    "n_fft": 64,
    "hop_length": 16,
    "win_length": 64,
    "window": "hann",
    "center": True,
    "pad_mode": "constant",
}


def _source() -> tuple[ChannelFrame, np.ndarray]:
    time = np.arange(_SAMPLES, dtype=np.float64) / _SAMPLING_RATE
    values = np.stack(
        [
            np.sin(2 * np.pi * 220 * time) + 0.5 * (np.arange(_SAMPLES) % 59 == 0),
            0.6 * np.sin(2 * np.pi * 440 * time) + 0.4 * (np.arange(_SAMPLES) % 71 == 0),
        ]
    )
    frame = ChannelFrame(
        da.from_array(values, chunks=(1, 64)),
        sampling_rate=_SAMPLING_RATE,
        label="recipe-source",
        metadata={"workflow": {"name": "hpss"}},
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
    return frame, values


@pytest.mark.parametrize(
    "method",
    ["hpss_harmonic", "hpss_percussive"],
)
def test_hpss_recipe_extract_serialize_deserialize_and_replay_preserves_contract(
    method: str,
) -> None:
    source, caller_values = _source()
    caller_values_before = caller_values.copy()
    source_values = channel_first_values(source).copy()
    source_metadata = source.metadata
    source_channels = source.channels.to_list()
    source_offsets = source.source_time_offset.copy()
    source_lineage = source.lineage
    processed = getattr(source, method)(**_PARAMS)

    plan = RecipePlan.from_frame(processed, input_names=("signal",))
    payload = json.loads(json.dumps(plan.to_dict(), allow_nan=False))
    loaded = RecipePlan.from_dict(payload)
    replayed = loaded.apply({"signal": source})

    assert payload["schema"] == "wandas.recipe"
    assert payload["version"] == 2
    assert payload["inputs"] == [{"id": "input-0", "name": "signal", "kind": "frame"}]
    assert len(payload["nodes"]) == 1
    assert payload["nodes"][0]["operation"] == f"wandas.audio.{method}"
    assert payload["nodes"][0]["version"] == 1
    assert loaded.to_dict() == plan.to_dict()

    assert isinstance(replayed, ChannelFrame)
    assert isinstance(replayed._data, DaArray)
    assert replayed.previous is source
    assert replayed.shape == processed.shape
    assert replayed._data.dtype == processed._data.dtype
    assert replayed._data.chunks == processed._data.chunks
    assert replayed.sampling_rate == processed.sampling_rate
    np.testing.assert_array_equal(replayed.time, processed.time)
    assert replayed.label == processed.label
    assert replayed.metadata == processed.metadata
    assert [channel.id for channel in replayed.channels] == ["first-id", "second-id"]
    assert replayed.labels == processed.labels
    assert [channel.calibration for channel in replayed.channels] == [
        channel.calibration for channel in processed.channels
    ]
    assert [channel.extra for channel in replayed.channels] == [channel.extra for channel in processed.channels]
    np.testing.assert_array_equal(replayed.source_time_offset, processed.source_time_offset)
    np.testing.assert_array_equal(replayed.source_time, processed.source_time)
    assert replayed.operation_history == processed.operation_history
    assert replayed.lineage.operation is not None
    assert replayed.lineage.operation.operation_id == f"wandas.audio.{method}"
    assert replayed.lineage.inputs == (source.lineage,)
    np.testing.assert_array_equal(
        channel_first_values(replayed),
        channel_first_values(processed),
    )

    np.testing.assert_array_equal(channel_first_values(source), source_values)
    np.testing.assert_array_equal(caller_values, caller_values_before)
    assert source.metadata == source_metadata
    assert source.channels.to_list() == source_channels
    np.testing.assert_array_equal(source.source_time_offset, source_offsets)
    assert source.operation_history == []
    assert source.lineage is source_lineage
