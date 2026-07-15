import json
from unittest.mock import patch

import dask.array as da
import numpy as np
from dask.array.core import Array as DaArray

from wandas.core.metadata import ChannelMetadata
from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipePlan
from wandas.processing import Calibration


def _source(data: np.ndarray, *, offset: float) -> ChannelFrame:
    return ChannelFrame(
        data=da.from_array(data, chunks=(1, -1)),
        sampling_rate=48_000,
        label="measurement",
        metadata={"session": "calibration-recipe"},
        channel_metadata=[
            ChannelMetadata(label="left", unit="V", ref=1.0, extra={"serial": "A"}),
            ChannelMetadata(label="right", unit="V", ref=1.0, extra={"serial": "B"}),
        ],
        source_time_offset=(offset, offset + 0.25),
    )


def test_calibrate_recipe_extracts_serializes_loads_and_applies_without_compute() -> None:
    calibration = Calibration.from_rms(
        (0.5, 0.25),
        target_rms=(1.0, 2.0),
        unit="Pa",
    )
    source = _source(np.array([[1.0, -1.0], [1.0, -1.0]]), offset=0.5)
    replay_source = _source(np.array([[2.0, -2.0], [0.5, -0.5]]), offset=1.5)

    with patch.object(DaArray, "compute", autospec=True, side_effect=AssertionError("unexpected compute")):
        calibrated = source.calibrate(calibration)
        plan = RecipePlan.from_frame(calibrated, input_names=("signal",))
        payload = plan.to_dict()
        loaded = RecipePlan.from_dict(payload)
        replayed = loaded.apply({"signal": replay_source})

    assert loaded.to_dict() == payload
    json.dumps(payload, allow_nan=False)
    assert [node.operation for node in loaded.nodes] == ["wandas.audio.calibrate"]
    assert isinstance(replayed, ChannelFrame)
    assert isinstance(replayed._data, DaArray)
    np.testing.assert_allclose(replayed.compute(), [[4.0, -4.0], [4.0, -4.0]], rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(replayed.source_time_offset, replay_source.source_time_offset)
    assert replayed.metadata == replay_source.metadata
    assert replayed.labels == ["cal(left)", "cal(right)"]
    assert [channel.unit for channel in replayed.channels] == ["Pa", "Pa"]
    assert [channel.ref for channel in replayed.channels] == [20e-6, 20e-6]
    assert [channel.extra for channel in replayed.channels] == [{"serial": "A"}, {"serial": "B"}]
    assert replayed.operation_history[-1]["params"] == {
        "measured_rms": [0.5, 0.25],
        "ref": 20e-6,
        "target_rms": [1.0, 2.0],
        "unit": "Pa",
    }
