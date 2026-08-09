from __future__ import annotations

import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipeExecutionError, RecipePlan


def test_astype_recipe_extracts_serializes_and_replays_normalized_dtype() -> None:
    source = ChannelFrame.from_numpy(
        np.array([[1.125, 2.25, 4.5]], dtype=np.float64),
        sampling_rate=8.0,
        metadata={"recording": "source"},
        ch_labels=["mic"],
    ).with_source_time_offset(0.25)
    expected = source.astype(np.float32)

    payload = RecipePlan.from_frame(expected, input_names=("signal",)).to_dict()
    replayed = RecipePlan.from_dict(payload).apply({"signal": source})

    assert payload["nodes"][-1]["operation"] == "wandas.frame.astype"
    assert payload["nodes"][-1]["params"] == {
        "$type": "map",
        "entries": [["dtype", "float32"]],
    }
    assert type(replayed) is type(expected)
    assert replayed._data.dtype == expected._data.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(replayed._data.compute(), expected._data.compute())
    assert replayed.metadata == expected.metadata
    assert replayed.channels.to_list() == expected.channels.to_list()
    np.testing.assert_array_equal(replayed.source_time_offset, expected.source_time_offset)


@pytest.mark.parametrize("dtype", ["float16", "int16", "bool", "object", "complex64"])
def test_astype_recipe_rejects_invalid_real_frame_target_at_apply(dtype: str) -> None:
    source = ChannelFrame.from_numpy(np.ones(8, dtype=np.float64), sampling_rate=8.0)
    payload = RecipePlan.from_frame(source.astype("float32"), input_names=("signal",)).to_dict()
    payload["nodes"][-1]["params"]["entries"][0][1] = dtype

    with pytest.raises((ValueError, RecipeExecutionError)):
        RecipePlan.from_dict(payload).apply({"signal": source})
