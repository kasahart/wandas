import struct

import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipePlan, RecipeSerializationError
from wandas.pipeline.errors import RecipeExecutionError


def _frame(channels: int = 2) -> ChannelFrame:
    return ChannelFrame.from_numpy(np.arange(channels * 8.0).reshape(channels, 8), sampling_rate=8)


def test_source_time_offset_recipe_roundtrip_normalizes_and_replays() -> None:
    caller_offsets = np.array([0.25, 0.5])
    source = _frame().with_source_time_offset(caller_offsets)
    caller_offsets[:] = 9
    plan = RecipePlan.from_dict(RecipePlan.from_frame(source).to_dict())
    replayed = plan.apply({"input_0": _frame()})
    np.testing.assert_array_equal(source.source_time_offset, np.array([0.25, 0.5]))
    np.testing.assert_array_equal(replayed.source_time_offset, np.array([0.25, 0.5]))
    assert replayed.operation_history[-1]["operation"] == "wandas.frame.with_source_time_offset"
    assert replayed.operation_history[-1]["params"] == {"value": [0.25, 0.5]}
    with pytest.raises(RecipeExecutionError, match="length must match"):
        plan.apply({"input_0": _frame(1)})


def test_source_time_offset_recipe_normalizes_integer_public_values_to_float_payload() -> None:
    source = _frame().with_source_time_offset([1, 2])

    assert source.operation_history[-1]["params"] == {"value": [1.0, 2.0]}

    replayed = RecipePlan.from_dict(RecipePlan.from_frame(source).to_dict()).apply({"input_0": _frame()})
    np.testing.assert_array_equal(replayed.source_time_offset, np.array([1.0, 2.0]))


def test_scalar_source_time_offset_recipe_replays_across_channel_arity() -> None:
    source = _frame(1).with_source_time_offset(1.25)
    plan = RecipePlan.from_dict(RecipePlan.from_frame(source).to_dict())
    assert source.operation_history[-1]["params"] == {"value": 1.25}
    replayed = plan.apply({"input_0": _frame(2)})
    np.testing.assert_array_equal(replayed.source_time_offset, np.array([1.25, 1.25]))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_scalar_source_time_offset_recipe_rejects_non_finite_value_when_loaded(value: float) -> None:
    payload = RecipePlan.from_frame(_frame(1).with_source_time_offset(1.25)).to_dict()
    encoded_value = payload["nodes"][0]["params"]["entries"][0][1]
    encoded_value["data"] = struct.pack(">d", value).hex()

    with pytest.raises(RecipeSerializationError, match="params violate"):
        RecipePlan.from_dict(payload)


def test_explicit_single_item_source_offset_vector_rejects_stereo_replay() -> None:
    plan = RecipePlan.from_frame(_frame(1).with_source_time_offset([1.25]))

    with pytest.raises(RecipeExecutionError, match="length must match"):
        plan.apply({"input_0": _frame(2)})


def test_annotations_are_not_recipe_intent_and_runtime_annotations_win() -> None:
    planned = _frame().with_label("planned").with_metadata({"planned": True}).normalize()
    plan = RecipePlan.from_frame(planned)
    runtime = _frame().with_label("runtime").with_metadata({"runtime": True}, replace=True)
    replayed = plan.apply({"input_0": runtime})
    assert replayed.label == "runtime"
    assert replayed.metadata == {"runtime": True}
    assert all(node.operation != "wandas.frame.with_annotations" for node in plan.nodes)


def test_rename_recipe_replays_before_following_name_selector() -> None:
    planned = _frame().rename_channels({0: "renamed"})["renamed"]
    plan = RecipePlan.from_dict(RecipePlan.from_frame(planned).to_dict())

    assert [node.operation for node in plan.nodes][-2:] == [
        "wandas.channel.rename_channels",
        "wandas.frame.index",
    ]
    replayed = plan.apply({"input_0": _frame()})

    assert replayed.labels == ["renamed"]
    assert replayed.operation_history[-2]["operation"] == "wandas.channel.rename_channels"
