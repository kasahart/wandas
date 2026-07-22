import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipePlan
from wandas.pipeline.errors import RecipeExecutionError


def _frame(channels: int = 2) -> ChannelFrame:
    return ChannelFrame.from_numpy(np.arange(channels * 8.0).reshape(channels, 8), sampling_rate=8)


def test_source_time_offset_recipe_roundtrip_normalizes_and_replays() -> None:
    source = _frame().with_source_time_offset(np.array([0.25, 0.5]))
    plan = RecipePlan.from_dict(RecipePlan.from_frame(source).to_dict())
    replayed = plan.apply({"input_0": _frame()})
    np.testing.assert_array_equal(replayed.source_time_offset, np.array([0.25, 0.5]))
    assert replayed.operation_history[-1]["operation"] == "wandas.frame.with_source_time_offset"
    assert replayed.operation_history[-1]["params"] == {"value": [0.25, 0.5]}
    with pytest.raises(RecipeExecutionError, match="length must match"):
        plan.apply({"input_0": _frame(1)})


def test_scalar_source_time_offset_recipe_captures_authored_channel_arity() -> None:
    source = _frame(1).with_source_time_offset(1.25)
    plan = RecipePlan.from_dict(RecipePlan.from_frame(source).to_dict())
    assert source.operation_history[-1]["params"] == {"value": [1.25]}
    replayed = plan.apply({"input_0": _frame(1)})
    np.testing.assert_array_equal(replayed.source_time_offset, np.array([1.25]))
    with pytest.raises(RecipeExecutionError, match="length must match"):
        plan.apply({"input_0": _frame(2)})


def test_annotations_are_not_recipe_intent_and_runtime_annotations_win() -> None:
    planned = _frame().with_annotations(label="planned", metadata={"planned": True}).normalize()
    plan = RecipePlan.from_frame(planned)
    runtime = _frame().with_annotations(label="runtime", metadata={"runtime": True})
    replayed = plan.apply({"input_0": runtime})
    assert replayed.label == "runtime"
    assert replayed.metadata == {"runtime": True}
    assert all(node.operation != "wandas.frame.with_annotations" for node in plan.nodes)
