from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame
from wandas.pipeline import (
    RecipeExecutionError,
    RecipeExtractionError,
    RecipePlan,
    RecipeSerializationError,
)


def _frame() -> ChannelFrame:
    return ChannelFrame.from_numpy(np.ones((1, 16)), sampling_rate=8000)


def _payload() -> dict:
    return RecipePlan.from_frame(_frame().normalize()).to_dict()


def test_from_frame_raises_extraction_error_for_non_frame_input() -> None:
    with pytest.raises(RecipeExtractionError, match="requires a Wandas frame"):
        RecipePlan.from_frame(np.ones((1, 4)))


def test_loader_wraps_graph_validation_as_serialization_error() -> None:
    payload = _payload()
    payload["nodes"].append(copy.deepcopy(payload["nodes"][0]))

    with pytest.raises(RecipeSerializationError, match="Invalid Recipe graph"):
        RecipePlan.from_dict(payload)


def test_loader_rejects_malformed_canonical_value_tree() -> None:
    payload = _payload()
    payload["nodes"][0]["params"] = {"$type": "map", "entries": [["x", {"$type": "unknown"}]]}

    with pytest.raises(RecipeSerializationError, match="Unknown Recipe value"):
        RecipePlan.from_dict(payload)


def test_executor_reports_operation_failure_with_node_context() -> None:
    plan = RecipePlan.from_frame(_frame().low_pass_filter(cutoff=1000), input_names=("signal",))
    incompatible_rate = ChannelFrame.from_numpy(np.ones((1, 16)), sampling_rate=1000)

    with pytest.raises(RecipeExecutionError, match="Recipe operation failed") as exc_info:
        plan.apply({"signal": incompatible_rate})

    error = exc_info.value
    assert error.node_id == plan.nodes[0].id
    assert error.operation == plan.nodes[0].operation == "wandas.audio.lowpass_filter"
    assert error.cause is not None
    assert error.cause is error.__cause__
    assert f"Node: {error.node_id!r}" in str(error)


def test_executor_input_mismatch_error_has_no_node_context() -> None:
    plan = RecipePlan.from_frame(_frame().normalize(), input_names=("signal",))

    with pytest.raises(RecipeExecutionError, match="Recipe inputs do not match the plan") as exc_info:
        plan.apply({"other": _frame()})

    assert exc_info.value.node_id is None
    assert exc_info.value.operation is None
    assert exc_info.value.cause is None


def test_recipe_execution_error_attributes_survive_pickling() -> None:
    cause = ValueError("bad cutoff")
    error = RecipeExecutionError(
        "Recipe operation failed", node_id="node-0", operation="wandas.spectral.welch", cause=cause
    )

    restored = pickle.loads(pickle.dumps(error))

    assert str(restored) == "Recipe operation failed"
    assert (restored.node_id, restored.operation) == ("node-0", "wandas.spectral.welch")
    assert isinstance(restored.cause, ValueError)
    assert str(restored.cause) == "bad cutoff"
