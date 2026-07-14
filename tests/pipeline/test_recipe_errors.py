from __future__ import annotations

import copy

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


def test_extraction_error_is_distinct_from_validation_and_execution_errors() -> None:
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

    with pytest.raises(RecipeExecutionError, match="Recipe operation failed"):
        plan.apply({"signal": incompatible_rate})
