"""Depth-independent lineage traversal and deterministic shared graph replay."""

import json
from unittest.mock import patch

import dask.array as da
import numpy as np

from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipeOperation, RecipePlan, default_recipe_registry
from wandas.processing.semantic import InputBinding, LineageNode, SemanticOperation, freeze_params


def test_deep_lineage_history_and_recipe_roundtrip_remain_lazy() -> None:
    source = ChannelFrame.from_numpy(np.arange(8.0), sampling_rate=8)
    result = source
    count = 1200
    with patch.object(da.Array, "compute", side_effect=AssertionError("must remain lazy")):
        for index in range(count):
            result = result.rename_channels({0: f"channel_{index}"})
        history = result.operation_history
        assert len(history) == count
        plan = RecipePlan.from_frame(result, input_names=("signal",))
        assert len(plan.nodes) == count
        payload = json.loads(json.dumps(plan.to_dict(), allow_nan=False))
        restored = RecipePlan.from_dict(payload)
        assert restored.to_dict() == payload
        replayed = restored.apply({"signal": source})
        assert replayed.labels == result.labels
        assert replayed.operation_history == history
        assert RecipePlan.from_frame(replayed, input_names=("signal",)).to_dict() == payload
    np.testing.assert_array_equal(replayed.data, source.data)


def test_shared_graph_keeps_depth_first_array_input_discovery_order() -> None:
    source = ChannelFrame.from_numpy(np.arange(8.0), sampling_rate=8)
    shared = source.rename_channels({0: "shared"})
    addend = np.ones(8)
    factor = np.full(8, 2.0)
    result = (shared + addend) + (shared * factor)
    plan = RecipePlan.from_frame(result, input_names=("signal", "addend", "factor"))
    assert [item.kind for item in plan.inputs] == ["frame", "array", "array"]
    assert [node.inputs for node in plan.nodes] == [
        ("input-0",),
        ("node-0", "input-1"),
        ("node-0", "input-2"),
        ("node-1", "node-2"),
    ]
    assert len(result.operation_history) == 4
    replayed = RecipePlan.from_dict(plan.to_dict()).apply({"signal": source, "addend": addend, "factor": factor})
    np.testing.assert_array_equal(replayed.data, np.arange(8.0) * 3 + 1)


def test_external_array_before_nested_frame_keeps_input_discovery_order() -> None:
    source = ChannelFrame.from_numpy(np.arange(8.0), sampling_rate=8)
    nested = source.rename_channels({0: "nested"})
    values = np.ones(8)
    bindings = (InputBinding("offset", "array"), InputBinding("frame", "frame"))
    definition = RecipeOperation("test.array_first", 1, (bindings,), lambda inputs, _params: inputs[1] + inputs[0])
    registry = default_recipe_registry().with_operation(definition)
    # A valid third-party declaration can bind an array before its receiver Frame.
    lineage = LineageNode(
        SemanticOperation(definition.operation_id, 1, bindings, freeze_params({})), (None, nested.lineage)
    )
    result = ChannelFrame(nested._data + values, sampling_rate=8, lineage=lineage)
    plan = RecipePlan.from_frame(result, input_names=("offset", "signal"), registry=registry)
    assert [item.kind for item in plan.inputs] == ["array", "frame"]
    assert [node.inputs for node in plan.nodes] == [("input-1",), ("input-0", "node-0")]
    replayed = RecipePlan.from_dict(plan.to_dict(), registry=registry).apply(
        {"offset": values, "signal": source}, registry=registry
    )
    np.testing.assert_array_equal(replayed.data, np.arange(8.0) + 1)
