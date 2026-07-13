from typing import Any, cast

import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame
from wandas.pipeline import AudioCall, RecipeInput, RecipeNode, RecipePlan, ScalarCall, TerminalCall


def _frame() -> ChannelFrame:
    return ChannelFrame.from_numpy(np.arange(32.0).reshape(2, 16), sampling_rate=8000)


def test_identity_frame_plan_is_valid() -> None:
    plan = RecipePlan((RecipeInput("source", "signal"),), (), "source")

    assert plan.apply({"signal": _frame()}).shape == (2, 16)


def test_node_edges_are_canonicalized_at_construction() -> None:
    edges = ["source"]
    node = RecipeNode("node", ScalarCall("+", 1), cast(Any, edges))
    plan = RecipePlan((RecipeInput("source", "signal"),), (node,), "node")
    edges[0] = "missing"

    assert plan.nodes[0].inputs == ("source",)


@pytest.mark.parametrize(
    "inputs,nodes,output,match",
    [
        ((RecipeInput("x", "signal"), RecipeInput("x", "other")), (), "x", "unique"),
        ((RecipeInput("x", "signal"),), (RecipeNode("n", ScalarCall("+", 1), ("missing",)),), "n", "unavailable"),
        (
            (RecipeInput("x", "signal"),),
            (RecipeNode("wanted", ScalarCall("+", 1), ("x",)), RecipeNode("dead", ScalarCall("+", 1), ("x",))),
            "wanted",
            "unreachable",
        ),
        (
            (RecipeInput("x", "signal"), RecipeInput("unused", "other")),
            (RecipeNode("n", ScalarCall("+", 1), ("x",)),),
            "n",
            "unreachable",
        ),
        ((RecipeInput("x", "array", "array"),), (), "x", "frame or terminal"),
    ],
)
def test_graph_invariants_fail_closed(inputs: Any, nodes: Any, output: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        RecipePlan(inputs, nodes, output)


def test_terminal_call_must_be_output() -> None:
    with pytest.raises(ValueError, match="Terminal"):
        RecipePlan(
            (RecipeInput("x", "signal"),),
            (
                RecipeNode("terminal", TerminalCall("rms"), ("x",)),
                RecipeNode("after", AudioCall("normalize"), ("terminal",)),
            ),
            "after",
        )


def test_missing_and_wrong_input_types_are_rejected() -> None:
    plan = RecipePlan((RecipeInput("x", "signal"),), (), "x")
    with pytest.raises(KeyError, match="missing"):
        plan.apply({})
    with pytest.raises(TypeError, match="Wandas frame"):
        plan.apply({"signal": np.ones((1, 4))})
