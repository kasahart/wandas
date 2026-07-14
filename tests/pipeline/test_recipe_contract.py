from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame
from wandas.pipeline import (
    RecipeExecutionError,
    RecipeOperation,
    RecipePlan,
    RecipeRegistry,
    RecipeSerializationError,
    default_recipe_registry,
)
from wandas.processing.semantic import InputBinding


def _frame(value: float = 1.0) -> ChannelFrame:
    return ChannelFrame.from_numpy(np.full((1, 16), value), sampling_rate=8000)


def _normalized_payload() -> dict[str, Any]:
    return RecipePlan.from_frame(_frame().normalize(), input_names=("signal",)).to_dict()


def test_identity_frame_plan_roundtrip_uses_public_api() -> None:
    source = _frame()
    plan = RecipePlan.from_frame(source, input_names=("signal",))

    assert plan.to_dict() == {
        "schema": "wandas.recipe",
        "version": 2,
        "inputs": [{"id": "input-0", "name": "signal", "kind": "frame"}],
        "nodes": [],
        "output": "input-0",
    }
    assert RecipePlan.from_dict(plan.to_dict()).apply({"signal": source}) is source


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda payload: payload["inputs"].append(copy.deepcopy(payload["inputs"][0])), "unique"),
        (lambda payload: payload["nodes"].append(copy.deepcopy(payload["nodes"][0])), "unique"),
        (lambda payload: payload["nodes"][0]["inputs"].append("missing"), "unavailable"),
        (lambda payload: payload.update(output="missing"), "unavailable"),
    ],
)
def test_complete_graph_validation_rejects_invalid_payloads(mutate: Any, match: str) -> None:
    payload = _normalized_payload()
    mutate(payload)

    with pytest.raises(RecipeSerializationError, match=match):
        RecipePlan.from_dict(payload)


def test_apply_requires_exact_named_inputs() -> None:
    plan = RecipePlan.from_frame(_frame().normalize(), input_names=("signal",))

    with pytest.raises(RecipeExecutionError, match="Recipe inputs do not match"):
        plan.apply({})
    with pytest.raises(RecipeExecutionError, match="Recipe inputs do not match"):
        plan.apply({"signal": _frame(), "extra": _frame()})
    with pytest.raises(RecipeExecutionError, match="Recipe frame input requires"):
        plan.apply({"signal": np.ones((1, 16))})


def test_registry_extension_returns_new_registry_without_mutating_base() -> None:
    def identity(inputs: tuple[Any, ...], _params: Mapping[str, Any]) -> Any:
        return inputs[0]

    operation = RecipeOperation(
        "tests.identity",
        1,
        ((InputBinding("frame", "frame"),),),
        identity,
    )
    base = RecipeRegistry()
    extended = base.with_operation(operation)

    with pytest.raises(KeyError):
        base.require("tests.identity", 1)
    assert extended.require("tests.identity", 1) is operation


def test_registry_snapshots_binding_pattern_container() -> None:
    def identity(inputs: tuple[Any, ...], _params: Mapping[str, Any]) -> Any:
        return inputs[0]

    patterns: Any = [(InputBinding("frame", "frame"),)]
    operation = RecipeOperation("tests.snapshot", 1, cast(Any, patterns), identity)
    patterns.clear()

    assert operation.binding_patterns == ((InputBinding("frame", "frame"),),)


def test_registry_rejects_ambiguous_binding_kind_signatures() -> None:
    def identity(inputs: tuple[Any, ...], _params: Mapping[str, Any]) -> Any:
        return inputs[0]

    with pytest.raises(ValueError, match="unique input kind signatures"):
        RecipeOperation(
            "tests.ambiguous",
            1,
            (
                (InputBinding("left", "frame"),),
                (InputBinding("base", "frame"),),
            ),
            identity,
        )


def test_registry_rejects_duplicate_operation_versions() -> None:
    existing = default_recipe_registry().require("wandas.audio.normalize", 1)

    with pytest.raises(ValueError, match="already registered"):
        RecipeRegistry((existing, existing))


def test_pipeline_public_surface_omits_removed_replay_models() -> None:
    import wandas.pipeline as pipeline

    removed = {
        "AudioCall",
        "BoundInput",
        "RecipeInput",
        "RecipeNode",
        "ReplayCodecRegistry",
        "replay_method",
    }

    assert not any(hasattr(pipeline, name) for name in removed)
