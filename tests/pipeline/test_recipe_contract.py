from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame
from wandas.pipeline import (
    OperationCapture,
    RecipeExecutionError,
    RecipeOperation,
    RecipePlan,
    RecipeRegistry,
    RecipeSerializationError,
    RecipeValidationError,
    default_recipe_registry,
    recipe_operation,
)
from wandas.pipeline.decorators import _freeze_display_params, _unary_capture
from wandas.pipeline.model import RecipeInput, RecipeNode, validate_recipe_plan
from wandas.processing.semantic import InputBinding, freeze_params


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


def test_registry_rejects_duplicate_roles_within_binding_pattern() -> None:
    def identity(inputs: tuple[Any, ...], _params: Mapping[str, Any]) -> Any:
        return inputs[0]

    with pytest.raises(ValueError, match="roles must be unique"):
        RecipeOperation(
            "tests.duplicate-roles",
            1,
            ((InputBinding("same", "frame"), InputBinding("same", "frame")),),
            identity,
        )


def test_registry_rejects_empty_binding_pattern() -> None:
    def identity(inputs: tuple[Any, ...], _params: Mapping[str, Any]) -> Any:
        return inputs[0]

    with pytest.raises(ValueError, match="at least one input"):
        RecipeOperation("tests.empty-binding", 1, ((),), identity)


def test_decorator_requires_handler_for_non_unary_frame_bindings() -> None:
    with pytest.raises(ValueError, match="require an explicit handler"):

        @recipe_operation(
            "tests.missing-handler",
            bindings=(InputBinding("left", "frame"), InputBinding("right", "frame")),
        )
        def combine(left: ChannelFrame, right: ChannelFrame) -> ChannelFrame:
            return left + right


def test_decorator_rejects_empty_binding_patterns() -> None:
    with pytest.raises(ValueError, match="at least one binding pattern"):
        recipe_operation("tests.empty-patterns", binding_patterns=())


def test_decorator_rejects_bindings_and_binding_patterns_together() -> None:
    binding = (InputBinding("frame", "frame"),)

    with pytest.raises(ValueError, match="either bindings or binding_patterns"):
        recipe_operation("tests.conflicting-bindings", bindings=binding, binding_patterns=(binding,))


def test_decorator_rejects_noncallable_capture_at_declaration() -> None:
    with pytest.raises(TypeError, match="capture must be callable"):
        recipe_operation("tests.invalid-capture", capture=cast(Any, 0))


def test_decorator_requires_capture_for_non_unary_frame_bindings() -> None:
    def handler(inputs: tuple[Any, ...], _params: Mapping[str, Any]) -> Any:
        return inputs[0] + inputs[1]

    with pytest.raises(ValueError, match="require an explicit capture"):

        @recipe_operation(
            "tests.missing-capture",
            bindings=(InputBinding("left", "frame"), InputBinding("right", "frame")),
            handler=handler,
        )
        def combine(left: ChannelFrame, right: ChannelFrame) -> ChannelFrame:
            return left + right


def test_default_handler_rejects_positional_only_parameters() -> None:
    with pytest.raises(ValueError, match="positional-only"):

        @recipe_operation("tests.positional-only")
        def scale(frame: ChannelFrame, gain: float, /) -> ChannelFrame:
            return frame * gain


def test_decorator_rejects_variadic_receiver() -> None:
    with pytest.raises(ValueError, match="positional Frame receiver"):

        @recipe_operation("tests.variadic-receiver")
        def variadic_scale(*args: Any) -> ChannelFrame:
            return args[0] * args[1]


def test_registry_equality_does_not_ignore_executable_behavior() -> None:
    def first_handler(inputs: tuple[Any, ...], _params: Mapping[str, Any]) -> Any:
        return inputs[0]

    def second_handler(inputs: tuple[Any, ...], _params: Mapping[str, Any]) -> Any:
        return inputs[0].normalize()

    bindings = ((InputBinding("frame", "frame"),),)
    first = RecipeRegistry((RecipeOperation("tests.behavior", 1, bindings, first_handler),))
    second = RecipeRegistry((RecipeOperation("tests.behavior", 1, bindings, second_handler),))

    assert first != second
    assert len({first, second}) == 2


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


def test_display_param_freezer_rejects_non_string_names() -> None:
    with pytest.raises(TypeError, match="parameter names must be strings"):
        _freeze_display_params(cast(Any, {1: "value"}))


def test_default_unary_capture_requires_frame_receiver() -> None:
    capture = _unary_capture(InputBinding("frame", "frame"))

    with pytest.raises(TypeError, match="require a Frame receiver"):
        capture((), {})


def test_decorated_call_rejects_bindings_outside_declared_contract() -> None:
    def mismatched_capture(_args: tuple[Any, ...], params: Mapping[str, Any]) -> OperationCapture:
        return OperationCapture((InputBinding("operand", "array"),), (None,), params)

    @recipe_operation("tests.mismatched-capture", capture=mismatched_capture)
    def copy_frame(frame: ChannelFrame) -> ChannelFrame:
        return frame

    with pytest.raises(RuntimeError, match="undeclared bindings"):
        copy_frame(_frame())


def _normalize_node(*, input_id: str = "input-0", version: int = 1, params: Any = None) -> RecipeNode:
    return RecipeNode(
        "node-0",
        "wandas.audio.normalize",
        version,
        (input_id,),
        freeze_params({}) if params is None else params,
    )


def test_plan_validation_rejects_blank_identifier() -> None:
    with pytest.raises(RecipeValidationError, match="non-blank string"):
        RecipePlan((RecipeInput("", "signal"),), (), "")


def test_plan_validation_rejects_registered_operation_input_kind_mismatch() -> None:
    with pytest.raises(RecipeValidationError, match="input kinds do not match"):
        RecipePlan(
            (RecipeInput("input-0", "operand", "array"),),
            (_normalize_node(),),
            "node-0",
        )


def test_plan_validation_wraps_parameter_validator_failure() -> None:
    def reject(_params: Mapping[str, Any]) -> None:
        raise ValueError("invalid params")

    operation = RecipeOperation(
        "tests.rejected-params",
        1,
        ((InputBinding("frame", "frame"),),),
        lambda inputs, _params: inputs[0],
        reject,
    )
    registry = RecipeRegistry((operation,))
    node = RecipeNode("node-0", operation.operation_id, 1, ("input-0",), freeze_params({}))

    with pytest.raises(RecipeValidationError, match="params violate"):
        RecipePlan((RecipeInput("input-0", "signal"),), (node,), "node-0", registry=registry)


@pytest.mark.parametrize(
    ("inputs", "nodes", "output", "message"),
    [
        ((), (), "missing", "at least one input"),
        ((RecipeInput("input-0", "value", cast(Any, "scalar")),), (), "input-0", "input kind"),
        (
            (RecipeInput("input-0", "signal"),),
            (_normalize_node(version=0),),
            "node-0",
            "version must be positive",
        ),
        (
            (RecipeInput("input-0", "signal"),),
            (_normalize_node(params={}),),
            "node-0",
            "params must be canonical",
        ),
        ((RecipeInput("input-0", "operand", "array"),), (), "input-0", "output must be a frame"),
        (
            (RecipeInput("input-0", "signal"), RecipeInput("input-1", "unused")),
            (),
            "input-0",
            "unreachable from output",
        ),
    ],
)
def test_plan_validation_rejects_graph_contract_edges(
    inputs: tuple[RecipeInput, ...],
    nodes: tuple[RecipeNode, ...],
    output: str,
    message: str,
) -> None:
    with pytest.raises(RecipeValidationError, match=message):
        RecipePlan(inputs, nodes, output)


def test_plan_validation_preserves_existing_validation_error() -> None:
    plan = RecipePlan.from_frame(_frame().normalize())

    class RejectingRegistry(RecipeRegistry):
        def require(self, operation_id: str, version: int) -> RecipeOperation:
            raise RecipeValidationError(f"rejected {operation_id} version {version}")

    with pytest.raises(RecipeValidationError, match="rejected wandas.audio.normalize"):
        validate_recipe_plan(plan, registry=RejectingRegistry())


@pytest.mark.parametrize(
    ("operation_id", "version", "patterns", "handler", "validator", "message"),
    [
        ("", 1, ((InputBinding("frame", "frame"),),), lambda *_args: None, lambda _params: None, "non-blank"),
        (
            "tests.invalid-version",
            0,
            ((InputBinding("frame", "frame"),),),
            lambda *_args: None,
            lambda _params: None,
            "positive integer",
        ),
        ("tests.no-patterns", 1, (), lambda *_args: None, lambda _params: None, "at least one binding pattern"),
        (
            "tests.bad-pattern",
            1,
            cast(Any, ([InputBinding("frame", "frame")],)),
            lambda *_args: None,
            lambda _params: None,
            "InputBinding tuples",
        ),
        (
            "tests.duplicate-pattern",
            1,
            ((InputBinding("frame", "frame"),), (InputBinding("frame", "frame"),)),
            lambda *_args: None,
            lambda _params: None,
            "must be unique",
        ),
        (
            "tests.bad-handler",
            1,
            ((InputBinding("frame", "frame"),),),
            cast(Any, None),
            lambda _params: None,
            "must be callable",
        ),
    ],
)
def test_recipe_operation_rejects_invalid_contract_fields(
    operation_id: str,
    version: int,
    patterns: tuple[tuple[InputBinding, ...], ...],
    handler: Any,
    validator: Any,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        RecipeOperation(operation_id, version, patterns, handler, validator)


def test_registry_rejects_non_operation_entry() -> None:
    with pytest.raises(TypeError, match="entries must be RecipeOperation"):
        RecipeRegistry(cast(Any, ("not-an-operation",)))
