"""Canonical public Recipe graph, validation, and execution."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from wandas.pipeline.codecs import ReplayCodecRegistry


class RecipeCall(Protocol):
    arity: int
    output_kind: str

    def accepts_input_kinds(self, kinds: tuple[str, ...]) -> bool: ...

    def invoke(self, inputs: tuple[Any, ...]) -> Any: ...

    def to_payload(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class RecipeInput:
    id: str
    name: str
    kind: Literal["frame", "array"] = "frame"


@dataclass(frozen=True)
class RecipeNode:
    id: str
    call: RecipeCall
    inputs: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", tuple(self.inputs))


@dataclass(frozen=True)
class RecipePlan:
    inputs: tuple[RecipeInput, ...]
    nodes: tuple[RecipeNode, ...]
    output: str

    def __init__(self, inputs: Iterable[RecipeInput], nodes: Iterable[RecipeNode], output: str) -> None:
        object.__setattr__(self, "inputs", tuple(inputs))
        object.__setattr__(self, "nodes", tuple(nodes))
        object.__setattr__(self, "output", output)
        validate_recipe_plan(self)

    @classmethod
    def from_frame(
        cls,
        frame: Any,
        *,
        input_names: tuple[str, ...] | None = None,
        registry: ReplayCodecRegistry | None = None,
    ) -> RecipePlan:
        from wandas.pipeline.compiler import LineageRecipeCompiler

        compiler = LineageRecipeCompiler(input_names=input_names, registry=registry)
        return compiler.compile_frame(frame)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RecipePlan:
        from wandas.pipeline.serialization import RecipeLoader

        return RecipeLoader().load(payload)

    def to_dict(self) -> dict[str, Any]:
        from wandas.pipeline.serialization import RecipeSerializer

        return RecipeSerializer().serialize(self)

    def apply(self, inputs: Mapping[str, Any]) -> Any:
        return RecipeExecutor().execute(self, inputs)


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-blank string")
    return value


def validate_recipe_plan(plan: RecipePlan) -> None:
    from wandas.pipeline.calls import CanonicalCall

    if not plan.inputs:
        raise ValueError("RecipePlan requires at least one input")
    ids = [_identifier(item.id, "Recipe input id") for item in plan.inputs]
    names = [_identifier(item.name, "Recipe input name") for item in plan.inputs]
    if len(set(ids)) != len(ids) or len(set(names)) != len(names):
        raise ValueError("Recipe input ids and names must be unique")
    if any(item.kind not in {"frame", "array"} for item in plan.inputs):
        raise ValueError("Recipe input kind must be 'frame' or 'array'")

    available = set(ids)
    kinds: dict[str, str] = {item.id: item.kind for item in plan.inputs}
    for node in plan.nodes:
        node_id = _identifier(node.id, "Recipe node id")
        if node_id in available:
            raise ValueError(f"Recipe node id must be unique\n  Node: {node_id!r}")
        if not isinstance(node.call, CanonicalCall):
            raise TypeError("Recipe nodes require a canonical call")
        references = tuple(_identifier(item, "Recipe edge reference") for item in node.inputs)
        missing = tuple(item for item in references if item not in available)
        if missing:
            raise ValueError(f"Recipe node references unavailable inputs\n  Node: {node_id!r}\n  Missing: {missing}")
        if len(references) != node.call.arity:
            raise ValueError(f"Recipe call arity mismatch\n  Node: {node_id!r}")
        input_kinds = tuple(kinds[item] for item in references)
        if not node.call.accepts_input_kinds(input_kinds):
            raise ValueError(f"Recipe call input kinds do not match\n  Node: {node_id!r}\n  Kinds: {input_kinds}")
        if node.call.output_kind not in {"frame", "terminal"}:
            raise ValueError(f"Recipe call output kind is invalid\n  Kind: {node.call.output_kind!r}")
        if node.call.output_kind == "terminal" and node_id != plan.output:
            raise ValueError("Terminal Recipe calls must be the plan output")
        available.add(node_id)
        kinds[node_id] = node.call.output_kind

    output = _identifier(plan.output, "Recipe output")
    if output not in available:
        raise ValueError(f"Recipe output is unavailable\n  Output: {output!r}")
    if kinds[output] not in {"frame", "terminal"}:
        raise ValueError("Recipe output must be a frame or terminal value")
    required = {output}
    for node in reversed(plan.nodes):
        if node.id in required:
            required.update(node.inputs)
    dead_nodes = tuple(node.id for node in plan.nodes if node.id not in required)
    unused_inputs = tuple(item.id for item in plan.inputs if item.id not in required)
    if dead_nodes or unused_inputs:
        raise ValueError(
            "RecipePlan contains graph elements unreachable from output\n"
            f"  Nodes: {dead_nodes}\n  Inputs: {unused_inputs}"
        )


class RecipeExecutor:
    """Single execution loop for every Recipe graph."""

    def execute(self, plan: RecipePlan, inputs: Mapping[str, Any]) -> Any:
        import numpy as np
        from dask.array.core import Array as DaArray

        from wandas.core.base_frame import BaseFrame

        environment: dict[str, Any] = {}
        for recipe_input in plan.inputs:
            if recipe_input.name not in inputs:
                raise KeyError(f"Recipe input is missing\n  Missing: {recipe_input.name!r}")
            value = inputs[recipe_input.name]
            if recipe_input.kind == "frame" and not isinstance(value, BaseFrame):
                raise TypeError(f"Recipe frame input requires a Wandas frame\n  Got: {type(value).__name__}")
            if recipe_input.kind == "array" and not isinstance(value, np.ndarray | DaArray):
                raise TypeError(f"Recipe array input requires NumPy or Dask\n  Got: {type(value).__name__}")
            environment[recipe_input.id] = value
        for node in plan.nodes:
            environment[node.id] = node.call.invoke(tuple(environment[item] for item in node.inputs))
        return environment[plan.output]


class RecipePlanBuilder:
    def __init__(self) -> None:
        self._inputs: list[RecipeInput] = []
        self._nodes: list[RecipeNode] = []

    def add_input(self, id: str, name: str, kind: Literal["frame", "array"] = "frame") -> str:
        self._inputs.append(RecipeInput(id, name, kind))
        return id

    def add_node(self, id: str, call: RecipeCall, inputs: Iterable[str]) -> str:
        self._nodes.append(RecipeNode(id, call, tuple(inputs)))
        return id

    def build(self, output: str) -> RecipePlan:
        return RecipePlan(self._inputs, self._nodes, output)
