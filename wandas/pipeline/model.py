"""Canonical Recipe graph, validation, and execution."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from wandas.pipeline.registry import RecipeOperation, RecipeRegistry
    from wandas.processing.semantic import FrozenMap


@dataclass(frozen=True)
class RecipeInput:
    id: str
    name: str
    kind: Literal["frame", "array"] = "frame"


@dataclass(frozen=True)
class RecipeNode:
    id: str
    operation: str
    version: int
    inputs: tuple[str, ...]
    params: FrozenMap

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", tuple(self.inputs))


@dataclass(frozen=True)
class RecipePlan:
    inputs: tuple[RecipeInput, ...]
    nodes: tuple[RecipeNode, ...]
    output: str

    def __init__(
        self,
        inputs: Iterable[RecipeInput],
        nodes: Iterable[RecipeNode],
        output: str,
        *,
        registry: RecipeRegistry | None = None,
    ) -> None:
        object.__setattr__(self, "inputs", tuple(inputs))
        object.__setattr__(self, "nodes", tuple(nodes))
        object.__setattr__(self, "output", output)
        validate_recipe_plan(self, registry=registry)

    @classmethod
    def from_frame(
        cls,
        frame: Any,
        *,
        input_names: tuple[str, ...] | None = None,
        registry: RecipeRegistry | None = None,
    ) -> RecipePlan:
        from wandas.pipeline.compiler import LineageRecipeCompiler

        return LineageRecipeCompiler(input_names=input_names, registry=registry).compile_frame(frame)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        registry: RecipeRegistry | None = None,
    ) -> RecipePlan:
        from wandas.pipeline.serialization import RecipeLoader

        return RecipeLoader(registry=registry).load(payload)

    def to_dict(self) -> dict[str, Any]:
        from wandas.pipeline.serialization import RecipeSerializer

        return RecipeSerializer().serialize(self)

    def apply(
        self,
        inputs: Mapping[str, Any],
        *,
        registry: RecipeRegistry | None = None,
    ) -> Any:
        return RecipeExecutor(registry=registry).execute(self, inputs)


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-blank string")
    return value


def _registry(registry: RecipeRegistry | None) -> RecipeRegistry:
    if registry is not None:
        return registry
    from wandas.pipeline.registry import default_recipe_registry

    return default_recipe_registry()


def _definition_for_node(
    node: RecipeNode,
    input_kinds: tuple[str, ...],
    registry: RecipeRegistry,
) -> tuple[RecipeOperation, tuple[Any, ...]]:
    try:
        definition = registry.require(node.operation, node.version)
    except KeyError as exc:
        raise ValueError(
            f"Recipe node uses an unregistered operation\n  Node: {node.id!r}\n  Operation: {node.operation!r}"
        ) from exc
    matching_patterns = tuple(
        pattern for pattern in definition.binding_patterns if tuple(binding.kind for binding in pattern) == input_kinds
    )
    if not matching_patterns:
        raise ValueError(
            "Recipe node input kinds do not match its registered operation\n"
            f"  Node: {node.id!r}\n  Operation: {node.operation!r}\n  Kinds: {input_kinds!r}"
        )
    from wandas.pipeline.registry import immutable_params

    try:
        definition.validate_params(immutable_params(node.params))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Recipe node params violate its registered contract\n  Node: {node.id!r}\n  Operation: {node.operation!r}"
        ) from exc
    return definition, matching_patterns[0]


def validate_recipe_plan(plan: RecipePlan, *, registry: RecipeRegistry | None = None) -> None:
    """Validate the complete graph against one explicit immutable registry."""
    from wandas.pipeline.errors import RecipeValidationError
    from wandas.processing.semantic import FrozenMap

    try:
        if not plan.inputs:
            raise ValueError("RecipePlan requires at least one input")
        input_ids = [_identifier(item.id, "Recipe input id") for item in plan.inputs]
        input_names = [_identifier(item.name, "Recipe input name") for item in plan.inputs]
        if len(set(input_ids)) != len(input_ids) or len(set(input_names)) != len(input_names):
            raise ValueError("Recipe input ids and names must be unique")
        if any(item.kind not in {"frame", "array"} for item in plan.inputs):
            raise ValueError("Recipe input kind must be 'frame' or 'array'")

        selected_registry = _registry(registry)
        available = set(input_ids)
        kinds: dict[str, str] = {item.id: item.kind for item in plan.inputs}
        node_ids: set[str] = set()
        for node in plan.nodes:
            node_id = _identifier(node.id, "Recipe node id")
            if node_id in available or node_id in node_ids:
                raise ValueError(f"Recipe node id must be unique\n  Node: {node_id!r}")
            _identifier(node.operation, "Recipe operation id")
            if type(node.version) is not int or node.version < 1:
                raise ValueError(f"Recipe operation version must be positive\n  Node: {node_id!r}")
            if not isinstance(node.params, FrozenMap):
                raise TypeError(f"Recipe node params must be canonical\n  Node: {node_id!r}")
            references = tuple(_identifier(item, "Recipe edge reference") for item in node.inputs)
            missing = tuple(item for item in references if item not in available)
            if missing:
                raise ValueError(
                    f"Recipe node references unavailable inputs\n  Node: {node_id!r}\n  Missing: {missing!r}"
                )
            definition, _pattern = _definition_for_node(
                node,
                tuple(kinds[item] for item in references),
                selected_registry,
            )
            if definition.output_kind == "terminal" and node_id != plan.output:
                raise ValueError("Terminal Recipe operations must be the plan output")
            available.add(node_id)
            node_ids.add(node_id)
            kinds[node_id] = definition.output_kind

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
                f"  Nodes: {dead_nodes!r}\n  Inputs: {unused_inputs!r}"
            )
    except RecipeValidationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise RecipeValidationError(str(exc)) from exc


class RecipeExecutor:
    """Single registry-driven execution loop for every Recipe operation."""

    def __init__(self, *, registry: RecipeRegistry | None = None) -> None:
        self._registry = _registry(registry)

    def execute(self, plan: RecipePlan, inputs: Mapping[str, Any]) -> Any:
        import numpy as np
        from dask.array.core import Array as DaArray

        from wandas.core.base_frame import BaseFrame
        from wandas.pipeline.errors import RecipeExecutionError
        from wandas.processing.semantic import LineageNode, SemanticOperation, semantic_lineage

        validate_recipe_plan(plan, registry=self._registry)
        expected_names = {recipe_input.name for recipe_input in plan.inputs}
        provided_names = set(inputs)
        missing_names = expected_names - provided_names
        extra_names = provided_names - expected_names
        if missing_names or extra_names:
            raise RecipeExecutionError(
                "Recipe inputs do not match the plan\n"
                f"  Missing: {sorted(missing_names)!r}\n"
                f"  Extra: {sorted(extra_names)!r}"
            )

        environment: dict[str, Any] = {}
        lineages: dict[str, LineageNode | None] = {}
        kinds: dict[str, str] = {}
        for recipe_input in plan.inputs:
            value = inputs[recipe_input.name]
            if recipe_input.kind == "frame" and not isinstance(value, BaseFrame):
                raise RecipeExecutionError(
                    "Recipe frame input requires a Wandas frame\n"
                    f"  Input: {recipe_input.name!r}\n"
                    f"  Got: {type(value).__name__}"
                )
            if recipe_input.kind == "array" and not isinstance(value, np.ndarray | DaArray):
                raise RecipeExecutionError(
                    "Recipe array input requires NumPy or Dask\n"
                    f"  Input: {recipe_input.name!r}\n"
                    f"  Got: {type(value).__name__}"
                )
            environment[recipe_input.id] = value
            lineages[recipe_input.id] = value.lineage if recipe_input.kind == "frame" else None
            kinds[recipe_input.id] = recipe_input.kind

        for node in plan.nodes:
            definition, bindings = _definition_for_node(
                node,
                tuple(kinds[reference] for reference in node.inputs),
                self._registry,
            )
            node_inputs = tuple(environment[reference] for reference in node.inputs)
            expected_lineage = LineageNode(
                SemanticOperation(node.operation, node.version, bindings, node.params, definition.output_kind),
                tuple(lineages[reference] for reference in node.inputs),
            )
            try:
                with semantic_lineage(expected_lineage):
                    result = definition.invoke(node_inputs, node.params)
            except Exception as exc:
                if isinstance(exc, RecipeExecutionError):
                    raise
                raise RecipeExecutionError(
                    f"Recipe operation failed\n  Node: {node.id!r}\n  Operation: {node.operation!r}\n  Cause: {exc}"
                ) from exc
            if definition.output_kind == "frame":
                if not isinstance(result, BaseFrame):
                    raise RecipeExecutionError(
                        f"Recipe frame operation returned {type(result).__name__}\n"
                        f"  Node: {node.id!r}\n"
                        f"  Operation: {node.operation!r}"
                    )
                if result.lineage is not expected_lineage:
                    raise RecipeExecutionError(
                        "Recipe frame operation did not preserve semantic lineage\n"
                        f"  Node: {node.id!r}\n"
                        f"  Operation: {node.operation!r}"
                    )
                lineages[node.id] = expected_lineage
            else:
                if isinstance(result, BaseFrame):
                    raise RecipeExecutionError(
                        "Recipe terminal operation returned a Frame\n"
                        f"  Node: {node.id!r}\n"
                        f"  Operation: {node.operation!r}"
                    )
                lineages[node.id] = None
            environment[node.id] = result
            kinds[node.id] = definition.output_kind
        return environment[plan.output]
