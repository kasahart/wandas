"""Identity-aware compiler from semantic lineage to a RecipePlan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from wandas.pipeline.errors import RecipeExtractionError
from wandas.pipeline.model import RecipeInput, RecipeNode, RecipePlan
from wandas.pipeline.registry import RecipeRegistry, default_recipe_registry
from wandas.processing.semantic import LineageNode


@dataclass
class LineageRecipeCompiler:
    input_names: tuple[str, ...] | None = None
    registry: RecipeRegistry | None = None
    _inputs: list[RecipeInput] = field(default_factory=list, init=False)
    _nodes: list[RecipeNode] = field(default_factory=list, init=False)
    _memo: dict[int, str] = field(default_factory=dict, init=False)

    def compile_frame(self, frame: object) -> RecipePlan:
        from wandas.core.base_frame import BaseFrame

        if not isinstance(frame, BaseFrame):
            raise RecipeExtractionError(f"RecipePlan.from_frame requires a Wandas frame\n  Got: {type(frame).__name__}")
        self._inputs.clear()
        self._nodes.clear()
        self._memo.clear()
        output = self._visit(frame.lineage)
        if self.input_names is not None and len(self.input_names) != len(self._inputs):
            raise RecipeExtractionError(
                "Recipe compilation requires one name per runtime input\n"
                f"  Expected: {len(self._inputs)}\n"
                f"  Got: {len(self.input_names)}"
            )
        try:
            return RecipePlan(self._inputs, self._nodes, output, registry=self._selected_registry)
        except ValueError as exc:
            raise RecipeExtractionError(f"Extracted Recipe graph is invalid\n  Cause: {exc}") from exc

    @property
    def _selected_registry(self) -> RecipeRegistry:
        return self.registry if self.registry is not None else default_recipe_registry()

    def _input(self, kind: Literal["frame", "array"]) -> str:
        index = len(self._inputs)
        if self.input_names is not None and index >= len(self.input_names):
            raise RecipeExtractionError("Recipe compilation requires one name per runtime input")
        name = self.input_names[index] if self.input_names is not None else f"input_{index}"
        reference = f"input-{index}"
        self._inputs.append(RecipeInput(reference, name, kind))
        return reference

    def _visit(self, lineage: LineageNode) -> str:
        identity = id(lineage)
        if identity in self._memo:
            return self._memo[identity]
        operation = lineage.operation
        if operation is None:
            reference = self._input("frame")
            self._memo[identity] = reference
            return reference
        if lineage.recipe_error is not None:
            raise RecipeExtractionError(
                "Recipe extraction rejected a public operation\n"
                f"  Operation: {operation.operation_id!r}\n"
                f"  Reason: {lineage.recipe_error}"
            )
        try:
            definition = self._selected_registry.require(operation.operation_id, operation.version)
        except KeyError as exc:
            raise RecipeExtractionError(
                "Recipe extraction found an unregistered operation\n"
                f"  Operation: {operation.operation_id!r}\n"
                f"  Version: {operation.version}"
            ) from exc
        if not definition.accepts(operation.bindings):
            raise RecipeExtractionError(
                f"Semantic operation disagrees with its registry contract\n  Operation: {operation.operation_id!r}"
            )
        references = tuple(
            self._visit(parent) if binding.kind == "frame" and parent is not None else self._input("array")
            for binding, parent in zip(operation.bindings, lineage.inputs)
        )
        node_id = f"node-{len(self._nodes)}"
        self._nodes.append(RecipeNode(node_id, operation.operation_id, operation.version, references, operation.params))
        self._memo[identity] = node_id
        return node_id
