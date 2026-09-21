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
    """Compile authoritative semantic lineage into a canonical Recipe graph.

    Args:
        input_names: Optional public names for discovered source and external-array
            inputs, in deterministic depth-first traversal order.
        registry: Registry that must contain every captured operation.
    """

    input_names: tuple[str, ...] | None = None
    registry: RecipeRegistry | None = None
    _inputs: list[RecipeInput] = field(default_factory=list, init=False)
    _nodes: list[RecipeNode] = field(default_factory=list, init=False)
    _memo: dict[int, str] = field(default_factory=dict, init=False)

    def compile_frame(self, frame: object) -> RecipePlan:
        """Compile one Frame without evaluating its lazy data.

        Args:
            frame: Frame whose semantic lineage defines the workflow.

        Returns:
            A validated Recipe plan.

        Raises:
            RecipeExtractionError: If ``frame`` or its lineage cannot be represented by
                the selected registry and portable Recipe contract.
        """
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
        """Return the explicit registry or the immutable built-in default."""
        return self.registry if self.registry is not None else default_recipe_registry()

    def _input(self, kind: Literal["frame", "array"]) -> str:
        """Append one discovered runtime input and return its graph reference."""
        index = len(self._inputs)
        if self.input_names is not None and index >= len(self.input_names):
            raise RecipeExtractionError("Recipe compilation requires one name per runtime input")
        name = self.input_names[index] if self.input_names is not None else f"input_{index}"
        reference = f"input-{index}"
        self._inputs.append(RecipeInput(reference, name, kind))
        return reference

    def _visit(self, lineage: LineageNode) -> str:
        """Compile depth-first using explicit enter/exit events, not recursion.

        External-array events share the traversal stack so input discovery keeps
        the original left-to-right order even when arrays precede Frame inputs.
        The result stack contains one reference per completed input edge.
        """
        pending: list[tuple[LineageNode | None, bool]] = [(lineage, False)]
        results: list[str] = []
        registry = self._selected_registry
        while pending:
            node, exiting = pending.pop()
            if node is None:
                results.append(self._input("array"))
                continue
            identity = id(node)
            if identity in self._memo:
                results.append(self._memo[identity])
                continue
            operation = node.operation
            if operation is None:
                reference = self._input("frame")
                self._memo[identity] = reference
                results.append(reference)
                continue
            if exiting:
                input_count = len(operation.bindings)
                references = tuple(results[-input_count:]) if input_count else ()
                if input_count:
                    del results[-input_count:]
                node_id = f"node-{len(self._nodes)}"
                self._nodes.append(
                    RecipeNode(node_id, operation.operation_id, operation.version, references, operation.params)
                )
                self._memo[identity] = node_id
                results.append(node_id)
                continue
            if node.recipe_error is not None:
                raise RecipeExtractionError(
                    "Recipe extraction rejected a public operation\n"
                    f"  Operation: {operation.operation_id!r}\n"
                    f"  Reason: {node.recipe_error}"
                )
            try:
                definition = registry.require(operation.operation_id, operation.version)
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
            pending.append((node, True))
            for binding, parent in reversed(list(zip(operation.bindings, node.inputs))):
                pending.append((parent if binding.kind == "frame" else None, False))
        return results[0]
