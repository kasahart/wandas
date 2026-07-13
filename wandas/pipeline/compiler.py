"""Single identity-aware compiler from semantic lineage to RecipePlan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from wandas.pipeline.codecs import ReplayCodecRegistry, default_codec_registry
from wandas.pipeline.errors import RecipeExtractionError
from wandas.pipeline.model import RecipeInput, RecipeNode, RecipePlan
from wandas.processing.base import LineageNode
from wandas.processing.semantic import SourceReplay


@dataclass
class LineageRecipeCompiler:
    input_names: tuple[str, ...] | None = None
    registry: ReplayCodecRegistry | None = None
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
        lineage = frame.lineage or frame._lineage_or_source()
        output = self._visit(lineage)
        if self.input_names is not None and len(self.input_names) != len(self._inputs):
            raise RecipeExtractionError("Recipe compilation requires one name per runtime input")
        return RecipePlan(self._inputs, self._nodes, output)

    def _input(self, kind: Literal["frame", "array"]) -> str:
        index = len(self._inputs)
        if self.input_names is not None and index >= len(self.input_names):
            raise RecipeExtractionError("Recipe compilation received too few input names")
        name = self.input_names[index] if self.input_names is not None else f"input_{index}"
        reference = f"input-{index}"
        self._inputs.append(RecipeInput(reference, name, kind))
        return reference

    def _visit(self, lineage: LineageNode) -> str:
        identity = id(lineage)
        if identity in self._memo:
            return self._memo[identity]
        if isinstance(lineage.replay, SourceReplay):
            reference = self._input("frame")
            self._memo[identity] = reference
            return reference
        registry = self.registry or default_codec_registry()
        encoding = registry.encode(lineage.replay, lineage.inputs)
        references = tuple(
            self._visit(binding.lineage) if binding.lineage is not None else self._input(binding.kind)
            for binding in encoding.bindings
        )
        node_id = f"node-{len(self._nodes)}"
        self._nodes.append(RecipeNode(node_id, encoding.call, references))
        self._memo[identity] = node_id
        return node_id
