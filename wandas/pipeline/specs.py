from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from wandas.pipeline.errors import RecipeExtractionError
from wandas.pipeline.extraction import (
    _add_channel_data_step_from_graph,
    _add_channel_step_from_graph,
    _axis_slices_from_params,
    _binary_frame_step_from_graph,
    _binary_operand_step_from_graph,
    _channel_key_from_parent_graph,
    _is_external_add_channel_data_graph,
    _is_external_operand_graph,
    _recipe_spec_steps_from_graph,
    _split_graph_at_binary_merge,
    _step_from_graph,
    _steps_from_graph,
)
from wandas.pipeline.params import _restore_history_value
from wandas.pipeline.steps import (
    AddChannelDataStep,
    AddChannelStep,
    BinaryFrameStep,
    BinaryOperandStep,
    GraphNodeSpec,
    IndexingStep,
    RecipeStep,
    _apply_recipe_step,
)


def _unique_preserving_order(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


@dataclass(frozen=True, init=False)
class RecipeSpec:
    """Serial recipe of replayable Wandas frame operations."""

    steps: tuple[RecipeStep, ...]

    def __init__(self, steps: Iterable[RecipeStep]) -> None:
        object.__setattr__(self, "steps", tuple(steps))

    def to_dict(self) -> dict[str, Any]:
        return {"steps": [step.to_dict() for step in self.steps]}

    @classmethod
    def from_frame(cls, frame: Any) -> RecipeSpec:
        from wandas.core.base_frame import BaseFrame

        if not isinstance(frame, BaseFrame):
            raise RecipeExtractionError(
                "Recipe extraction requires a Wandas frame\n"
                f"  Got: {type(frame).__name__}\n"
                "  Pass a processed Wandas frame with operation_graph lineage."
            )
        graph = frame.operation_graph
        if graph is None:
            return cls(())
        return cls(_recipe_spec_steps_from_graph(cast(Mapping[str, Any], graph)))

    def apply(self, frame: Any) -> Any:
        result: Any = frame
        for step in self.steps:
            result = _apply_recipe_step(step, result)
        return result


@dataclass(frozen=True, init=False)
class GraphRecipeSpec:
    """Explicit multi-input recipe with one binary frame merge and an optional linear tail."""

    input_recipes: tuple[tuple[str, RecipeSpec], ...]
    output: BinaryFrameStep
    tail_recipe: RecipeSpec

    def __init__(
        self,
        input_recipes: Mapping[str, RecipeSpec],
        output: BinaryFrameStep,
        tail_recipe: RecipeSpec | None = None,
    ) -> None:
        if not input_recipes:
            raise ValueError("GraphRecipeSpec requires at least one named input recipe")
        if not isinstance(output, BinaryFrameStep):
            raise TypeError(f"GraphRecipeSpec output must be a BinaryFrameStep\n  Got: {type(output).__name__}")
        if output.left == output.right:
            raise ValueError(
                "GraphRecipeSpec output requires two distinct inputs\n"
                f"  Left input: {output.left!r}\n"
                f"  Right input: {output.right!r}"
            )
        if tail_recipe is None:
            tail_recipe = RecipeSpec(())
        if not isinstance(tail_recipe, RecipeSpec):
            raise TypeError(f"GraphRecipeSpec tail_recipe must be a RecipeSpec\n  Got: {type(tail_recipe).__name__}")
        frozen_inputs: list[tuple[str, RecipeSpec]] = []
        for name, recipe in input_recipes.items():
            if not isinstance(name, str) or not name:
                raise TypeError("GraphRecipeSpec input names must be non-empty strings")
            if not isinstance(recipe, RecipeSpec):
                raise TypeError(
                    "GraphRecipeSpec input recipes must be RecipeSpec instances\n"
                    f"  Input: {name}\n"
                    f"  Got: {type(recipe).__name__}"
                )
            frozen_inputs.append((name, recipe))
        input_names = {name for name, _recipe in frozen_inputs}
        missing_output_inputs = [name for name in (output.left, output.right) if name not in input_names]
        if missing_output_inputs:
            raise ValueError(
                "GraphRecipeSpec output references unknown input\n"
                f"  Missing input references: {missing_output_inputs}\n"
                f"  Available inputs: {sorted(input_names)}"
            )
        output_inputs = {output.left, output.right}
        extra_inputs = input_names - output_inputs
        if extra_inputs:
            raise ValueError(
                "GraphRecipeSpec input recipes must exactly match output inputs\n"
                f"  Extra input recipes: {sorted(extra_inputs)}\n"
                f"  Output inputs: {sorted(output_inputs)}"
            )
        object.__setattr__(self, "input_recipes", tuple(frozen_inputs))
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "tail_recipe", tail_recipe)

    def to_dict(self) -> dict[str, Any]:
        recipe_dict = {
            "inputs": {name: recipe.to_dict() for name, recipe in self.input_recipes},
            "output": self.output.to_dict(),
        }
        if self.tail_recipe.steps:
            recipe_dict["tail"] = self.tail_recipe.to_dict()
        return recipe_dict

    @classmethod
    def from_frame(cls, frame: Any, *, input_names: tuple[str, ...] | None = None) -> GraphRecipeSpec:
        from wandas.core.base_frame import BaseFrame

        if not isinstance(frame, BaseFrame):
            raise RecipeExtractionError(
                f"GraphRecipeSpec extraction requires a Wandas frame\n  Got: {type(frame).__name__}"
            )
        graph = frame.operation_graph
        if graph is None:
            raise RecipeExtractionError("GraphRecipeSpec extraction requires operation_graph lineage")

        merge_graph, tail_steps = _split_graph_at_binary_merge(cast(Mapping[str, Any], graph))
        inputs = tuple(merge_graph.get("inputs", ()))
        resolved_input_names = (
            tuple(f"input_{index}" for index in range(len(inputs))) if input_names is None else input_names
        )
        if len(inputs) != 2 or resolved_input_names is None or len(resolved_input_names) != 2:
            raise RecipeExtractionError(
                "GraphRecipeSpec extraction requires one input name per parent\n"
                f"  Parent count: {len(inputs)}\n"
                f"  Input names: {list(resolved_input_names or ())}"
            )
        if resolved_input_names[0] == resolved_input_names[1]:
            raise RecipeExtractionError(
                f"GraphRecipeSpec extraction requires distinct input names\n  Input names: {list(resolved_input_names)}"
            )

        operation = str(merge_graph["operation"])
        params = cast(Mapping[str, Any], _restore_history_value(merge_graph.get("params", {})))
        left, right = resolved_input_names
        return cls(
            {
                left: RecipeSpec(_steps_from_graph(cast(Mapping[str, Any], inputs[0]))),
                right: RecipeSpec(_steps_from_graph(cast(Mapping[str, Any], inputs[1]))),
            },
            _binary_frame_step_from_graph(operation, params, left, right),
            RecipeSpec(tail_steps),
        )

    def apply(self, inputs: Mapping[str, Any]) -> Any:
        frames: dict[str, Any] = {}
        for name, recipe in self.input_recipes:
            try:
                frame = inputs[name]
            except KeyError as exc:
                raise KeyError(
                    f"GraphRecipeSpec input is missing\n  Missing input: {name!r}\n  Available inputs: {sorted(inputs)}"
                ) from exc
            frames[name] = recipe.apply(frame)
        return self.tail_recipe.apply(self.output.apply(frames))


@dataclass(frozen=True, init=False)
class NodeGraphRecipeSpec:
    """Replayable tree-shaped graph recipe with named external inputs."""

    inputs: tuple[str, ...]
    nodes: tuple[GraphNodeSpec, ...]
    output: str

    def __init__(self, inputs: Iterable[str], nodes: Iterable[GraphNodeSpec], output: str) -> None:
        frozen_inputs = tuple(inputs)
        if not frozen_inputs or not all(isinstance(input_name, str) and input_name for input_name in frozen_inputs):
            raise ValueError("NodeGraphRecipeSpec inputs must be non-empty strings")
        if len(set(frozen_inputs)) != len(frozen_inputs):
            raise ValueError(f"NodeGraphRecipeSpec inputs must be unique\n  Inputs: {list(frozen_inputs)}")
        frozen_nodes = tuple(nodes)
        if not frozen_nodes:
            raise ValueError("NodeGraphRecipeSpec requires at least one node")
        if not isinstance(output, str) or not output:
            raise ValueError("NodeGraphRecipeSpec output must be a non-empty string")

        available_refs = set(frozen_inputs)
        for node in frozen_nodes:
            if not isinstance(node, GraphNodeSpec):
                raise TypeError(
                    f"NodeGraphRecipeSpec nodes must be GraphNodeSpec instances\n  Got: {type(node).__name__}"
                )
            if node.id in available_refs:
                raise ValueError(
                    f"NodeGraphRecipeSpec node id duplicates an existing reference\n  Node id: {node.id!r}"
                )
            missing_inputs = [input_id for input_id in node.inputs if input_id not in available_refs]
            if missing_inputs:
                raise ValueError(
                    "NodeGraphRecipeSpec node references unknown inputs\n"
                    f"  Node id: {node.id!r}\n"
                    f"  Missing refs: {missing_inputs}\n"
                    f"  Available refs: {sorted(available_refs)}"
                )
            available_refs.add(node.id)
        if output not in available_refs:
            raise ValueError(
                "NodeGraphRecipeSpec output references unknown node or input\n"
                f"  Output: {output!r}\n"
                f"  Available refs: {sorted(available_refs)}"
            )
        node_ids = {node.id for node in frozen_nodes}
        if output not in node_ids:
            raise ValueError(
                "NodeGraphRecipeSpec output must reference a graph node\n"
                f"  Output: {output!r}\n"
                f"  Node ids: {sorted(node_ids)}"
            )
        final_node_id = frozen_nodes[-1].id
        if output != final_node_id:
            raise ValueError(
                "NodeGraphRecipeSpec output must reference the final graph node\n"
                f"  Output: {output!r}\n"
                f"  Final node: {final_node_id!r}"
            )
        object.__setattr__(self, "inputs", frozen_inputs)
        object.__setattr__(self, "nodes", frozen_nodes)
        object.__setattr__(self, "output", output)

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": list(self.inputs),
            "nodes": [node.to_dict() for node in self.nodes],
            "output": self.output,
        }

    @classmethod
    def from_frame(cls, frame: Any, *, input_names: tuple[str, ...] | None = None) -> NodeGraphRecipeSpec:
        from wandas.core.base_frame import BaseFrame

        if not isinstance(frame, BaseFrame):
            raise RecipeExtractionError(
                f"NodeGraphRecipeSpec extraction requires a Wandas frame\n  Got: {type(frame).__name__}"
            )
        graph = frame.operation_graph
        if graph is None:
            raise RecipeExtractionError("NodeGraphRecipeSpec extraction requires operation_graph lineage")

        nodes: list[GraphNodeSpec] = []
        source_names: list[str] = []
        next_source_index = 0

        def next_source_name() -> str:
            nonlocal next_source_index
            source_index = next_source_index
            next_source_index += 1
            if input_names is not None:
                if source_index >= len(input_names):
                    raise RecipeExtractionError(
                        "NodeGraphRecipeSpec extraction requires one input name per source leaf\n"
                        f"  Source leaf count so far: {source_index + 1}\n"
                        f"  Input names: {list(input_names)}"
                    )
                name = input_names[source_index]
            else:
                name = f"input_{source_index}"
            if not isinstance(name, str) or not name:
                raise RecipeExtractionError(
                    "NodeGraphRecipeSpec input names must be non-empty strings\n"
                    f"  Input names: {list(input_names or ())}"
                )
            source_names.append(name)
            return name

        def next_node_id() -> str:
            return f"n{len(nodes)}"

        def extract_node(node_graph: Mapping[str, Any]) -> str:
            kind = cast(str | None, node_graph.get("kind"))
            if kind == "source":
                return next_source_name()

            operation = str(node_graph["operation"])
            params = cast(Mapping[str, Any], _restore_history_value(node_graph.get("params", {})))
            input_graphs = tuple(node_graph.get("inputs", ()))

            if operation == "add_channel" and len(input_graphs) != 2:
                if _is_external_add_channel_data_graph(operation, params):
                    if len(input_graphs) > 1:
                        raise RecipeExtractionError(
                            "add_channel data recipe extraction requires at most one frame parent\n"
                            f"  Parent count: {len(input_graphs)}"
                        )
                    base_ref = (
                        extract_node(cast(Mapping[str, Any], input_graphs[0])) if input_graphs else next_source_name()
                    )
                    data_ref = next_source_name()
                    step = _add_channel_data_step_from_graph(params, base_ref, data_ref)
                    node_id = next_node_id()
                    nodes.append(GraphNodeSpec(node_id, step, (base_ref, data_ref)))
                    return node_id
                raise RecipeExtractionError(
                    "add_channel recipe extraction only supports ChannelFrame or external raw data inputs\n"
                    f"  Parent count: {len(input_graphs)}\n"
                    f"  Input kind: {params.get('input_kind')!r}"
                )

            if _is_external_operand_graph(operation, params):
                if len(input_graphs) > 1:
                    raise RecipeExtractionError(
                        "Binary operand recipe extraction requires at most one frame parent\n"
                        f"  Operation: {operation}\n"
                        f"  Parent count: {len(input_graphs)}"
                    )
                frame_ref = (
                    extract_node(cast(Mapping[str, Any], input_graphs[0])) if input_graphs else next_source_name()
                )
                operand_ref = next_source_name()
                step = _binary_operand_step_from_graph(operation, params, frame_ref, operand_ref)
                node_id = next_node_id()
                nodes.append(GraphNodeSpec(node_id, step, (frame_ref, operand_ref)))
                return node_id

            if operation == "__getitem__" and params.get("indexing") == "multidimensional_slice":
                if len(input_graphs) != 1:
                    raise RecipeExtractionError(
                        "Multidimensional indexing recipe extraction requires one channel-selection parent\n"
                        f"  Parent count: {len(input_graphs)}"
                    )
                parent_graph = cast(Mapping[str, Any], input_graphs[0])
                parent_inputs = tuple(parent_graph.get("inputs", ()))
                if len(parent_inputs) > 1:
                    raise RecipeExtractionError(
                        "Graph operation requires graph recipe support\n"
                        f"  Operation: {parent_graph['operation']}\n"
                        f"  Parent count: {len(parent_inputs)}\n"
                        "  Multidimensional indexing requires one replayable parent chain."
                    )
                parent_ref = (
                    extract_node(cast(Mapping[str, Any], parent_inputs[0])) if parent_inputs else next_source_name()
                )
                step = IndexingStep((_channel_key_from_parent_graph(parent_graph), *_axis_slices_from_params(params)))
                node_id = next_node_id()
                nodes.append(GraphNodeSpec(node_id, step, (parent_ref,)))
                return node_id

            if len(input_graphs) == 0:
                parent_ref = next_source_name()
                step = _step_from_graph(
                    operation,
                    params,
                    kind,
                    cast(Mapping[str, Any] | None, node_graph.get("custom")),
                )
                node_id = next_node_id()
                nodes.append(GraphNodeSpec(node_id, step, (parent_ref,)))
                return node_id
            if len(input_graphs) == 1:
                parent_ref = extract_node(cast(Mapping[str, Any], input_graphs[0]))
                step = _step_from_graph(
                    operation,
                    params,
                    kind,
                    cast(Mapping[str, Any] | None, node_graph.get("custom")),
                )
                node_id = next_node_id()
                nodes.append(GraphNodeSpec(node_id, step, (parent_ref,)))
                return node_id
            if len(input_graphs) == 2:
                left_ref = extract_node(cast(Mapping[str, Any], input_graphs[0]))
                right_ref = extract_node(cast(Mapping[str, Any], input_graphs[1]))
                step = (
                    _add_channel_step_from_graph(params, left_ref, right_ref)
                    if operation == "add_channel"
                    else _binary_frame_step_from_graph(operation, params, left_ref, right_ref)
                )
                node_id = next_node_id()
                nodes.append(GraphNodeSpec(node_id, step, (left_ref, right_ref)))
                return node_id
            raise RecipeExtractionError(
                "NodeGraphRecipeSpec extraction only supports unary and binary frame graph nodes\n"
                f"  Operation: {operation}\n"
                f"  Parent count: {len(input_graphs)}"
            )

        output = extract_node(cast(Mapping[str, Any], graph))
        if input_names is not None and next_source_index != len(input_names):
            raise RecipeExtractionError(
                "NodeGraphRecipeSpec extraction requires one input name per source leaf\n"
                f"  Source leaf count: {next_source_index}\n"
                f"  Input names: {list(input_names)}"
            )
        return cls(_unique_preserving_order(source_names), nodes, output)

    def apply(self, inputs: Mapping[str, Any]) -> Any:
        env: dict[str, Any] = {}
        for input_name in self.inputs:
            try:
                env[input_name] = inputs[input_name]
            except KeyError as exc:
                raise KeyError(
                    "NodeGraphRecipeSpec input is missing\n"
                    f"  Missing input: {input_name!r}\n"
                    f"  Available inputs: {sorted(inputs)}"
                ) from exc

        for node in self.nodes:
            if isinstance(node.step, BinaryFrameStep | BinaryOperandStep | AddChannelStep | AddChannelDataStep):
                env[node.id] = node.step.apply({input_ref: env[input_ref] for input_ref in node.inputs})
            else:
                env[node.id] = _apply_recipe_step(cast(RecipeStep, node.step), env[node.inputs[0]])
        return env[self.output]
