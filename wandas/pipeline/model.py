"""Canonical Recipe graph, validation, and execution."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from wandas.pipeline.registry import RecipeOperation, RecipeRegistry
    from wandas.processing.semantic import FrozenMap, InputBinding


@dataclass(frozen=True)
class RecipeInput:
    """Named runtime input required by a :class:`RecipePlan`.

    Args:
        id: Graph-local identifier used by node edges.
        name: Public name callers pass to :meth:`RecipePlan.apply`.
        kind: Required runtime value category, either ``"frame"`` or ``"array"``.
    """

    id: str
    name: str
    kind: Literal["frame", "array"] = "frame"


@dataclass(frozen=True)
class RecipeNode:
    """One immutable operation invocation in a Recipe graph.

    Args:
        id: Graph-local identifier for this result.
        operation: Stable operation identifier resolved through a Recipe registry.
        version: Positive version of the registered operation contract.
        inputs: Ordered graph references matching the operation bindings.
        params: Canonical immutable operation parameters.
    """

    id: str
    operation: str
    version: int
    inputs: tuple[str, ...]
    params: FrozenMap

    def __post_init__(self) -> None:
        """Snapshot input references as an immutable tuple."""
        object.__setattr__(self, "inputs", tuple(self.inputs))


@dataclass(frozen=True)
class RecipePlan:
    """Portable, validated graph of public Wandas Frame operations.

    A plan stores operation intent and named input slots, never Frame samples or a
    Dask task graph. Constructing, serializing, loading, and applying a plan therefore
    remain lazy; computation occurs only when the returned Frame is computed.

    User workflows create plans with :meth:`from_frame` or :meth:`from_dict`, then use
    :meth:`apply` and :meth:`to_dict`. The direct ``inputs``/``nodes``/``output``
    constructor is an internal graph-assembly interface, not a public compatibility
    surface.

    Args:
        inputs: Named Frame or NumPy/Dask array inputs used by the graph.
        nodes: Topologically ordered Recipe operations.
        output: Identifier of the Frame input or node returned by the plan.
        registry: Immutable operation registry used to validate the graph. Uses the
            built-in registry when omitted.

    Raises:
        RecipeValidationError: If the graph or an operation contract is invalid.
    """

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
        """Snapshot and validate a complete Recipe graph."""
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
        """Extract a Recipe from a Frame's semantic lineage.

        Args:
            frame: Result Frame produced through public Recipe-capable operations.
            input_names: Optional names assigned to discovered runtime inputs in
                deterministic traversal order. When omitted, names are generated as
                ``input_0``, ``input_1``, and so on.
            registry: Registry used to resolve every captured operation. Uses the
                built-in registry when omitted.

        Returns:
            A validated plan whose output reproduces ``frame``'s public workflow.

        Raises:
            RecipeExtractionError: If the value is not a Frame, an operation is not
                portable or registered, or ``input_names`` has the wrong length.
        """
        from wandas.pipeline.compiler import LineageRecipeCompiler

        return LineageRecipeCompiler(input_names=input_names, registry=registry).compile_frame(frame)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        registry: RecipeRegistry | None = None,
    ) -> RecipePlan:
        """Load and validate a Recipe schema-2 mapping.

        Args:
            payload: Mapping produced by :meth:`to_dict` or an equivalent decoded
                JSON object.
            registry: Registry used to validate operation identifiers and versions.

        Returns:
            A new immutable Recipe plan.

        Raises:
            RecipeSerializationError: If the payload violates the schema or graph
                contract.
        """
        from wandas.pipeline.serialization import RecipeLoader

        return RecipeLoader(registry=registry).load(payload)

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic JSON-compatible Recipe schema.

        Returns:
            A fresh mapping using schema ``wandas.recipe`` version 2.
        """
        from wandas.pipeline.serialization import RecipeSerializer

        return RecipeSerializer().serialize(self)

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Save this plan as a standalone ``.recipe.json`` artifact.

        The artifact contains operation intent and named inputs only. Frame samples,
        WDF data, and Dask graphs remain outside the Recipe persistence boundary.

        Args:
            path: Target path. ``.recipe.json`` is appended when absent.
            overwrite: Replace an existing artifact when true.

        Returns:
            The normalized artifact path written to disk.
        """
        from wandas.pipeline.artifacts import save_recipe_artifact

        return save_recipe_artifact(self, path, overwrite=overwrite)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        registry: RecipeRegistry | None = None,
    ) -> RecipePlan:
        """Load and validate a standalone ``.recipe.json`` artifact."""
        from wandas.pipeline.artifacts import load_recipe_artifact

        return load_recipe_artifact(path, registry=registry)

    def apply(
        self,
        inputs: Mapping[str, Any],
        *,
        registry: RecipeRegistry | None = None,
    ) -> Any:
        """Build the Recipe output lazily from named runtime inputs.

        Args:
            inputs: Exact mapping from each :class:`RecipeInput` name to a Wandas
                Frame or NumPy/Dask array of the declared kind.
            registry: Registry used for validation and operation execution. Supply
                the same extension registry used during extraction and loading.

        Returns:
            The output Frame with semantic lineage, metadata, and Dask laziness
            preserved by its public operations. An identity plan with no nodes returns
            its input Frame unchanged.

        Raises:
            RecipeExecutionError: If names or runtime value kinds do not match, an
                operation fails, or an operation violates the Frame lineage contract.
            RecipeValidationError: If the plan is invalid for ``registry``.
        """
        return RecipeExecutor(registry=registry).execute(self, inputs)


def _identifier(value: object, label: str) -> str:
    """Validate and return a non-blank graph identifier."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-blank string")
    return value


def _registry(registry: RecipeRegistry | None) -> RecipeRegistry:
    """Resolve an optional registry to the immutable built-in registry."""
    if registry is not None:
        return registry
    from wandas.pipeline.registry import default_recipe_registry

    return default_recipe_registry()


def _definition_for_node(
    node: RecipeNode,
    input_kinds: tuple[str, ...],
    registry: RecipeRegistry,
) -> tuple[RecipeOperation, tuple[InputBinding, ...]]:
    """Resolve a node definition and its matching ordered binding pattern."""
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
    return definition, matching_patterns[0]


def _validate_node_params(node: RecipeNode, definition: RecipeOperation) -> None:
    """Validate canonical node parameters against one registered definition."""
    from wandas.pipeline.registry import immutable_params

    try:
        definition.validate_params(immutable_params(node.params))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Recipe node params violate its registered contract\n  Node: {node.id!r}\n  Operation: {node.operation!r}"
        ) from exc


def validate_recipe_plan(plan: RecipePlan, *, registry: RecipeRegistry | None = None) -> None:
    """Validate a complete Recipe graph against one immutable registry.

    Validation covers identifiers, topological references, input kinds, registered
    operation versions, canonical parameters, Frame-only output, and graph reachability.

    Args:
        plan: Plan to validate.
        registry: Registry defining the accepted operations. Uses built-ins when
            omitted.

    Raises:
        RecipeValidationError: If any graph or operation invariant is violated.
    """
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
            _validate_node_params(node, definition)
            available.add(node_id)
            node_ids.add(node_id)
            kinds[node_id] = "frame"

        output = _identifier(plan.output, "Recipe output")
        if output not in available:
            raise ValueError(f"Recipe output is unavailable\n  Output: {output!r}")
        if kinds[output] != "frame":
            raise ValueError("Recipe output must be a frame")
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
    """Registry-driven executor shared by every Recipe operation.

    Args:
        registry: Registry used for graph validation and handler lookup. Uses the
            built-in immutable registry when omitted.
    """

    def __init__(self, *, registry: RecipeRegistry | None = None) -> None:
        """Select the registry used for subsequent executions."""
        self._registry = _registry(registry)

    def execute(self, plan: RecipePlan, inputs: Mapping[str, Any]) -> Any:
        """Validate and lazily execute a plan with exact named inputs.

        Args:
            plan: Recipe graph to execute.
            inputs: Mapping whose keys exactly match the plan's public input names.

        Returns:
            The Frame referenced by ``plan.output``.

        Raises:
            RecipeExecutionError: If runtime inputs mismatch or an operation fails
                or returns a value that violates its semantic Frame contract.
            RecipeValidationError: If ``plan`` is invalid for this executor's registry.
        """
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
                SemanticOperation(node.operation, node.version, bindings, node.params),
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
            environment[node.id] = result
            kinds[node.id] = "frame"
        return environment[plan.output]
