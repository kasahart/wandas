"""Deterministic schema-2 RecipePlan persistence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from wandas.pipeline.errors import RecipeSerializationError
from wandas.pipeline.model import RecipeInput, RecipeNode, RecipePlan
from wandas.pipeline.registry import RecipeRegistry
from wandas.processing.semantic import FrozenMap, value_from_json, value_to_json

SCHEMA = "wandas.recipe"
SCHEMA_VERSION = 2


class RecipeSerializer:
    """Encode validated Recipe plans using the deterministic schema-2 grammar."""

    def serialize(self, plan: RecipePlan) -> dict[str, Any]:
        """Encode a Recipe plan as a fresh JSON-compatible mapping.

        Args:
            plan: Validated plan to encode.

        Returns:
            A mapping with explicit schema, version, inputs, nodes, and output fields.
        """
        return {
            "schema": SCHEMA,
            "version": SCHEMA_VERSION,
            "inputs": [{"id": item.id, "name": item.name, "kind": item.kind} for item in plan.inputs],
            "nodes": [
                {
                    "id": node.id,
                    "operation": node.operation,
                    "version": node.version,
                    "inputs": list(node.inputs),
                    "params": value_to_json(node.params),
                }
                for node in plan.nodes
            ],
            "output": plan.output,
        }


@dataclass(frozen=True)
class RecipeLoader:
    """Strict schema-2 Recipe loader.

    Args:
        registry: Registry used to validate decoded operation identifiers, versions,
            input kinds, and parameters. Uses the built-in registry when omitted.
    """

    registry: RecipeRegistry | None = None

    def load(self, payload: Mapping[str, Any]) -> RecipePlan:
        """Decode and validate one Recipe mapping.

        Args:
            payload: Decoded JSON-like mapping using the exact schema-2 fields.

        Returns:
            A new immutable Recipe plan.

        Raises:
            RecipeSerializationError: If fields, canonical values, or the decoded
                graph violate the persistence contract.
        """
        if not isinstance(payload, Mapping) or set(payload) != {"schema", "version", "inputs", "nodes", "output"}:
            raise RecipeSerializationError("Recipe payload fields do not match schema 2")
        if payload.get("schema") != SCHEMA or type(payload.get("version")) is not int:
            raise RecipeSerializationError("Unknown Recipe schema or invalid schema version")
        if payload["version"] != SCHEMA_VERSION:
            raise RecipeSerializationError(f"Unsupported Recipe schema version\n  Got: {payload['version']!r}")
        raw_inputs = payload.get("inputs")
        raw_nodes = payload.get("nodes")
        output = payload.get("output")
        if not isinstance(raw_inputs, list) or not isinstance(raw_nodes, list) or not isinstance(output, str):
            raise RecipeSerializationError("Recipe graph collections or output are malformed")
        try:
            inputs = tuple(self._input(item) for item in raw_inputs)
            nodes = tuple(self._node(item) for item in raw_nodes)
            return RecipePlan(inputs, nodes, output, registry=self.registry)
        except RecipeSerializationError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise RecipeSerializationError(f"Invalid Recipe graph\n  Cause: {exc}") from exc

    @staticmethod
    def _input(value: Any) -> RecipeInput:
        """Decode one strictly shaped Recipe input record."""
        if not isinstance(value, Mapping) or set(value) != {"id", "name", "kind"}:
            raise RecipeSerializationError("Recipe input fields are malformed")
        if (
            not isinstance(value["id"], str)
            or not isinstance(value["name"], str)
            or value["kind"]
            not in {
                "frame",
                "array",
            }
        ):
            raise RecipeSerializationError("Recipe input values are malformed")
        return RecipeInput(value["id"], value["name"], value["kind"])

    @staticmethod
    def _node(value: Any) -> RecipeNode:
        """Decode one strictly shaped Recipe node record."""
        expected = {"id", "operation", "version", "inputs", "params"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise RecipeSerializationError("Recipe node fields are malformed")
        if not isinstance(value["id"], str) or not isinstance(value["operation"], str):
            raise RecipeSerializationError("Recipe node id or operation is malformed")
        if type(value["version"]) is not int or value["version"] < 1:
            raise RecipeSerializationError("Recipe node version is malformed")
        if not isinstance(value["inputs"], list) or not all(isinstance(item, str) for item in value["inputs"]):
            raise RecipeSerializationError("Recipe node inputs are malformed")
        params = value_from_json(value["params"])
        if not isinstance(params, FrozenMap):
            raise RecipeSerializationError("Recipe node params must be a canonical map")
        return RecipeNode(value["id"], value["operation"], value["version"], tuple(value["inputs"]), params)
