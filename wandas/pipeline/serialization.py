"""Deterministic RecipePlan persistence (writes schema 3 or 4, reads schemas 2 to 4)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from wandas.pipeline.errors import RecipeSerializationError
from wandas.pipeline.model import RecipeInput, RecipeNode, RecipePlan
from wandas.pipeline.registry import RecipeRegistry
from wandas.processing.semantic import FrozenMap, value_from_json, value_to_json

SCHEMA = "wandas.recipe"
SCHEMA_VERSION = 3
# Schema 4 adds named steps; it is written only for plans that have steps, so a plan
# without steps stays readable by Wandas versions that know schema 3.
STEPS_SCHEMA_VERSION = 4
# Schema 2 stored every Python float in the tagged hex form.
_READABLE_SCHEMA_VERSIONS = frozenset({2, SCHEMA_VERSION, STEPS_SCHEMA_VERSION})
_FIELDS = frozenset({"schema", "version", "inputs", "nodes", "output"})


class RecipeSerializer:
    """Encode validated Recipe plans using the deterministic schema-3 or schema-4 grammar."""

    def serialize(self, plan: RecipePlan) -> dict[str, Any]:
        """Encode a Recipe plan as a fresh JSON-compatible mapping.

        Args:
            plan: Validated plan to encode.

        Returns:
            A mapping with explicit schema, version, inputs, nodes, and output fields.
            A plan with steps uses schema 4 and adds a ``steps`` object.
        """
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "version": STEPS_SCHEMA_VERSION if plan.steps else SCHEMA_VERSION,
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
        if plan.steps:
            payload["steps"] = dict(plan.steps)
        return payload


@dataclass(frozen=True)
class RecipeLoader:
    """Strict Recipe loader for schemas 2, 3, and 4.

    Args:
        registry: Registry used to validate decoded operation identifiers, versions,
            input kinds, and parameters. Uses the built-in registry when omitted.
    """

    registry: RecipeRegistry | None = None

    def load(self, payload: Mapping[str, Any]) -> RecipePlan:
        """Decode and validate one Recipe mapping.

        Args:
            payload: Decoded JSON-like mapping using the exact schema fields.

        Returns:
            A new immutable Recipe plan.

        Raises:
            RecipeSerializationError: If fields, canonical values, or the decoded
                graph violate the persistence contract.
        """
        if not isinstance(payload, Mapping):
            raise RecipeSerializationError("Recipe payload fields do not match the Recipe schema")
        if payload.get("schema") != SCHEMA or type(payload.get("version")) is not int:
            raise RecipeSerializationError("Unknown Recipe schema or invalid schema version")
        if payload["version"] not in _READABLE_SCHEMA_VERSIONS:
            raise RecipeSerializationError(f"Unsupported Recipe schema version\n  Got: {payload['version']!r}")
        expected_fields = _FIELDS | {"steps"} if payload["version"] == STEPS_SCHEMA_VERSION else _FIELDS
        if set(payload) != expected_fields:
            raise RecipeSerializationError("Recipe payload fields do not match the Recipe schema")
        steps = self._steps(payload["steps"]) if payload["version"] == STEPS_SCHEMA_VERSION else ()
        raw_inputs = payload.get("inputs")
        raw_nodes = payload.get("nodes")
        output = payload.get("output")
        if not isinstance(raw_inputs, list) or not isinstance(raw_nodes, list) or not isinstance(output, str):
            raise RecipeSerializationError("Recipe graph collections or output are malformed")
        try:
            inputs = tuple(self._input(item) for item in raw_inputs)
            plain_floats = payload["version"] >= 3
            nodes = tuple(self._node(item, plain_floats=plain_floats) for item in raw_nodes)
            return RecipePlan(inputs, nodes, output, steps=steps, registry=self.registry)
        except RecipeSerializationError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise RecipeSerializationError(f"Invalid Recipe graph\n  Cause: {exc}") from exc

    @staticmethod
    def _steps(value: Any) -> tuple[tuple[str, str], ...]:
        """Decode the schema-4 mapping of step names to graph references."""
        if (
            not isinstance(value, Mapping)
            or not value
            or not all(isinstance(name, str) and isinstance(reference, str) for name, reference in value.items())
        ):
            raise RecipeSerializationError("Recipe steps must be a non-empty object of names to references")
        return tuple(value.items())

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
    def _node(value: Any, *, plain_floats: bool) -> RecipeNode:
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
        params = value_from_json(value["params"], plain_floats=plain_floats)
        if not isinstance(params, FrozenMap):
            raise RecipeSerializationError("Recipe node params must be a canonical map")
        return RecipeNode(value["id"], value["operation"], value["version"], tuple(value["inputs"]), params)
