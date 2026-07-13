"""Versioned RecipePlan serializer and fail-closed loader."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from wandas.pipeline.calls import load_call
from wandas.pipeline.errors import RecipeSerializationError
from wandas.pipeline.model import RecipeInput, RecipeNode, RecipePlan

SCHEMA = "wandas.recipe"
SCHEMA_VERSION = 1


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


class RecipeSerializer:
    def serialize(self, plan: RecipePlan) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "version": SCHEMA_VERSION,
            "inputs": [{"id": item.id, "name": item.name, "kind": item.kind} for item in plan.inputs],
            "nodes": [
                {"id": node.id, "call": _json_value(node.call.to_payload()), "inputs": list(node.inputs)}
                for node in plan.nodes
            ],
            "output": plan.output,
        }


class RecipeLoader:
    def load(self, payload: Mapping[str, Any]) -> RecipePlan:
        if not isinstance(payload, Mapping) or set(payload) != {"schema", "version", "inputs", "nodes", "output"}:
            raise RecipeSerializationError("Recipe payload fields do not match canonical schema")
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
            return RecipePlan(inputs, nodes, output)
        except RecipeSerializationError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise RecipeSerializationError(f"Invalid Recipe graph\n  Cause: {exc}") from exc

    @staticmethod
    def _input(value: Any) -> RecipeInput:
        if not isinstance(value, Mapping) or set(value) != {"id", "name", "kind"}:
            raise RecipeSerializationError("Recipe input fields are malformed")
        return RecipeInput(value["id"], value["name"], value["kind"])  # type: ignore[arg-type]

    @staticmethod
    def _node(value: Any) -> RecipeNode:
        if not isinstance(value, Mapping) or set(value) != {"id", "call", "inputs"}:
            raise RecipeSerializationError("Recipe node fields are malformed")
        if not isinstance(value["id"], str) or not isinstance(value["call"], Mapping):
            raise RecipeSerializationError("Recipe node id or call is malformed")
        if not isinstance(value["inputs"], list) or not all(isinstance(item, str) for item in value["inputs"]):
            raise RecipeSerializationError("Recipe node inputs are malformed")
        return RecipeNode(value["id"], load_call(value["call"]), tuple(value["inputs"]))
