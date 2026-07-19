from __future__ import annotations

import copy
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipePlan, RecipeSerializationError
from wandas.processing.semantic import thaw_params


def _frame() -> ChannelFrame:
    return ChannelFrame.from_numpy(np.arange(8.0).reshape(1, 8), sampling_rate=8000)


def _plan_for_operand(operand: Any) -> RecipePlan:
    return RecipePlan.from_frame(_frame().__add__(operand), input_names=("signal",))


@pytest.mark.parametrize(
    "operand",
    [
        2**100,
        -0.0,
        float("inf"),
        float("-inf"),
        float("nan"),
        complex(-0.0, float("inf")),
        np.int8(-7),
        np.uint64(2**63 + 5),
        np.float32(-0.0),
        np.complex64(complex(1.25, -2.5)),
    ],
)
def test_schema_2_roundtrips_lossless_numeric_params(operand: Any) -> None:
    original = _plan_for_operand(operand)
    loaded = RecipePlan.from_dict(json.loads(json.dumps(original.to_dict(), allow_nan=False)))

    decoded = thaw_params(loaded.nodes[0].params)["operand"]
    if isinstance(operand, np.generic):
        assert type(decoded) is type(operand)
        assert np.asarray(decoded).tobytes() == np.asarray(operand).tobytes()
    elif type(operand) is float:
        assert struct.pack(">d", decoded) == struct.pack(">d", operand)
    elif type(operand) is complex:
        assert struct.pack(">dd", decoded.real, decoded.imag) == struct.pack(">dd", operand.real, operand.imag)
    else:
        assert decoded == operand


def test_schema_2_serialization_is_deterministic_and_strict_json() -> None:
    processed = _frame().rename_channels({0: "mono"}).normalize(norm=float("inf"))
    plan = RecipePlan.from_frame(processed, input_names=("signal",))

    first = plan.to_dict()
    second = RecipePlan.from_dict(copy.deepcopy(first)).to_dict()

    assert first == second
    assert json.loads(json.dumps(first, sort_keys=True, allow_nan=False)) == first


@pytest.mark.parametrize("version", [1, 0, 3, "2", True])
def test_loader_rejects_non_schema_2_payloads(version: object) -> None:
    payload = RecipePlan.from_frame(_frame().normalize()).to_dict()
    payload["version"] = version

    with pytest.raises(RecipeSerializationError, match="schema version|schema or invalid"):
        RecipePlan.from_dict(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(extra=True),
        lambda payload: payload["inputs"][0].update(extra=True),
        lambda payload: payload["nodes"][0].update(extra=True),
        lambda payload: payload["nodes"][0].update(version=True),
        lambda payload: payload["nodes"][0].update(inputs=[1]),
        lambda payload: payload["nodes"][0].update(params={"$type": "unknown"}),
    ],
)
def test_loader_rejects_malformed_schema_fields(mutation: Any) -> None:
    payload = RecipePlan.from_frame(_frame().normalize()).to_dict()
    mutation(payload)

    with pytest.raises(RecipeSerializationError):
        RecipePlan.from_dict(payload)


@pytest.mark.parametrize("operand", [1.5, complex(1.0, 2.0)])
def test_loader_wraps_truncated_python_number_payloads(operand: float | complex) -> None:
    payload = _plan_for_operand(operand).to_dict()
    payload["nodes"][0]["params"]["entries"][0][1]["data"] = ""

    with pytest.raises(RecipeSerializationError, match="data does not match its kind"):
        RecipePlan.from_dict(payload)


def test_loaded_plan_does_not_retain_mutable_payload_containers() -> None:
    payload = RecipePlan.from_frame(_frame().rename_channels({0: "renamed"})).to_dict()
    loaded = RecipePlan.from_dict(payload)
    expected = loaded.to_dict()

    payload["nodes"][0]["params"]["entries"].clear()
    payload["nodes"][0]["inputs"].clear()

    assert loaded.to_dict() == expected


def test_canonical_map_tag_cannot_collide_with_user_mapping_keys() -> None:
    from wandas.processing.semantic import freeze_params, value_from_json, value_to_json

    params = freeze_params({"payload": {"$type": "number", "kind": "user", "data": "unchanged"}})

    assert value_from_json(value_to_json(params)) == params


def test_loader_rejects_unsorted_top_level_canonical_map() -> None:
    payload = RecipePlan.from_frame(_frame().normalize(norm=2.0, axis=-1, threshold=0.1, fill=True)).to_dict()
    payload["nodes"][0]["params"]["entries"].reverse()

    with pytest.raises(RecipeSerializationError, match="sorted"):
        RecipePlan.from_dict(payload)


def test_loader_rejects_unsorted_nested_canonical_map() -> None:
    payload = RecipePlan.from_frame(_frame().rename_channels({0: "mono"})).to_dict()
    encoded_key = payload["nodes"][0]["params"]["entries"][0][1]["items"][0]["items"][0]
    encoded_key["entries"].reverse()

    with pytest.raises(RecipeSerializationError, match="sorted"):
        RecipePlan.from_dict(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(inputs=tuple(payload["inputs"])), "collections or output"),
        (lambda payload: payload["inputs"][0].update(id=1), "input values"),
        (lambda payload: payload["nodes"][0].update(operation=1), "node id or operation"),
        (
            lambda payload: payload["nodes"][0].update(params={"$type": "list", "items": []}),
            "params must be a canonical map",
        ),
    ],
)
def test_loader_rejects_well_shaped_records_with_invalid_values(mutation: Any, message: str) -> None:
    payload = RecipePlan.from_frame(_frame().normalize()).to_dict()
    mutation(payload)

    with pytest.raises(RecipeSerializationError, match=message):
        RecipePlan.from_dict(payload)


def test_recipe_json_artifact_roundtrip_is_executable(tmp_path: Path) -> None:
    plan = RecipePlan.from_frame(_frame().normalize())

    path = plan.save(tmp_path / "analysis")
    loaded = RecipePlan.load(tmp_path / "analysis")
    replayed = loaded.apply({"input_0": _frame()})

    assert path == tmp_path / "analysis.recipe.json"
    assert loaded.to_dict() == plan.to_dict()
    assert replayed.operation_history[-1]["operation"] == "wandas.audio.normalize"


def test_recipe_json_artifact_is_strict_deterministic_json(tmp_path: Path) -> None:
    plan = RecipePlan.from_frame(_frame().normalize())

    path = plan.save(tmp_path / "analysis.recipe.json")

    assert path.read_text(encoding="utf-8").endswith("\n")
    assert json.loads(path.read_text(encoding="utf-8")) == plan.to_dict()


def test_recipe_json_artifact_refuses_overwrite_by_default(tmp_path: Path) -> None:
    plan = RecipePlan.from_frame(_frame().normalize())
    path = plan.save(tmp_path / "analysis")

    with pytest.raises(FileExistsError):
        plan.save(path)

    plan.save(path, overwrite=True)
    assert RecipePlan.load(path).to_dict() == plan.to_dict()


def test_recipe_json_artifact_rejects_nonfinite_json(tmp_path: Path) -> None:
    path = tmp_path / "invalid.recipe.json"
    path.write_text('{"schema": NaN}\n', encoding="utf-8")

    with pytest.raises(RecipeSerializationError, match="strict JSON"):
        RecipePlan.load(path)


def test_recipe_json_artifact_wraps_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "malformed.recipe.json"
    path.write_text('{"schema":', encoding="utf-8")

    with pytest.raises(RecipeSerializationError, match="Invalid Recipe JSON artifact"):
        RecipePlan.load(path)


def test_recipe_json_artifact_missing_file_preserves_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError) as exc_info:
        RecipePlan.load(missing)

    assert exc_info.value.filename == str(tmp_path / "missing.recipe.json")
