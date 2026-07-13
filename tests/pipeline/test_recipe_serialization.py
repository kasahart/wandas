import copy
import json
from typing import Any, cast

import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame
from wandas.pipeline import AudioCall, RecipePlan, RecipeSerializationError


def _plan() -> RecipePlan:
    source = ChannelFrame.from_numpy(np.ones((1, 16)), sampling_rate=8000)
    return RecipePlan.from_frame(source.normalize(), input_names=("signal",))


def test_canonical_schema_roundtrip_is_json_serializable() -> None:
    payload = _plan().to_dict()

    json.dumps(payload, allow_nan=False)
    restored = RecipePlan.from_dict(payload)

    assert restored.to_dict() == payload
    assert payload["schema"] == "wandas.recipe" and payload["version"] == 1


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda value: value.update(version=2), "schema version"),
        (lambda value: value.update(extra=True), "fields"),
        (lambda value: value["nodes"][0]["call"].update(type="unknown"), "Unknown Recipe call"),
        (lambda value: value["nodes"][0]["call"].update(version=True), "version"),
        (lambda value: value["nodes"][0]["call"].update(extra=True), "fields"),
    ],
)
def test_loader_fails_closed_for_schema_and_call_contracts(mutate: Any, match: str) -> None:
    payload = copy.deepcopy(_plan().to_dict())
    mutate(payload)

    with pytest.raises(RecipeSerializationError, match=match):
        RecipePlan.from_dict(payload)


def test_runtime_values_are_deeply_snapshotted() -> None:
    params = {"nested": {"values": [1]}}
    call = AudioCall("normalize", params)
    params["nested"]["values"].append(2)

    assert call.to_payload()["params"] == (
        "mapping",
        (("nested", ("mapping", (("values", ("list", (("int", 1),))),))),),
    )


def test_arbitrary_objects_and_non_string_mapping_keys_are_rejected() -> None:
    with pytest.raises(RecipeSerializationError, match="Unsupported"):
        AudioCall("normalize", {"value": object()})
    with pytest.raises(RecipeSerializationError, match="string-keyed"):
        AudioCall("normalize", cast(Any, {1: "value"}))


def test_persisted_ids_do_not_use_python_object_identity() -> None:
    payload = _plan().to_dict()
    serialized = json.dumps(payload)

    assert "node-0" in serialized and "input-0" in serialized
    assert str(id(_plan())) not in serialized
