import copy
import json
from typing import Any, cast

import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame
from wandas.pipeline import AudioCall, RecipeInput, RecipeNode, RecipePlan, RecipeSerializationError
from wandas.pipeline.calls import CustomCall, IndexCall, MethodCall, MultiInputCall, TerminalCall


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
    with pytest.raises(RecipeSerializationError, match="External arrays"):
        AudioCall("normalize", {"weights": np.arange(4)})


def test_persisted_ids_do_not_use_python_object_identity() -> None:
    payload = _plan().to_dict()
    serialized = json.dumps(payload)

    assert "node-0" in serialized and "input-0" in serialized
    assert str(id(_plan())) not in serialized


@pytest.mark.parametrize("build", [lambda frame: 10 - frame, lambda frame: frame + frame])
def test_edge_free_empty_params_use_the_canonical_value_tree(build: Any) -> None:
    source = ChannelFrame.from_numpy(np.arange(16, dtype=float).reshape(1, 16) + 1, sampling_rate=8000)
    plan = RecipePlan.from_frame(build(source))

    restored = RecipePlan.from_dict(plan.to_dict())

    assert restored.to_dict() == plan.to_dict()


def test_public_call_constructors_share_fail_closed_contracts() -> None:
    with pytest.raises(RecipeSerializationError, match="public member"):
        MethodCall("_lineage_or_source", "wandas.core.base_frame.BaseFrame._lineage_or_source")
    with pytest.raises(RecipeSerializationError, match="contract mismatch"):
        MethodCall("fft", "wandas.frames.mixins.channel_transform_mixin.ChannelTransformMixin.fft", version=99)
    with pytest.raises(RecipeSerializationError, match="selection intent"):
        IndexCall({"bad": "mapping"})
    with pytest.raises(RecipeSerializationError, match="public member"):
        TerminalCall("_lineage_or_source", "wandas.core.base_frame.BaseFrame._lineage_or_source")
    with pytest.raises(RecipeSerializationError, match="contract mismatch"):
        MethodCall("to_numpy", "wandas.core.base_frame.BaseFrame.to_numpy")
    with pytest.raises(RecipeSerializationError, match="contract mismatch"):
        TerminalCall("compute", "wandas.core.base_frame.BaseFrame.compute")


def test_custom_call_direct_constructor_deeply_freezes_params() -> None:
    params = {"gain": [1]}
    call = CustomCall("tests.pipeline.test_recipe_serialization._stable_custom", params)
    params["gain"].append(2)

    assert call.to_payload()["params"] == ("mapping", (("gain", ("list", (("int", 1),))),))


def test_frozen_value_tree_cannot_bypass_external_array_guard() -> None:
    frozen_array = (
        "mapping",
        (("weights", ("ndarray", "int64", (2,), ("list", (("int", 1), ("int", 2))))),),
    )

    with pytest.raises(RecipeSerializationError, match="External arrays"):
        CustomCall("tests.pipeline.test_recipe_serialization._stable_custom", frozen_array)


def _stable_custom(data: Any, *, gain: list[int]) -> Any:
    return data * gain[0]


def test_truncated_value_tree_is_normalized_to_serialization_error() -> None:
    payload = copy.deepcopy(_plan().to_dict())
    payload["nodes"][0]["call"]["params"] = ["mapping"]

    with pytest.raises(RecipeSerializationError, match="mapping value"):
        RecipePlan.from_dict(payload)


def test_valid_terminal_property_roundtrips_and_executes() -> None:
    source = ChannelFrame.from_numpy(np.ones((1, 16)), sampling_rate=8000)
    plan = RecipePlan(
        (RecipeInput("input-0", "signal"),),
        (
            RecipeNode(
                "node-0",
                TerminalCall("rms", "wandas.frames.channel.ChannelFrame.rms"),
                ("input-0",),
            ),
        ),
        "node-0",
    )

    restored = RecipePlan.from_dict(plan.to_dict())

    np.testing.assert_allclose(restored.apply({"signal": source}), [1.0])


def test_multi_input_direct_constructor_normalizes_mutable_value_tree() -> None:
    child: list[Any] = ["list", []]
    params = ("mapping", (("snr", child),))
    call = MultiInputCall(
        "add_with_snr",
        ("signal", "noise"),
        "wandas.pipeline.calls.apply_add_with_snr",
        cast(Any, params),
    )
    child[1].append(("float", 3.0))

    assert call.params == ("mapping", (("snr", ("list", ())),))


def test_audio_invoke_rechecks_runtime_operation_version(monkeypatch: pytest.MonkeyPatch) -> None:
    from wandas.processing import get_operation

    call = AudioCall("normalize")
    monkeypatch.setattr(get_operation("normalize"), "operation_version", 2, raising=False)

    with pytest.raises(RecipeSerializationError, match="version"):
        call.invoke((ChannelFrame.from_numpy(np.ones((1, 8)), sampling_rate=8000),))
