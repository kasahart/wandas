from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from fractions import Fraction
from typing import Any

import dask.array as da
import numpy as np
import pytest

from tests.frame_helpers import channel_first_values
from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipeExecutionError, RecipePlan
from wandas.processing.semantic import freeze_value, semantic_lineage, value_to_json


def _frame(
    values: np.ndarray[Any, Any],
    *,
    labels: list[str],
    offsets: list[float] | None = None,
) -> ChannelFrame:
    return ChannelFrame(
        da.from_array(values, chunks=(1, -1)),
        sampling_rate=8_000,
        channel_metadata=[{"label": label} for label in labels],
        source_time_offset=offsets if offsets is not None else [0.0] * len(labels),
    )


def _payload(result: ChannelFrame, names: tuple[str, ...]) -> dict[str, Any]:
    return RecipePlan.from_frame(result, input_names=names).to_dict()


def _replace_params(payload: dict[str, Any], params: dict[str, Any]) -> None:
    payload["nodes"][0]["params"] = value_to_json(freeze_value(params))


def test_add_channel_v2_round_trip_replays_lazy_external_array() -> None:
    base = _frame(np.zeros((1, 8)), labels=["base"], offsets=[1.0])
    array = da.arange(8, chunks=4)
    result = base.add_channel(array, label="raw", source_time_offset=2.5)

    payload = _payload(result, ("base", "array"))
    replayed = RecipePlan.from_dict(payload).apply({"base": base, "array": array})

    assert payload["nodes"][0]["operation"] == "wandas.channel.add_channel"
    assert payload["nodes"][0]["version"] == 2
    assert [item["kind"] for item in payload["inputs"]] == ["frame", "array"]
    assert isinstance(replayed._data, da.Array)
    assert replayed.labels == ["base", "raw"]
    np.testing.assert_array_equal(replayed.source_time_offset, [1.0, 2.5])
    np.testing.assert_array_equal(channel_first_values(replayed), channel_first_values(result))


def test_add_channel_v2_round_trip_accepts_explicit_default_offset() -> None:
    base = _frame(np.zeros((1, 8)), labels=["base"], offsets=[1.0])
    array = np.ones(8)
    result = base.add_channel(array, source_time_offset=None)

    payload = _payload(result, ("base", "array"))
    replayed = RecipePlan.from_dict(payload).apply({"base": base, "array": array})

    encoded_offset = dict(payload["nodes"][0]["params"]["entries"])["source_time_offset"]
    assert encoded_offset["kind"] == "python-float"
    np.testing.assert_array_equal(replayed.source_time_offset, [1.0, 0.0])


@pytest.mark.parametrize("array_kind", ["numpy", "dask"])
@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (None, 0.0),
        (1.25, 1.25),
        ([1.5], 1.5),
        (np.array([1.75]), 1.75),
        (range(1), 0.0),
        (memoryview(b"\x02"), 2.0),
        (Fraction(9, 4), 2.25),
    ],
)
def test_add_channel_accepted_offsets_round_trip_for_array_kinds(
    array_kind: str,
    offset: Any,
    expected: float,
) -> None:
    base = _frame(np.zeros((1, 8)), labels=["base"], offsets=[1.0])
    array: Any = np.ones(8) if array_kind == "numpy" else da.ones(8, chunks=4)

    result = base.add_channel(array, source_time_offset=offset)
    payload = _payload(result, ("base", "array"))
    loaded = RecipePlan.from_dict(payload)
    replayed = loaded.apply({"base": base, "array": array})

    assert payload["nodes"][0]["version"] == 2
    encoded_offset = dict(payload["nodes"][0]["params"]["entries"])["source_time_offset"]
    assert encoded_offset["kind"] == "python-float"
    assert type(result.operation_history[-1]["params"]["source_time_offset"]) is float
    np.testing.assert_array_equal(replayed.source_time_offset, [1.0, expected])
    np.testing.assert_array_equal(channel_first_values(replayed), channel_first_values(result))


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (range(1), 0.0),
        (memoryview(b"\x02"), 2.0),
        (Fraction(9, 4), 2.25),
    ],
)
def test_add_channel_accepted_offsets_normalize_inside_active_lineage(
    offset: Any,
    expected: float,
) -> None:
    base = _frame(np.zeros((1, 8)), labels=["base"], offsets=[1.0])

    with semantic_lineage(base.lineage):
        result = base.add_channel(np.ones(8), source_time_offset=offset)

    np.testing.assert_array_equal(result.source_time_offset, [1.0, expected])


def test_add_channel_offset_outside_float_range_has_contract_error() -> None:
    base = _frame(np.zeros((1, 8)), labels=["base"], offsets=[1.0])

    with pytest.raises(ValueError, match="representable as a finite float"):
        base.add_channel(np.ones(8), source_time_offset=10**1000)


def test_concat_frame_v1_round_trip_preserves_frame_contract() -> None:
    base = _frame(np.zeros((1, 8)), labels=["base"], offsets=[1.0])
    other = _frame(np.arange(16).reshape(2, 8), labels=["left", "right"], offsets=[2.0, 3.0])
    other = other.with_calibration(
        {
            0: other.channels[0].calibration.with_ref(1e-6),
            1: other.channels[1].calibration.with_ref(2e-6),
        }
    )
    base_ids = list(base._channel_ids)
    other_ids = list(other._channel_ids)

    result = base.concat_frame(other, label_prefix="copy")
    payload = _payload(result, ("base", "other"))
    replayed = RecipePlan.from_dict(payload).apply({"base": base, "other": other})

    concat_node = next(node for node in payload["nodes"] if node["operation"] == "wandas.channel.concat_frame")
    assert concat_node["version"] == 1
    assert [item["kind"] for item in payload["inputs"]] == ["frame", "frame"]
    assert "source_time_offset" not in repr(concat_node["params"])
    assert result.labels == ["base", "copy_left", "copy_right"]
    assert [channel.calibration.ref for channel in result.channels] == [1.0, 1e-6, 2e-6]
    assert base._channel_ids == base_ids
    assert other._channel_ids == other_ids
    assert result._channel_ids[:1] == base_ids
    assert len(set(result._channel_ids)) == 3
    assert sum(item["operation"] == "wandas.channel.concat_frame" for item in result.operation_history) == 1
    assert isinstance(replayed._data, da.Array)
    np.testing.assert_array_equal(replayed.source_time_offset, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(channel_first_values(replayed), channel_first_values(result))


@pytest.mark.parametrize(
    ("method", "params"),
    [
        ("add", {"align": "invalid"}),
        ("add", {"label": 1}),
        ("add", {"suffix_on_dup": 1}),
        ("add", {"source_time_offset": "invalid"}),
        ("add", {"source_time_offset": []}),
        ("add", {"source_time_offset": [1.0, 2.0]}),
        ("add", {"source_time_offset": [[1.0]]}),
        ("add", {"source_time_offset": True}),
        ("add", {"source_time_offset": np.bool_(True)}),
        ("add", {"source_time_offset": float("nan")}),
        ("add", {"source_time_offset": float("inf")}),
        ("add", {"unknown": True}),
        ("concat", {"align": "invalid"}),
        ("concat", {"label_prefix": 1}),
        ("concat", {"suffix_on_dup": 1}),
        ("concat", {"source_time_offset": 1}),
        ("concat", {"unknown": True}),
    ],
)
def test_channel_recipe_params_are_rejected_at_load_time(method: str, params: dict[str, Any]) -> None:
    base = _frame(np.zeros((1, 8)), labels=["base"])
    if method == "add":
        result = base.add_channel(np.ones(8))
        payload = _payload(result, ("base", "array"))
    else:
        result = base.concat_frame(_frame(np.ones((1, 8)), labels=["other"]))
        payload = _payload(result, ("base", "other"))
    _replace_params(payload, params)

    with pytest.raises(ValueError, match="params violate its registered contract"):
        RecipePlan.from_dict(payload)


@pytest.mark.parametrize(
    ("method", "params"),
    [
        ("add", {"align": "invalid"}),
        ("add", {"label": 1}),
        ("add", {"suffix_on_dup": 1}),
        ("add", {"source_time_offset": []}),
        ("add", {"source_time_offset": [1.0, 2.0]}),
        ("add", {"source_time_offset": np.array([])}),
        ("add", {"source_time_offset": np.array([1.0, 2.0])}),
        ("add", {"source_time_offset": np.array([[1.0]])}),
        ("add", {"source_time_offset": True}),
        ("add", {"source_time_offset": np.bool_(True)}),
        ("add", {"source_time_offset": "invalid"}),
        ("add", {"source_time_offset": b"invalid"}),
        ("add", {"source_time_offset": float("nan")}),
        ("add", {"source_time_offset": float("-inf")}),
        ("concat", {"align": "invalid"}),
        ("concat", {"label_prefix": 1}),
        ("concat", {"suffix_on_dup": 1}),
    ],
)
@pytest.mark.parametrize("nested", [False, True], ids=["public", "active-lineage"])
def test_channel_public_parameter_validation_is_independent_of_runtime_conditions(
    method: str,
    params: dict[str, Any],
    nested: bool,
) -> None:
    base = _frame(np.zeros((1, 8)), labels=["base"])
    context = semantic_lineage(base.lineage) if nested else nullcontext()

    with context, pytest.raises((TypeError, ValueError)):
        if method == "add":
            base.add_channel(np.ones(8), **params)
        else:
            other = _frame(np.ones((1, 8)), labels=["other"])
            base.concat_frame(other, **params)


@pytest.mark.parametrize("kind", ["array", "frame"])
def test_legacy_add_channel_v1_recipes_load_and_replay(kind: str) -> None:
    base = _frame(np.zeros((1, 8)), labels=["base"], offsets=[1.0])
    if kind == "array":
        external: Any = np.ones(8)
        result = base.add_channel(external, label="legacy", source_time_offset=2.0)
        payload = _payload(result, ("base", "data"))
        payload["nodes"][0]["version"] = 1
    else:
        external = _frame(np.ones((1, 8)), labels=["other"], offsets=[2.0])
        result = base.concat_frame(external, label_prefix="legacy")
        payload = _payload(result, ("base", "data"))
        payload["nodes"][0]["operation"] = "wandas.channel.add_channel"
        payload["nodes"][0]["params"]["entries"] = [
            ["label", "legacy"],
            ["source_time_offset", None],
        ]

    loaded = RecipePlan.from_dict(deepcopy(payload))
    replayed = loaded.apply({"base": base, "data": external})

    assert loaded.nodes[0].version == 1
    expected_labels = ["base", "legacy_other"] if kind == "frame" else ["base", "legacy"]
    assert replayed.labels == expected_labels
    np.testing.assert_array_equal(replayed.source_time_offset, [1.0, 2.0])


def test_legacy_add_channel_v1_rejects_frame_offset_that_was_never_valid() -> None:
    base = _frame(np.zeros((1, 8)), labels=["base"])
    other = _frame(np.ones((1, 8)), labels=["other"])
    payload = _payload(base.concat_frame(other), ("base", "data"))
    payload["nodes"][0]["operation"] = "wandas.channel.add_channel"
    payload["nodes"][0]["params"]["entries"] = [
        [
            "source_time_offset",
            {"$type": "number", "kind": "python-float", "data": "3ff0000000000000"},
        ]
    ]

    loaded = RecipePlan.from_dict(payload)

    with pytest.raises(RecipeExecutionError, match="source_time_offset is only supported for array input"):
        loaded.apply({"base": base, "data": other})
