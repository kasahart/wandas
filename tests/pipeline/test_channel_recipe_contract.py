from __future__ import annotations

from copy import deepcopy
from typing import Any

import dask.array as da
import numpy as np
import pytest

from tests.frame_helpers import channel_first_values
from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipePlan


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

    assert ["source_time_offset", None] in payload["nodes"][0]["params"]["entries"]
    np.testing.assert_array_equal(replayed.source_time_offset, [1.0, 0.0])


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
    payload["nodes"][0]["params"] = {
        "$type": "map",
        "entries": [[key, value] for key, value in sorted(params.items())],
    }

    with pytest.raises(ValueError, match="params violate its registered contract"):
        RecipePlan.from_dict(payload)


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
