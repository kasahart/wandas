"""Named steps: one plan, one returned Frame per apply, chosen by name."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

import wandas as wd
from wandas.frames.noct import NOctFrame
from wandas.frames.spectral import SpectralFrame
from wandas.pipeline import (
    RecipeExecutionError,
    RecipeExtractionError,
    RecipePlan,
    RecipeSerializationError,
    RecipeValidationError,
)

_SR = 48_000


def _signal(seed: int = 0) -> wd.ChannelFrame:
    samples = np.random.default_rng(seed).standard_normal((1, _SR // 4))
    return wd.from_numpy(samples, sampling_rate=_SR)


def _plan() -> RecipePlan:
    weighted = _signal().high_pass_filter(cutoff=50).a_weighting()
    return RecipePlan.from_frame(
        weighted.welch(n_fft=1024),
        steps={"weighted": weighted, "noct": weighted.noct_spectrum(fmin=100, fmax=5_000)},
    )


def _count_operations(plan: RecipePlan) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in plan.nodes:
        counts[node.operation] = counts.get(node.operation, 0) + 1
    return counts


def test_steps_share_preprocessing_and_keep_order() -> None:
    plan = _plan()

    assert plan.step_names == ("weighted", "noct")
    counts = _count_operations(plan)
    assert counts["wandas.audio.highpass_filter"] == 1
    assert counts["wandas.audio.a_weighting"] == 1


def test_apply_returns_output_or_named_step() -> None:
    plan = _plan()
    other = _signal(seed=1)
    weighted = other.high_pass_filter(cutoff=50).a_weighting()

    output = plan.apply(other)
    noct = plan.apply(other, step="noct")
    step = plan.apply(other, step="weighted")

    assert isinstance(output, SpectralFrame)
    assert isinstance(noct, NOctFrame)
    np.testing.assert_allclose(output.data, weighted.welch(n_fft=1024).data)
    np.testing.assert_allclose(noct.data, weighted.noct_spectrum(fmin=100, fmax=5_000).data)
    np.testing.assert_allclose(step.data, weighted.data)


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        ("weighted", ["wandas.audio.highpass_filter", "wandas.audio.a_weighting"]),
        ("noct", ["wandas.audio.highpass_filter", "wandas.audio.a_weighting", "wandas.audio.noct_spectrum"]),
        (None, ["wandas.audio.highpass_filter", "wandas.audio.a_weighting", "wandas.audio.welch"]),
    ],
)
def test_apply_builds_only_the_operations_the_target_needs(
    monkeypatch: pytest.MonkeyPatch, step: str | None, expected: list[str]
) -> None:
    from wandas.pipeline.registry import RecipeOperation

    plan = _plan()
    invoked: list[str] = []
    original = RecipeOperation.invoke

    def recording_invoke(self: RecipeOperation, inputs: tuple[Any, ...], params: Any) -> Any:
        invoked.append(self.operation_id)
        return original(self, inputs, params)

    monkeypatch.setattr(RecipeOperation, "invoke", recording_invoke)
    plan.apply(_signal(seed=1), step=step)

    assert invoked == expected


def test_for_step_keeps_only_the_inputs_that_step_uses() -> None:
    first = _signal()
    second = _signal(seed=1)
    filtered = first.high_pass_filter(cutoff=50)
    plan = RecipePlan.from_frame(filtered + second, input_names=("a", "b"), steps={"hp": filtered})
    a = _signal(seed=2)

    hp_plan = plan.for_step("hp")

    assert [item.name for item in hp_plan.inputs] == ["a"]
    np.testing.assert_array_equal(
        hp_plan.apply({"a": a}).data,
        plan.apply({"a": a, "b": second}, step="hp").data,
    )


def test_step_must_reference_a_frame() -> None:
    signal = _signal()
    plan = RecipePlan.from_frame(signal + np.ones((1, signal.n_samples)), input_names=("signal", "offset"))
    array_input = next(item.id for item in plan.inputs if item.kind == "array")

    with pytest.raises(RecipeValidationError, match="must be a frame"):
        RecipePlan(plan.inputs, plan.nodes, plan.output, steps=[("offset", array_input)])


def test_loader_rejects_a_non_mapping_payload() -> None:
    with pytest.raises(RecipeSerializationError, match="fields"):
        RecipePlan.from_dict([])  # ty: ignore[invalid-argument-type]


def test_unknown_step_names_the_available_steps() -> None:
    with pytest.raises(RecipeExecutionError, match="no step") as error:
        _plan().apply(_signal(), step="missing")

    assert "'weighted', 'noct'" in str(error.value)


def test_for_step_returns_a_plan_with_only_that_step() -> None:
    plan = _plan()

    noct_plan = plan.for_step("noct")

    assert noct_plan.steps == ()
    assert "wandas.audio.welch" not in _count_operations(noct_plan)
    np.testing.assert_array_equal(noct_plan.apply(_signal(1)).data, plan.apply(_signal(1), step="noct").data)


def test_plan_without_steps_still_writes_schema_3() -> None:
    plan = RecipePlan.from_frame(_signal().high_pass_filter(cutoff=50))

    payload = plan.to_dict()

    assert payload["version"] == 3
    assert "steps" not in payload
    assert plan.steps == ()


def test_steps_round_trip_through_schema_4(tmp_path: Path) -> None:
    plan = _plan()

    payload = plan.to_dict()
    path = plan.save(tmp_path / "steps")
    loaded = RecipePlan.load(path)

    assert payload["version"] == 4
    assert list(payload["steps"]) == ["weighted", "noct"]
    assert RecipePlan.from_dict(json.loads(json.dumps(payload))) == plan
    assert loaded == plan
    np.testing.assert_array_equal(loaded.apply(_signal(1), step="noct").data, plan.apply(_signal(1), step="noct").data)


@pytest.mark.parametrize(
    "change",
    [
        lambda payload: payload.pop("steps"),
        lambda payload: payload.__setitem__("steps", {}),
        lambda payload: payload.__setitem__("steps", ["weighted"]),
        lambda payload: payload.__setitem__("steps", {"weighted": 3}),
        lambda payload: payload.__setitem__("version", 3),
    ],
    ids=["missing", "empty", "list", "non-string", "schema-3-with-steps"],
)
def test_schema_4_steps_are_strict(change: Callable[[dict[str, Any]], object]) -> None:
    payload = _plan().to_dict()
    change(payload)

    with pytest.raises(RecipeSerializationError):
        RecipePlan.from_dict(payload)


def test_loaded_step_must_reference_a_frame_in_the_graph() -> None:
    payload = _plan().to_dict()
    payload["steps"] = {"weighted": "node-99"}

    with pytest.raises(RecipeSerializationError, match="unavailable"):
        RecipePlan.from_dict(payload)


def test_constructor_rejects_duplicate_and_blank_step_names() -> None:
    plan = _plan()
    reference = dict(plan.steps)["weighted"]

    with pytest.raises(RecipeValidationError, match="unique"):
        RecipePlan(plan.inputs, plan.nodes, plan.output, steps=[("a", reference), ("a", reference)])
    with pytest.raises(RecipeValidationError, match="step name"):
        RecipePlan(plan.inputs, plan.nodes, plan.output, steps=[(" ", reference)])
    with pytest.raises(RecipeValidationError, match="pair"):
        RecipePlan(plan.inputs, plan.nodes, plan.output, steps=[("a", reference, "extra")])  # ty: ignore[invalid-argument-type]


def test_steps_keep_their_nodes_reachable() -> None:
    plan = _plan()
    noct_reference = dict(plan.steps)["noct"]

    with pytest.raises(RecipeValidationError, match="unreachable"):
        RecipePlan(plan.inputs, plan.nodes, plan.output, steps=[(n, r) for n, r in plan.steps if r != noct_reference])


def test_step_from_another_source_is_rejected() -> None:
    signal = _signal()

    with pytest.raises(RecipeExtractionError, match="runtime input"):
        RecipePlan.from_frame(signal.welch(), steps={"other": _signal(1).welch()})


@pytest.mark.parametrize(
    ("steps", "match"),
    [
        ({"": None}, "non-blank"),
        ({"x": np.ones(3)}, "Wandas frame"),
        ([("x", None)], "mapping"),
    ],
)
def test_from_frame_rejects_invalid_steps(steps: object, match: str) -> None:
    signal = _signal()
    if isinstance(steps, dict) and "" in steps:
        steps = {"": signal}

    with pytest.raises(RecipeExtractionError, match=match):
        RecipePlan.from_frame(signal.welch(), steps=steps)  # ty: ignore[invalid-argument-type]


def test_step_may_name_the_source_frame() -> None:
    signal = _signal()
    plan = RecipePlan.from_frame(signal.high_pass_filter(cutoff=50), steps={"raw": signal})
    other = _signal(1)

    assert plan.apply(other, step="raw") is other


def test_apply_files_saves_the_named_step(tmp_path: Path) -> None:
    recording = tmp_path / "a.wav"
    sf.write(recording, _signal(1).data, _SR, subtype="DOUBLE")
    plan = _plan()

    table = plan.apply_files([recording], tmp_path / "out", step="noct")

    assert table["status"].tolist() == ["ok"]
    saved = wd.load(table.loc[0, "output"])
    np.testing.assert_allclose(saved.data, plan.apply(wd.read(recording), step="noct").data)


def test_apply_files_checks_the_step_before_reading(tmp_path: Path) -> None:
    with pytest.raises(RecipeExecutionError, match="no step"):
        _plan().apply_files([tmp_path / "missing.wav"], tmp_path / "out", step="missing")
