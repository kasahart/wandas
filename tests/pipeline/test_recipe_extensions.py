"""Executable public-path probes for the three Recipe extension KPI families."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pytest

import wandas.processing.base as processing_base
from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipePlan
from wandas.pipeline.decorators import multi_input_handler, replay_method
from wandas.processing.base import AudioOperation


class RecipeProbeUnary(AudioOperation[Any, Any]):
    name = "recipe_probe_unary"
    supports_generic_replay = True

    def _process(self, data: Any) -> Any:
        return data


class RecipeProbeTransitionFrame(ChannelFrame):
    """Test-only target type for a persisted Frame transition."""


class RecipeProbeFrame(ChannelFrame):
    @replay_method(version=2)
    def probe_transition(self) -> RecipeProbeTransitionFrame:
        return RecipeProbeTransitionFrame(
            data=self._data,
            sampling_rate=self.sampling_rate,
            label=self.label,
            metadata=self.metadata,
            channel_metadata=self.channels.to_list(),
            channel_ids=self._channel_ids,
            source_time_offset=self.source_time_offset,
            lineage=self._required_semantic_lineage(),
        )

    def probe_multi(self, other: RecipeProbeFrame, *, gain: float) -> RecipeProbeFrame:
        operation = RecipeProbeMulti(self.sampling_rate, gain=gain)
        data = operation.process(self._data, other._data)
        lineage = self._lineage_with_operation(
            operation,
            self._lineage_or_source(),
            other._lineage_or_source(),
        )
        return self._create_new_instance(data=data, lineage=lineage)


class RecipeProbeMulti(AudioOperation[Any, Any]):
    name = "recipe_probe_multi"
    _expected_input_count = 2
    input_roles = ("signal", "noise")
    replay_handler_path = f"{__name__}.probe_multi_handler"

    def _process(self, data: Any, noise: Any) -> Any:
        return data + noise * self._config_value("gain")


@multi_input_handler("recipe_probe_multi", version=1, roles=("signal", "noise"))
def probe_multi_handler(inputs: tuple[Any, ...], params: Mapping[str, Any]) -> Any:
    return inputs[0] + inputs[1] * params["gain"]


def _frame(value: float = 1.0) -> RecipeProbeFrame:
    return cast(RecipeProbeFrame, RecipeProbeFrame.from_numpy(np.full((1, 8), value), sampling_rate=8000))


def _persist_and_apply(result: ChannelFrame, inputs: Mapping[str, ChannelFrame]) -> ChannelFrame:
    plan = RecipePlan.from_frame(result, input_names=tuple(inputs))
    loaded = RecipePlan.from_dict(plan.to_dict())
    replayed = loaded.apply(inputs)
    assert isinstance(replayed, ChannelFrame)
    np.testing.assert_allclose(replayed.compute(), result.compute())
    return replayed


def test_new_unary_same_frame_runs_complete_public_recipe_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(processing_base._OPERATION_REGISTRY, RecipeProbeUnary.name, RecipeProbeUnary)
    source = _frame()

    replayed = _persist_and_apply(source.apply_operation(RecipeProbeUnary.name), {"signal": source})

    assert type(replayed) is RecipeProbeFrame


def test_new_typed_transition_runs_complete_public_recipe_path() -> None:
    source = _frame()

    replayed = _persist_and_apply(source.probe_transition(), {"signal": source})

    assert type(replayed) is RecipeProbeTransitionFrame


def test_new_true_multi_input_runs_complete_public_recipe_path() -> None:
    signal = _frame(1.0)
    noise = _frame(2.0)

    replayed = _persist_and_apply(
        signal.probe_multi(noise, gain=0.5),
        {"signal": signal, "noise": noise},
    )

    assert type(replayed) is RecipeProbeFrame
