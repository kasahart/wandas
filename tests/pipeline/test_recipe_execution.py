from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.frames.channel import ChannelFrame
from wandas.pipeline import (
    RecipeExecutionError,
    RecipeOperation,
    RecipePlan,
    RecipeRegistry,
    RecipeValidationError,
    default_recipe_registry,
)
from wandas.processing.semantic import InputBinding


def _frame(value: float = 1.0, *, sampling_rate: int = 8000) -> ChannelFrame:
    return ChannelFrame.from_numpy(np.full((1, 32), value), sampling_rate=sampling_rate)


def test_typed_frame_transition_replays_lazily() -> None:
    source = _frame()
    processed = source.fft(n_fft=16)
    replayed = RecipePlan.from_frame(processed).apply({"input_0": source})

    assert type(replayed) is type(processed)
    assert replayed.shape == processed.shape
    assert isinstance(replayed._data, DaArray)


def test_typed_transition_after_true_frame_merge_replays() -> None:
    left = _frame(1.0)
    right = _frame(2.0)
    processed = (left + right).fft(n_fft=16)
    replayed = RecipePlan.from_frame(processed, input_names=("left", "right")).apply({"left": left, "right": right})

    np.testing.assert_allclose(replayed.compute(), processed.compute())


def test_external_numpy_and_dask_inputs_remain_lazy_until_user_compute() -> None:
    source = _frame()
    numpy_operand = np.arange(32.0)
    dask_operand = da.from_array(numpy_operand, chunks=8)

    for operand in (numpy_operand, dask_operand):
        plan = RecipePlan.from_frame(source + operand, input_names=("signal", "operand"))
        replayed = plan.apply({"signal": source, "operand": operand})

        assert isinstance(replayed._data, DaArray)


def test_mix_replays_true_multi_frame_operation_in_role_order() -> None:
    signal = _frame(4.0)
    noise = _frame(2.0)
    processed = signal.mix(noise, snr_db=6.0)
    plan = RecipePlan.from_frame(processed, input_names=("signal", "noise"))
    replayed = plan.apply({"signal": signal, "noise": noise})

    assert plan.to_dict()["nodes"][-1]["operation"] == "wandas.audio.mix"
    np.testing.assert_allclose(replayed.compute(), processed.compute(), rtol=1e-12, atol=0.0)


def test_executor_rejects_array_input_of_wrong_runtime_kind() -> None:
    source = _frame()
    plan = RecipePlan.from_frame(source + np.ones(32), input_names=("signal", "operand"))

    with pytest.raises(RecipeExecutionError, match="Recipe array input requires"):
        plan.apply({"signal": source, "operand": [1.0] * 32})


class ExternalChannelFrame(ChannelFrame):
    """Test-only external BaseFrame subclass."""


def test_executor_detects_lineage_mismatch_for_external_base_frame_subclass() -> None:
    operation_id = "tests.broken-external-result"

    def return_input(inputs: tuple[Any, ...], _params: Mapping[str, Any]) -> Any:
        return inputs[0]

    operation = RecipeOperation(
        operation_id,
        1,
        ((InputBinding("frame", "frame"),),),
        "frame",
        return_input,
    )
    registry = default_recipe_registry().with_operation(operation)
    payload = {
        "schema": "wandas.recipe",
        "version": 2,
        "inputs": [{"id": "input-0", "name": "signal", "kind": "frame"}],
        "nodes": [
            {
                "id": "node-0",
                "operation": operation_id,
                "version": 1,
                "inputs": ["input-0"],
                "params": {"$type": "map", "entries": []},
            }
        ],
        "output": "node-0",
    }
    plan = RecipePlan.from_dict(payload, registry=registry)
    source = ExternalChannelFrame.from_numpy(np.ones((1, 8)), sampling_rate=8000)

    with pytest.raises(RecipeExecutionError, match="did not preserve semantic lineage"):
        plan.apply({"signal": source}, registry=registry)


def test_execution_revalidates_plan_with_selected_registry() -> None:
    plan = RecipePlan.from_frame(_frame().normalize())

    with pytest.raises(RecipeValidationError, match="unregistered"):
        # A registry lacking built-ins must never silently fall back to a global registry.
        plan.apply({"input_0": _frame()}, registry=RecipeRegistry())
