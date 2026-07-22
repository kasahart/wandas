from __future__ import annotations

import operator
from collections.abc import Mapping
from typing import Any

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

from tests.frame_helpers import channel_first_values
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


def test_fft_ifft_typed_transition_chain_replays() -> None:
    source = _frame()
    processed = source.fft(n_fft=32).ifft()
    plan = RecipePlan.from_frame(processed, input_names=("signal",))
    replayed = RecipePlan.from_dict(plan.to_dict()).apply({"signal": source})

    assert [node.operation for node in plan.nodes] == ["wandas.audio.fft", "wandas.spectral.ifft"]
    np.testing.assert_allclose(channel_first_values(replayed), channel_first_values(processed))


def test_remove_dc_channel_wise_execution_recipe_roundtrip_replays_lazily() -> None:
    source = ChannelFrame(
        da.from_array(
            np.array(
                [
                    [1.0, 2.0, 4.0, 8.0],
                    [8.0, 4.0, 2.0, 1.0],
                ]
            ),
            chunks=(1, -1),
        ),
        sampling_rate=8_000,
        source_time_offset=[0.25, 0.5],
    )
    processed = source.remove_dc()

    plan = RecipePlan.from_frame(processed, input_names=("signal",))
    replayed = RecipePlan.from_dict(plan.to_dict()).apply({"signal": source})

    assert [node.operation for node in plan.nodes] == ["wandas.audio.remove_dc"]
    assert isinstance(replayed._data, DaArray)
    assert replayed.shape == processed.shape
    np.testing.assert_array_equal(replayed.source_time_offset, processed.source_time_offset)
    np.testing.assert_allclose(channel_first_values(replayed), channel_first_values(processed))


def test_typed_transition_after_true_frame_merge_replays() -> None:
    left = _frame(1.0)
    right = _frame(2.0)
    processed = (left + right).fft(n_fft=16)
    replayed = RecipePlan.from_frame(processed, input_names=("left", "right")).apply({"left": left, "right": right})

    np.testing.assert_allclose(channel_first_values(replayed), channel_first_values(processed))


def test_external_numpy_and_dask_inputs_remain_lazy_until_user_compute() -> None:
    source = _frame()
    numpy_operand = np.arange(32.0)
    dask_operand = da.from_array(numpy_operand, chunks=8)

    for operand in (numpy_operand, dask_operand):
        plan = RecipePlan.from_frame(source + operand, input_names=("signal", "operand"))
        replayed = plan.apply({"signal": source, "operand": operand})

        assert isinstance(replayed._data, DaArray)


@pytest.mark.parametrize(
    ("operation", "operation_id"),
    [
        (lambda frame: frame + 2.0, "wandas.operator.add"),
        (lambda frame: 2.0 + frame, "wandas.operator.reverse_add"),
        (lambda frame: frame - 2.0, "wandas.operator.subtract"),
        (lambda frame: 2.0 - frame, "wandas.operator.reverse_subtract"),
        (lambda frame: frame * 2.0, "wandas.operator.multiply"),
        (lambda frame: 2.0 * frame, "wandas.operator.reverse_multiply"),
        (lambda frame: frame / 2.0, "wandas.operator.divide"),
        (lambda frame: 2.0 / frame, "wandas.operator.reverse_divide"),
        (lambda frame: frame**2.0, "wandas.operator.power"),
        (lambda frame: 2.0**frame, "wandas.operator.reverse_power"),
    ],
)
def test_scalar_operator_roundtrip_preserves_operand_order(operation: Any, operation_id: str) -> None:
    source = _frame()
    expected = operation(source)
    plan = RecipePlan.from_frame(expected, input_names=("signal",))
    replayed = RecipePlan.from_dict(plan.to_dict()).apply({"signal": source})

    assert plan.nodes[-1].operation == operation_id
    np.testing.assert_allclose(channel_first_values(replayed), channel_first_values(expected))


@pytest.mark.parametrize("array_operation", [operator.sub, operator.mul, operator.truediv, operator.pow])
def test_nonadditive_external_array_roundtrip_stays_lazy(array_operation: Any) -> None:
    source = _frame()
    numpy_operand = np.full((1, 32), 2.0)

    for operand in (numpy_operand, da.from_array(numpy_operand, chunks=(1, 8))):
        expected = array_operation(source, operand)
        plan = RecipePlan.from_frame(expected, input_names=("signal", "operand"))
        replayed = RecipePlan.from_dict(plan.to_dict()).apply({"signal": source, "operand": operand})

        assert isinstance(replayed._data, DaArray)
        np.testing.assert_allclose(channel_first_values(replayed), channel_first_values(expected))


def test_mix_replays_true_multi_frame_operation_in_role_order() -> None:
    signal = _frame(4.0)
    noise = _frame(2.0)
    processed = signal.mix(noise, snr_db=6.0)
    plan = RecipePlan.from_frame(processed, input_names=("signal", "noise"))
    replayed = plan.apply({"signal": signal, "noise": noise})

    assert plan.to_dict()["nodes"][-1]["operation"] == "wandas.audio.mix"
    np.testing.assert_allclose(channel_first_values(replayed), channel_first_values(processed), rtol=1e-12, atol=0.0)


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


def _plan_for_test_operation(operation: RecipeOperation, registry: RecipeRegistry) -> RecipePlan:
    payload = {
        "schema": "wandas.recipe",
        "version": 2,
        "inputs": [{"id": "input-0", "name": "signal", "kind": "frame"}],
        "nodes": [
            {
                "id": "node-0",
                "operation": operation.operation_id,
                "version": operation.version,
                "inputs": ["input-0"],
                "params": {"$type": "map", "entries": []},
            }
        ],
        "output": "node-0",
    }
    return RecipePlan.from_dict(payload, registry=registry)


def test_executor_preserves_nested_recipe_execution_error() -> None:
    def fail(_inputs: tuple[Any, ...], _params: Mapping[str, Any]) -> Any:
        raise RecipeExecutionError("nested execution failure")

    operation = RecipeOperation(
        "tests.nested-execution-error",
        1,
        ((InputBinding("frame", "frame"),),),
        fail,
    )
    registry = RecipeRegistry((operation,))
    plan = _plan_for_test_operation(operation, registry)

    with pytest.raises(RecipeExecutionError, match="nested execution failure"):
        plan.apply({"signal": _frame()}, registry=registry)


def test_executor_rejects_non_frame_operation_result() -> None:
    operation = RecipeOperation(
        "tests.non-frame-result",
        1,
        ((InputBinding("frame", "frame"),),),
        lambda _inputs, _params: 42,
    )
    registry = RecipeRegistry((operation,))
    plan = _plan_for_test_operation(operation, registry)

    with pytest.raises(RecipeExecutionError, match="returned int"):
        plan.apply({"signal": _frame()}, registry=registry)
