from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any, cast
from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

from tests.frame_helpers import channel_first_values
from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipePlan
from wandas.processing.semantic import FrozenMap, LineageNode


def _frame(value: float = 1.0, *, channels: int = 1) -> ChannelFrame:
    data = np.full((channels, 16), value, dtype=float)
    labels = [f"channel-{index}" for index in range(channels)]
    return ChannelFrame.from_numpy(data, sampling_rate=8000, ch_labels=labels)


def _names(frame: ChannelFrame) -> list[str]:
    return [record["operation"] for record in frame.operation_history]


def test_new_frame_has_explicit_source_lineage() -> None:
    frame = _frame()

    assert isinstance(frame.lineage, LineageNode)
    assert frame.lineage.operation is None
    assert frame.lineage.inputs == ()
    assert frame.operation_history == []


def test_removed_parallel_provenance_views_are_absent() -> None:
    frame = _frame().normalize()

    assert not hasattr(frame, "operations")
    assert not hasattr(frame, "operation_graph")
    assert not hasattr(frame, "operation_summaries")


def test_unary_operation_uses_one_semantic_node_for_history_and_recipe() -> None:
    source = _frame()
    result = source.normalize(norm=2.0)
    operation = result.lineage.operation

    assert operation is not None
    assert operation.operation_id == "wandas.audio.normalize"
    assert operation.version == 1
    assert operation.params == result.lineage.operation.params
    assert result.lineage.inputs == (source.lineage,)
    assert _names(result) == ["wandas.audio.normalize"]
    assert RecipePlan.from_frame(result).nodes[0].params is operation.params


def test_frame_binary_operation_preserves_operand_order_and_both_parents() -> None:
    left = _frame(5.0)
    right = _frame(2.0)
    result = left - right
    operation = result.lineage.operation

    assert operation is not None
    assert operation.operation_id == "wandas.operator.subtract"
    assert [(binding.role, binding.kind) for binding in operation.bindings] == [
        ("left", "frame"),
        ("right", "frame"),
    ]
    assert result.lineage.inputs == (left.lineage, right.lineage)
    np.testing.assert_allclose(channel_first_values(result), 3.0)


def test_external_array_binary_operation_has_no_array_lineage_parent() -> None:
    source = _frame()
    operand = da.ones(16, chunks=4)
    result = source + operand
    operation = result.lineage.operation

    assert operation is not None
    assert [(binding.role, binding.kind) for binding in operation.bindings] == [
        ("left", "frame"),
        ("right", "array"),
    ]
    assert result.lineage.inputs == (source.lineage, None)
    assert operation.params == FrozenMap(())


def test_scalar_binary_operation_stores_only_canonical_scalar_param() -> None:
    source = _frame()
    result = source * 2.0
    operation = result.lineage.operation

    assert operation is not None
    assert [(binding.role, binding.kind) for binding in operation.bindings] == [("left", "frame")]
    assert result.lineage.inputs == (source.lineage,)
    assert result.operation_history[-1]["params"] == {"operand": 2.0}


@pytest.mark.parametrize(
    ("build", "operation_id", "expected"),
    [
        (lambda frame: 2.0 + frame, "wandas.operator.reverse_add", 3.0),
        (lambda frame: 2.0 - frame, "wandas.operator.reverse_subtract", 1.0),
        (lambda frame: 2.0 * frame, "wandas.operator.reverse_multiply", 2.0),
        (lambda frame: 2.0 / frame, "wandas.operator.reverse_divide", 2.0),
        (lambda frame: 2.0**frame, "wandas.operator.reverse_power", 2.0),
    ],
)
def test_reverse_scalar_operations_preserve_public_intent(
    build: Callable[[ChannelFrame], ChannelFrame], operation_id: str, expected: float
) -> None:
    result = build(_frame())

    assert result.lineage.operation is not None
    assert result.lineage.operation.operation_id == operation_id
    np.testing.assert_allclose(channel_first_values(result), expected)


def test_depth_first_history_deduplicates_shared_source_nodes() -> None:
    source = _frame()
    left = source.normalize()
    right = source.remove_dc()
    result = left + right

    assert _names(result) == [
        "wandas.audio.normalize",
        "wandas.audio.remove_dc",
        "wandas.operator.add",
    ]


def test_depth_first_history_keeps_independent_branch_order() -> None:
    left = _frame().normalize()
    right = _frame(2.0).remove_dc()
    result = left + right

    assert _names(result) == [
        "wandas.audio.normalize",
        "wandas.audio.remove_dc",
        "wandas.operator.add",
    ]
    assert result.lineage.inputs == (left.lineage, right.lineage)


def test_history_is_strict_json_and_does_not_expose_numpy_or_dask_containers() -> None:
    result = _frame() + da.ones(16, chunks=4)

    encoded = json.dumps(result.operation_history, allow_nan=False)

    assert "numpy" not in encoded.lower()
    assert "dask" not in encoded.lower()
    assert "chunks" not in encoded.lower()


def test_history_params_are_frozen_at_public_call_entry() -> None:
    source = _frame(channels=2)
    mapping: dict[int | str, str] = {0: "left"}
    result = source.rename_channels(mapping)
    expected_history = copy.deepcopy(result.operation_history)
    expected_plan = RecipePlan.from_frame(result).to_dict()

    mapping[0] = "mutated"
    mapping[1] = "right"

    assert result.operation_history == expected_history
    assert RecipePlan.from_frame(result).to_dict() == expected_plan


def test_operation_history_projection_is_defensive() -> None:
    result = _frame().normalize(norm=2.0)
    first = result.operation_history
    first[0]["params"]["norm"] = 3.0
    first.append({"operation": "fake", "version": 1, "params": {}})

    assert result.operation_history == [{"operation": "wandas.audio.normalize", "version": 1, "params": {"norm": 2.0}}]


def test_public_operation_and_history_access_do_not_compute() -> None:
    source = _frame()

    with patch.object(DaArray, "compute", autospec=True, side_effect=AssertionError("unexpected compute")):
        result = source.normalize().remove_dc()
        history = result.operation_history
        plan = RecipePlan.from_frame(result)

    assert len(history) == 2
    assert len(plan.nodes) == 2


def test_indexing_uses_one_semantic_node_for_multidimensional_selection() -> None:
    source = _frame(channels=2)
    source.source_time_offset = np.array([0.25, 0.5])

    result = source[:, 4:12]

    assert _names(result) == ["wandas.frame.index"]
    assert result.lineage.inputs == (source.lineage,)
    np.testing.assert_allclose(result.source_time_offset, np.array([0.2505, 0.5005]))


def test_mix_history_preserves_both_branch_operations_before_mix() -> None:
    signal = _frame(4.0).normalize()
    noise = _frame(2.0).remove_dc()

    result = signal.mix(noise, snr_db=6.0)

    assert _names(result) == [
        "wandas.audio.normalize",
        "wandas.audio.remove_dc",
        "wandas.audio.mix",
    ]
    assert result.lineage.inputs == (signal.lineage, noise.lineage)


def test_source_history_prefix_is_copied_and_extended_without_executable_recipe() -> None:
    prefix = [{"operation": "persisted", "version": 1, "params": {"gain": 2.0}}]
    frame = ChannelFrame(
        da.ones((1, 16), chunks=(1, 4)),
        sampling_rate=8000,
        operation_history_prefix=prefix,
    )
    prefix[0]["params"]["gain"] = 99.0

    result = frame.normalize()

    assert result.operation_history == [
        {"operation": "persisted", "version": 1, "params": {"gain": 2.0}},
        {"operation": "wandas.audio.normalize", "version": 1, "params": {}},
    ]
    plan = RecipePlan.from_frame(result)
    assert len(plan.inputs) == 1
    assert len(plan.nodes) == 1


def test_source_history_prefix_rejects_malformed_records() -> None:
    with pytest.raises(ValueError, match="record is malformed"):
        ChannelFrame(
            da.ones((1, 8), chunks=(1, 4)),
            sampling_rate=8000,
            operation_history_prefix=[{"operation": "missing-fields"}],
        )


def test_existing_lineage_rejects_source_history_prefix() -> None:
    source = _frame()

    with pytest.raises(ValueError, match="valid only for a new source Frame"):
        ChannelFrame(
            da.ones((1, 8), chunks=(1, 4)),
            sampling_rate=8000,
            lineage=source.lineage,
            operation_history_prefix=[{"operation": "persisted", "version": 1, "params": {}}],
        )


@pytest.mark.parametrize(
    "selector",
    [
        (["channel-0"], slice(2, 8)),
        (np.array([True, False]), slice(2, 8)),
    ],
)
def test_multidimensional_label_and_mask_selectors_replay(selector: tuple[object, slice]) -> None:
    source = _frame(channels=2)
    selected = source[cast(Any, selector)]
    plan = RecipePlan.from_frame(selected, input_names=("signal",))

    replayed = plan.apply({"signal": source})

    assert replayed.shape == selected.shape
    np.testing.assert_array_equal(channel_first_values(replayed), channel_first_values(selected))
