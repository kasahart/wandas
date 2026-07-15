from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipeExecutionError, RecipePlan


def _frame() -> ChannelFrame:
    data = np.arange(72.0).reshape(3, 24)
    frame = ChannelFrame.from_numpy(
        data,
        sampling_rate=8,
        ch_labels=["left", "right", "aux"],
        ch_units=["Pa", "Pa", "V"],
    )
    frame.metadata["owner"] = {"team": "audio"}
    frame.source_time_offset = np.array([0.25, 0.5, 0.75])
    return frame


def _operation_names(frame: ChannelFrame) -> list[str]:
    return [record["operation"] for record in frame.operation_history]


def test_public_processing_call_creates_one_atomic_semantic_record() -> None:
    result = _frame().normalize()

    assert _operation_names(result) == ["wandas.audio.normalize"]
    assert result.lineage.operation is not None
    assert result.lineage.operation.operation_id == "wandas.audio.normalize"
    assert len(result.lineage.inputs) == 1


def test_rename_mapping_mutation_does_not_change_history_or_plan() -> None:
    mapping: dict[int | str, str] = {0: "renamed"}
    result = _frame().rename_channels(mapping)
    expected_history = copy.deepcopy(result.operation_history)
    expected_plan = RecipePlan.from_frame(result).to_dict()

    mapping[0] = "mutated"
    mapping[1] = "new"

    assert result.operation_history == expected_history
    assert RecipePlan.from_frame(result).to_dict() == expected_plan


def test_source_time_offset_mutation_does_not_change_history_or_plan() -> None:
    offsets = np.array([2.5])
    result = _frame().add_channel(np.ones(24), label="extra", source_time_offset=offsets)
    expected_history = copy.deepcopy(result.operation_history)
    expected_plan = RecipePlan.from_frame(result).to_dict()

    offsets[0] = 99.0

    assert result.operation_history == expected_history
    assert RecipePlan.from_frame(result).to_dict() == expected_plan
    np.testing.assert_allclose(result.source_time_offset[-1], 2.5)


@pytest.mark.parametrize(
    "selector",
    [
        0,
        -1,
        slice(0, 2),
        [0, 2],
        ["left", "aux"],
        np.array([0, 2]),
        np.array([True, False, True]),
        "right",
        (slice(0, 2), slice(2, 10)),
    ],
)
def test_each_supported_index_form_creates_exactly_one_record(selector: Any) -> None:
    selected = _frame()[selector]

    assert _operation_names(selected) == ["wandas.frame.index"]
    assert len(selected.lineage.inputs) == 1


def test_multidimensional_index_roundtrip_preserves_data_metadata_and_offset() -> None:
    source = _frame()
    selected = source[[0, 2], 4:12]
    plan = RecipePlan.from_frame(selected, input_names=("signal",))
    replayed = plan.apply({"signal": source})

    np.testing.assert_allclose(replayed.compute(), selected.compute())
    assert replayed.metadata == selected.metadata
    assert replayed.labels == selected.labels
    np.testing.assert_allclose(replayed.source_time_offset, selected.source_time_offset)


def test_non_time_axis_step_roundtrips_after_typed_transition() -> None:
    source = _frame()
    selected = source.fft(n_fft=24)[:, ::2]
    plan = RecipePlan.from_frame(selected, input_names=("signal",))

    replayed = RecipePlan.from_dict(plan.to_dict()).apply({"signal": source})

    np.testing.assert_allclose(replayed.compute(), selected.compute())
    assert replayed.metadata == selected.metadata
    assert replayed.labels == selected.labels
    np.testing.assert_allclose(replayed.source_time_offset, selected.source_time_offset)


def test_non_time_axis_point_requires_one_element_slice_and_roundtrips() -> None:
    source = _frame()
    spectral = source.fft(n_fft=24)

    with pytest.raises(ValueError, match="one-element slice"):
        spectral[:, 0]

    selected = spectral[:, 0:1]
    plan = RecipePlan.from_frame(selected, input_names=("signal",))
    replayed = RecipePlan.from_dict(plan.to_dict()).apply({"signal": source})

    np.testing.assert_allclose(replayed.compute(), selected.compute())
    assert replayed.metadata == selected.metadata
    assert replayed.labels == selected.labels
    np.testing.assert_allclose(replayed.source_time_offset, selected.source_time_offset)


def test_get_channel_all_false_boolean_mask_roundtrips_by_intent() -> None:
    source = _frame()
    mask = np.array([False, False, False])
    selected = source.get_channel(mask)
    mask[:] = True
    plan = RecipePlan.from_frame(selected, input_names=("signal",))

    replayed = RecipePlan.from_dict(plan.to_dict()).apply({"signal": source})

    assert replayed.n_channels == 0
    assert replayed.shape == selected.shape
    assert replayed.labels == []
    np.testing.assert_allclose(replayed.source_time_offset, selected.source_time_offset)


def test_get_channel_boolean_mask_revalidates_runtime_channel_count() -> None:
    source = _frame()
    selected = source.get_channel(np.array([True, False, True]))
    plan = RecipePlan.from_frame(selected, input_names=("signal",))
    runtime = ChannelFrame.from_numpy(np.arange(48.0).reshape(2, 24), sampling_rate=8)

    with pytest.raises(RecipeExecutionError, match="Boolean mask length 3 does not match number of channels 2"):
        plan.apply({"signal": runtime})


@pytest.mark.parametrize("time_slice", [slice(None, None, 2), slice(None, None, -1)])
def test_time_axis_step_or_reverse_is_rejected_at_public_boundary(time_slice: slice) -> None:
    with pytest.raises(ValueError, match="continuous forward slicing"):
        _frame()[:, time_slice]


def test_literal_metadata_query_replays_against_runtime_input() -> None:
    source = _frame()
    selected = source.get_channel(query={"unit": "Pa"})
    plan = RecipePlan.from_frame(selected, input_names=("signal",))

    runtime = _frame().rename_channels({0: "runtime-left", 1: "runtime-right"})
    replayed = plan.apply({"signal": runtime})

    assert replayed.labels == ["runtime-left", "runtime-right"]


def test_list_valued_metadata_query_roundtrips_public_value_shape() -> None:
    source = _frame()
    source.channels[0].extra["tags"] = ["x", "y"]
    selected = source.get_channel(query={"tags": ["x", "y"]})
    plan = RecipePlan.from_frame(selected, input_names=("signal",))

    replayed = RecipePlan.from_dict(plan.to_dict()).apply({"signal": source})

    assert replayed.labels == ["left"]


def test_tuple_valued_metadata_query_roundtrips_public_value_shape() -> None:
    source = _frame()
    source.channels[0].extra["tags"] = ["x", "y"]
    source.channels[1].extra["tags"] = ("x", "y")
    selected = source.get_channel(query={"tags": ("x", "y")})
    plan = RecipePlan.from_frame(selected, input_names=("signal",))

    replayed = RecipePlan.from_dict(plan.to_dict()).apply({"signal": source})

    assert replayed.labels == ["right"]


def test_external_array_history_omits_backend_and_payload_details() -> None:
    operand = da.ones((3, 24), chunks=(1, 6))
    result = _frame() + operand
    history_json = json.dumps(result.operation_history, allow_nan=False)

    assert _operation_names(result) == ["wandas.operator.add"]
    assert "dask" not in history_json.lower()
    assert "numpy" not in history_json.lower()
    assert "chunks" not in history_json.lower()


def test_array_operand_mutation_does_not_change_history_or_plan() -> None:
    operand = np.ones((3, 24))
    result = _frame() + operand
    expected_history = copy.deepcopy(result.operation_history)
    expected_plan = RecipePlan.from_frame(result).to_dict()

    operand[:] = 99.0

    assert result.operation_history == expected_history
    assert RecipePlan.from_frame(result).to_dict() == expected_plan


def test_public_call_extraction_and_apply_do_not_compute() -> None:
    source = _frame()
    operand = da.ones((3, 24), chunks=(1, 6))

    with patch.object(DaArray, "compute", autospec=True, side_effect=AssertionError("unexpected compute")):
        processed = source + operand
        plan = RecipePlan.from_frame(processed, input_names=("signal", "operand"))
        replayed = RecipePlan.from_dict(plan.to_dict()).apply({"signal": source, "operand": operand})

    assert isinstance(replayed._data, DaArray)


def test_persist_keeps_one_lineage_authority_and_history() -> None:
    processed = _frame().normalize().remove_dc()
    expected = processed.operation_history

    persisted = processed.persist()

    assert persisted.lineage is processed.lineage
    assert persisted.operation_history == expected


def test_operation_history_returns_a_defensive_projection() -> None:
    processed = _frame().normalize(norm=2.0)
    returned = processed.operation_history
    returned[0]["params"]["norm"] = 99.0

    assert processed.operation_history[0]["params"]["norm"] == 2.0


def test_all_index_forms_roundtrip_through_schema_2() -> None:
    selectors: tuple[Callable[[ChannelFrame], ChannelFrame], ...] = (
        lambda frame: frame[0],
        lambda frame: frame["right"],
        lambda frame: frame[[0, 2]],
        lambda frame: frame[np.array([True, False, True])],
        lambda frame: frame[:, 3:9],
    )
    source = _frame()

    for select in selectors:
        expected = select(source)
        plan = RecipePlan.from_frame(expected, input_names=("signal",))
        replayed = RecipePlan.from_dict(plan.to_dict()).apply({"signal": source})
        np.testing.assert_allclose(replayed.compute(), expected.compute())
