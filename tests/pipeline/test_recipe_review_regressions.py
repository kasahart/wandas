from __future__ import annotations

import re
from collections.abc import Callable
from fractions import Fraction
from typing import Any, cast

import dask.array as da
import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipePlan
from wandas.pipeline.calls import MethodCall
from wandas.pipeline.decorators import replay_method
from wandas.pipeline.errors import RecipeExtractionError, RecipeSerializationError
from wandas.processing.base import BinaryOperation, FrameMethodOperation, IndexOperation, LineageNode


class ExternalBrokenFrame(ChannelFrame):
    @replay_method()
    def discard_semantic_lineage(self) -> ExternalBrokenFrame:
        return self._create_new_instance(data=self._data, lineage=self._lineage_or_source())


def _frame(channels: int = 3, samples: int = 256) -> ChannelFrame:
    data = np.arange(channels * samples, dtype=float).reshape(channels, samples) + 1.0
    return ChannelFrame.from_numpy(data, sampling_rate=8000, label="source")


def _custom_callback(data: Any, *, callback: Callable[[Any], Any]) -> Any:
    return callback(data)


def _roundtrip_replay(source: ChannelFrame, processed: Any) -> Any:
    plan = RecipePlan.from_dict(RecipePlan.from_frame(processed, input_names=("signal",)).to_dict())
    replayed = plan.apply({"signal": source})
    np.testing.assert_allclose(replayed.compute(), processed.compute())
    assert replayed.labels == processed.labels
    return replayed


def test_boolean_mask_get_channel_replays_as_public_channel_idx() -> None:
    source = _frame()
    processed = source.get_channel(np.array([True, False, True]))

    _roundtrip_replay(source, processed)


def test_non_literal_get_channel_query_is_an_extraction_boundary() -> None:
    source = _frame()
    processed = source.get_channel(query=re.compile("ch1"))

    with pytest.raises(RecipeExtractionError, match="not portable"):
        RecipePlan.from_frame(processed)


@pytest.mark.parametrize(
    "query",
    [lambda channel: channel.label == "ch1", {"label": re.compile("ch1")}],
)
def test_runtime_search_queries_remain_valid_but_nonportable(query: Any) -> None:
    processed = _frame().get_channel(query=query)
    assert processed.labels == ["ch1"]
    with pytest.raises(RecipeExtractionError, match="not portable"):
        RecipePlan.from_frame(processed)


def test_literal_channel_query_is_re_evaluated_on_recipe_input() -> None:
    source = _frame()
    plan = RecipePlan.from_frame(source.get_channel(query="ch1"), input_names=("signal",))
    reordered = source[[1, 0, 2]]

    assert plan.apply({"signal": reordered}).labels == ["ch1"]


def test_query_recipe_drops_ignored_channel_index_argument() -> None:
    source = _frame()
    processed = source.get_channel(channel_idx=np.array([0]), query="ch1")

    _roundtrip_replay(source, processed)


def test_method_runtime_params_are_snapshotted_for_history_and_graph() -> None:
    source = _frame(channels=1)
    params = {"nested": {"value": 1}}
    lineage = LineageNode(FrameMethodOperation("probe", params), (source._lineage_or_source(),))
    processed = source._create_new_instance(data=source._data, lineage=lineage)
    params["nested"]["value"] = 9
    returned_nested_params = lineage.operation.method_params["nested"]
    returned_nested_params["value"] = 7

    assert lineage.operation.to_params() == {"nested": {"value": 1}}
    assert processed.operation_history[-1]["params"] == {"nested": {"value": 1}}
    assert processed.operation_graph is not None
    assert processed.operation_graph["params"] == {"nested": {"value": 1}}


def test_public_method_runtime_params_cannot_diverge_from_summary_or_recipe() -> None:
    source = _frame()
    processed = source.get_channel([0])
    assert processed.lineage is not None
    operation = processed.lineage.operation
    assert isinstance(operation, FrameMethodOperation)
    history = processed.operation_history
    summaries = processed.operation_summaries
    plan_payload = RecipePlan.from_frame(processed).to_dict()

    returned_channel_indices = operation.method_params["channel_idx"]
    returned_channel_indices.append(1)

    assert operation.method_params["channel_idx"] == [0]
    assert processed.operation_history == history
    assert processed.operation_summaries == summaries
    assert RecipePlan.from_frame(processed).to_dict() == plan_payload


def test_index_runtime_params_cannot_diverge_from_summary_or_recipe() -> None:
    source = _frame()
    processed = source[0, 2:8]
    assert processed.lineage is not None
    operation = processed.lineage.operation
    assert isinstance(operation, IndexOperation)
    history = processed.operation_history
    summaries = processed.operation_summaries
    plan_payload = RecipePlan.from_frame(processed).to_dict()

    returned_channel_selector = operation.params["channel"]
    returned_channel_selector["index"] = 2

    assert operation.params["channel"]["index"] == 0
    assert processed.operation_history == history
    assert processed.operation_summaries == summaries
    assert RecipePlan.from_frame(processed).to_dict() == plan_payload


def test_external_base_frame_subclass_cannot_discard_semantic_lineage() -> None:
    source = cast(ExternalBrokenFrame, ExternalBrokenFrame.from_numpy(np.ones((1, 8)), sampling_rate=8000))

    with pytest.raises(RuntimeError, match="did not preserve semantic lineage"):
        source.discard_semantic_lineage()


def test_array_operand_mutation_cannot_change_history_summary_or_recipe() -> None:
    source = _frame(channels=1)
    operand = np.arange(source.n_samples, dtype=float).reshape(1, -1)
    processed = source + operand
    history = processed.operation_history
    summaries = processed.operation_summaries
    plan_payload = RecipePlan.from_frame(processed, input_names=("signal", "operand")).to_dict()

    operand[:] = -1

    assert processed.operation_history == history
    assert processed.operation_summaries == summaries
    assert RecipePlan.from_frame(processed, input_names=("signal", "operand")).to_dict() == plan_payload
    assert history[-1]["params"]["operand"]["type"] == "array"


def test_non_source_operation_requires_its_frame_parent_lineage() -> None:
    source = _frame(channels=1)
    normalized = source.normalize()
    assert normalized.lineage is not None
    broken_lineage = LineageNode(normalized.lineage.operation, ())
    broken = normalized._create_new_instance(data=normalized._data, lineage=broken_lineage)

    with pytest.raises(RecipeExtractionError, match="frame bindings and lineage inputs disagree"):
        RecipePlan.from_frame(broken)


def test_add_channel_offset_mutation_cannot_change_history_summary_or_recipe() -> None:
    source = _frame(channels=1)
    added = np.ones((1, source.n_samples))
    source_time_offset = np.array([1.25])
    processed = source.add_channel(added, label="added", source_time_offset=source_time_offset)
    history = processed.operation_history
    summaries = processed.operation_summaries
    plan_payload = RecipePlan.from_frame(processed, input_names=("signal", "added")).to_dict()

    source_time_offset[:] = 9.0

    assert processed.operation_history == history
    assert processed.operation_summaries == summaries
    assert RecipePlan.from_frame(processed, input_names=("signal", "added")).to_dict() == plan_payload
    np.testing.assert_allclose(processed.source_time_offset, [0.0, 1.25])
    assert "input_kind" not in history[-1]["params"]


def test_add_channel_offset_keeps_canonical_list_payload() -> None:
    source = _frame(channels=1)
    processed = source.add_channel(
        np.ones((1, source.n_samples)),
        source_time_offset=np.array([1.25]),
    )

    call = RecipePlan.from_frame(processed).to_dict()["nodes"][0]["call"]
    params = dict(call["params"][1])

    assert params["source_time_offset"] == ["list", [["float", 1.25]]]


def test_binary_runtime_descriptor_is_deeply_read_only() -> None:
    processed = _frame(channels=1) + np.ones((1, 256))
    assert processed.lineage is not None
    operation = processed.lineage.operation
    history = processed.operation_history

    with pytest.raises(TypeError):
        operation.operand["shape"][0] = 99

    assert processed.operation_history == history


def test_add_channel_runtime_params_expose_only_defensive_values() -> None:
    source = _frame(channels=1)
    processed = source.add_channel(
        np.ones((1, source.n_samples)),
        source_time_offset=np.array([1.25]),
    )
    assert processed.lineage is not None
    operation = processed.lineage.operation
    history = processed.operation_history

    returned_offset = operation.params["source_time_offset"]
    returned_offset[0] = 9.0

    assert processed.operation_history == history
    assert operation.params["source_time_offset"] == [1.25]


def test_direct_to_channel_frame_and_istft_keep_distinct_public_identity() -> None:
    source = _frame(channels=1)
    spectrogram = source.stft(n_fft=64, hop_length=16, win_length=64)
    direct = spectrogram.to_channel_frame()
    alias = spectrogram.istft()

    assert direct.operation_history[-1]["operation"] == "to_channel_frame"
    assert alias.operation_history[-1]["operation"] == "istft"
    direct_call = RecipePlan.from_frame(direct).nodes[-1].call
    alias_call = RecipePlan.from_frame(alias).nodes[-1].call
    assert isinstance(direct_call, MethodCall) and direct_call.operation == "to_channel_frame"
    assert isinstance(alias_call, MethodCall) and alias_call.operation == "istft"


def test_fixed_call_loaders_reject_tampered_operation() -> None:
    source = _frame()
    plans = [
        RecipePlan.from_frame(source[0]).to_dict(),
        RecipePlan.from_frame(source.add_channel(np.ones((1, source.n_samples)))).to_dict(),
    ]
    for payload in plans:
        payload["nodes"][0]["call"]["operation"] = "unknown"
        with pytest.raises(RecipeSerializationError, match="operation must be"):
            RecipePlan.from_dict(payload)


def test_integer_key_rename_mapping_survives_serialization() -> None:
    source = _frame()
    processed = source.rename_channels({0: "left", 2: "right"})

    _roundtrip_replay(source, processed)


def test_inplace_public_intent_replays_without_mutating_recipe_input() -> None:
    captured = _frame()
    processed = captured.rename_channels({0: "left"}, inplace=True)
    plan = RecipePlan.from_dict(RecipePlan.from_frame(processed, input_names=("signal",)).to_dict())
    runtime_input = _frame()

    replayed = plan.apply({"signal": runtime_input})

    assert replayed is not runtime_input
    assert replayed.labels[0] == "left"
    assert runtime_input.labels[0] == "ch0"
    assert processed.operation_history[-1]["params"]["inplace"] is True


def test_scalar_branches_share_the_source_recipe_input() -> None:
    source = _frame()
    processed = (source + 1) + (source + 2)
    plan = RecipePlan.from_frame(processed, input_names=("signal",))

    assert len(plan.inputs) == 1
    np.testing.assert_allclose(plan.apply({"signal": source}).compute(), processed.compute())


def test_integer_scalar_replay_preserves_labels_and_operand_type() -> None:
    source = _frame()
    processed = source + 1
    plan = RecipePlan.from_frame(processed)
    payload = plan.to_dict()

    assert payload["nodes"][0]["call"]["operand"] == 1
    assert type(payload["nodes"][0]["call"]["operand"]) is int
    assert plan.apply({"input_0": source}).labels == processed.labels


def test_generic_real_scalar_roundtrips_from_canonical_operand_params() -> None:
    source = _frame(channels=1)
    processed = source + cast(Any, Fraction(1, 2))
    plan = RecipePlan.from_dict(RecipePlan.from_frame(processed).to_dict())

    assert plan.to_dict()["nodes"][0]["call"]["operand"] == 0.5
    expected = np.asarray(processed.compute(), dtype=float)
    np.testing.assert_allclose(plan.apply({"input_0": source}).compute(), expected)


@pytest.mark.parametrize("operand", [np.inf, -np.inf, np.nan])
def test_nonfinite_scalar_operands_roundtrip(operand: float) -> None:
    source = _frame(channels=1)
    processed = source + operand
    plan = RecipePlan.from_dict(RecipePlan.from_frame(processed).to_dict())

    np.testing.assert_allclose(
        plan.apply({"input_0": source}).compute(),
        processed.compute(),
        equal_nan=True,
    )


@pytest.mark.parametrize("method", ["sum", "mean"])
def test_channel_reductions_emit_replayable_public_method_lineage(method: str) -> None:
    source = _frame()

    _roundtrip_replay(source, getattr(source, method)())


def test_raw_snr_noise_remains_an_external_array_input() -> None:
    source = _frame(channels=1)
    noise = np.linspace(0.25, 1.25, source.n_samples).reshape(1, -1)
    processed = source.add(noise, snr=6.0)
    plan = RecipePlan.from_dict(RecipePlan.from_frame(processed, input_names=("signal", "noise")).to_dict())

    assert [item.kind for item in plan.inputs] == ["frame", "array"]
    replayed = plan.apply({"signal": source, "noise": noise})
    np.testing.assert_allclose(replayed.compute(), processed.compute())


def test_raw_add_without_snr_remains_an_external_array_input() -> None:
    source = _frame(channels=1)
    operand = np.linspace(0.25, 1.25, source.n_samples // 2).reshape(1, -1)
    processed = source.add(operand)
    plan = RecipePlan.from_frame(processed, input_names=("signal", "operand"))

    assert [item.kind for item in plan.inputs] == ["frame", "array"]
    replayed = plan.apply({"signal": source, "operand": operand})
    np.testing.assert_allclose(replayed.compute(), processed.compute())


@pytest.mark.parametrize("use_dask_operand", [False, True], ids=["numpy", "dask"])
def test_raw_add_uses_one_operation_for_lineage_graph_and_recipe(
    use_dask_operand: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _frame(channels=1)
    source.metadata["contract"] = "preserved"
    source.source_time_offset = [0.25]
    raw_operand = np.linspace(0.25, 1.25, source.n_samples // 2).reshape(1, -1)
    operand = da.from_array(raw_operand, chunks=(1, 32)) if use_dask_operand else raw_operand

    def fail_compute(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("raw add graph construction must stay lazy")

    monkeypatch.setattr(da.Array, "compute", fail_compute)
    processed = source.add(operand)
    assert processed.lineage is not None
    assert isinstance(processed.lineage.operation, BinaryOperation)
    graph_operations = processed.operations
    assert len(graph_operations) == 1
    assert graph_operations[0] is processed.lineage.operation
    assert len(processed.operation_history) == 1
    plan = RecipePlan.from_dict(RecipePlan.from_frame(processed, input_names=("signal", "operand")).to_dict())
    replayed = plan.apply({"signal": source, "operand": operand})
    monkeypatch.undo()

    expected_operand = np.pad(raw_operand, ((0, 0), (0, source.n_samples - raw_operand.shape[1])))
    np.testing.assert_allclose(processed.compute(), source.compute() + expected_operand)
    np.testing.assert_allclose(replayed.compute(), processed.compute())
    assert [item.kind for item in plan.inputs] == ["frame", "array"]
    assert processed.label == "(source + array_data)"
    assert processed.metadata == source.metadata
    np.testing.assert_allclose(processed.source_time_offset, source.source_time_offset)


def test_raw_add_recipe_accepts_dask_operand_without_eager_compute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _frame(channels=1)
    operand = np.linspace(0.25, 1.25, source.n_samples // 2).reshape(1, -1)
    dask_operand = da.from_array(operand, chunks=(1, 32))
    processed = source.add(operand)
    plan = RecipePlan.from_frame(processed, input_names=("signal", "operand"))

    def fail_compute(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Recipe graph construction must stay lazy")

    monkeypatch.setattr(da.Array, "compute", fail_compute)
    replayed = plan.apply({"signal": source, "operand": dask_operand})
    monkeypatch.undo()

    np.testing.assert_allclose(replayed.compute(), processed.compute())


def test_add_channel_accepts_numpy_source_time_offset_metadata() -> None:
    source = _frame(channels=1)
    added = np.ones((1, source.n_samples))
    processed = source.add_channel(added, label="added", source_time_offset=np.array([1.25]))
    plan = RecipePlan.from_dict(RecipePlan.from_frame(processed, input_names=("signal", "added")).to_dict())

    replayed = plan.apply({"signal": source, "added": added})
    np.testing.assert_allclose(replayed.source_time_offset, processed.source_time_offset)


def test_raw_snr_recipe_accepts_dask_noise_without_eager_compute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _frame(channels=1)
    noise = np.linspace(0.25, 1.25, source.n_samples).reshape(1, -1)
    dask_noise = da.from_array(noise, chunks=(1, 64))
    processed = source.add(noise, snr=6.0)
    plan = RecipePlan.from_frame(processed, input_names=("signal", "noise"))

    def fail_compute(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Recipe graph construction must stay lazy")

    monkeypatch.setattr(da.Array, "compute", fail_compute)
    replayed = plan.apply({"signal": source, "noise": dask_noise})
    monkeypatch.undo()

    np.testing.assert_allclose(replayed.compute(), processed.compute())


def test_fix_length_emits_replayable_public_method_lineage() -> None:
    source = _frame(samples=128)

    _roundtrip_replay(source, source.fix_length(length=96))


@pytest.mark.parametrize(
    "build",
    [
        lambda frame: frame.rms_trend(frame_length=32, hop_length=16),
        lambda frame: frame.sound_level(freq_weighting="Z", time_weighting="Fast"),
    ],
)
def test_public_trend_methods_have_stable_replay_targets(
    build: Callable[[ChannelFrame], ChannelFrame],
) -> None:
    source = _frame(channels=1, samples=2048)

    _roundtrip_replay(source, build(source))


def test_integer_channel_difference_emits_method_lineage() -> None:
    source = _frame()

    _roundtrip_replay(source, source.channel_difference(0))


def test_spectrogram_abs_replays_public_label_semantics() -> None:
    source = _frame(channels=1)
    processed = source.stft(n_fft=64, hop_length=16, win_length=64).abs()

    _roundtrip_replay(source, processed)


def test_complex_scalar_roundtrips_without_losing_value_or_labels() -> None:
    source = _frame(channels=1)
    processed = source + (1 + 2j)
    plan = RecipePlan.from_dict(RecipePlan.from_frame(processed).to_dict())
    replayed = plan.apply({"input_0": source})

    np.testing.assert_allclose(replayed.compute(), processed.compute())
    assert replayed.labels == processed.labels


@pytest.mark.parametrize("operand", [True, np.bool_(True)])
def test_boolean_scalar_roundtrips_without_complex_coercion(operand: Any) -> None:
    source = _frame(channels=1)
    processed = source + operand
    plan = RecipePlan.from_dict(RecipePlan.from_frame(processed).to_dict())
    replayed = plan.apply({"input_0": source})

    np.testing.assert_allclose(replayed.compute(), processed.compute())
    assert replayed.labels == processed.labels


def test_integer_index_replaces_loaded_operation_summary() -> None:
    source = ChannelFrame(
        da.from_array(np.ones((2, 16)), chunks=(1, 8)),
        sampling_rate=8000,
        operation_summaries_snapshot=({"operation": "loaded", "params": {}},),
    )

    selected = source[0]

    assert selected.operation_summaries[-1]["operation"] == "__getitem__"


def test_multidimensional_index_replaces_loaded_operation_summary() -> None:
    source = ChannelFrame(
        da.from_array(np.ones((2, 16)), chunks=(1, 8)),
        sampling_rate=8000,
        operation_summaries_snapshot=({"operation": "loaded", "params": {}},),
    )

    selected = source[:, 2:5]

    assert selected.operation_summaries[-1]["operation"] == "__getitem__"
    assert selected.operation_summaries[-1]["params"]["indexing"] == "multidimensional_slice"


def test_raw_add_replacement_summary_uses_external_array_intent() -> None:
    source = ChannelFrame(
        da.from_array(np.ones((1, 16)), chunks=(1, 8)),
        sampling_rate=8000,
        operation_summaries_snapshot=({"operation": "loaded", "params": {}},),
    )

    processed = source.add(np.ones((1, 8)))

    assert processed.operation_summaries[-1]["params"]["operand"]["type"] == "array"


@pytest.mark.parametrize("operation", ["rms_trend", "sound_level"])
def test_array_backed_operations_require_public_method_replay(operation: str) -> None:
    source = _frame(channels=1)
    processed = source.apply_operation(operation, ref=np.array([1.0]))

    with pytest.raises(Exception, match="opted into generic"):
        RecipePlan.from_frame(processed)


def test_nonportable_custom_params_fail_at_extraction_not_runtime() -> None:
    source = _frame(channels=1)
    processed = source.apply(_custom_callback, callback=lambda value: value)
    np.testing.assert_allclose(processed.compute(), source.compute())

    with pytest.raises(RecipeExtractionError, match="not portable"):
        RecipePlan.from_frame(processed)
