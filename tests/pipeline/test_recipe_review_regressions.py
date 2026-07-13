from __future__ import annotations

from collections.abc import Callable
from typing import Any

import dask.array as da
import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipePlan


def _frame(channels: int = 3, samples: int = 256) -> ChannelFrame:
    data = np.arange(channels * samples, dtype=float).reshape(channels, samples) + 1.0
    return ChannelFrame.from_numpy(data, sampling_rate=8000, label="source")


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


def test_integer_key_rename_mapping_survives_serialization() -> None:
    source = _frame()
    processed = source.rename_channels({0: "left", 2: "right"})

    _roundtrip_replay(source, processed)


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
