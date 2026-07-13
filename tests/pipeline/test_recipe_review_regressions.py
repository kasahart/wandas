from __future__ import annotations

from collections.abc import Callable

import dask.array as da
import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipePlan


def _frame(channels: int = 3, samples: int = 256) -> ChannelFrame:
    data = np.arange(channels * samples, dtype=float).reshape(channels, samples) + 1.0
    return ChannelFrame.from_numpy(data, sampling_rate=8000, label="source")


def _roundtrip_replay(source: ChannelFrame, processed: ChannelFrame) -> ChannelFrame:
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
