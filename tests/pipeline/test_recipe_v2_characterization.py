from __future__ import annotations

import inspect
from typing import Any, cast

import dask.array as da
import numpy as np
import pytest

from tests.frame_helpers import channel_first_values
from wandas.frames.channel import ChannelFrame
from wandas.pipeline import RecipePlan, recipe_definition


def _frame(
    value: float = 1.0,
    *,
    channels: int = 2,
    samples: int = 12,
    sampling_rate: int = 8,
) -> ChannelFrame:
    labels = [f"channel-{index}" for index in range(channels)]
    frame = ChannelFrame.from_numpy(
        np.full((channels, samples), value),
        sampling_rate=sampling_rate,
        ch_labels=labels,
    )
    frame.metadata["owner"] = "left"
    frame.source_time_offset = np.arange(channels, dtype=float) + 0.25
    return frame


def test_mix_ignores_source_time_offsets_and_preserves_left_contract() -> None:
    left = _frame(1.0)
    right = _frame(2.0)
    right.source_time_offset = np.array([100.0, -50.0])
    right.metadata["owner"] = "right"

    mixed = left.mix(right)

    np.testing.assert_allclose(channel_first_values(mixed), 3.0)
    np.testing.assert_allclose(mixed.source_time_offset, left.source_time_offset)
    assert mixed.metadata == left.metadata
    assert mixed.labels == left.labels


def test_mix_mono_other_broadcasts_across_left_channels() -> None:
    mixed = _frame(1.0).mix(_frame(2.0, channels=1))

    np.testing.assert_allclose(channel_first_values(mixed), 3.0)
    assert mixed.n_channels == 2


@pytest.mark.parametrize(
    ("align", "other_samples", "expected_tail"),
    [
        ("pad", 8, 1.0),
        ("truncate", 16, 3.0),
    ],
)
def test_mix_alignment_is_directional(align: str, other_samples: int, expected_tail: float) -> None:
    mixed = _frame(1.0).mix(_frame(2.0, samples=other_samples), align=align)

    np.testing.assert_allclose(channel_first_values(mixed)[:, -1], expected_tail)


@pytest.mark.parametrize(
    ("align", "other_samples", "match"),
    [
        ("strict", 8, "equal lengths"),
        ("pad", 12, "shorter"),
        ("pad", 16, "shorter"),
        ("truncate", 12, "longer"),
        ("truncate", 8, "longer"),
    ],
)
def test_mix_rejects_opposite_or_ambiguous_alignment_direction(align: str, other_samples: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _frame().mix(_frame(samples=other_samples), align=align)


def test_mix_rejects_scalar_and_sampling_rate_mismatch() -> None:
    with pytest.raises(TypeError, match="ChannelFrame, NumPy array, or Dask array"):
        _frame().mix(cast(Any, 1.0))
    with pytest.raises(ValueError, match="Sampling rate mismatch"):
        _frame().mix(_frame(sampling_rate=16))


def test_mix_array_input_roundtrips_as_external_array() -> None:
    source = _frame()
    other = da.full((1, 12), 2.0, chunks=(1, 4))
    processed = source.mix(other)
    plan = RecipePlan.from_frame(processed, input_names=("base", "other"))
    replayed = plan.apply({"base": source, "other": other})

    assert [item["kind"] for item in plan.to_dict()["inputs"]] == ["frame", "array"]
    np.testing.assert_allclose(channel_first_values(replayed), 3.0)


def test_mix_with_silent_noise_is_finite_and_leaves_signal_unchanged() -> None:
    source = _frame(3.0)
    silent_noise = da.zeros((1, 12), chunks=(1, 4))
    processed = source.mix(silent_noise, snr_db=6.0)
    plan = RecipePlan.from_frame(processed, input_names=("signal", "noise"))
    replayed = RecipePlan.from_dict(plan.to_dict()).apply({"signal": source, "noise": silent_noise})

    result = channel_first_values(replayed)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, channel_first_values(source))


def test_binary_frame_operation_requires_exact_rate_shape_and_semantic_axes() -> None:
    left = _frame()

    with pytest.raises(ValueError, match="Sampling rate mismatch"):
        _ = left + _frame(sampling_rate=16)
    with pytest.raises(ValueError, match="Frame shape mismatch"):
        _ = left + _frame(samples=8)
    with pytest.raises(ValueError, match="Channel count mismatch"):
        _ = left + _frame(channels=1)


def test_add_channel_raw_input_accepts_only_one_channel() -> None:
    base = _frame(channels=1)

    assert base.add_channel(np.ones(12)).n_channels == 2
    assert base.add_channel(np.ones((1, 12))).n_channels == 2
    with pytest.raises(ValueError, match="Raw add_channel input"):
        base.add_channel(np.ones((2, 12)))


def test_add_channel_frame_input_accepts_multiple_channels() -> None:
    base = _frame(channels=1)
    other = _frame(channels=2)

    added = base.add_channel(other, label="other")

    assert added.n_channels == 3


def test_add_channel_declares_its_data_role() -> None:
    definition = recipe_definition(ChannelFrame.add_channel)

    assert [[binding.role for binding in pattern] for pattern in definition.binding_patterns] == [
        ["base", "data"],
        ["base", "data"],
    ]


def test_removed_add_and_inplace_entrypoints_are_absent() -> None:
    frame = _frame()

    assert not hasattr(frame, "add")
    for method_name in ("add_channel", "remove_channel", "rename_channels"):
        assert "inplace" not in inspect.signature(getattr(frame, method_name)).parameters
