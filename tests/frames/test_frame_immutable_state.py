from collections.abc import Callable
from typing import Any, cast

import dask.array as da
import numpy as np
import pytest

from wandas.core.base_frame import BaseFrame
from wandas.core.metadata import ChannelCalibration
from wandas.frames.channel import ChannelFrame
from wandas.frames.noct import NOctFrame
from wandas.frames.roughness import RoughnessFrame


def _frame() -> ChannelFrame:
    return ChannelFrame.from_numpy(
        np.arange(16.0).reshape(2, 8),
        sampling_rate=8,
        label="source",
        metadata={"nested": {"items": [1]}},
        ch_labels=["left", "right"],
    )


def _frame_family_factories() -> list[tuple[str, Callable[[], BaseFrame[Any]]]]:
    return [
        ("channel", _frame),
        ("spectral", lambda: _frame().fft(n_fft=8)),
        ("spectrogram", lambda: _frame().stft(n_fft=8, hop_length=2)),
        ("cepstral", lambda: _frame().cepstrum(n_fft=8)),
        ("cepstrogram", lambda: _frame().stft(n_fft=8, hop_length=2).cepstrum()),
        (
            "noct",
            lambda: NOctFrame(
                da.arange(8.0, chunks=8).reshape(2, 4),
                sampling_rate=8,
                fmin=1,
                fmax=4,
                channel_metadata=[{"label": "left"}, {"label": "right"}],
            ),
        ),
        (
            "roughness",
            lambda: RoughnessFrame(
                da.arange(282.0, chunks=282).reshape(2, 47, 3),
                sampling_rate=8,
                bark_axis=np.linspace(0.5, 23.5, 47),
                overlap=0.5,
                channel_metadata=[{"label": "left"}, {"label": "right"}],
            ),
        ),
    ]


@pytest.mark.parametrize(
    "frame_factory",
    [pytest.param(factory, id=name) for name, factory in _frame_family_factories()],
)
def test_annotation_updates_preserve_every_frame_family(
    frame_factory: Callable[[], BaseFrame[Any]],
) -> None:
    frame = frame_factory()
    state = frame._get_additional_init_kwargs()
    updated = (
        frame.with_label("annotated")
        .with_metadata({"new": {"items": [2]}})
        .with_channel_extra(0, {"sensor": {"serial": "42"}})
    )

    assert type(updated) is type(frame)
    assert updated is not frame
    assert updated._data is frame._data
    assert isinstance(updated._data, da.Array)
    assert updated._channel_ids == frame._channel_ids
    assert updated.labels == frame.labels
    assert updated.lineage is frame.lineage
    assert updated.operation_history == frame.operation_history
    np.testing.assert_array_equal(updated.source_time_offset, frame.source_time_offset)
    assert updated.metadata == {**frame.metadata, "new": {"items": [2]}}
    assert updated.channels[0].extra == {"sensor": {"serial": "42"}}
    for name, expected in state.items():
        actual = updated._get_additional_init_kwargs()[name]
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual == expected


def test_annotation_updates_copy_caller_values_once_at_reconstruction_boundary() -> None:
    frame = _frame()
    metadata = {"new": {"items": [2]}}
    extra = {"sensor": {"serials": [3]}}

    updated = frame.with_metadata(metadata).with_channel_extra("left", extra)
    metadata["new"]["items"].append(99)
    extra["sensor"]["serials"].append(99)

    assert updated.metadata["new"]["items"] == [2]
    assert updated.channels[0].extra["sensor"]["serials"] == [3]
    assert frame.metadata == {"nested": {"items": [1]}}
    assert frame.channels[0].extra == {}


def test_public_mutable_paths_are_read_only_and_nested_snapshots_are_detached() -> None:
    frame = _frame().with_channel_extra(0, {"nested": {"items": [1]}})
    metadata = frame.metadata
    extra = frame.channels[0].extra
    offsets = frame.source_time_offset
    metadata["nested"]["items"].append(2)
    extra["nested"]["items"].append(2)
    offsets[0] = 9

    assert frame.metadata == {"nested": {"items": [1]}}
    assert frame.channels[0].extra == {"nested": {"items": [1]}}
    np.testing.assert_array_equal(frame.source_time_offset, np.zeros(2))
    for attribute, value in [
        ("metadata", {}),
        ("label", "changed"),
        ("sampling_rate", 16),
        ("source_time_offset", [1, 2]),
    ]:
        with pytest.raises(AttributeError):
            setattr(frame, attribute, value)
    for attribute, value in [("label", "changed"), ("extra", {}), ("unit", "Pa"), ("ref", 1.0)]:
        with pytest.raises(AttributeError):
            setattr(frame.channels[0], attribute, value)


def test_channel_name_and_stable_id_lookup_use_distinct_paths() -> None:
    frame = ChannelFrame(
        da.arange(16.0, chunks=16).reshape(2, 8),
        sampling_rate=8,
        channel_ids=["left-id", "collision"],
        channel_metadata=[{"label": "collision"}, {"label": "right"}],
    )

    assert frame.channels["collision"].id == "left-id"
    assert frame.channels.by_id("collision").label == "right"
    updated = frame.with_channel_extra("collision", {"selected": "name"})
    assert updated.channels[0].extra == {"selected": "name"}
    assert updated.channels[1].extra == {}


def test_empty_and_duplicate_stable_ids_are_rejected() -> None:
    data = da.arange(16.0, chunks=16).reshape(2, 8)
    with pytest.raises(ValueError, match="non-empty"):
        ChannelFrame(data, sampling_rate=8, channel_ids=["", "c1"])
    with pytest.raises(ValueError, match="unique"):
        ChannelFrame(data, sampling_rate=8, channel_ids=["same", "same"])


def test_calibration_partial_updates_preserve_factor_and_reset_ref_for_unit() -> None:
    calibration = ChannelCalibration(factor=2, unit="Pa", ref=0.5)

    changed_unit = calibration.with_unit("V")
    custom_ref = changed_unit.with_ref(0.25)

    assert changed_unit == ChannelCalibration(factor=2, unit="V")
    assert custom_ref == ChannelCalibration(factor=2, unit="V", ref=0.25)
    assert calibration.with_factor(3) == ChannelCalibration(factor=3, unit="Pa", ref=0.5)


@pytest.mark.parametrize(
    ("invoke", "error", "message"),
    [
        (lambda frame: frame.with_label(cast(Any, 1)), TypeError, "Frame label"),
        (lambda frame: frame.with_metadata(cast(Any, [])), TypeError, "Frame metadata"),
        (lambda frame: frame.with_channel_extra(cast(Any, True), {}), TypeError, "Channel selector"),
        (lambda frame: frame.with_source_time_offset(cast(Any, "1.0")), TypeError, "finite numeric"),
        (lambda frame: frame.rename_channels({0: cast(Any, 1)}), TypeError, "Channel label"),
    ],
)
def test_immutable_updates_reject_invalid_inputs(
    invoke: Callable[[ChannelFrame], Any],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        invoke(_frame())


def test_rename_channels_is_lazy_and_available_on_derived_frames() -> None:
    frame = _frame().fft(n_fft=8)
    renamed = frame.rename_channels({0: "renamed"})

    assert type(renamed) is type(frame)
    assert renamed._data is frame._data
    assert renamed.labels == ["renamed", "right"]
    np.testing.assert_array_equal(renamed.freqs, frame.freqs)
