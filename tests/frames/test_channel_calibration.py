"""Per-channel calibration UX and numerical behavior."""

import json

import dask.array as da
import numpy as np
import pytest

from wandas.core.metadata import ChannelCalibration, ChannelMetadata
from wandas.frames.channel import ChannelFrame, _validate_with_calibration_recipe
from wandas.pipeline import RecipeExecutionError, RecipePlan
from wandas.utils.types import NDArrayReal


def _frame(*, channel_count: int = 2) -> ChannelFrame:
    data = np.arange(1, channel_count * 8 + 1, dtype=float).reshape(channel_count, 8)
    labels = ["microphone", "accelerometer"] if channel_count == 2 else [f"sensor-{i:03d}" for i in range(100)]
    return ChannelFrame(
        data=da.from_array(data, chunks=(1, -1)),
        sampling_rate=8_000,
        label="raw acquisition",
        metadata={"asset": "rig-7"},
        channel_metadata=[
            ChannelMetadata(label=label, extra={"serial": f"SN-{index:03d}"}) for index, label in enumerate(labels)
        ],
        channel_ids=[f"sensor-id-{index:03d}" for index in range(channel_count)],
        source_time_offset=np.arange(channel_count, dtype=float),
    )


def test_channel_calibration_defaults_to_identity() -> None:
    frame = _frame()

    assert [channel.calibration for channel in frame.channels] == [ChannelCalibration(), ChannelCalibration()]
    assert frame.raw_data is frame._data
    np.testing.assert_array_equal(frame.compute(), frame.raw_data.compute())


def test_list_replaces_every_factor_in_current_channel_order() -> None:
    frame = _frame()

    configured = frame.with_calibration([2.0, 0.5])

    np.testing.assert_array_equal(configured.raw_data.compute(), frame.raw_data.compute())
    np.testing.assert_array_equal(
        configured.compute(),
        frame.raw_data.compute() * np.array([[2.0], [0.5]]),
    )
    assert [channel.calibration.factor for channel in configured.channels] == [2.0, 0.5]


def test_numpy_array_replaces_factors_in_current_channel_order() -> None:
    frame = _frame()
    factors = np.array([2.0, 0.5])

    configured = frame.with_calibration(factors)

    assert [channel.calibration.factor for channel in configured.channels] == [2.0, 0.5]
    np.testing.assert_array_equal(factors, np.array([2.0, 0.5]))


@pytest.mark.parametrize("values", [np.array(2.0), np.ones((2, 1))])
def test_numpy_array_requires_one_dimension(values: NDArrayReal) -> None:
    with pytest.raises(ValueError, match="Invalid calibration array shape"):
        _frame().with_calibration(values)


def test_label_and_index_mapping_support_partial_mixed_updates() -> None:
    configured = _frame().with_calibration(
        {
            "microphone": ChannelCalibration(0.02, "Pa"),
            -1: ChannelCalibration(9.81, "m/s^2", 1.0),
        }
    )
    updated = configured.with_calibration({0: 0.04})

    assert updated.channels[0].calibration == ChannelCalibration(0.04, "Pa")
    assert updated.channels[1].calibration == ChannelCalibration(9.81, "m/s^2", 1.0)
    assert configured.channels[0].calibration.factor == 0.02


def test_calibration_replacement_never_compounds_and_preserves_frame_state() -> None:
    frame = _frame()
    first = frame.with_calibration([2.0, 3.0])
    second = first.with_calibration([5.0, 7.0])

    np.testing.assert_array_equal(second.compute(), frame.raw_data.compute() * np.array([[5.0], [7.0]]))
    assert [channel.calibration.factor for channel in first.channels] == [2.0, 3.0]
    assert [record["operation"] for record in second.operation_history] == [
        "wandas.channel.with_calibration",
        "wandas.channel.with_calibration",
    ]
    assert second.metadata == {"asset": "rig-7"}
    assert "calibration" not in second.metadata
    assert second._channel_ids == frame._channel_ids
    np.testing.assert_array_equal(second.source_time_offset, frame.source_time_offset)
    assert [channel.extra for channel in second.channels] == [channel.extra for channel in frame.channels]


def test_selection_and_reordering_keep_calibration_aligned() -> None:
    configured = _frame().with_calibration(
        [
            ChannelCalibration(0.02, "Pa"),
            ChannelCalibration(9.81, "m/s^2"),
        ]
    )

    reordered = configured.get_channel([1, 0])

    assert reordered.labels == ["accelerometer", "microphone"]
    assert [channel.calibration.factor for channel in reordered.channels] == [9.81, 0.02]
    np.testing.assert_array_equal(reordered.compute(), configured.compute()[[1, 0]])


def test_hundred_channel_list_and_mapping_are_practical() -> None:
    frame = _frame(channel_count=100)
    factors = [1.0 + index / 100 for index in range(100)]

    configured = frame.with_calibration(factors)
    updated = configured.with_calibration({f"sensor-{index:03d}": 2.0 for index in range(0, 100, 10)})

    assert configured.n_channels == 100
    assert updated.channels[0].calibration.factor == 2.0
    assert updated.channels[1].calibration.factor == factors[1]
    assert updated.channels[90].calibration.factor == 2.0
    np.testing.assert_array_equal(updated.raw_data.compute(), frame.raw_data.compute())


def test_configuration_and_analysis_stay_lazy_until_a_result_is_requested() -> None:
    configured = _frame().with_calibration([ChannelCalibration(2.0, "Pa"), ChannelCalibration(3.0, "m/s^2")])
    spectrum = configured.fft(n_fft=8, window="boxcar")

    assert isinstance(configured.raw_data, da.Array)
    assert isinstance(configured._effective_data, da.Array)
    assert isinstance(spectrum._data, da.Array)
    assert [channel.calibration.factor for channel in spectrum.channels] == [1.0, 1.0]
    assert [channel.unit for channel in spectrum.channels] == ["Pa", "m/s^2"]


def test_compute_data_rms_fft_stft_and_sound_level_use_effective_data() -> None:
    frame = _frame()
    calibrations = [
        ChannelCalibration(2.0, "Pa"),
        ChannelCalibration(3.0, "Pa"),
    ]
    configured = frame.with_calibration(calibrations)
    physical = ChannelFrame(
        data=configured._effective_data,
        sampling_rate=frame.sampling_rate,
        channel_metadata=[
            ChannelMetadata(label="microphone", unit="Pa"),
            ChannelMetadata(label="accelerometer", unit="Pa"),
        ],
    )

    np.testing.assert_allclose(configured.compute(), physical.compute())
    np.testing.assert_allclose(configured.data, physical.data)
    np.testing.assert_allclose(configured.rms, physical.rms)
    np.testing.assert_allclose(
        configured.fft(n_fft=8, window="boxcar").compute(),
        physical.fft(n_fft=8, window="boxcar").compute(),
    )
    np.testing.assert_allclose(
        configured.stft(n_fft=4, hop_length=2, window="boxcar").compute(),
        physical.stft(n_fft=4, hop_length=2, window="boxcar").compute(),
    )
    np.testing.assert_allclose(
        configured.sound_level(freq_weighting="Z", time_weighting="Fast").compute(),
        physical.sound_level(freq_weighting="Z", time_weighting="Fast").compute(),
    )


def test_existing_derived_frame_does_not_change_after_replacement() -> None:
    frame = _frame()
    first = frame.with_calibration([2.0, 3.0])
    first_result = first * 2.0
    replacement = first.with_calibration([5.0, 7.0])

    np.testing.assert_array_equal(first_result.compute(), frame.raw_data.compute() * np.array([[4.0], [6.0]]))
    np.testing.assert_array_equal(replacement.compute(), frame.raw_data.compute() * np.array([[5.0], [7.0]]))
    assert [channel.calibration.factor for channel in first_result.channels] == [1.0, 1.0]


def test_history_and_recipe_store_stable_ids_and_replay_after_reordering() -> None:
    source = _frame()
    expected = source.get_channel([1, 0]).with_calibration(
        [
            ChannelCalibration(9.81, "m/s^2"),
            ChannelCalibration(0.02, "Pa"),
        ]
    )

    history_params = expected.operation_history[-1]["params"]
    assert set(history_params["calibrations"]) == {"sensor-id-000", "sensor-id-001"}
    plan = RecipePlan.from_frame(expected, input_names=("signal",))
    payload = plan.to_dict()
    json.dumps(payload)
    replayed = RecipePlan.from_dict(payload).apply({"signal": source})

    assert replayed.labels == ["accelerometer", "microphone"]
    assert [channel.calibration.factor for channel in replayed.channels] == [9.81, 0.02]
    np.testing.assert_array_equal(replayed.compute(), expected.compute())


def test_recipe_replay_rejects_frames_without_captured_stable_ids() -> None:
    source = _frame()
    plan = RecipePlan.from_frame(source.with_calibration([2.0, 3.0]), input_names=("signal",))
    incompatible = ChannelFrame.from_numpy(
        source.raw_data.compute(),
        source.sampling_rate,
        ch_labels=source.labels,
    )

    with pytest.raises(RecipeExecutionError, match="Recipe operation failed"):
        plan.apply({"signal": incompatible})


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"calibrations": {}},
        {"calibrations": {"": {"factor": 1.0, "unit": "", "ref": 1.0}}},
    ],
)
def test_calibration_recipe_validator_rejects_malformed_params(params: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="with_calibration Recipe"):
        _validate_with_calibration_recipe(params)


def test_label_mapping_rejects_ambiguous_duplicate_labels() -> None:
    frame = ChannelFrame.from_numpy(
        np.ones((2, 4)),
        8_000,
        ch_labels=["duplicate", "duplicate"],
    )

    with pytest.raises(ValueError, match="Ambiguous calibration channel label"):
        frame.with_calibration({"duplicate": 2.0})


@pytest.mark.parametrize(
    ("values", "error", "message"),
    [
        ([], ValueError, "Calibration list length mismatch"),
        ([1.0], ValueError, "Calibration list length mismatch"),
        ({}, ValueError, "Empty calibration update"),
        ({"missing": 1.0}, KeyError, "Unknown calibration channel label"),
        ({2: 1.0}, IndexError, "Calibration channel index out of range"),
        ({True: 1.0}, TypeError, "Invalid calibration channel reference"),
        ({1.5: 1.0}, TypeError, "Invalid calibration channel reference"),
        ({"microphone": 1.0, 0: 2.0}, ValueError, "Duplicate calibration channel reference"),
        ({"microphone": "bad"}, TypeError, "Invalid channel calibration value"),
        ("bad", TypeError, "Invalid calibration values"),
    ],
)
def test_with_calibration_rejects_invalid_intent(
    values: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _frame().with_calibration(values)  # ty: ignore[invalid-argument-type]


def test_mapping_input_is_not_mutated() -> None:
    values: dict[str | int, float | ChannelCalibration] = {"microphone": 2.0}

    _frame().with_calibration(values)

    assert values == {"microphone": 2.0}
