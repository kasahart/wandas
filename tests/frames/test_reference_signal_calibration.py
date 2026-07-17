"""Reference-signal calibration derivation and application contracts."""

import dask.array as da
import numpy as np
import pytest

from wandas.core.metadata import ChannelCalibration, ChannelMetadata
from wandas.frames.channel import ChannelFrame


def _frame(data: np.ndarray, labels: list[str], *, units: list[str] | None = None) -> ChannelFrame:
    channel_units = units or ["V"] * len(labels)
    return ChannelFrame(
        data=da.from_array(data, chunks=(1, -1)),
        sampling_rate=8_000,
        channel_metadata=[
            ChannelMetadata(label=label, unit=unit, extra={"serial": f"SN-{index:03d}"})
            for index, (label, unit) in enumerate(zip(labels, channel_units, strict=True))
        ],
        metadata={"asset": "rig-7"},
    )


def test_derive_calibration_returns_label_mapping_without_mutating_signal() -> None:
    calibration_signal = _frame(
        np.array(
            [
                [0.5, -0.5, 0.5, -0.5],
                [0.25, -0.25, 0.25, -0.25],
            ]
        ),
        ["microphone-left", "microphone-right"],
    )
    original_data = calibration_signal.data.copy()
    original_channels = calibration_signal.channels.to_list()

    calibrations = calibration_signal.derive_calibration(target_level=94.0, unit="Pa")

    target_rms = 2e-5 * 10 ** (94.0 / 20.0)
    assert list(calibrations) == ["microphone-left", "microphone-right"]
    np.testing.assert_allclose(
        [calibration.factor for calibration in calibrations.values()],
        [target_rms / 0.5, target_rms / 0.25],
        rtol=1e-12,
        atol=0.0,
    )
    assert [calibration.unit for calibration in calibrations.values()] == ["Pa", "Pa"]
    assert [calibration.ref for calibration in calibrations.values()] == [2e-5, 2e-5]
    np.testing.assert_array_equal(calibration_signal.data, original_data)
    assert calibration_signal.channels.to_list() == original_channels
    assert calibration_signal.operation_history == []


def test_derive_calibration_broadcasts_one_target_to_many_channels() -> None:
    labels = [f"sensor-{index:03d}" for index in range(100)]
    recorded_rms = np.arange(1.0, 101.0)
    calibration_signal = _frame(
        np.stack([recorded_rms, -recorded_rms], axis=1),
        labels,
    )
    measurement = _frame(np.ones((100, 2)), labels)

    calibrations = calibration_signal.derive_calibration(target_rms=1.0, unit="m/s^2")
    calibrated = measurement.with_calibration(calibrations)

    assert len(calibrations) == 100
    np.testing.assert_allclose(calibrated.data[:, 0], 1.0 / recorded_rms)


def test_derive_calibration_normalizes_unit_and_requires_physical_domain() -> None:
    calibration_signal = _frame(np.array([[0.5, -0.5]]), ["microphone"])

    calibrations = calibration_signal.derive_calibration(target_level=94.0, unit=" Pa ")

    assert calibrations["microphone"].unit == "Pa"
    assert calibrations["microphone"].ref == 2e-5
    with pytest.raises(ValueError, match="Invalid calibration unit"):
        calibration_signal.derive_calibration(target_rms=1.0, unit="")


def test_separately_recorded_sound_and_acceleration_calibrations_apply_together() -> None:
    microphone_calibration_signal = _frame(
        np.array([[0.5, -0.5, 0.5, -0.5]]),
        ["microphone"],
    )
    accelerometer_calibration_signal = _frame(
        np.array([[0.25, -0.25, 0.25, -0.25]]),
        ["accelerometer"],
    )
    measurement = _frame(
        np.array(
            [
                [1.0, -1.0],
                [2.0, -2.0],
            ]
        ),
        ["microphone", "accelerometer"],
    )

    calibrations = {
        **microphone_calibration_signal.derive_calibration(target_level=94.0, unit="Pa"),
        **accelerometer_calibration_signal.derive_calibration(target_rms=1.0, unit="m/s^2", ref=1.0),
    }
    calibrated = measurement.with_calibration(calibrations)

    microphone_factor = 2e-5 * 10 ** (94.0 / 20.0) / 0.5
    np.testing.assert_allclose(
        calibrated.data,
        np.array(
            [
                [microphone_factor, -microphone_factor],
                [8.0, -8.0],
            ]
        ),
        rtol=1e-12,
        atol=0.0,
    )
    assert [channel.unit for channel in calibrated.channels] == ["Pa", "m/s^2"]
    np.testing.assert_array_equal(measurement.data, [[1.0, -1.0], [2.0, -2.0]])
    assert calibrated.operation_history[-1]["operation"] == "wandas.channel.with_calibration"


def test_derive_calibration_rejects_ambiguous_or_already_calibrated_signal() -> None:
    duplicate_labels = _frame(np.ones((2, 4)), ["sensor", "sensor"])
    already_calibrated = _frame(np.ones((1, 4)), ["sensor"]).with_calibration([ChannelCalibration(2.0, "Pa")])

    with pytest.raises(ValueError, match="unique non-empty channel labels"):
        duplicate_labels.derive_calibration(target_rms=1.0, unit="Pa")
    with pytest.raises(ValueError, match="already has calibration factors"):
        already_calibrated.derive_calibration(target_rms=1.0, unit="Pa")


def test_derive_calibration_rejects_reference_that_consumed_prior_calibration() -> None:
    raw_reference = _frame(np.array([[0.5, -0.5, 0.5, -0.5]]), ["microphone"])
    consumed_reference = raw_reference.with_calibration([ChannelCalibration(2.0, "Pa")]).remove_dc()

    assert consumed_reference.channels[0].calibration.factor == 1.0
    assert [record["operation"] for record in consumed_reference.operation_history] == [
        "wandas.channel.with_calibration",
        "wandas.audio.remove_dc",
    ]
    with pytest.raises(ValueError, match="unprocessed source Frame"):
        consumed_reference.derive_calibration(target_rms=1.0, unit="Pa")


def test_derive_calibration_rejects_processed_source_with_calibration_metadata() -> None:
    source_reference = ChannelFrame(
        data=da.from_array(np.array([[0.5, -0.5, 0.5, -0.5]]), chunks=(1, -1)),
        sampling_rate=8_000,
        channel_metadata=[
            ChannelMetadata(
                label="microphone",
                calibration=ChannelCalibration(2.0, "Pa"),
            )
        ],
    )
    consumed_reference = source_reference.remove_dc()

    assert consumed_reference.channels[0].calibration.factor == 1.0
    assert [record["operation"] for record in consumed_reference.operation_history] == ["wandas.audio.remove_dc"]
    with pytest.raises(ValueError, match="unprocessed source Frame"):
        consumed_reference.derive_calibration(target_rms=1.0, unit="Pa")
