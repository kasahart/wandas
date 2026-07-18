"""Reference-signal calibration contracts."""

from collections.abc import Mapping

import dask.array as da
import numpy as np
import pytest

from wandas.core.metadata import ChannelCalibration, ChannelMetadata
from wandas.frames.channel import ChannelFrame


def _frame(data: np.ndarray, labels: list[str], factors: list[float] | None = None) -> ChannelFrame:
    factors = factors or [1.0] * len(labels)
    return ChannelFrame(
        da.from_array(data, chunks=(1, -1)),
        sampling_rate=8_000,
        channel_metadata=[
            ChannelMetadata(label=label, calibration=ChannelCalibration(factor))
            for label, factor in zip(labels, factors, strict=True)
        ],
    )


def test_separate_reference_events_combine_for_measurement() -> None:
    microphone = _frame(np.array([[0.5, -0.5]]), ["microphone"])
    accelerometer = _frame(np.array([[0.25, -0.25]]), ["accelerometer"])
    measurement = _frame(np.array([[1.0, -1.0], [2.0, -2.0]]), ["microphone", "accelerometer"])

    calibrations: Mapping[str | int, float | ChannelCalibration] = {
        **microphone.derive_calibration(target_level=94.0, unit="Pa"),
        **accelerometer.derive_calibration(target_rms=1.0, unit="m/s^2"),
    }
    calibrated = measurement.with_calibration(calibrations)

    microphone_factor = 2e-5 * 10 ** (94.0 / 20.0) / 0.5
    np.testing.assert_allclose(calibrated.data, [[microphone_factor, -microphone_factor], [8.0, -8.0]])


def test_common_target_broadcast_and_many_event_aggregation() -> None:
    reference = _frame(np.array([[1.0, -1.0], [2.0, -2.0]]), ["a", "b"])
    factors = reference.derive_calibration(target_rms=1.0, unit="m/s^2")
    assert factors == {
        "a": ChannelCalibration(1.0, "m/s^2"),
        "b": ChannelCalibration(0.5, "m/s^2"),
    }

    aggregated: dict[str, ChannelCalibration] = {}
    for index in range(100):
        event = _frame(np.array([[index + 1.0, -(index + 1.0)]]), [f"sensor-{index:03d}"])
        aggregated.update(event.derive_calibration(target_rms=1.0, unit="m/s^2"))
    assert len(aggregated) == 100


def test_existing_factor_and_processed_data_produce_absolute_factor() -> None:
    reference = _frame(np.array([[0.5, -0.5]]), ["sensor"], factors=[2.0]).remove_dc()
    original_data = reference.data.copy()
    original_history = reference.operation_history.copy()

    result = reference.derive_calibration(target_rms=4.0, unit="Pa")

    assert result == {reference.labels[0]: ChannelCalibration(4.0, "Pa")}
    np.testing.assert_array_equal(reference.data, original_data)
    assert reference.operation_history == original_history


@pytest.mark.parametrize("value", [True, 1 + 0j, [1.0], np.array([1.0]), np.ma.array(1.0), np.nan, np.inf])
def test_target_is_narrow_finite_real_scalar(value: object) -> None:
    reference = _frame(np.array([[1.0, -1.0]]), ["sensor"])
    with pytest.raises((TypeError, ValueError), match="target_rms"):
        reference.derive_calibration(target_rms=value, unit="Pa")  # ty: ignore[invalid-argument-type]


def test_invalid_rms_labels_and_target_choice_are_rejected() -> None:
    with pytest.raises(ValueError, match="Reference RMS"):
        _frame(np.zeros((1, 2)), ["sensor"]).derive_calibration(target_rms=1.0, unit="Pa")
    with pytest.raises(ValueError, match="unique non-empty"):
        _frame(np.ones((2, 2)), ["sensor", "sensor"]).derive_calibration(target_rms=1.0, unit="Pa")
    with pytest.raises(ValueError, match="unique non-empty"):
        _frame(np.ones((1, 2)), [""]).derive_calibration(target_rms=1.0, unit="Pa")
    with pytest.raises(ValueError, match="Exactly one"):
        _frame(np.ones((1, 2)), ["sensor"]).derive_calibration(unit="Pa")
    with pytest.raises(ValueError, match="Exactly one"):
        _frame(np.ones((1, 2)), ["sensor"]).derive_calibration(target_rms=1.0, target_level=94.0, unit="Pa")
