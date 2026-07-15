from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

from wandas.core.metadata import ChannelMetadata
from wandas.frames.channel import ChannelFrame
from wandas.processing import Calibration


def _frame(data: np.ndarray, *, label: str = "measurement") -> ChannelFrame:
    channel_count = data.shape[0]
    channel_metadata = [
        ChannelMetadata(
            label=("left", "right")[index] if channel_count == 2 else f"ch{index}",
            unit="V",
            ref=1.0,
            extra={"sensor": f"mic-{index}"},
        )
        for index in range(channel_count)
    ]
    return ChannelFrame(
        data=da.from_array(data, chunks=(1, -1)),
        sampling_rate=48_000,
        label=label,
        metadata={"recording": "anechoic-room"},
        channel_metadata=channel_metadata,
        source_time_offset=np.arange(channel_count, dtype=float) * 0.25,
    )


def test_derive_calibration_known_rms_returns_auditable_factors() -> None:
    calibration_signal = _frame(
        np.array(
            [
                [0.5, -0.5, 0.5, -0.5],
                [0.25, -0.25, 0.25, -0.25],
            ]
        ),
        label="calibrator",
    )

    calibration = calibration_signal.derive_calibration(
        target_rms=(1.0, 2.0),
        unit="Pa",
    )

    assert isinstance(calibration, Calibration)
    assert calibration.measured_rms == (0.5, 0.25)
    assert calibration.target_rms == (1.0, 2.0)
    assert calibration.factors == (2.0, 8.0)
    assert calibration.ref == pytest.approx(20e-6)
    assert calibration_signal.operation_history == []


def test_derive_calibration_sound_pressure_level_uses_channel_reference() -> None:
    calibration_signal = _frame(np.full((1, 16), 0.5), label="94 dB calibrator")

    calibration = calibration_signal.derive_calibration(
        target_level=94.0,
        unit="Pa",
    )

    expected_rms = 20e-6 * 10 ** (94.0 / 20.0)
    np.testing.assert_allclose(calibration.target_rms, (expected_rms,), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(calibration.factors, (expected_rms / 0.5,), rtol=1e-12, atol=0.0)


def test_sound_pressure_calibration_recovers_known_spl_end_to_end() -> None:
    sampling_rate = 48_000
    time = np.arange(sampling_rate, dtype=float) / sampling_rate
    raw_tone = 0.5 * np.sqrt(2.0) * np.sin(2.0 * np.pi * 1_000.0 * time)
    calibration_signal = _frame(raw_tone.reshape(1, -1), label="94 dB calibrator")

    calibration = calibration_signal.derive_calibration(target_level=94.0, unit="Pa")
    pressure = calibration_signal.calibrate(calibration)
    recovered_level = 20.0 * np.log10(pressure.rms / pressure.channels[0].ref)
    meter_level = pressure.sound_level(freq_weighting="Z", time_weighting="Fast", dB=True).compute()

    # Integer-cycle sine RMS and the amplitude-level formula are analytical.
    np.testing.assert_allclose(recovered_level, [94.0], rtol=0.0, atol=1e-12)
    # One second is eight Fast time constants; allow the remaining exponential settling error.
    np.testing.assert_allclose(meter_level[0, -1000:].mean(), 94.0, rtol=0.0, atol=2e-3)
    assert pressure.channels[0].unit == "Pa"


def test_calibrate_is_lazy_immutable_and_updates_physical_metadata_atomically() -> None:
    calibration = Calibration.from_rms(
        (0.5, 0.25),
        target_rms=(1.0, 2.0),
        unit="Pa",
    )
    signal = _frame(np.array([[1.0, -1.0], [1.0, -1.0]]))
    original_data = signal.compute().copy()
    original_metadata = signal.metadata.copy()
    original_channels = signal.channels.to_list()
    original_offsets = signal.source_time_offset.copy()
    original_lineage = signal.lineage

    with patch.object(DaArray, "compute", autospec=True, side_effect=AssertionError("unexpected compute")):
        calibrated = signal.calibrate(calibration)

    assert calibrated is not signal
    assert isinstance(calibrated, ChannelFrame)
    assert isinstance(calibrated._data, DaArray)
    assert calibrated.sampling_rate == signal.sampling_rate
    assert calibrated.metadata == original_metadata
    assert calibrated.metadata is not signal.metadata
    np.testing.assert_array_equal(calibrated.source_time_offset, original_offsets)
    assert calibrated.labels == ["cal(left)", "cal(right)"]
    assert [channel.unit for channel in calibrated.channels] == ["Pa", "Pa"]
    assert [channel.ref for channel in calibrated.channels] == pytest.approx([20e-6, 20e-6])
    assert [channel.extra for channel in calibrated.channels] == [
        {"sensor": "mic-0"},
        {"sensor": "mic-1"},
    ]
    np.testing.assert_allclose(calibrated.compute(), [[2.0, -2.0], [8.0, -8.0]], rtol=0.0, atol=0.0)

    np.testing.assert_array_equal(signal.compute(), original_data)
    assert signal.metadata == original_metadata
    assert signal.channels.to_list() == original_channels
    np.testing.assert_array_equal(signal.source_time_offset, original_offsets)
    assert signal.lineage is original_lineage
    assert len(calibrated.operation_history) == len(signal.operation_history) + 1
    assert calibrated.operation_history[-1] == {
        "operation": "wandas.audio.calibrate",
        "version": 1,
        "params": {
            "measured_rms": [0.5, 0.25],
            "ref": 20e-6,
            "target_rms": [1.0, 2.0],
            "unit": "Pa",
        },
    }


def test_calibrate_single_channel_calibration_broadcasts_and_chains() -> None:
    calibration = Calibration.from_rms((0.5,), target_rms=1.0, unit="Pa")
    signal = _frame(np.array([[1.0, -1.0], [2.0, -2.0]]))

    result = signal.calibrate(calibration).remove_dc()

    assert isinstance(result, ChannelFrame)
    assert isinstance(result._data, DaArray)
    np.testing.assert_allclose(result.compute(), [[2.0, -2.0], [4.0, -4.0]], rtol=0.0, atol=0.0)
    assert [entry["operation"] for entry in result.operation_history] == [
        "wandas.audio.calibrate",
        "wandas.audio.remove_dc",
    ]


def test_calibrate_channel_mismatch_raises_error_before_graph_creation() -> None:
    calibration = Calibration.from_rms(
        (0.5, 0.25, 0.125),
        target_rms=1.0,
        unit="Pa",
    )
    signal = _frame(np.ones((2, 8)))

    with pytest.raises(ValueError, match="Calibration channel mismatch"):
        signal.calibrate(calibration)


def test_calibrate_non_calibration_value_raises_error() -> None:
    signal = _frame(np.ones((1, 8)))

    with pytest.raises(TypeError, match="calibration must be a Calibration"):
        signal.calibrate(2.0)  # ty: ignore[invalid-argument-type]
