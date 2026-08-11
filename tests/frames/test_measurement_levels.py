"""Frame-context measurement level contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import wandas as wd
from wandas.core import ChannelMetadata
from wandas.frames.channel import ChannelFrame


def _measurement() -> ChannelFrame:
    return ChannelFrame.from_numpy(
        np.array(
            [
                [1.0, -1.0, 0.0, 0.0],
                [0.5, -0.5, 0.5, -0.5],
            ]
        ),
        sampling_rate=8,
        ch_labels=["mic", "sensor"],
    ).with_calibration(
        {
            "mic": wd.ChannelCalibration(factor=2e-5, unit="Pa"),
            "sensor": wd.ChannelCalibration(factor=2.0, unit="m/s^2", ref=1.0),
        }
    )


def test_rms_and_peak_stay_linear_while_level_properties_use_channel_context() -> None:
    frame = _measurement()
    history = frame.operation_history

    np.testing.assert_allclose(frame.rms, [np.sqrt(0.5) * 2e-5, 1.0])
    np.testing.assert_allclose(frame.peak, [2e-5, 1.0])
    np.testing.assert_allclose(frame.rms_level, [20.0 * np.log10(np.sqrt(0.5)), 0.0])
    np.testing.assert_allclose(frame.peak_level, [0.0, 0.0])
    assert frame.operation_history == history
    assert [channel.unit for channel in frame.channels] == ["Pa", "m/s^2"]


def test_zero_rms_and_peak_share_the_public_minus_240_db_floor() -> None:
    frame = ChannelFrame.from_numpy(np.zeros((2, 4)), sampling_rate=8).with_calibration(
        [
            wd.ChannelCalibration(unit="Pa"),
            wd.ChannelCalibration(unit="m/s^2", ref=1.0),
        ]
    )

    np.testing.assert_array_equal(frame.rms_level, [-240.0, -240.0])
    np.testing.assert_array_equal(frame.peak_level, [-240.0, -240.0])


def test_fft_and_stft_db_use_the_same_channel_level_reference() -> None:
    frame = _measurement()

    for spectral in (
        frame.fft(n_fft=4, window="boxcar"),
        frame.stft(n_fft=4, hop_length=2, window="boxcar"),
    ):
        expected = np.stack(
            [
                channel.level_reference.to_level(magnitude)
                for channel, magnitude in zip(
                    spectral.channels,
                    spectral.magnitude,
                    strict=True,
                )
            ]
        )
        np.testing.assert_allclose(spectral.dB, expected)


def test_identity_numpy_data_is_not_inferred_to_be_full_scale() -> None:
    frame = wd.from_numpy(np.array([0.5, -0.5]), sampling_rate=8)

    assert frame.channels[0].calibration == wd.ChannelCalibration()
    assert frame.channels[0].level_reference.unit == "dB"
    assert frame.channels[0].level_reference.label == "dB re 1 input unit"


def test_explicit_full_scale_context_flows_to_spectral_and_temporal_levels() -> None:
    frame = wd.from_numpy(
        np.array([1.0, -1.0, 1.0, -1.0]),
        sampling_rate=8,
        ch_units="FS",
    )

    assert frame.rms_level[0] == 0.0
    assert frame.peak_level[0] == 0.0
    assert frame.fft(n_fft=4, window="boxcar").channels[0].level_reference.label == "dBFS"
    assert frame.rms_trend(frame_length=4, hop_length=2, dB=True).channels[0].unit == "dBFS"
    assert frame.sound_level(dB=True).channels[0].unit == "dBFS"


def test_channel_metadata_view_exposes_level_reference_without_new_frame_state() -> None:
    frame = wd.from_numpy(np.ones(4), sampling_rate=8).with_calibration([wd.ChannelCalibration(unit="Pa")])

    assert frame.channels[0].level_reference == ChannelMetadata(unit="Pa").level_reference
    assert "level_reference" not in frame.channels[0].extra


def test_level_reference_is_rederived_from_existing_wdf_unit_and_ref(tmp_path: Path) -> None:
    frame = _measurement()
    path = tmp_path / "measurement.wdf"

    frame.save(path)
    loaded = wd.load(path)

    assert isinstance(loaded, ChannelFrame)
    assert [channel.level_reference for channel in loaded.channels] == [
        channel.level_reference for channel in frame.channels
    ]
    np.testing.assert_allclose(loaded.rms_level, frame.rms_level)
