"""Frame-context measurement-level contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import wandas as wd
from wandas.frames.channel import ChannelFrame
from wandas.io.wdf_io import WDF_FORMAT_VERSION


def _measurement() -> ChannelFrame:
    return wd.from_numpy(
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


def test_public_reference_converts_existing_linear_rms_without_new_reduction_api() -> None:
    frame = _measurement()

    rms_levels = np.asarray(
        [channel.level_reference.to_level(rms) for channel, rms in zip(frame.channels, frame.rms, strict=True)]
    )

    np.testing.assert_allclose(rms_levels, [20.0 * np.log10(np.sqrt(0.5)), 0.0])
    assert [channel.unit for channel in frame.channels] == ["Pa", "m/s^2"]


def test_fft_and_stft_db_share_level_reference_numerics_and_public_shapes() -> None:
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
        assert spectral.dB.shape == spectral.magnitude.shape


def test_numpy_identity_data_remains_generic() -> None:
    frame = wd.from_numpy(np.array([0.5, -0.5]), sampling_rate=8)

    assert frame.channels[0].calibration == wd.ChannelCalibration()
    assert frame.channels[0].level_reference.unit == "dB"
    assert frame.channels[0].level_reference.label == "dB re 1 input unit"


def test_linear_calibration_rederives_level_reference_after_wdf_04_roundtrip(
    tmp_path: Path,
) -> None:
    frame = _measurement()
    path = tmp_path / "measurement.wdf"

    frame.save(path)
    loaded = wd.load(path)

    assert WDF_FORMAT_VERSION == "0.4"
    assert isinstance(loaded, ChannelFrame)
    assert [channel.level_reference for channel in loaded.channels] == [
        channel.level_reference for channel in frame.channels
    ]
