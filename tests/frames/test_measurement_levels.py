"""Frame-context measurement level contracts."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import dask.array as da
import numpy as np
import pytest

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


@pytest.mark.parametrize("amplitude", [1e300, 1e-200])
def test_rms_and_rms_level_are_range_safe_for_dask_backed_extremes(amplitude: float) -> None:
    frame = ChannelFrame(
        da.from_array(np.full((1, 5), amplitude), chunks=(1, 2)),
        sampling_rate=8,
        channel_metadata=[
            ChannelMetadata(
                label="extreme",
                calibration=wd.ChannelCalibration(unit="V", ref=amplitude),
            )
        ],
    )

    np.testing.assert_array_equal(frame.rms, [amplitude])
    np.testing.assert_array_equal(frame.rms_level, [0.0])


@pytest.mark.parametrize("property_name", ["rms", "rms_level", "peak", "peak_level", "crest_factor"])
def test_scalar_reductions_reject_channels_without_samples(property_name: str) -> None:
    frame = ChannelFrame(
        da.from_array(np.empty((2, 0)), chunks=(1, 0)),
        sampling_rate=8,
    )

    with pytest.raises(ValueError, match="at least one sample per channel"):
        getattr(frame, property_name)


def test_frame_scalar_properties_delegate_to_processing_reductions() -> None:
    frame = wd.from_numpy(np.ones((2, 4)), sampling_rate=8)
    results = {
        "_channel_rms": np.array([1.0, 2.0]),
        "_channel_peak": np.array([3.0, 4.0]),
        "_channel_crest_factor": np.array([5.0, 6.0]),
    }

    for helper_name, expected in results.items():
        with mock.patch(
            f"wandas.processing.stats.{helper_name}",
            return_value=da.from_array(expected, chunks=1),
        ) as helper:
            property_name = {
                "_channel_rms": "rms",
                "_channel_peak": "peak",
                "_channel_crest_factor": "crest_factor",
            }[helper_name]

            np.testing.assert_array_equal(getattr(frame, property_name), expected)
            helper.assert_called_once()
            assert helper.call_args.args[0].shape == frame.shape


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


def test_empty_unit_wdf_remains_generic_until_full_scale_is_explicit(tmp_path: Path) -> None:
    frame = wd.from_numpy(np.array([0.5, -0.5]), sampling_rate=8)
    path = tmp_path / "legacy-empty-unit.wdf"

    frame.save(path)
    loaded = wd.load(path)

    assert isinstance(loaded, ChannelFrame)
    assert loaded.channels[0].calibration == wd.ChannelCalibration()
    assert loaded.channels[0].level_reference.label == "dB re 1 input unit"


@pytest.mark.parametrize("operation", ["rms_trend", "sound_level"])
def test_level_frame_wdf_roundtrip_preserves_exact_source_reference(
    operation: str,
    tmp_path: Path,
) -> None:
    reference = 0.12345678901234566
    source = wd.from_numpy(np.ones(8), sampling_rate=8).with_calibration(
        [wd.ChannelCalibration(unit="V", ref=reference)]
    )
    if operation == "rms_trend":
        level = source.rms_trend(frame_length=4, hop_length=2, dB=True)
    else:
        level = source.sound_level(freq_weighting="Z", time_weighting="Fast", dB=True)
    path = tmp_path / f"{operation}.wdf"

    level.save(path)
    loaded = wd.load(path)

    expected_unit = f"dB re {reference!r} V"
    assert loaded.channels[0].unit == expected_unit
    assert float(loaded.channels[0].unit.removeprefix("dB re ").split(" ", 1)[0]) == reference
    with pytest.raises(ValueError, match="already-level channel"):
        _ = loaded.channels[0].level_reference


@pytest.mark.parametrize("operation", ["rms_trend", "sound_level"])
def test_already_level_channel_rejects_a_second_level_reference(operation: str) -> None:
    source = wd.from_numpy(np.ones(8), sampling_rate=8).with_calibration([wd.ChannelCalibration(unit="Pa")])
    if operation == "rms_trend":
        level = source.rms_trend(frame_length=4, hop_length=2, dB=True)
    else:
        level = source.sound_level(freq_weighting="Z", time_weighting="Fast", dB=True)

    assert level.channels[0].unit == "dB SPL re 2e-05 Pa"
    with pytest.raises(ValueError, match="retain the linear source Frame"):
        _ = level.channels[0].level_reference
