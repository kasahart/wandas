"""Channel level kind: amplitude (20 log10), power (10 log10) or none (no dB)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import dask.array as da
import numpy as np
import pytest
import xarray as xr

import wandas as wd
from wandas.core.metadata import ChannelCalibration, ChannelMetadata, LevelReference
from wandas.frames.channel import ChannelFrame

_SR = 8000.0


def _sine(amplitude: float = 0.2, n: int = 4096) -> np.ndarray:
    times = np.arange(n) / _SR
    return amplitude * np.sin(2 * np.pi * 1000.0 * times)


def _frame(calibrations: list[ChannelCalibration]) -> ChannelFrame:
    values = np.vstack([_sine() for _ in calibrations])
    return ChannelFrame(
        da.from_array(values, chunks=(1, -1)),
        sampling_rate=_SR,
        channel_metadata=[
            ChannelMetadata(label=f"ch{index}", calibration=calibration)
            for index, calibration in enumerate(calibrations)
        ],
    )


_PA = ChannelCalibration(1.0, "Pa")
_PA2 = ChannelCalibration(1.0, "Pa^2", ref=4e-10, level="power")
_SONE = ChannelCalibration(1.0, "sone", ref=1.0, level="none")


class TestCalibrationContract:
    def test_default_is_amplitude_and_replacements_keep_the_kind(self) -> None:
        assert ChannelCalibration().level == "amplitude"
        assert _PA2.with_factor(3.0).level == "power"
        assert _PA2.with_unit("W").level == "power"
        assert _PA2.with_ref(1.0).level == "power"
        assert _PA.with_level("power") == ChannelCalibration(1.0, "Pa", level="power")

    def test_unknown_kind_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid channel level kind"):
            ChannelCalibration(1.0, "Pa", level=cast(Any, "energy"))

    def test_snapshot_writes_level_only_when_not_amplitude(self) -> None:
        assert _PA.to_dict() == {"factor": 1.0, "unit": "Pa", "ref": 2e-5}
        assert _PA2.to_dict() == {"factor": 1.0, "unit": "Pa^2", "ref": 4e-10, "level": "power"}
        assert ChannelCalibration.from_dict(_PA2.to_dict()) == _PA2
        assert ChannelCalibration.from_dict({"factor": 1.0, "unit": "Pa", "ref": 2e-5}) == _PA
        with pytest.raises(ValueError, match="Invalid channel calibration snapshot"):
            ChannelCalibration.from_dict({"factor": 1.0, "unit": "Pa", "ref": 2e-5, "kind": "power"})


class TestLevelConversion:
    def test_spectral_level_uses_10_log10_for_power_channels(self) -> None:
        spectrum = _frame([_PA, _PA2]).fft()
        magnitude = np.abs(spectrum.data)
        level = spectrum.dB
        np.testing.assert_allclose(level[0], 20 * np.log10(np.maximum(magnitude[0] / 2e-5, 1e-12)), rtol=1e-9)
        np.testing.assert_allclose(level[1], 10 * np.log10(np.maximum(magnitude[1] / 4e-10, 1e-12)), rtol=1e-9)

    def test_level_of_a_none_channel_raises(self) -> None:
        spectrum = _frame([_SONE]).fft()
        with pytest.raises(ValueError, match="A dB level is not defined for this channel"):
            _ = spectrum.dB
        with pytest.raises(ValueError, match="A dB level is not defined"):
            _ = _frame([_SONE]).channels[0].level_reference.to_level(1.0)

    def test_level_reference_carries_the_kind(self) -> None:
        reference = _frame([_PA2]).channels[0].level_reference
        assert reference == LevelReference(4e-10, "Pa^2", "power")
        assert reference.to_level(4e-8) == pytest.approx(20.0)
        assert LevelReference(2e-5, "Pa").to_level(2e-4) == pytest.approx(20.0)

    def test_noct_level_follows_the_kind(self) -> None:
        pytest.importorskip("mosqito")
        frame = ChannelFrame(
            da.from_array(np.vstack([_sine(n=48000 // 2)] * 2), chunks=(1, -1)),
            sampling_rate=48000.0,
            channel_metadata=[
                ChannelMetadata(label="a", calibration=_PA),
                ChannelMetadata(label="b", calibration=_PA2),
            ],
        )
        bands = frame.noct_spectrum(fmin=500.0, fmax=2000.0, n=3)
        values = np.abs(bands.data)
        np.testing.assert_allclose(bands.dB[0], 20 * np.log10(values[0] / 2e-5), rtol=1e-9)
        np.testing.assert_allclose(bands.dB[1], 10 * np.log10(values[1] / 4e-10), rtol=1e-9)

    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda frame: frame.rms_trend(frame_length=256, hop_length=128, dB=True), id="rms_trend"),
            pytest.param(lambda frame: frame.sound_level(dB=True), id="sound_level"),
            pytest.param(lambda frame: frame.leq(), id="leq"),
        ],
    )
    def test_rms_based_levels_need_amplitude_channels(self, call: Any) -> None:
        call(_frame([_PA]))
        for calibration in (_PA2, _SONE):
            with pytest.raises(ValueError, match="needs amplitude quantities"):
                call(_frame([calibration]))

    def test_level_results_have_no_further_level(self) -> None:
        level = _frame([_PA]).rms_trend(frame_length=256, hop_length=128, dB=True)
        assert level.channels[0].calibration.level == "none"


class TestPropagation:
    def test_storage_selection_and_derived_frames_keep_the_kind(self) -> None:
        frame = _frame([_PA, _PA2])
        assert [channel.calibration.level for channel in frame.channels] == ["amplitude", "power"]
        assert frame.get_channel(1).channels[0].calibration.level == "power"
        assert frame.high_pass_filter(cutoff=100.0).channels[1].calibration.level == "power"
        assert frame.fft().channels[1].calibration.level == "power"
        assert frame.with_calibration([_PA2, _PA2]).sum().channels[0].calibration.level == "power"
        assert frame.sum().channels[0].calibration.level == "none"

    def test_recipe_replays_a_power_calibration(self) -> None:
        from wandas.pipeline import RecipePlan

        source = _frame([_PA])
        configured = source.with_calibration([_PA2])
        replayed = RecipePlan.from_frame(configured, input_names=("signal",)).apply({"signal": source})
        assert replayed.channels[0].calibration == _PA2

    def test_wdf_round_trip_and_amplitude_files_keep_their_schema(self, tmp_path: Path) -> None:
        amplitude_path = tmp_path / "amplitude.wdf"
        _frame([_PA]).save(amplitude_path)
        with xr.open_dataset(amplitude_path, engine="h5netcdf") as dataset:
            assert "channel_level" not in dataset.data_vars

        path = tmp_path / "power.wdf"
        _frame([_PA, _PA2, _SONE]).save(path)
        loaded = wd.load(path)
        assert [channel.calibration for channel in loaded.channels] == [_PA, _PA2, _SONE]
