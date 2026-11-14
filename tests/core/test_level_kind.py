"""Channel level kind: amplitude (20 log10), power (10 log10) or none (no dB)."""

from __future__ import annotations

import warnings
from typing import Any, cast

import dask.array as da
import numpy as np
import pytest

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

    def test_level_of_a_none_channel_warns_and_uses_the_amplitude_level(self) -> None:
        spectrum = _frame([_SONE]).fft()
        with pytest.warns(UserWarning, match="A dB level is not defined for this channel") as record:
            level = spectrum.dB
        assert record[0].filename == __file__
        np.testing.assert_allclose(level, 20 * np.log10(np.maximum(np.abs(spectrum.data), 1e-12)), rtol=1e-9)
        with pytest.warns(UserWarning, match="A dB level is not defined"):
            assert _frame([_SONE]).channels[0].level_reference.to_level(10.0) == pytest.approx(20.0)

    def test_level_reference_carries_the_kind(self) -> None:
        reference = _frame([_PA2]).channels[0].level_reference
        assert reference == LevelReference(4e-10, "Pa^2", "power")
        assert reference.to_level(4e-8) == pytest.approx(20.0)
        assert LevelReference(2e-5, "Pa").to_level(2e-4) == pytest.approx(20.0)

    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda frame: frame.rms_trend(frame_length=256, hop_length=128, dB=True), id="rms_trend"),
            pytest.param(lambda frame: frame.sound_level(dB=True), id="sound_level"),
            pytest.param(lambda frame: frame.leq(), id="leq"),
        ],
    )
    def test_rms_based_levels_warn_for_non_amplitude_channels(self, call: Any) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            call(_frame([_PA]))
        for calibration in (_PA2, _SONE):
            with pytest.warns(UserWarning, match="needs amplitude quantities") as record:
                call(_frame([calibration]))
            assert record[0].filename == __file__

    def test_level_results_have_no_further_level(self) -> None:
        level = _frame([_PA]).rms_trend(frame_length=256, hop_length=128, dB=True)
        assert level.channels[0].calibration.level == "none"
        with pytest.warns(UserWarning, match="A dB level is not defined"):
            _ = level.fft().dB


class TestPropagation:
    def test_storage_selection_and_derived_frames_keep_the_kind(self) -> None:
        frame = _frame([_PA, _PA2])
        assert [channel.calibration.level for channel in frame.channels] == ["amplitude", "power"]
        assert frame.get_channel(1).channels[0].calibration.level == "power"
        assert frame.high_pass_filter(cutoff=100.0).channels[1].calibration.level == "power"
        assert frame.fft().channels[1].calibration.level == "power"
        assert frame.with_calibration([_PA2, _PA2]).sum().channels[0].calibration.level == "power"
        assert frame.sum().channels[0].calibration.level == "none"

    def test_frame_arithmetic_of_different_kinds_has_no_level_in_either_order(self) -> None:
        amplitude, power = _frame([_PA]), _frame([_PA2])
        with pytest.warns(UserWarning, match="different units"):
            assert (amplitude + power).channels[0].calibration.level == "none"
        with pytest.warns(UserWarning, match="different units"):
            assert (power + amplitude).channels[0].calibration.level == "none"
        assert (power + power).channels[0].calibration.level == "power"
        assert (power * 2.0).channels[0].calibration.level == "power"

    def test_storage_written_before_the_level_kind_reads_as_amplitude(self) -> None:
        frame = _frame([_PA2])
        frame._xr = frame._xr.drop_vars("channel_level")
        assert frame.channels[0].calibration == ChannelCalibration(1.0, "Pa^2", ref=4e-10)

    def test_recipe_replays_a_power_calibration(self) -> None:
        from wandas.pipeline import RecipePlan

        source = _frame([_PA])
        configured = source.with_calibration([_PA2])
        replayed = RecipePlan.from_frame(configured, input_names=("signal",)).apply({"signal": source})
        assert replayed.channels[0].calibration == _PA2
