"""WDF files written before the canonical unit form still load (#651)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import dask.array as da
import numpy as np
import pytest
import xarray as xr

import wandas as wd
from wandas.core.metadata import ChannelCalibration, ChannelMetadata
from wandas.frames.channel import ChannelFrame


def _source() -> ChannelFrame:
    rng = np.random.default_rng(651)
    return ChannelFrame(
        da.from_array(rng.standard_normal((2, 2048)), chunks=(1, -1)),
        sampling_rate=8000.0,
        channel_metadata=[
            ChannelMetadata(label="p", calibration=ChannelCalibration(1.0, "Pa")),
            ChannelMetadata(label="v", calibration=ChannelCalibration(1.0, "m/s")),
        ],
    )


def _respell(path: Path, old_path: Path, replacements: dict[str, str]) -> None:
    """Write a copy of a WDF file whose pair units use the released spelling."""
    with xr.open_dataset(path, engine="h5netcdf", decode_cf=False) as opened:
        dataset = opened.load()
    constructor = dataset.attrs["constructor_json"]
    units = [str(unit) for unit in dataset["channel_unit"].values]
    for new, old in replacements.items():
        constructor = constructor.replace(f'"{new}"', f'"{old}"')
        units = [old if unit == new else unit for unit in units]
    assert constructor != dataset.attrs["constructor_json"]
    dataset.attrs["constructor_json"] = constructor
    dataset["channel_unit"] = ("channel", units)
    dataset.to_netcdf(old_path, engine="h5netcdf")


def test_csd_and_transfer_files_with_the_released_unit_spelling_load_in_the_current_form(tmp_path: Path) -> None:
    source = _source()
    csd = source.csd(n_fft=256, win_length=256, hop_length=128)
    transfer = source.transfer_function(n_fft=256, win_length=256, hop_length=128)
    assert [channel.unit for channel in csd.channels] == ["Pa^2", "m*Pa/s", "Pa*m/s", "m^2/s^2"]
    assert [channel.unit for channel in transfer.channels] == ["1", "Pa*s/m", "m/(s*Pa)", "1"]

    for frame, replacements in (
        (csd, {"Pa^2": "Pa*Pa", "m*Pa/s": "m/s*Pa", "m^2/s^2": "m/s*m/s"}),
        (transfer, {"Pa*s/m": "Pa/m/s", "m/(s*Pa)": "m/s/Pa"}),
    ):
        path = tmp_path / f"{type(frame).__name__}.wdf"
        old_path = tmp_path / f"{type(frame).__name__}-released.wdf"
        frame.save(path)
        _respell(path, old_path, replacements)
        loaded = cast(Any, wd.load(old_path))
        assert [channel.unit for channel in loaded.channels] == [channel.unit for channel in frame.channels]
        assert [record.domain for record in loaded.pair_state] == [record.domain for record in frame.pair_state]
        np.testing.assert_array_equal(np.asarray(loaded.data), np.asarray(frame.data))


@pytest.mark.parametrize(
    ("kind", "replacements"),
    [
        # A spectrum CSD never had the density spelling or "1".
        pytest.param("csd", {"Pa^2": "Pa*Pa/Hz"}, id="csd-density-spelling"),
        pytest.param("csd", {"Pa^2": "1"}, id="csd-dimensionless"),
        # The released spelling of the other transfer direction.
        pytest.param("transfer", {"Pa*s/m": "m/s/Pa"}, id="transfer-other-direction"),
    ],
)
def test_units_that_were_never_written_for_the_stored_quantity_are_rejected(
    tmp_path: Path, kind: str, replacements: dict[str, str]
) -> None:
    source = _source()
    if kind == "csd":
        frame: Any = source.csd(n_fft=256, win_length=256, hop_length=128, scaling="spectrum")
    else:
        frame = source.transfer_function(n_fft=256, win_length=256, hop_length=128)
    path = tmp_path / "current.wdf"
    old_path = tmp_path / "respelled.wdf"
    frame.save(path)
    _respell(path, old_path, replacements)
    with pytest.raises(ValueError):
        wd.load(old_path)


@pytest.mark.parametrize(
    ("kind", "replacements"),
    [
        pytest.param("csd-density", {"Pa^2/Hz": "Pa*Pa"}, id="density-csd-spectrum-spelling"),
        pytest.param("transfer", {"Pa*s/m": "m/s*Pa"}, id="transfer-csd-spelling"),
    ],
)
def test_spellings_of_another_scaling_or_quantity_are_rejected(
    tmp_path: Path, kind: str, replacements: dict[str, str]
) -> None:
    source = _source()
    if kind == "csd-density":
        frame: Any = source.csd(n_fft=256, win_length=256, hop_length=128, scaling="density")
    else:
        frame = source.transfer_function(n_fft=256, win_length=256, hop_length=128)
    path = tmp_path / "current.wdf"
    old_path = tmp_path / "respelled.wdf"
    frame.save(path)
    _respell(path, old_path, replacements)
    with pytest.raises(ValueError):
        wd.load(old_path)


def test_released_spellings_of_uncalibrated_and_equal_units() -> None:
    from wandas.processing.spectral_contracts import (
        OrderedSpectralPair,
        SpectralChannelRole,
        released_csd_unit,
        released_transfer_unit,
    )

    def pair(output: str, input_: str) -> OrderedSpectralPair:
        return OrderedSpectralPair(
            output=SpectralChannelRole(0, "out", output, 1.0),
            input=SpectralChannelRole(1, "in", input_, 1.0),
            n_channels=2,
        )

    assert released_csd_unit(pair("", ""), "spectrum") == "1"
    assert released_csd_unit(pair("Pa", ""), "density") == "Pa/Hz"
    assert released_transfer_unit(pair("Pa", "Pa")) == "1"
    assert released_transfer_unit(pair("Pa", "")) == "Pa"
    assert released_transfer_unit(pair("", "Pa")) == "1/Pa"
    assert released_transfer_unit(pair("Pa", "m/s"), "output") == "m/s/Pa"
