"""Level kinds of band levels and WDF files (see tests/core/test_level_kind.py)."""

from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

import wandas as wd
from wandas.core.metadata import ChannelCalibration, ChannelMetadata
from wandas.frames.channel import ChannelFrame

_PA = ChannelCalibration(1.0, "Pa")
_PA2 = ChannelCalibration(1.0, "Pa^2", ref=4e-10, level="power")
_SONE = ChannelCalibration(1.0, "sone", ref=1.0, level="none")


def _frame(calibrations: list[ChannelCalibration], sampling_rate: float = 8000.0, n: int = 4096) -> ChannelFrame:
    times = np.arange(n) / sampling_rate
    values = np.vstack([0.2 * np.sin(2 * np.pi * 1000.0 * times) for _ in calibrations])
    return ChannelFrame(
        da.from_array(values, chunks=(1, -1)),
        sampling_rate=sampling_rate,
        channel_metadata=[
            ChannelMetadata(label=f"ch{index}", calibration=calibration)
            for index, calibration in enumerate(calibrations)
        ],
    )


def test_wdf_round_trip_and_amplitude_files_keep_their_schema(tmp_path: Path) -> None:
    amplitude_path = tmp_path / "amplitude.wdf"
    _frame([_PA]).save(amplitude_path)
    with xr.open_dataset(amplitude_path, engine="h5netcdf") as dataset:
        assert "channel_level" not in dataset.data_vars

    path = tmp_path / "power.wdf"
    _frame([_PA, _PA2, _SONE]).save(path)
    loaded = wd.load(path)
    assert [channel.calibration for channel in loaded.channels] == [_PA, _PA2, _SONE]
