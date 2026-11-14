"""Level kinds of band levels and WDF files (see tests/core/test_level_kind.py)."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest

from wandas.core.metadata import ChannelCalibration, ChannelMetadata
from wandas.frames.channel import ChannelFrame

_PA = ChannelCalibration(1.0, "Pa")
_PA2 = ChannelCalibration(1.0, "Pa^2", ref=4e-10, level="power")


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


def test_noct_level_follows_the_kind() -> None:
    pytest.importorskip("mosqito")
    frame = _frame([_PA, _PA2], sampling_rate=48000.0, n=48000 // 2)
    bands = frame.noct_spectrum(fmin=500.0, fmax=2000.0, n=3)
    values = np.abs(bands.data)
    np.testing.assert_allclose(bands.dB[0], 20 * np.log10(values[0] / 2e-5), rtol=1e-9)
    np.testing.assert_allclose(bands.dB[1], 10 * np.log10(values[1] / 4e-10), rtol=1e-9)
