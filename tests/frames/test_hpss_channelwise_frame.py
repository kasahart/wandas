"""Public Frame contracts for channel-wise HPSS execution."""

from copy import deepcopy
from unittest import mock

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

from tests.frame_helpers import channel_first_values
from wandas.core.metadata import ChannelCalibration
from wandas.frames.channel import ChannelFrame, ChannelMetadata
from wandas.utils.optional_imports import require_librosa_effects

_SAMPLING_RATE = 8_000
_SAMPLES = 512
_PARAMS = {
    "kernel_size": (7, 9),
    "power": 2.0,
    "margin": (1.0, 2.0),
    "n_fft": 64,
    "hop_length": 16,
    "win_length": 64,
    "window": "hann",
    "center": True,
    "pad_mode": "constant",
}
_HISTORY_PARAMS = {
    **_PARAMS,
    "kernel_size": [7, 9],
    "margin": [1.0, 2.0],
}


def _source() -> tuple[ChannelFrame, np.ndarray]:
    time = np.arange(_SAMPLES, dtype=np.float64) / _SAMPLING_RATE
    values = np.stack(
        [
            np.sin(2 * np.pi * 220 * time) + 0.5 * (np.arange(_SAMPLES) % 83 == 0),
            0.7 * np.sin(2 * np.pi * 330 * time) + 0.4 * (np.arange(_SAMPLES) % 97 == 0),
            0.4 * np.sin(2 * np.pi * 440 * time) + 0.3 * (np.arange(_SAMPLES) % 109 == 0),
        ]
    ).astype(np.float32)
    frame = ChannelFrame(
        da.from_array(values, chunks=(1, 128)),
        sampling_rate=_SAMPLING_RATE,
        label="measurement",
        metadata={"session": {"id": "hpss-contract"}},
        channel_metadata=[
            ChannelMetadata(
                label="left",
                calibration=ChannelCalibration(factor=2.0, unit="Pa", ref=2e-5),
                extra={"sensor": "mic-1"},
            ),
            ChannelMetadata(
                label="right",
                calibration=ChannelCalibration(factor=0.5, unit="V", ref=1.0),
                extra={"sensor": "mic-2"},
            ),
            ChannelMetadata(
                label="aux",
                calibration=ChannelCalibration(factor=1.5, unit="m/s^2", ref=1.0),
                extra={"sensor": "accel"},
            ),
        ],
        channel_ids=["left-id", "right-id", "aux-id"],
        source_time_offset=[0.25, 0.5, 0.75],
    )
    return frame, values


@pytest.mark.parametrize(
    ("method", "display", "extract_func"),
    [
        pytest.param("hpss_harmonic", "Hrm", "harmonic", id="harmonic"),
        pytest.param("hpss_percussive", "Prc", "percussive", id="percussive"),
    ],
)
def test_hpss_public_frame_preserves_contract_and_consumes_calibration_lazily(
    method: str,
    display: str,
    extract_func: str,
) -> None:
    source, caller_values = _source()
    caller_values_before = caller_values.copy()
    source_values = channel_first_values(source).copy()
    source_metadata = deepcopy(source.metadata)
    source_channels = source.channels.to_list()
    source_offsets = source.source_time_offset.copy()
    source_history = deepcopy(source.operation_history)
    source_lineage = source.lineage

    with mock.patch.object(DaArray, "compute") as compute:
        result = getattr(source, method)(**_PARAMS)
        compute.assert_not_called()

    assert isinstance(result, ChannelFrame)
    assert result is not source
    assert result.previous is source
    assert isinstance(result._data, DaArray)
    assert result.shape == (3, _SAMPLES)
    assert result._data.dtype == np.dtype(np.float64)
    assert result._data.chunks == ((1, 1, 1), (_SAMPLES,))
    assert result.sampling_rate == source.sampling_rate
    np.testing.assert_array_equal(result.time, source.time)
    assert result.label == source.label
    assert result.metadata == source_metadata
    assert [channel.id for channel in result.channels] == ["left-id", "right-id", "aux-id"]
    assert result.labels == [f"{display}(left)", f"{display}(right)", f"{display}(aux)"]
    assert [channel.unit for channel in result.channels] == ["Pa", "V", "m/s^2"]
    assert [channel.ref for channel in result.channels] == [2e-5, 1.0, 1.0]
    assert [channel.calibration.factor for channel in result.channels] == [1.0, 1.0, 1.0]
    assert [channel.extra for channel in result.channels] == [
        {"sensor": "mic-1"},
        {"sensor": "mic-2"},
        {"sensor": "accel"},
    ]
    np.testing.assert_array_equal(result.source_time_offset, source_offsets)
    np.testing.assert_array_equal(result.source_time, source_offsets[:, None] + source.time[None, :])
    assert result.operation_history == [
        {
            "operation": f"wandas.audio.{method}",
            "version": 1,
            "params": _HISTORY_PARAMS,
        }
    ]
    assert result.lineage.operation is not None
    assert result.lineage.operation.operation_id == f"wandas.audio.{method}"
    assert result.lineage.inputs == (source.lineage,)

    calibrated = caller_values.astype(np.float64) * np.array([[2.0], [0.5], [1.5]])
    librosa_effects = require_librosa_effects(method)
    expected = getattr(librosa_effects, extract_func)(calibrated, **_PARAMS)
    np.testing.assert_array_equal(channel_first_values(result), expected)

    chained = result.abs()
    assert isinstance(chained, ChannelFrame)
    assert isinstance(chained._data, DaArray)

    np.testing.assert_array_equal(channel_first_values(source), source_values)
    np.testing.assert_array_equal(caller_values, caller_values_before)
    assert source.metadata == source_metadata
    assert source.channels.to_list() == source_channels
    np.testing.assert_array_equal(source.source_time_offset, source_offsets)
    assert source.operation_history == source_history
    assert source.lineage is source_lineage
