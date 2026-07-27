"""Public Frame contracts for channel-wise Trim execution."""

from copy import deepcopy
from unittest import mock

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

from tests.frame_helpers import channel_first_values
from wandas.core.metadata import ChannelCalibration
from wandas.frames.channel import ChannelFrame, ChannelMetadata


def _source() -> ChannelFrame:
    values = np.arange(24, dtype=np.float32).reshape(3, 8)
    return ChannelFrame(
        da.from_array(values, chunks=(1, 4)),
        sampling_rate=8,
        label="measurement",
        metadata={"session": {"id": "trim-contract"}},
        channel_metadata=[
            ChannelMetadata(
                label="left",
                calibration=ChannelCalibration(factor=2.0, unit="Pa", ref=2e-5),
                extra={"sensor": "mic-1"},
            ),
            ChannelMetadata(
                label="right",
                calibration=ChannelCalibration(factor=3.0, unit="V", ref=1.0),
                extra={"sensor": "mic-2"},
            ),
            ChannelMetadata(
                label="aux",
                calibration=ChannelCalibration(factor=0.5, unit="m/s^2", ref=1.0),
                extra={"sensor": "accel"},
            ),
        ],
        channel_ids=["left-id", "right-id", "aux-id"],
        source_time_offset=[0.25, 0.5, 0.75],
    )


def test_trim_public_frame_preserves_state_and_consumes_calibration_lazily() -> None:
    source = _source()
    source_values = channel_first_values(source).copy()
    source_metadata = deepcopy(source.metadata)
    source_channels = source.channels.to_list()
    source_history = deepcopy(source.operation_history)
    source_lineage = source.lineage

    with mock.patch.object(DaArray, "compute") as compute:
        result = source.trim(start=0.25, end=0.75)
        compute.assert_not_called()

    assert isinstance(result, ChannelFrame)
    assert result is not source
    assert result.previous is source
    assert isinstance(result._data, DaArray)
    assert result.shape == (3, 4)
    assert result._data.dtype == np.dtype(np.float64)
    assert result._data.chunks == ((1, 1, 1), (4,))
    assert result.sampling_rate == source.sampling_rate
    assert result.label == source.label
    assert result.metadata == source_metadata
    assert [channel.id for channel in result.channels] == ["left-id", "right-id", "aux-id"]
    assert result.labels == ["trim(left)", "trim(right)", "trim(aux)"]
    assert [channel.unit for channel in result.channels] == ["Pa", "V", "m/s^2"]
    assert [channel.ref for channel in result.channels] == [2e-5, 1.0, 1.0]
    assert [channel.calibration.factor for channel in result.channels] == [1.0, 1.0, 1.0]
    assert [channel.extra for channel in result.channels] == [
        {"sensor": "mic-1"},
        {"sensor": "mic-2"},
        {"sensor": "accel"},
    ]
    np.testing.assert_array_equal(result.source_time_offset, np.array([0.5, 0.75, 1.0]))
    assert result.operation_history == [
        {
            "operation": "wandas.audio.trim",
            "version": 1,
            "params": {"start": 0.25, "end": 0.75},
        }
    ]
    assert result.lineage is not source.lineage

    raw = np.arange(24, dtype=np.float32).reshape(3, 8)
    expected = raw[:, 2:6] * np.array([[2.0], [3.0], [0.5]], dtype=np.float64)
    np.testing.assert_array_equal(channel_first_values(result), expected)

    np.testing.assert_array_equal(channel_first_values(source), source_values)
    assert source.metadata == source_metadata
    assert source.channels.to_list() == source_channels
    assert source.operation_history == source_history
    assert source.lineage is source_lineage


@pytest.mark.parametrize(
    ("start", "end", "expected_slice", "expected_offset"),
    [
        pytest.param(-0.25, 1.0, slice(-2, 8), np.array([1.0, 1.25, 1.5]), id="negative-start"),
        pytest.param(1.25, 1.75, slice(10, 14), np.array([1.25, 1.5, 1.75]), id="start-after-input"),
    ],
)
def test_trim_public_frame_slice_boundaries_stay_lazy_and_match_metadata(
    start: float,
    end: float,
    expected_slice: slice,
    expected_offset: np.ndarray,
) -> None:
    source = _source()
    raw = np.arange(24, dtype=np.float32).reshape(3, 8)
    expected = raw[:, expected_slice] * np.array([[2.0], [3.0], [0.5]], dtype=np.float64)

    with mock.patch.object(DaArray, "compute") as compute:
        result = source.trim(start=start, end=end)
        compute.assert_not_called()

    assert isinstance(result._data, DaArray)
    assert result.shape == expected.shape
    assert result._data.chunks == ((1, 1, 1), (expected.shape[-1],))
    np.testing.assert_array_equal(channel_first_values(result), expected)
    np.testing.assert_array_equal(result.source_time_offset, expected_offset)
    assert result.source_time.shape == expected.shape
    if expected.shape[-1]:
        np.testing.assert_array_equal(result.source_time[:, 0], expected_offset)
    assert result.operation_history[-1] == {
        "operation": "wandas.audio.trim",
        "version": 1,
        "params": {"start": start, "end": end},
    }
