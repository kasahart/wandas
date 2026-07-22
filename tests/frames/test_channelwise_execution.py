"""Frame contracts for channel-wise AudioOperation execution."""

import dask.array as da
import numpy as np
from dask.array.core import Array as DaArray

from wandas.core.metadata import ChannelCalibration, ChannelMetadata
from wandas.frames.channel import ChannelFrame


def _calibrated_frame() -> ChannelFrame:
    return ChannelFrame(
        da.from_array(
            np.array(
                [
                    [1.0, 2.0, 4.0, 8.0],
                    [8.0, 4.0, 2.0, 1.0],
                ]
            ),
            chunks=(1, -1),
        ),
        sampling_rate=8_000,
        label="measurement",
        metadata={"site": "lab-a"},
        channel_metadata=[
            ChannelMetadata(
                label="microphone",
                calibration=ChannelCalibration(factor=2.0, unit="Pa", ref=2e-5),
                extra={"serial": "mic-1"},
            ),
            ChannelMetadata(
                label="accelerometer",
                calibration=ChannelCalibration(factor=0.5, unit="m/s^2", ref=1.0),
                extra={"serial": "acc-1"},
            ),
        ],
        channel_ids=["sensor-mic", "sensor-acc"],
        source_time_offset=[0.25, 0.5],
    )


def test_remove_dc_channel_wise_execution_preserves_frame_contract() -> None:
    source = _calibrated_frame()
    source_values = source.data.copy()
    source_metadata = source.metadata.copy()
    source_history = source.operation_history.copy()

    result = source.remove_dc()

    expected = source_values - source_values.mean(axis=-1, keepdims=True)
    assert result is not source
    assert isinstance(result._data, DaArray)
    np.testing.assert_allclose(result.data, expected)
    np.testing.assert_array_equal(source.data, source_values)
    assert source.metadata == source_metadata
    assert source.operation_history == source_history

    assert result.shape == source.shape
    assert result._data.dtype == source._effective_data.dtype
    assert result.sampling_rate == source.sampling_rate
    assert result.metadata == source.metadata
    assert [channel.id for channel in result.channels] == ["sensor-mic", "sensor-acc"]
    assert result.labels == ["dcRM(microphone)", "dcRM(accelerometer)"]
    assert [channel.calibration.factor for channel in result.channels] == [1.0, 1.0]
    assert [channel.unit for channel in result.channels] == ["Pa", "m/s^2"]
    assert [channel.ref for channel in result.channels] == [2e-5, 1.0]
    assert [channel.extra for channel in result.channels] == [
        {"serial": "mic-1"},
        {"serial": "acc-1"},
    ]
    np.testing.assert_array_equal(result.source_time_offset, np.array([0.25, 0.5]))
    assert result.operation_history == [{"operation": "wandas.audio.remove_dc", "version": 1, "params": {}}]


def test_remove_dc_empty_frame_preserves_audio_operation_contract() -> None:
    source = ChannelFrame(
        da.from_array(np.empty((0, 4)), chunks=(0, 4)),
        sampling_rate=8_000,
    )

    result = source.remove_dc()

    assert result is not source
    assert isinstance(result._data, DaArray)
    assert result.shape == (0, 4)
    np.testing.assert_array_equal(result.data, np.empty((0, 4)))
    assert result.operation_history == [{"operation": "wandas.audio.remove_dc", "version": 1, "params": {}}]
