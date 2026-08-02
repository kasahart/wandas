"""Independent numerical and provenance contracts for SpectrogramFrame ISTFT."""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from unittest import mock

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray

from tests.processing.istft_oracle_fixtures import (
    _CASES,
    _SAMPLING_RATE,
    _make_independent_oracle,
    _OracleCase,
)
from wandas.core.metadata import ChannelCalibration, ChannelMetadata
from wandas.frames.channel import ChannelFrame
from wandas.frames.spectrogram import SpectrogramFrame
from wandas.processing.semantic import source_lineage

_SOURCE_HISTORY = [{"operation": "fixture.independent_istft", "version": 1, "params": {}}]


def _frame_metadata(case: _OracleCase) -> tuple[dict[str, Any], list[ChannelMetadata], list[str], np.ndarray]:
    """Build metadata that makes ownership and channel alignment observable."""

    channels = [
        ChannelMetadata(
            label=f"oracle-{index}",
            calibration=ChannelCalibration(
                factor=1.0 + 0.25 * index,
                unit="Pa",
                ref=2e-5 * (index + 1),
            ),
            extra={"sensor": f"sensor-{index}", "case": case.window},
        )
        for index in range(case.channels)
    ]
    channel_ids = [f"oracle-id-{index}" for index in range(case.channels)]
    source_time_offset = np.asarray(
        [0.125 + 0.75 * index for index in range(case.channels)],
        dtype=np.float64,
    )
    metadata = {
        "recording": "independent-istft-oracle",
        "analysis": {"fft": case.n_fft, "hop": case.hop_length},
    }
    return metadata, channels, channel_ids, source_time_offset


def _channel_metadata_snapshot(frame: SpectrogramFrame | ChannelFrame) -> list[tuple[Any, ...]]:
    """Return a value snapshot of public channel metadata fields."""

    return [
        (
            channel.label,
            channel.unit,
            channel.ref,
            channel.calibration,
            channel.extra.copy(),
        )
        for channel in frame._channel_metadata
    ]


@pytest.mark.parametrize(
    ("case", "channels"),
    [
        (_CASES[0], 1),
        (_CASES[1], 2),
        (_CASES[2], 1),
        (_CASES[3], 2),
    ],
)
def test_spectrogram_frame_public_istft_matches_independent_oracle(
    case: _OracleCase,
    channels: int,
) -> None:
    """Both public inverse paths preserve full values and public mono shape."""

    case = replace(case, channels=channels)
    scipy_sft, scipy_domain, normalized_wandas = _make_independent_oracle(case)
    expected_raw = scipy_sft.istft(scipy_domain)
    metadata, channel_metadata, channel_ids, source_time_offset = _frame_metadata(case)
    input_lineage = source_lineage(_SOURCE_HISTORY)
    input_snapshot = normalized_wandas.copy()

    frame = SpectrogramFrame(
        data=da.from_array(
            normalized_wandas.copy(),
            chunks=(1, normalized_wandas.shape[1], normalized_wandas.shape[2]),
        ),
        sampling_rate=_SAMPLING_RATE,
        n_fft=case.n_fft,
        hop_length=case.hop_length,
        win_length=case.win_length,
        window=case.window,
        label="independent-oracle",
        metadata=metadata,
        channel_metadata=channel_metadata,
        channel_ids=channel_ids,
        source_time_offset=source_time_offset,
        lineage=input_lineage,
    )
    input_history = frame.operation_history
    input_metadata = frame.metadata
    input_channel_metadata = _channel_metadata_snapshot(frame)

    assert isinstance(frame._data, DaArray)
    assert frame._data.shape == normalized_wandas.shape
    assert frame.shape == (
        (normalized_wandas.shape[1], normalized_wandas.shape[2]) if channels == 1 else normalized_wandas.shape
    )

    with mock.patch.object(DaArray, "compute") as compute:
        actual_istft = frame.istft()
        actual_to_channel = frame.to_channel_frame()
        compute.assert_not_called()

    for actual in (actual_istft, actual_to_channel):
        assert isinstance(actual, ChannelFrame)
        assert isinstance(actual._data, DaArray)
        assert actual._data.shape == expected_raw.shape
        assert actual._data.dtype == np.dtype(np.float64)
        assert actual.previous is frame
        assert actual.sampling_rate == _SAMPLING_RATE
        assert actual.label == "istft(independent-oracle)"
        assert actual.labels == frame.labels
        assert actual._channel_ids == channel_ids
        np.testing.assert_array_equal(actual.source_time_offset, source_time_offset)
        assert actual.metadata == metadata
        assert actual.lineage.inputs == (frame.lineage,)
        assert actual.lineage.operation is not None
        assert actual.lineage.operation.operation_id == "wandas.spectrogram.to_channel_frame"
        assert actual.operation_history == [
            *_SOURCE_HISTORY,
            {"operation": "wandas.spectrogram.to_channel_frame", "version": 1, "params": {}},
        ]
        assert _channel_metadata_snapshot(actual) == input_channel_metadata
        for source_channel, output_channel in zip(frame.channels, actual.channels, strict=True):
            assert output_channel.unit == source_channel.unit
            assert output_channel.ref == source_channel.ref
            assert output_channel.calibration == source_channel.calibration
            assert output_channel.extra == source_channel.extra

        # The internal tensor is the direct SciPy oracle. Public data also applies
        # the preserved per-channel calibration factor and hides mono's axis.
        np.testing.assert_allclose(actual._data.compute(), expected_raw, rtol=2e-12, atol=2e-12)
        calibration_factors = np.asarray(
            [channel.calibration.factor for channel in channel_metadata],
            dtype=np.float64,
        )
        expected_public = expected_raw * calibration_factors[:, None]
        public_data = actual.data
        if channels == 1:
            assert public_data.shape == expected_public.shape[1:]
            np.testing.assert_allclose(public_data, expected_public[0], rtol=2e-12, atol=2e-12)
        else:
            assert public_data.shape == expected_public.shape
            np.testing.assert_allclose(public_data, expected_public, rtol=2e-12, atol=2e-12)

    assert actual_istft is not actual_to_channel
    np.testing.assert_allclose(actual_istft._data.compute(), actual_to_channel._data.compute(), rtol=0.0, atol=0.0)

    # Constructing either public result must not alter the source Frame or its
    # caller-owned values, metadata, offsets, or semantic provenance.
    np.testing.assert_array_equal(frame._data.compute(), input_snapshot)
    assert frame.metadata == input_metadata
    assert frame.operation_history == input_history
    assert frame.lineage is input_lineage
    assert _channel_metadata_snapshot(frame) == input_channel_metadata
    np.testing.assert_array_equal(frame.source_time_offset, source_time_offset)
