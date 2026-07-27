"""Public Frame contracts for channel-wise N-octave spectrum execution."""

from copy import deepcopy
from unittest import mock

import dask.array as da
import numpy as np
import pytest
from dask.array.core import Array as DaArray
from mosqito.sound_level_meter import noct_spectrum

from tests.frame_helpers import channel_first_values
from wandas.core.metadata import ChannelCalibration
from wandas.frames.channel import ChannelFrame, ChannelMetadata
from wandas.frames.noct import NOctFrame

_SAMPLING_RATE = 8_192
_SAMPLES = 2_048
_FMIN = 250.0
_FMAX = 2_000.0
_N = 3
_G = 10
_FR = 1_000


def _source() -> tuple[ChannelFrame, np.ndarray]:
    time = np.arange(_SAMPLES, dtype=np.float64) / _SAMPLING_RATE
    values = np.stack(
        [
            0.6 * np.sin(2 * np.pi * 375 * time) + 0.2 * np.cos(2 * np.pi * 750 * time),
            0.5 * np.sin(2 * np.pi * 625 * time) + 0.15 * np.cos(2 * np.pi * 1_250 * time),
            0.4 * np.sin(2 * np.pi * 875 * time) + 0.1 * np.cos(2 * np.pi * 1_750 * time),
        ]
    ).astype(np.float32)
    frame = ChannelFrame(
        da.from_array(values, chunks=(1, 512)),
        sampling_rate=_SAMPLING_RATE,
        label="measurement",
        metadata={"session": {"id": "noct-spectrum-contract"}},
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


def _direct_mosqito(
    values: np.ndarray,
    *,
    fmin: float = _FMIN,
    fmax: float = _FMAX,
) -> tuple[np.ndarray, np.ndarray]:
    spectrum, frequencies = noct_spectrum(
        sig=values.T,
        fs=_SAMPLING_RATE,
        fmin=fmin,
        fmax=fmax,
        n=_N,
        G=_G,
        fr=_FR,
    )
    spectrum = np.asarray(spectrum)
    channels = values.shape[0]
    channel_first = spectrum.reshape(-1, channels).T if channels > 0 else spectrum.T
    return channel_first, np.asarray(frequencies)


def test_noct_spectrum_public_frame_preserves_analysis_contract_and_consumes_calibration_lazily() -> None:
    source, caller_values = _source()
    caller_values_before = caller_values.copy()
    source_values = channel_first_values(source).copy()
    source_metadata = deepcopy(source.metadata)
    source_channels = source.channels.to_list()
    source_offsets = source.source_time_offset.copy()
    source_history = deepcopy(source.operation_history)
    source_lineage = source.lineage

    with mock.patch.object(DaArray, "compute") as compute:
        result = source.noct_spectrum(
            fmin=_FMIN,
            fmax=_FMAX,
            n=_N,
            G=_G,
            fr=_FR,
        )
        compute.assert_not_called()

    calibrated = caller_values.astype(np.float64) * np.array([[2.0], [0.5], [1.5]])
    expected, expected_frequencies = _direct_mosqito(calibrated)

    assert isinstance(result, NOctFrame)
    assert result is not source
    assert result.previous is source
    assert isinstance(result._data, DaArray)
    assert result.shape == expected.shape
    assert result._data.dtype == np.dtype(np.float64)
    assert result._data.chunks == ((1, 1, 1), (expected.shape[1],))
    assert result._xr.dims == ("channel", "band")
    assert result.sampling_rate == source.sampling_rate
    assert result.fmin == _FMIN
    assert result.fmax == _FMAX
    assert result.n == _N
    assert result.G == _G
    assert result.fr == _FR
    np.testing.assert_array_equal(result.freqs, expected_frequencies)
    assert result.label == "1/3Oct of measurement"
    assert result.metadata == source_metadata
    assert [channel.id for channel in result.channels] == ["left-id", "right-id", "aux-id"]
    assert result.labels == ["left", "right", "aux"]
    assert [channel.unit for channel in result.channels] == ["Pa", "V", "m/s^2"]
    assert [channel.ref for channel in result.channels] == [2e-5, 1.0, 1.0]
    assert [channel.calibration.factor for channel in result.channels] == [1.0, 1.0, 1.0]
    assert [channel.extra for channel in result.channels] == [
        {"sensor": "mic-1"},
        {"sensor": "mic-2"},
        {"sensor": "accel"},
    ]
    np.testing.assert_array_equal(result.source_time_offset, source_offsets)
    assert result.operation_history == [
        {
            "operation": "wandas.audio.noct_spectrum",
            "version": 1,
            "params": {
                "fmin": _FMIN,
                "fmax": _FMAX,
                "n": _N,
                "G": _G,
                "fr": _FR,
            },
        }
    ]
    assert result.lineage.operation is not None
    assert result.lineage.operation.operation_id == "wandas.audio.noct_spectrum"
    assert result.lineage.inputs == (source.lineage,)
    np.testing.assert_array_equal(channel_first_values(result), expected)

    np.testing.assert_array_equal(channel_first_values(source), source_values)
    np.testing.assert_array_equal(caller_values, caller_values_before)
    assert source.metadata == source_metadata
    assert source.channels.to_list() == source_channels
    np.testing.assert_array_equal(source.source_time_offset, source_offsets)
    assert source.operation_history == source_history
    assert source.lineage is source_lineage


def test_noct_spectrum_public_frame_preserves_single_band_shape_axes_dtype_and_laziness() -> None:
    source, caller_values = _source()

    with mock.patch.object(DaArray, "compute") as compute:
        result = source.noct_spectrum(
            fmin=1_000.0,
            fmax=1_000.0,
            n=_N,
            G=_G,
            fr=_FR,
        )
        compute.assert_not_called()

    calibrated = caller_values.astype(np.float64) * np.array([[2.0], [0.5], [1.5]])
    expected, expected_frequencies = _direct_mosqito(
        calibrated,
        fmin=1_000.0,
        fmax=1_000.0,
    )

    assert isinstance(result, NOctFrame)
    assert isinstance(result._data, DaArray)
    assert result.previous is source
    assert result.shape == expected.shape == (3, 1)
    assert result._data.dtype == expected.dtype == np.dtype(np.float64)
    assert result._data.chunks == ((1, 1, 1), (1,))
    assert result._xr.dims == ("channel", "band")
    np.testing.assert_array_equal(result.freqs, expected_frequencies)
    np.testing.assert_array_equal(channel_first_values(result), expected)


def test_noct_spectrum_public_frame_rejects_empty_band_range_without_computing() -> None:
    source, _ = _source()

    with mock.patch.object(DaArray, "compute") as compute:
        with pytest.raises(ValueError) as exc_info:
            source.noct_spectrum(
                fmin=2_000.0,
                fmax=1_000.0,
                n=_N,
                G=_G,
                fr=_FR,
            )
        compute.assert_not_called()
    assert str(exc_info.value) == (
        "Invalid frequency bounds for NOctFrame\n"
        "  Got: fmin=2000.0, fmax=1000.0\n"
        "  Expected: 0 <= fmin <= fmax\n"
        "Use the frequency bounds of the N-octave analysis."
    )
