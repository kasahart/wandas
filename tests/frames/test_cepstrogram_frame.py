"""Public contracts for time-varying real-cepstrum analysis."""

from typing import Any, cast

import dask.array as da
import numpy as np
import pytest

import wandas as wd
from wandas.core.metadata import ChannelMetadata
from wandas.frames.cepstrogram import CepstrogramFrame
from wandas.frames.channel import ChannelFrame
from wandas.frames.spectrogram import SpectrogramFrame

_SAMPLING_RATE = 8_000
_N_FFT = 16
_HOP_LENGTH = 4


def _source_frame(sample_count: int = 64) -> ChannelFrame:
    time = np.arange(sample_count, dtype=float) / _SAMPLING_RATE
    data = np.stack(
        [
            np.sin(2 * np.pi * 500 * time) + 0.2 * np.cos(2 * np.pi * 1_250 * time),
            0.5 * np.cos(2 * np.pi * 1_000 * time) + 0.1,
        ]
    )
    return ChannelFrame(
        data=da.from_array(data, chunks=(1, -1)),
        sampling_rate=_SAMPLING_RATE,
        label="voice",
        metadata={"recording": {"room": "A"}},
        channel_metadata=[
            ChannelMetadata(label="left", unit="Pa", ref=2e-5),
            ChannelMetadata(label="right", unit="Pa", ref=2e-5),
        ],
        source_time_offset=np.array([0.25, 0.5]),
    )


def _spectrogram() -> SpectrogramFrame:
    return _source_frame().stft(
        n_fft=_N_FFT,
        hop_length=_HOP_LENGTH,
        win_length=_N_FFT,
        window="boxcar",
    )


def _cepstrogram() -> CepstrogramFrame:
    return _spectrogram().cepstrum(floor=1e-9)


def test_spectrogram_cepstrum_returns_lazy_typed_frame_with_atomic_state() -> None:
    spectrogram = _spectrogram()
    original_history = spectrogram.operation_history

    result = spectrogram.cepstrum(floor=1e-9)

    assert isinstance(result, CepstrogramFrame)
    assert isinstance(result._data, da.Array)
    assert result._data.shape == (2, _N_FFT, spectrogram.n_frames)
    assert result.n_fft == spectrogram.n_fft
    assert result.hop_length == spectrogram.hop_length
    assert result.win_length == spectrogram.win_length
    assert result.window == spectrogram.window
    assert result.labels == spectrogram.labels
    assert result._channel_ids == spectrogram._channel_ids
    assert result.metadata == spectrogram.metadata
    assert result.metadata is not spectrogram.metadata
    np.testing.assert_array_equal(result.source_time_offset, spectrogram.source_time_offset)
    np.testing.assert_allclose(result.quefrencies, np.arange(_N_FFT) / _SAMPLING_RATE)
    np.testing.assert_allclose(result.times, spectrogram.times)
    np.testing.assert_allclose(result.source_times, result.source_time_offset[:, None] + result.times[None, :])
    assert result.n_frames == spectrogram.n_frames
    assert result.to_xarray().dims == ("channel", "quefrency", "time")
    rebuilt_coords = result._xarray_coords(result._data)
    np.testing.assert_allclose(rebuilt_coords["quefrency"][1], result.quefrencies)
    np.testing.assert_allclose(rebuilt_coords["time"][1], result.times)
    assert result.operation_history[-1] == {
        "operation": "wandas.spectrogram.cepstrum",
        "version": 1,
        "params": {"floor": 1e-9},
    }
    assert len(result.operation_history) == len(original_history) + 1
    assert spectrogram.operation_history == original_history
    assert wd.CepstrogramFrame is CepstrogramFrame
    with pytest.raises(AttributeError, match=r"sampling_rate is immutable"):
        result.sampling_rate = _SAMPLING_RATE / 2


def test_cepstrogram_workflow_preserves_state_and_reconstructs_stft_magnitude() -> None:
    spectrogram = _spectrogram()
    cepstrogram = spectrogram.cepstrum()
    liftered = cepstrogram.lifter(cutoff=2 / _SAMPLING_RATE)
    envelope = liftered.to_spectral_envelope()

    assert isinstance(liftered, CepstrogramFrame)
    assert isinstance(envelope, SpectrogramFrame)
    assert isinstance(envelope._data, da.Array)
    assert envelope.n_fft == spectrogram.n_fft
    assert envelope.hop_length == spectrogram.hop_length
    assert envelope.win_length == spectrogram.win_length
    assert envelope.window == spectrogram.window
    assert envelope.labels == spectrogram.labels
    assert envelope._channel_ids == spectrogram._channel_ids
    assert envelope.metadata == spectrogram.metadata
    np.testing.assert_array_equal(envelope.source_time_offset, spectrogram.source_time_offset)
    assert [entry["operation"] for entry in envelope.operation_history[-3:]] == [
        "wandas.spectrogram.cepstrum",
        "wandas.cepstrogram.lifter",
        "wandas.cepstrogram.to_spectral_envelope",
    ]

    reconstructed = cepstrogram.to_spectral_envelope().magnitude
    expected = np.maximum(spectrogram.magnitude, 1e-12)
    # FFT/IFFT round-off is the only expected error in the analytical round trip.
    np.testing.assert_allclose(reconstructed, expected, rtol=1e-12, atol=1e-12)


def test_cepstrogram_slicing_preserves_axes_and_source_time() -> None:
    frame = _cepstrogram()

    selected_quefrencies = frame[:, 2:8:2, :]
    selected_time = frame[:, :, 2:6]

    np.testing.assert_array_equal(selected_quefrencies.quefrencies, frame.quefrencies[2:8:2])
    np.testing.assert_array_equal(selected_time.quefrencies, frame.quefrencies)
    np.testing.assert_allclose(selected_time.times, np.arange(4) * _HOP_LENGTH / _SAMPLING_RATE)
    np.testing.assert_allclose(
        selected_time.source_time_offset,
        frame.source_time_offset + 2 * _HOP_LENGTH / _SAMPLING_RATE,
    )
    assert isinstance(selected_time.lifter(2 / _SAMPLING_RATE), CepstrogramFrame)
    with pytest.raises(ValueError, match=r"complete, unsliced quefrency axis"):
        selected_quefrencies.lifter(2 / _SAMPLING_RATE)


def test_cepstrogram_constructor_rejects_invalid_domain_data() -> None:
    two_dimensional = CepstrogramFrame(
        da.zeros((8, 3), chunks=(-1, -1)),
        _SAMPLING_RATE,
        n_fft=8,
        hop_length=2,
    )

    assert two_dimensional._data.shape == (1, 8, 3)
    with pytest.raises(ValueError, match=r"Invalid data shape for CepstrogramFrame"):
        CepstrogramFrame(da.zeros((8,), chunks=-1), _SAMPLING_RATE, n_fft=8, hop_length=2)
    with pytest.raises(TypeError, match=r"real-valued coefficients"):
        CepstrogramFrame(
            da.zeros((1, 8, 3), chunks=(1, -1, -1), dtype=np.complex128),
            _SAMPLING_RATE,
            n_fft=8,
            hop_length=2,
        )
    with pytest.raises(TypeError, match=r"Invalid n_fft for CepstrogramFrame"):
        CepstrogramFrame(
            da.zeros((1, 8, 3), chunks=(1, -1, -1)),
            _SAMPLING_RATE,
            n_fft=True,
            hop_length=2,
        )
    with pytest.raises(ValueError, match=r"Invalid hop_length for CepstrogramFrame"):
        CepstrogramFrame(
            da.zeros((1, 8, 3), chunks=(1, -1, -1)),
            _SAMPLING_RATE,
            n_fft=8,
            hop_length=0,
        )
    with pytest.raises(ValueError, match=r"more quefrency bins than n_fft"):
        CepstrogramFrame(
            da.zeros((1, 9, 3), chunks=(1, -1, -1)),
            _SAMPLING_RATE,
            n_fft=8,
            hop_length=2,
        )
    with pytest.raises(ValueError, match=r"Invalid hop_length for CepstrogramFrame"):
        CepstrogramFrame(
            da.zeros((1, 8, 3), chunks=(1, -1, -1)),
            _SAMPLING_RATE,
            n_fft=8,
            hop_length=9,
            win_length=8,
        )
    with pytest.raises(ValueError, match=r"Invalid win_length for CepstrogramFrame"):
        CepstrogramFrame(
            da.zeros((1, 8, 3), chunks=(1, -1, -1)),
            _SAMPLING_RATE,
            n_fft=8,
            hop_length=2,
            win_length=9,
        )
    with pytest.raises(TypeError, match=r"window must be a non-empty string"):
        CepstrogramFrame(
            da.zeros((1, 8, 3), chunks=(1, -1, -1)),
            _SAMPLING_RATE,
            n_fft=8,
            hop_length=2,
            window="",
        )


def test_cepstrogram_binary_frame_requires_matching_domain_state() -> None:
    left = _cepstrogram()
    matching = _cepstrogram()
    different_state = CepstrogramFrame(
        data=left._data,
        sampling_rate=left.sampling_rate,
        n_fft=left.n_fft,
        hop_length=2,
        win_length=left.win_length,
        window=left.window,
        channel_metadata=left.channels.to_list(),
    )
    shifted_time = _cepstrogram()
    shifted_time._xr = shifted_time._xr.assign_coords(time=("time", shifted_time.times + 1.0))

    result = left + matching

    assert isinstance(result, CepstrogramFrame)
    assert isinstance(result._data, da.Array)
    with pytest.raises(TypeError, match=r"requires another CepstrogramFrame"):
        _ = left + _source_frame()
    with pytest.raises(ValueError, match=r"analysis state must match exactly"):
        _ = left + different_state
    with pytest.raises(ValueError, match=r"quefrency coordinates must match exactly"):
        _ = left[:, :8, :] + left[:, 8:, :]
    with pytest.raises(ValueError, match=r"time coordinates must match exactly"):
        _ = left + shifted_time


def test_cepstrogram_dataframe_conversion_is_explicitly_unsupported() -> None:
    frame = _cepstrogram()

    for conversion in (frame._get_dataframe_index, frame.to_dataframe):
        with pytest.raises(NotImplementedError, match=r"DataFrame conversion is not supported"):
            conversion()


def test_cepstrogram_plot_uses_time_and_quefrency_axes() -> None:
    import matplotlib.pyplot as plt

    frame = _cepstrogram()[0]
    figure, supplied_axis = plt.subplots()
    axis = cast(Any, frame.plot(ax=supplied_axis))

    assert axis is supplied_axis
    assert axis.get_xlabel() == "Time [s]"
    assert axis.get_ylabel() == "Quefrency [s]"
    assert len(axis.collections) == 1
    with pytest.raises(ValueError, match=r"supports only plot_type='cepstrogram'"):
        frame.plot("spectrogram")
    with pytest.raises(ValueError, match=r"requested quefrency plot range contains no bins"):
        frame.plot(qmin=1.0, qmax=2.0)

    multi_channel = _cepstrogram()
    with pytest.raises(ValueError, match=r"explicit axes can plot only one"):
        multi_channel.plot(ax=supplied_axis)
    channel_axes = list(cast(Any, multi_channel.plot(title="Cepstrum")))
    assert [target.get_title() for target in channel_axes] == ["Cepstrum — left", "Cepstrum — right"]

    plt.close(figure)
    for target in channel_axes:
        plt.close(target.figure)
