"""Public contracts for the quefrency-domain ``CepstralFrame`` workflow."""

from typing import Any, cast

import dask.array as da
import numpy as np
import pytest

from wandas.core.metadata import ChannelMetadata
from wandas.frames.cepstral import CepstralFrame
from wandas.frames.channel import ChannelFrame
from wandas.frames.spectral import SpectralFrame

_SAMPLING_RATE = 8_000


def _source_frame(sample_count: int = 16) -> ChannelFrame:
    time = np.arange(sample_count, dtype=float) / _SAMPLING_RATE
    data = np.stack(
        [
            np.sin(2 * np.pi * 500 * time),
            0.5 * np.cos(2 * np.pi * 1_000 * time),
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


def _cepstral_frame() -> CepstralFrame:
    return _source_frame().cepstrum(n_fft=32, window="boxcar", floor=1e-9)


def test_channel_cepstrum_returns_lazy_typed_frame_with_atomic_state() -> None:
    source = _source_frame()
    source_history = source.operation_history
    source_data = source.compute().copy()

    result = source.cepstrum(n_fft=32, window="boxcar", floor=1e-9)

    assert isinstance(result, CepstralFrame)
    assert isinstance(result._data, da.Array)
    assert result.shape == (2, 32)
    assert result.n_fft == 32
    assert result.window == "boxcar"
    assert result.labels == source.labels
    assert result._channel_ids == source._channel_ids
    assert result.metadata == source.metadata
    assert result.metadata is not source.metadata
    np.testing.assert_array_equal(result.source_time_offset, source.source_time_offset)
    np.testing.assert_allclose(result.quefrencies, np.arange(32) / _SAMPLING_RATE)
    assert result.operation_history == [
        {
            "operation": "wandas.audio.cepstrum",
            "version": 1,
            "params": {"floor": 1e-9, "n_fft": 32, "window": "boxcar"},
        }
    ]
    assert source.operation_history == source_history
    np.testing.assert_array_equal(source.compute(), source_data)


def test_cepstral_workflow_preserves_metadata_and_matches_fft_envelope() -> None:
    source = _source_frame()

    cepstrum = source.cepstrum(n_fft=32, window="boxcar")
    liftered = cepstrum.lifter(cutoff=2 / _SAMPLING_RATE)
    envelope = liftered.to_spectral_envelope()

    assert isinstance(liftered, CepstralFrame)
    assert isinstance(envelope, SpectralFrame)
    assert isinstance(envelope._data, da.Array)
    assert envelope.n_fft == 32
    assert envelope.window == "boxcar"
    assert liftered.labels == source.labels
    assert envelope.labels == source.labels
    assert envelope._channel_ids == source._channel_ids
    assert envelope.metadata == source.metadata
    np.testing.assert_array_equal(envelope.source_time_offset, source.source_time_offset)
    assert [entry["operation"] for entry in envelope.operation_history] == [
        "wandas.audio.cepstrum",
        "wandas.cepstral.lifter",
        "wandas.cepstral.to_spectral_envelope",
    ]
    assert len(cepstrum.operation_history) == 1
    assert len(liftered.operation_history) == 2
    assert len(envelope.operation_history) == 3

    expected = np.abs(source.fft(n_fft=32, window="boxcar").compute())
    unfiltered = cepstrum.to_spectral_envelope().compute().real
    # FFT/IFFT round-off is the only expected error in the analytical round trip.
    np.testing.assert_allclose(unfiltered, expected, rtol=1e-12, atol=1e-12)


def test_lifter_rejects_invalid_cutoff_before_recording_history() -> None:
    cepstrum = _source_frame().cepstrum(n_fft=16, window="boxcar")
    history = cepstrum.operation_history

    with pytest.raises(ValueError, match=r"Invalid lifter cutoff for this cepstrum length"):
        cepstrum.lifter(cutoff=8 / _SAMPLING_RATE)

    assert cepstrum.operation_history == history


def test_cepstral_axes_survive_selection_and_derived_frames() -> None:
    frame = _cepstral_frame()

    selected = frame[:, 4:12:2]
    shifted = selected + 1.0

    expected = frame.quefrencies[4:12:2]
    np.testing.assert_array_equal(selected.quefrencies, expected)
    np.testing.assert_array_equal(shifted.quefrencies, expected)
    np.testing.assert_array_equal(selected.to_dataframe().index.to_numpy(), expected)
    assert selected.to_xarray().dims == ("channel", "quefrency")


@pytest.mark.parametrize("method_name", ["lifter", "to_spectral_envelope"])
def test_cepstral_transform_on_sliced_axis_raises_value_error(method_name: str) -> None:
    selected = _cepstral_frame()[:, 2:10]

    with pytest.raises(ValueError, match=r"complete, unsliced quefrency axis"):
        if method_name == "lifter":
            selected.lifter(cutoff=2 / _SAMPLING_RATE)
        else:
            selected.to_spectral_envelope()


def test_cepstral_xarray_export_isolates_quefrency_coordinates() -> None:
    frame = _cepstral_frame()
    expected = frame.quefrencies
    exported = frame.to_xarray()

    exported.coords["quefrency"].values[0] = 1.0

    np.testing.assert_array_equal(frame.quefrencies, expected)


def test_cepstral_constructor_rejects_invalid_domain_data() -> None:
    with pytest.raises(ValueError, match=r"Invalid data shape for CepstralFrame"):
        CepstralFrame(da.zeros((1, 2, 3), chunks=(1, -1, -1)), _SAMPLING_RATE, n_fft=3)
    with pytest.raises(TypeError, match=r"real-valued coefficients"):
        CepstralFrame(da.zeros((1, 8), chunks=(1, -1), dtype=np.complex128), _SAMPLING_RATE, n_fft=8)
    with pytest.raises(ValueError, match=r"more quefrency bins than n_fft"):
        CepstralFrame(da.zeros((1, 8), chunks=(1, -1)), _SAMPLING_RATE, n_fft=4)


def test_cepstral_binary_frame_requires_matching_domain_state() -> None:
    left = _cepstral_frame()
    right = CepstralFrame(
        data=left._data,
        sampling_rate=left.sampling_rate,
        n_fft=left.n_fft,
        window="hann",
        channel_metadata=left.channels.to_list(),
    )

    with pytest.raises(ValueError, match=r"analysis window mismatch"):
        _ = left + right


def test_cepstral_plot_uses_quefrency_axis_and_coefficients() -> None:
    import matplotlib.pyplot as plt

    frame = _cepstral_frame()[0]
    axis = cast(Any, frame.plot())

    np.testing.assert_array_equal(np.asarray(axis.lines[0].get_xdata()), frame.quefrencies)
    np.testing.assert_allclose(np.asarray(axis.lines[0].get_ydata()), frame.compute()[0])
    assert axis.get_xlabel() == "Quefrency [s]"
    plt.close(axis.figure)
