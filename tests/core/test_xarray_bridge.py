import numpy as np
import numpy.testing as npt

import wandas as wd
from wandas.core.metadata import ChannelMetadata
from wandas.frames.spectral import SpectralFrame
from wandas.frames.spectrogram import SpectrogramFrame
from wandas.utils.dask_helpers import da_from_array


def test_channel_frame_to_xarray_uses_wandas_schema() -> None:
    frame = wd.ChannelFrame.from_numpy(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        sampling_rate=10.0,
        label="signal",
        metadata={"source": "fixture"},
        ch_labels=["left", "right"],
        ch_units=["Pa", "Pa"],
    )
    frame.operation_history.append({"operation": "gain", "factor": 2.0})

    xr_data = frame.to_xarray()

    assert xr_data.dims == ("channel", "time")
    assert xr_data.name == "signal"
    assert xr_data.attrs["wandas_frame_type"] == "ChannelFrame"
    assert xr_data.attrs["sampling_rate"] == 10.0
    assert xr_data.attrs["label"] == "signal"
    assert xr_data.attrs["metadata"] == {"source": "fixture"}
    assert xr_data.attrs["operation_history"] == [{"operation": "gain", "factor": 2.0}]
    assert xr_data.chunks == ((1, 1), (3,))
    npt.assert_array_equal(xr_data.coords["channel"].values, np.array(["left", "right"], dtype=object))
    npt.assert_allclose(xr_data.coords["time"].values, np.array([0.0, 0.1, 0.2]))
    npt.assert_array_equal(xr_data.coords["unit"].values, np.array(["Pa", "Pa"], dtype=object))
    npt.assert_allclose(xr_data.coords["ref"].values, np.array([20e-6, 20e-6]))


def test_xr_property_returns_dataarray_copy_by_default() -> None:
    frame = wd.ChannelFrame.from_numpy(np.array([[1.0, 2.0]]), sampling_rate=2.0)

    xr_data = frame.xr
    xr_data.attrs["label"] = "changed"

    assert frame.label == "numpy_data"
    assert frame.xr.attrs["label"] == "numpy_data"


def test_spectral_frame_to_xarray_uses_frequency_dimension() -> None:
    data = np.ones((2, 5), dtype=np.complex128)
    frame = SpectralFrame(
        data=da_from_array(data, chunks=(1, -1)),
        sampling_rate=16.0,
        n_fft=8,
        window="hann",
        label="spectrum",
        channel_metadata=[ChannelMetadata(label="a", unit="Pa"), ChannelMetadata(label="b", unit="Pa")],
    )

    xr_data = frame.to_xarray()

    assert xr_data.dims == ("channel", "frequency")
    assert xr_data.attrs["wandas_frame_type"] == "SpectralFrame"
    assert xr_data.attrs["n_fft"] == 8
    assert xr_data.attrs["window"] == "hann"
    assert xr_data.chunks == ((1, 1), (5,))
    npt.assert_allclose(xr_data.coords["frequency"].values, frame.freqs)


def test_spectrogram_frame_to_xarray_uses_frequency_and_time_dimensions() -> None:
    data = np.ones((2, 5, 3), dtype=np.complex128)
    frame = SpectrogramFrame.from_numpy(
        data,
        sampling_rate=16.0,
        n_fft=8,
        hop_length=2,
        label="stft",
        channel_metadata=[ChannelMetadata(label="a"), ChannelMetadata(label="b")],
    )

    xr_data = frame.to_xarray()

    assert xr_data.dims == ("channel", "frequency", "time")
    assert xr_data.attrs["wandas_frame_type"] == "SpectrogramFrame"
    assert xr_data.attrs["n_fft"] == 8
    assert xr_data.attrs["hop_length"] == 2
    assert xr_data.attrs["win_length"] == 8
    assert xr_data.attrs["window"] == "hann"
    assert xr_data.chunks == ((1, 1), (5,), (3,))
    npt.assert_allclose(xr_data.coords["frequency"].values, frame.freqs)
    npt.assert_allclose(xr_data.coords["time"].values, frame.times)


def test_from_xarray_round_trips_channel_frame_metadata_and_data() -> None:
    original = wd.ChannelFrame.from_numpy(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        sampling_rate=10.0,
        label="signal",
        metadata={"source": "fixture"},
        ch_labels=["left", "right"],
        ch_units=["Pa", "g"],
    )
    original.operation_history.append({"operation": "normalize"})

    restored = wd.from_xarray(original.to_xarray())

    assert isinstance(restored, wd.ChannelFrame)
    assert restored.sampling_rate == original.sampling_rate
    assert restored.label == original.label
    assert restored.metadata == original.metadata
    assert restored.operation_history == original.operation_history
    assert restored.labels == original.labels
    assert [ch.unit for ch in restored.channels] == ["Pa", "g"]
    npt.assert_allclose(restored.compute(), original.compute())
    assert restored._data.chunks == ((1, 1), (3,))


def test_frame_keeps_internal_xarray_storage_in_sync_with_data_property() -> None:
    frame = wd.ChannelFrame.from_numpy(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        sampling_rate=4.0,
        ch_labels=["a", "b"],
    )

    assert hasattr(frame, "_xr")
    assert frame._xr.dims == ("channel", "time")
    assert frame._data is frame._xr.data

    replacement = da_from_array(np.array([[5.0, 6.0]]), chunks=(1, -1))
    frame._data = replacement

    assert frame._xr.dims == ("channel", "time")
    assert frame._data is frame._xr.data
    assert frame._xr.chunks == ((1,), (2,))
    npt.assert_allclose(frame.compute(), np.array([[5.0, 6.0]]))
