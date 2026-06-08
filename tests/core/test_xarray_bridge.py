import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

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


def test_strict_time_core_operation_rejects_split_time_chunks() -> None:
    frame = wd.ChannelFrame.from_numpy(np.arange(64.0).reshape(1, -1), sampling_rate=128.0)
    frame._data = frame._data.rechunk((1, 16))

    with pytest.raises(ValueError, match="requires contiguous chunks along time"):
        frame.low_pass_filter(10.0)


def test_elementwise_operation_allows_split_time_chunks() -> None:
    frame = wd.ChannelFrame.from_numpy(np.array([[-1.0, 2.0, -3.0, 4.0]]), sampling_rate=4.0)
    frame._data = frame._data.rechunk((1, 2))

    result = frame.abs()

    npt.assert_allclose(result.compute(), np.array([[1.0, 2.0, 3.0, 4.0]]))


def test_netcdf_round_trip_preserves_channel_frame_schema(tmp_path) -> None:
    original = wd.ChannelFrame.from_numpy(
        np.array([[1.0, 2.0, 3.0]]),
        sampling_rate=12.0,
        label="io-signal",
        metadata={"source": "netcdf-test"},
        ch_labels=["mic"],
        ch_units=["Pa"],
    )
    original.operation_history.append({"operation": "gain", "factor": 0.5})
    path = tmp_path / "frame.nc"

    original.to_netcdf(path)
    restored = wd.open_netcdf(path)

    assert isinstance(restored, wd.ChannelFrame)
    assert restored.sampling_rate == original.sampling_rate
    assert restored.label == original.label
    assert restored.metadata == original.metadata
    assert restored.operation_history == original.operation_history
    assert restored.labels == original.labels
    assert [ch.unit for ch in restored.channels] == ["Pa"]
    npt.assert_allclose(restored.compute(), original.compute())


def test_from_xarray_preserves_channel_extra_metadata() -> None:
    frame = wd.ChannelFrame(
        data=da_from_array(np.array([[1.0, 2.0], [3.0, 4.0]]), chunks=(1, -1)),
        sampling_rate=10.0,
        channel_metadata=[
            ChannelMetadata(label="front", unit="Pa", extra={"role": "reference", "axis": "x"}),
            ChannelMetadata(label="rear", unit="g", extra={"role": "response", "axis": "z"}),
        ],
    )

    restored = wd.from_xarray(frame.to_xarray())

    assert [ch.extra for ch in restored.channels] == [ch.extra for ch in frame.channels]
    assert restored.get_channel(query={"role": "response"}).labels == ["rear"]


def test_from_xarray_rejects_sliced_spectral_frequency_axis() -> None:
    frame = SpectralFrame(
        data=da_from_array(np.ones((1, 5), dtype=np.complex128), chunks=(1, -1)),
        sampling_rate=16.0,
        n_fft=8,
    )
    sliced = frame.to_xarray().isel(frequency=slice(0, 2))

    with pytest.raises(ValueError, match="frequency dimension length"):
        wd.from_xarray(sliced)


def test_noct_frame_round_trips_through_xarray() -> None:
    from wandas.frames.noct import NOctFrame

    frame = NOctFrame(
        data=da_from_array(np.ones((1, 4)), chunks=(1, -1)),
        sampling_rate=48_000,
        fmin=100.0,
        fmax=1000.0,
        n=3,
        G=10,
        fr=1000,
        channel_metadata=[ChannelMetadata(label="mic", unit="Pa")],
    )

    restored = wd.from_xarray(frame.to_xarray())

    assert isinstance(restored, NOctFrame)
    assert restored.fmin == frame.fmin
    assert restored.fmax == frame.fmax
    assert restored.n == frame.n
    assert restored.G == frame.G
    assert restored.fr == frame.fr
    npt.assert_allclose(restored.compute(), frame.compute())


def test_normalize_axis_zero_allows_split_time_chunks() -> None:
    frame = wd.ChannelFrame.from_numpy(np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]), sampling_rate=4.0)
    frame._data = frame._data.rechunk((1, 2))

    result = frame.normalize(axis=0)

    npt.assert_allclose(result.compute(), np.array([[0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0]]))


def test_netcdf_round_trip_encodes_nested_numpy_attrs(tmp_path) -> None:
    frame = wd.ChannelFrame.from_numpy(np.array([[1.0, 2.0]]), sampling_rate=2.0)
    frame.operation_history.append(
        {"operation": "custom", "params": {"ref": np.array([1.0, 2.0]), "scale": np.float64(0.5)}}
    )
    path = tmp_path / "numpy_attrs.nc"

    frame.to_netcdf(path)
    restored = wd.open_netcdf(path)

    assert restored.operation_history == [{"operation": "custom", "params": {"ref": [1.0, 2.0], "scale": 0.5}}]


def test_as_dask_channelwise_handles_scalar_and_1d_dataarrays() -> None:
    from wandas.xarray_bridge import _as_dask_channelwise

    scalar = _as_dask_channelwise(xr.DataArray(np.array(1.0)))
    one_dim = _as_dask_channelwise(xr.DataArray(np.array([1.0, 2.0]), dims=("time",)))

    assert scalar.shape == ()
    assert scalar.chunks == ()
    assert one_dim.shape == (2,)
    assert one_dim.chunks == ((2,),)
