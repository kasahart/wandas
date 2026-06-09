from typing import Any, cast

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
        data=da_from_array(np.ones((1, 11)), chunks=(1, -1)),
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


def test_from_xarray_infers_channel_frame_when_schema_type_missing() -> None:
    original = wd.ChannelFrame.from_numpy(
        np.array([[1.0, 2.0, 3.0]]),
        sampling_rate=10.0,
        label="signal",
        ch_labels=["mic"],
    )
    xr_data = original.to_xarray()
    xr_data.attrs.pop("wandas_frame_type")

    restored = wd.from_xarray(xr_data)

    assert isinstance(restored, wd.ChannelFrame)
    assert restored.labels == ["mic"]
    npt.assert_allclose(restored.compute(), original.compute())


def test_from_xarray_round_trips_spectral_frame() -> None:
    original = SpectralFrame(
        data=da_from_array(np.ones((1, 5), dtype=np.complex128), chunks=(1, -1)),
        sampling_rate=16.0,
        n_fft=8,
        window="boxcar",
        label="spectrum",
    )

    restored = wd.from_xarray(original.to_xarray())

    assert isinstance(restored, SpectralFrame)
    assert restored.n_fft == 8
    assert restored.window == "boxcar"
    npt.assert_allclose(restored.compute(), original.compute())


def test_from_xarray_infers_spectral_frame_when_schema_type_missing() -> None:
    original = SpectralFrame(
        data=da_from_array(np.ones((1, 5), dtype=np.complex128), chunks=(1, -1)),
        sampling_rate=16.0,
        n_fft=8,
    )
    xr_data = original.to_xarray()
    xr_data.attrs.pop("wandas_frame_type")

    restored = wd.from_xarray(xr_data)

    assert isinstance(restored, SpectralFrame)
    npt.assert_allclose(restored.compute(), original.compute())


def test_from_xarray_round_trips_spectrogram_frame() -> None:
    original = SpectrogramFrame.from_numpy(
        np.ones((1, 5, 3), dtype=np.complex128),
        sampling_rate=16.0,
        n_fft=8,
        hop_length=2,
        win_length=4,
        window="boxcar",
        label="stft",
    )

    restored = wd.from_xarray(original.to_xarray())

    assert isinstance(restored, SpectrogramFrame)
    assert restored.n_fft == 8
    assert restored.hop_length == 2
    assert restored.win_length == 4
    assert restored.window == "boxcar"
    npt.assert_allclose(restored.compute(), original.compute())


def test_from_xarray_infers_spectrogram_frame_when_schema_type_missing() -> None:
    original = SpectrogramFrame.from_numpy(
        np.ones((1, 5, 3), dtype=np.complex128),
        sampling_rate=16.0,
        n_fft=8,
        hop_length=2,
    )
    xr_data = original.to_xarray()
    xr_data.attrs.pop("wandas_frame_type")

    restored = wd.from_xarray(xr_data)

    assert isinstance(restored, SpectrogramFrame)
    npt.assert_allclose(restored.compute(), original.compute())


def test_from_xarray_rejects_unknown_untyped_dims() -> None:
    xr_data = xr.DataArray(
        np.ones((2, 3)),
        dims=("sensor", "sample"),
        attrs={"sampling_rate": 10.0},
    )

    with pytest.raises(ValueError, match="Cannot infer Wandas frame type"):
        wd.from_xarray(xr_data)


def test_from_xarray_rejects_invalid_typed_dims() -> None:
    xr_data = xr.DataArray(
        np.ones((2, 3)),
        dims=("time", "channel"),
        attrs={"wandas_frame_type": "ChannelFrame", "sampling_rate": 10.0},
    )

    with pytest.raises(ValueError, match="Invalid dims for ChannelFrame"):
        wd.from_xarray(xr_data)


def test_from_xarray_rejects_unsupported_typed_frame() -> None:
    xr_data = xr.DataArray(
        np.ones((2, 3)),
        dims=("row", "column"),
        attrs={"wandas_frame_type": "UnknownFrame", "sampling_rate": 10.0},
    )

    with pytest.raises(ValueError, match="Unsupported Wandas xarray frame type"):
        wd.from_xarray(xr_data)


def test_to_xarray_uses_source_file_attr_from_metadata() -> None:
    frame = wd.ChannelFrame.from_numpy(np.array([[1.0, 2.0]]), sampling_rate=2.0)
    frame.metadata.source_file = "input.wav"

    restored = wd.from_xarray(frame.to_xarray())

    assert restored.metadata.source_file == "input.wav"


def test_from_xarray_ignores_malformed_channel_extra_metadata() -> None:
    frame = wd.ChannelFrame.from_numpy(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        sampling_rate=10.0,
        ch_labels=["front", "rear"],
    )
    xr_data = frame.to_xarray()
    xr_data.attrs["channel_extra"] = [{"role": "reference"}]

    restored = wd.from_xarray(xr_data)

    assert [ch.extra for ch in restored.channels] == [{}, {}]


def test_noct_frame_to_xarray_includes_optional_band_coordinate() -> None:
    from wandas.frames.noct import NOctFrame

    frame = NOctFrame(
        data=da_from_array(np.ones((1, 4)), chunks=(1, -1)),
        sampling_rate=48_000,
        fmin=100.0,
        fmax=1000.0,
        n=3,
        G=10,
        fr=1000,
    )
    bands = np.array([100.0, 125.0, 160.0, 200.0])
    setattr(frame, "bands", bands)

    xr_data = frame.to_xarray()

    npt.assert_allclose(xr_data.coords["band"].values, bands)


def test_coord_values_falls_back_for_missing_and_scalar_coords() -> None:
    from wandas.xarray_bridge import _coord_values

    xr_data = xr.DataArray(np.array([1.0, 2.0]), dims=("time",), coords={"unit": "Pa"})

    assert _coord_values(xr_data, "missing", ["fallback"]) == ["fallback"]
    assert _coord_values(xr_data, "unit", ["", ""]) == ["Pa", "Pa"]


def test_strict_chunk_policy_allows_unchunked_xarray_data() -> None:
    from wandas.processing.chunk_policy import validate_strict_chunks

    class DummyFrame:
        def to_xarray(self, *, copy: bool = False) -> xr.DataArray:
            return xr.DataArray(np.ones((1, 4)), dims=("channel", "time"))

    validate_strict_chunks(DummyFrame(), "lowpass_filter")


def test_roughness_frame_to_xarray_uses_bark_time_schema() -> None:
    from wandas.frames.roughness import RoughnessFrame

    bark_axis = np.linspace(0.5, 23.5, 47)
    frame = RoughnessFrame(
        data=da_from_array(np.ones((47, 3)), chunks=(47, -1)),
        sampling_rate=10.0,
        bark_axis=bark_axis,
        overlap=0.5,
    )

    xr_data = frame.to_xarray()

    assert xr_data.dims == ("bark", "time")
    npt.assert_allclose(xr_data.coords["bark"].values, bark_axis)
    npt.assert_allclose(xr_data.coords["time"].values, frame.time)


def test_multichannel_roughness_frame_to_xarray_uses_channel_bark_time_schema() -> None:
    from wandas.frames.roughness import RoughnessFrame

    bark_axis = np.linspace(0.5, 23.5, 47)
    frame = RoughnessFrame(
        data=da_from_array(np.ones((2, 47, 3)), chunks=(1, 47, -1)),
        sampling_rate=10.0,
        bark_axis=bark_axis,
        overlap=0.5,
        channel_metadata=[ChannelMetadata(label="left"), ChannelMetadata(label="right")],
    )

    xr_data = frame.to_xarray()

    assert xr_data.dims == ("channel", "bark", "time")
    npt.assert_array_equal(xr_data.coords["channel"].values, np.array(["left", "right"], dtype=object))
    npt.assert_allclose(xr_data.coords["bark"].values, bark_axis)
    npt.assert_allclose(xr_data.coords["time"].values, frame.time)


def test_dims_for_frame_falls_back_to_positional_dim_names() -> None:
    from types import SimpleNamespace

    from wandas.xarray_bridge import _dims_for_frame

    frame = SimpleNamespace(_data=np.ones((2, 3, 4)))

    assert _dims_for_frame(cast(Any, frame)) == ("dim_0", "dim_1", "dim_2")


def test_transform_methods_reject_split_time_chunks() -> None:
    frame = wd.ChannelFrame.from_numpy(np.arange(64.0).reshape(1, -1), sampling_rate=128.0)
    frame._data = frame._data.rechunk((1, 16))

    with pytest.raises(ValueError, match="Operation 'fft' requires contiguous chunks along time"):
        frame.fft(n_fft=64)

    with pytest.raises(ValueError, match="Operation 'stft' requires contiguous chunks along time"):
        frame.stft(n_fft=16, hop_length=4)

    with pytest.raises(ValueError, match="Operation 'welch' requires contiguous chunks along time"):
        frame.welch(n_fft=16)


def test_cross_channel_transform_rejects_split_time_chunks() -> None:
    frame = wd.ChannelFrame.from_numpy(np.vstack([np.arange(64.0), np.arange(64.0)]), sampling_rate=128.0)
    frame._data = frame._data.rechunk((1, 16))

    with pytest.raises(ValueError, match="Operation 'coherence' requires contiguous chunks along time"):
        frame.coherence(n_fft=16)


def test_normalize_axis_one_rejects_split_time_chunks() -> None:
    frame = wd.ChannelFrame.from_numpy(np.array([[1.0, 2.0, 3.0, 4.0]]), sampling_rate=4.0)
    frame._data = frame._data.rechunk((1, 2))

    with pytest.raises(ValueError, match="Operation 'normalize' requires contiguous chunks along time"):
        frame.normalize(axis=1)


def test_roughness_frame_round_trips_through_netcdf(tmp_path) -> None:
    from wandas.frames.roughness import RoughnessFrame

    bark_axis = np.linspace(0.5, 23.5, 47)
    original = RoughnessFrame(
        data=da_from_array(np.ones((47, 3)), chunks=(47, -1)),
        sampling_rate=10.0,
        bark_axis=bark_axis,
        overlap=0.5,
        label="roughness",
    )
    path = tmp_path / "roughness.nc"

    original.to_netcdf(path)
    restored = wd.open_netcdf(path)

    assert isinstance(restored, RoughnessFrame)
    assert restored.overlap == original.overlap
    npt.assert_allclose(restored.bark_axis, bark_axis)
    npt.assert_allclose(restored.compute(), original.compute())


def test_from_xarray_rejects_sliced_noct_band_axis() -> None:
    from wandas.frames.noct import NOctFrame

    frame = NOctFrame(
        data=da_from_array(np.ones((1, 11)), chunks=(1, -1)),
        sampling_rate=48_000,
        fmin=100.0,
        fmax=1000.0,
        n=3,
        G=10,
        fr=1000,
    )
    sliced = frame.to_xarray().isel(band=slice(0, 2))

    with pytest.raises(ValueError, match="band dimension length"):
        wd.from_xarray(sliced)


def test_complex_spectral_frame_round_trips_through_netcdf(tmp_path) -> None:
    original = SpectralFrame(
        data=da_from_array(np.array([[1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j]]), chunks=(1, -1)),
        sampling_rate=8.0,
        n_fft=4,
        window="hann",
        label="complex-spectrum",
    )
    path = tmp_path / "spectrum.nc"

    original.to_netcdf(path)
    restored = wd.open_netcdf(path)

    assert isinstance(restored, SpectralFrame)
    assert restored.n_fft == original.n_fft
    npt.assert_allclose(restored.compute(), original.compute())


def test_from_xarray_realigns_channel_extra_after_channel_selection() -> None:
    frame = wd.ChannelFrame(
        data=da_from_array(np.array([[1.0, 2.0], [3.0, 4.0]]), chunks=(1, -1)),
        sampling_rate=10.0,
        channel_metadata=[
            ChannelMetadata(label="left", extra={"side": "left"}),
            ChannelMetadata(label="right", extra={"side": "right"}),
        ],
    )
    selected = frame.to_xarray().sel(channel=["right"])

    restored = wd.from_xarray(selected)

    assert restored.labels == ["right"]
    assert [ch.extra for ch in restored.channels] == [{"side": "right"}]


def test_from_xarray_rejects_reversed_spectral_frequency_coordinate() -> None:
    frame = SpectralFrame(
        data=da_from_array(np.ones((1, 5), dtype=np.complex128), chunks=(1, -1)),
        sampling_rate=16.0,
        n_fft=8,
    )
    reversed_frequency = frame.to_xarray().isel(frequency=slice(None, None, -1))

    with pytest.raises(ValueError, match="frequency coordinate"):
        wd.from_xarray(reversed_frequency)


def test_from_xarray_decodes_netcdf_attrs_loaded_by_xarray(tmp_path) -> None:
    original = wd.ChannelFrame.from_numpy(
        np.array([[1.0, 2.0]]),
        sampling_rate=2.0,
        metadata={"source": "direct-xarray"},
        ch_labels=["mic"],
    )
    original.operation_history.append({"operation": "gain", "factor": 2.0})
    path = tmp_path / "direct_xarray.nc"
    original.to_netcdf(path)

    xr_data = xr.open_dataarray(path).load()
    try:
        restored = wd.from_xarray(xr_data)
    finally:
        xr_data.close()

    assert restored.metadata == original.metadata
    assert restored.operation_history == original.operation_history
    assert restored.labels == ["mic"]


def test_from_xarray_rejects_noncanonical_channel_time_coordinate() -> None:
    frame = wd.ChannelFrame.from_numpy(np.arange(6.0).reshape(1, -1), sampling_rate=2.0)
    decimated = frame.to_xarray().isel(time=slice(None, None, 2))

    with pytest.raises(ValueError, match="time coordinate"):
        wd.from_xarray(decimated)


def test_from_xarray_rejects_nonzero_start_channel_time_coordinate() -> None:
    frame = wd.ChannelFrame.from_numpy(np.arange(6.0).reshape(1, -1), sampling_rate=2.0)
    shifted = frame.to_xarray().isel(time=slice(1, None))

    with pytest.raises(ValueError, match="time coordinate"):
        wd.from_xarray(shifted)


def test_from_xarray_infers_roughness_frame_without_schema_type() -> None:
    from wandas.frames.roughness import RoughnessFrame

    bark_axis = np.linspace(0.5, 23.5, 47)
    original = RoughnessFrame(
        data=da_from_array(np.ones((47, 3)), chunks=(47, -1)),
        sampling_rate=10.0,
        bark_axis=bark_axis,
        overlap=0.5,
    )
    xr_data = original.to_xarray()
    xr_data.attrs.pop("wandas_frame_type")

    restored = wd.from_xarray(xr_data)

    assert isinstance(restored, RoughnessFrame)
    npt.assert_allclose(restored.compute(), original.compute())


def test_from_xarray_rejects_invalid_roughness_dims() -> None:
    xr_data = xr.DataArray(
        np.ones((47, 3)),
        dims=("bark", "sample"),
        attrs={"wandas_frame_type": "RoughnessFrame", "sampling_rate": 10.0, "metadata": {"overlap": 0.5}},
    )

    with pytest.raises(ValueError, match="Invalid dims for RoughnessFrame"):
        wd.from_xarray(xr_data)


def test_channel_extra_list_fallback_still_rejects_malformed_lengths() -> None:
    frame = wd.ChannelFrame.from_numpy(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        sampling_rate=10.0,
        ch_labels=["front", "rear"],
    )
    xr_data = frame.to_xarray()
    xr_data.attrs.pop("channel_extra_by_label")
    xr_data.attrs["channel_extra"] = [{"role": "reference"}]

    restored = wd.from_xarray(xr_data)

    assert [ch.extra for ch in restored.channels] == [{}, {}]


def test_channel_metadata_attrs_ignores_malformed_entries() -> None:
    from wandas.xarray_bridge import _channel_metadata_from_attrs

    restored = _channel_metadata_from_attrs(["bad", {"label": "mic", "extra": ["bad"]}])

    assert len(restored) == 1
    assert restored[0].label == "mic"
    assert restored[0].extra == {}


def test_axis_targets_dim_handles_none_string_and_nonmatching_axis() -> None:
    from wandas.processing.chunk_policy import _axis_targets_dim

    assert _axis_targets_dim(None, ("channel", "time"), "time")
    assert _axis_targets_dim("time", ("channel", "time"), "time")
    assert not _axis_targets_dim("frequency", ("channel", "time"), "time")


def test_noct_frame_to_xarray_uses_frequency_band_coordinate_by_default() -> None:
    from wandas.frames.noct import NOctFrame

    frame = NOctFrame(
        data=da_from_array(np.ones((1, 11)), chunks=(1, -1)),
        sampling_rate=48_000,
        fmin=100.0,
        fmax=1000.0,
        n=3,
        G=10,
        fr=1000,
    )

    xr_data = frame.to_xarray()

    npt.assert_allclose(xr_data.coords["band"].values, frame.freqs)


def test_from_xarray_rejects_reordered_noct_band_coordinate() -> None:
    from wandas.frames.noct import NOctFrame

    frame = NOctFrame(
        data=da_from_array(np.ones((1, 11)), chunks=(1, -1)),
        sampling_rate=48_000,
        fmin=100.0,
        fmax=1000.0,
        n=3,
        G=10,
        fr=1000,
    )
    reordered = frame.to_xarray().isel(band=slice(None, None, -1))

    with pytest.raises(ValueError, match="band coordinate"):
        wd.from_xarray(reordered)


def test_noct_synthesis_rejects_split_frequency_chunks() -> None:
    frame = SpectralFrame(
        data=da_from_array(np.ones((1, 5), dtype=np.complex128), chunks=(1, -1)),
        sampling_rate=48_000,
        n_fft=8,
    )
    frame._data = frame._data.rechunk((1, 2))

    with pytest.raises(ValueError, match="Operation 'noct_synthesis' requires contiguous chunks along frequency"):
        frame.noct_synthesis(fmin=100.0, fmax=1000.0)


def test_mono_roughness_frame_preserves_channel_metadata_through_netcdf(tmp_path) -> None:
    from wandas.frames.roughness import RoughnessFrame

    bark_axis = np.linspace(0.5, 23.5, 47)
    original = RoughnessFrame(
        data=da_from_array(np.ones((47, 3)), chunks=(47, -1)),
        sampling_rate=10.0,
        bark_axis=bark_axis,
        overlap=0.5,
        channel_metadata=[ChannelMetadata(label="rough", unit="asper", ref=2.0, extra={"source": "mono"})],
    )
    path = tmp_path / "mono_roughness_metadata.nc"

    original.to_netcdf(path)
    restored = wd.open_netcdf(path)

    assert restored.channels[0].label == "rough"
    assert restored.channels[0].unit == "asper"
    assert restored.channels[0].ref == 2.0
    assert restored.channels[0].extra == {"source": "mono"}
