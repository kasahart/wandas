from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr
from dask import array as da
from dask import delayed

from tests.frame_helpers import channel_first_values
from wandas import ChannelFrame
from wandas.core.base_frame import BaseFrame
from wandas.core.metadata import ChannelMetadata
from wandas.frames.noct import NOctFrame
from wandas.frames.roughness import RoughnessFrame
from wandas.frames.spectral import SpectralFrame
from wandas.frames.spectrogram import SpectrogramFrame
from wandas.processing.semantic import source_lineage


def _lazy_frame_with_counter(calls: list[str]) -> ChannelFrame:
    def build() -> np.ndarray:
        calls.append("computed")
        return np.array([[1.0, 2.0, 3.0, 4.0]])

    lazy_data = da.from_delayed(delayed(build)(), shape=(1, 4), dtype=float)
    return ChannelFrame(data=lazy_data, sampling_rate=4.0)


class AxisOnlyFrame(BaseFrame[np.ndarray]):
    _channel_axis = -3

    def plot(self, plot_type: str = "default", ax=None, **kwargs):
        raise NotImplementedError

    def _get_dataframe_index(self):
        return None


class SuffixOnlyFrame(BaseFrame[np.ndarray]):
    _xarray_dim_suffix = ("channel", "frequency", "time")

    def plot(self, plot_type: str = "default", ax=None, **kwargs):
        raise NotImplementedError

    def _get_dataframe_index(self):
        return None


def test_base_frame_channel_axis_drives_default_metadata_and_n_channels() -> None:
    data = da.ones((2, 3, 4), chunks=(1, 3, 4))

    frame = AxisOnlyFrame(data=data, sampling_rate=1.0)

    assert frame.n_channels == 2
    assert len(frame.channels) == 2


def test_n_channels_prefers_xarray_channel_dim_size() -> None:
    frame = SuffixOnlyFrame(
        data=da.ones((2, 3, 4), chunks=(1, 3, 4)),
        sampling_rate=1.0,
    )

    assert frame._xr.dims == ("channel", "frequency", "time")
    assert frame.n_channels == 2
    assert len(frame.channels) == 2


def test_base_frame_owns_xarray_dataarray() -> None:
    data = da.ones((2, 8), chunks=(2, 4))
    frame = ChannelFrame(
        data=data,
        sampling_rate=4.0,
        label="signal",
        channel_metadata=[
            ChannelMetadata(label="left", unit="Pa"),
            ChannelMetadata(label="right", unit="Pa"),
        ],
    )

    assert isinstance(frame._xr, xr.DataArray)
    assert frame._xr.dims == ("channel", "time")
    assert frame._data is frame._xr.data
    assert frame._data.chunks == ((1, 1), (8,))
    assert list(frame._xr.coords["channel"].values) == ["c0", "c1"]
    assert list(frame._xr.coords["channel_label"].values) == ["left", "right"]
    assert "time" not in frame._xr.coords


def test_target_frames_use_semantic_suffix_dims() -> None:
    channel = ChannelFrame.from_numpy(np.ones((2, 8)), sampling_rate=8.0)
    spectral = SpectralFrame(
        data=da.ones((2, 5), chunks=(1, 5)) + 0j,
        sampling_rate=8.0,
        n_fft=8,
    )
    spectrogram = SpectrogramFrame(
        data=da.ones((2, 5, 3), chunks=(1, 5, 3)) + 0j,
        sampling_rate=8.0,
        n_fft=8,
        hop_length=2,
    )
    noct = NOctFrame(
        data=da.ones((2, 4), chunks=(1, 4)),
        sampling_rate=8.0,
        fmin=20.0,
        fmax=2000.0,
    )

    assert channel._xr.dims == ("channel", "time")
    assert spectral._xr.dims == ("channel", "frequency")
    assert spectrogram._xr.dims == ("channel", "frequency", "time")
    assert noct._xr.dims == ("channel", "band")


def test_constructor_dimension_constraints_remain_unchanged() -> None:
    with pytest.raises(ValueError, match="Invalid data shape for ChannelFrame"):
        ChannelFrame(data=da.ones((1, 2, 3), chunks=(1, 2, 3)), sampling_rate=8.0)

    with pytest.raises(ValueError, match="Data must be 1-dimensional or 2-dimensional"):
        SpectralFrame(
            data=da.ones((1, 2, 3), chunks=(1, 2, 3)) + 0j,
            sampling_rate=8.0,
            n_fft=8,
        )

    with pytest.raises(ValueError, match="Invalid data dimensions"):
        SpectrogramFrame(
            data=da.ones((1, 2, 5, 3), chunks=(1, 2, 5, 3)) + 0j,
            sampling_rate=8.0,
            n_fft=8,
            hop_length=2,
        )

    # NOctFrame preserves existing behavior and accepts 3D inputs.
    # This is intentionally kept to avoid introducing new constraints in this PR.
    noct = NOctFrame(
        data=da.ones((1, 2, 3), chunks=(1, 2, 3)),
        sampling_rate=8.0,
        fmin=20.0,
        fmax=2000.0,
    )
    assert noct._xr.dims == ("dim_0", "dim_1", "dim_2")
    assert "channel" not in noct._xr.dims


def test_spectral_frame_adds_channel_coord_without_frequency_coord() -> None:
    frame = SpectralFrame(
        data=da.ones((2, 5), chunks=(1, 5)) + 0j,
        sampling_rate=8.0,
        n_fft=8,
        channel_metadata=[
            ChannelMetadata(label="left"),
            ChannelMetadata(label="right"),
        ],
    )

    assert list(frame._xr.coords["channel"].values) == ["c0", "c1"]
    assert list(frame._xr.coords["channel_label"].values) == ["left", "right"]
    assert "frequency" not in frame._xr.coords


def test_spectrogram_frame_adds_channel_coord_without_frequency_or_time_coords() -> None:
    frame = SpectrogramFrame(
        data=da.ones((2, 5, 3), chunks=(1, 5, 3)) + 0j,
        sampling_rate=8.0,
        n_fft=8,
        hop_length=2,
        channel_metadata=[
            ChannelMetadata(label="left"),
            ChannelMetadata(label="right"),
        ],
    )

    assert list(frame._xr.coords["channel"].values) == ["c0", "c1"]
    assert list(frame._xr.coords["channel_label"].values) == ["left", "right"]
    assert "frequency" not in frame._xr.coords
    assert "time" not in frame._xr.coords


def test_noct_frame_adds_channel_coord_without_band_coord() -> None:
    frame = NOctFrame(
        data=da.ones((2, 4), chunks=(1, 4)),
        sampling_rate=8.0,
        fmin=20.0,
        fmax=2000.0,
        channel_metadata=[
            ChannelMetadata(label="left"),
            ChannelMetadata(label="right"),
        ],
    )

    assert list(frame._xr.coords["channel"].values) == ["c0", "c1"]
    assert list(frame._xr.coords["channel_label"].values) == ["left", "right"]
    assert "band" not in frame._xr.coords


@pytest.mark.parametrize(
    ("frame_factory",),
    [
        pytest.param(
            lambda: SpectralFrame(
                data=da.ones((2, 5), chunks=(1, 5)) + 0j,
                sampling_rate=8.0,
                n_fft=8,
                channel_metadata=[ChannelMetadata(label="only-one")],
            ),
            id="spectral",
        ),
        pytest.param(
            lambda: SpectrogramFrame(
                data=da.ones((2, 5, 3), chunks=(1, 5, 3)) + 0j,
                sampling_rate=8.0,
                n_fft=8,
                hop_length=2,
                channel_metadata=[ChannelMetadata(label="only-one")],
            ),
            id="spectrogram",
        ),
        pytest.param(
            lambda: NOctFrame(
                data=da.ones((2, 4), chunks=(1, 4)),
                sampling_rate=8.0,
                fmin=20.0,
                fmax=2000.0,
                channel_metadata=[ChannelMetadata(label="only-one")],
            ),
            id="noct",
        ),
    ],
)
def test_channel_coord_uses_padded_metadata_for_target_frames(
    frame_factory: Callable[[], BaseFrame[np.ndarray]],
) -> None:
    frame = frame_factory()

    assert frame.labels == ["only-one", "ch1"]
    assert list(frame._xr.coords["channel"].values) == ["c0", "c1"]
    assert list(frame._xr.coords["channel_label"].values) == ["only-one", "ch1"]


def test_data_alias_is_read_only() -> None:
    frame = ChannelFrame.from_numpy(np.array([1.0, 2.0, 3.0]), sampling_rate=3.0)

    with pytest.raises(AttributeError):
        setattr(frame, "_data", da.zeros((1, 3), chunks=(1, -1)))


def test_replace_data_preserves_xarray_attrs_backed_frame_state() -> None:
    frame = ChannelFrame(
        data=da.from_array(np.array([[1.0, 2.0, 3.0]]), chunks=(1, -1)),
        sampling_rate=3.0,
        label="original",
        metadata={"source": "test", "nested": {"x": 1}},
        channel_metadata=[ChannelMetadata(label="mic")],
        lineage=source_lineage([{"operation": "wandas.audio.normalize", "version": 1, "params": {}}]),
    )
    frame.channels[0].unit = "Pa"
    frame.channels[0].ref = 0.25
    frame.channels[0].extra = {"gain": 12}
    replacement = da.full((1, 4), 2.0, chunks=(1, -1))

    frame._replace_data(replacement)

    assert frame._data is frame._xr.data
    assert frame._data.shape == (1, 4)
    assert frame.sampling_rate == 3.0
    assert frame.label == "original"
    assert frame._xr.attrs["sampling_rate"] == 3.0
    assert frame._xr.attrs["label"] == "original"
    assert frame._xr.attrs["metadata"] == {"source": "test", "nested": {"x": 1}}
    assert frame.metadata == {"source": "test", "nested": {"x": 1}}
    assert frame.operation_history[0]["operation"] == "wandas.audio.normalize"
    assert frame.channels[0].id == "c0"
    assert frame.channels[0].label == "mic"
    assert frame.channels[0].unit == "Pa"
    assert frame.channels[0].ref == 0.25
    assert frame.channels[0].extra == {"gain": 12}


def test_internal_xarray_attrs_are_frame_state_source_of_truth() -> None:
    frame = ChannelFrame.from_numpy(
        np.array([[1.0, 2.0]]),
        sampling_rate=2.0,
        label="owned-by-attrs",
        metadata={"gain": 1.5},
    )

    frame._xr.attrs["label"] = "authoritative"
    frame._xr.attrs["metadata"] = {"gain": 999}

    assert frame.label == "authoritative"
    assert frame.metadata["gain"] == 999


def test_base_frame_keeps_roughness_mono_dims_neutral() -> None:
    bark_axis = np.linspace(0.5, 23.5, 47)
    data = da.ones((47, 5), chunks=(47, 5))

    frame = RoughnessFrame(
        data=data,
        sampling_rate=10.0,
        bark_axis=bark_axis,
        overlap=0.5,
        channel_metadata=[ChannelMetadata(label="mono", unit="asper")],
    )

    assert frame.n_channels == 1
    assert frame._xr.dims == ("dim_0", "dim_1")
    assert "channel" not in frame._xr.coords


def test_base_frame_does_not_invent_spectrogram_time_coords() -> None:
    data = da.ones((513, 6), chunks=(513, 3)) + 0j

    frame = SpectrogramFrame(
        data=data,
        sampling_rate=48_000.0,
        n_fft=1024,
        hop_length=256,
    )

    assert frame._xr.dims == ("channel", "frequency", "time")
    assert "time" not in frame._xr.coords
    assert "frequency" not in frame._xr.coords


def test_default_spectrogram_channel_metadata_matches_n_channels() -> None:
    data = da.ones((2, 513, 6), chunks=(1, 513, 3)) + 0j

    frame = SpectrogramFrame(
        data=data,
        sampling_rate=48_000.0,
        n_fft=1024,
        hop_length=256,
    )

    assert frame.n_channels == 2
    assert len(frame.channels) == 2


def test_default_roughness_mono_channel_metadata_matches_n_channels() -> None:
    bark_axis = np.linspace(0.5, 23.5, 47)

    frame = RoughnessFrame(
        data=da.ones((47, 5), chunks=(47, 5)),
        sampling_rate=10.0,
        bark_axis=bark_axis,
        overlap=0.5,
    )

    assert frame.n_channels == 1
    assert len(frame.channels) == 1


def test_default_roughness_stereo_channel_metadata_matches_n_channels() -> None:
    bark_axis = np.linspace(0.5, 23.5, 47)

    frame = RoughnessFrame(
        data=da.ones((2, 47, 5), chunks=(1, 47, 5)),
        sampling_rate=10.0,
        bark_axis=bark_axis,
        overlap=0.5,
    )

    assert frame.n_channels == 2
    assert len(frame.channels) == 2


def test_channel_frame_refreshes_xarray_channel_coord_after_label_update() -> None:
    frame = ChannelFrame.from_numpy(
        np.ones((2, 4)),
        sampling_rate=4.0,
        ch_labels=["L", "R"],
    )

    assert list(frame._xr.coords["channel"].values) == ["c0", "c1"]
    assert list(frame._xr.coords["channel_label"].values) == ["L", "R"]


def test_large_lazy_channel_frame_does_not_create_internal_time_coord() -> None:
    data = da.ones((2, 1_000_000), chunks=(1, 100_000))

    frame = ChannelFrame(data=data, sampling_rate=48_000.0)

    assert frame._xr.dims == ("channel", "time")
    assert "channel" in frame._xr.coords
    assert "time" not in frame._xr.coords


def test_channel_frame_pads_channel_metadata_and_creates_channel_coord() -> None:
    data = da.ones((2, 8), chunks=(1, 8))

    frame = ChannelFrame(
        data=data,
        sampling_rate=8.0,
        channel_metadata=[{"label": "sig", "unit": "", "extra": {}}],
    )

    assert frame._xr.dims == ("channel", "time")
    assert frame.labels == ["sig", "ch1"]
    assert list(frame._xr.coords["channel"].values) == ["c0", "c1"]
    assert list(frame._xr.coords["channel_label"].values) == ["sig", "ch1"]


def test_rename_channels_returns_new_xarray_channel_coord() -> None:
    frame = ChannelFrame.from_numpy(
        np.ones((2, 4)),
        sampling_rate=4.0,
        ch_labels=["L", "R"],
    )

    result = frame.rename_channels({"L": "Left"})

    assert result is not frame
    assert frame.labels == ["L", "R"]
    assert result.labels == ["Left", "R"]
    assert list(result._xr.coords["channel"].values) == ["c0", "c1"]
    assert list(result._xr.coords["channel_label"].values) == ["Left", "R"]


def test_add_channel_returns_new_xarray_without_compute() -> None:
    calls: list[str] = []

    def build() -> list[list[float]]:
        calls.append("computed")
        return [[1.0, 2.0, 3.0]]

    lazy_data = da.from_delayed(
        delayed(build)(),
        shape=(1, 3),
        dtype=float,
    )
    frame = ChannelFrame(data=lazy_data, sampling_rate=3.0)

    result = frame.add_channel(np.array([4.0, 5.0, 6.0]), label="extra")

    assert result is not frame
    assert calls == []
    assert frame._data.shape == (1, 3)
    assert result._data is result._xr.data
    assert result._data.shape == (2, 3)
    assert list(result._xr.coords["channel"].values) == ["c0", "c1"]
    assert list(result._xr.coords["channel_label"].values) == ["ch0", "extra"]


def test_construction_and_private_storage_access_do_not_compute() -> None:
    calls: list[str] = []
    frame = _lazy_frame_with_counter(calls)

    _ = frame._data
    stored = frame._xr

    assert calls == []
    assert stored.data is frame._data


def test_selection_and_operation_do_not_compute_until_compute() -> None:
    calls: list[str] = []
    frame = _lazy_frame_with_counter(calls)

    selected = frame.get_channel(0)
    normalized = selected.normalize()

    assert calls == []

    result = channel_first_values(normalized)

    assert calls == ["computed"]
    assert result.shape == (1, 4)


def test_transform_methods_remain_lazy() -> None:
    calls: list[str] = []
    frame = _lazy_frame_with_counter(calls)

    spectrum = frame.fft()
    spectrogram = frame.stft(n_fft=4, hop_length=2)

    assert calls == []
    assert spectrum._data is spectrum._xr.data
    assert spectrogram._data is spectrogram._xr.data


def test_remove_channel_returns_new_xarray_without_compute() -> None:
    calls: list[str] = []

    def build() -> list[list[float]]:
        calls.append("computed")
        return [[1.0, 2.0], [3.0, 4.0]]

    lazy_data = da.from_delayed(
        delayed(build)(),
        shape=(2, 2),
        dtype=float,
    )
    frame = ChannelFrame(
        data=lazy_data,
        sampling_rate=2.0,
        channel_metadata=[
            ChannelMetadata(label="left"),
            ChannelMetadata(label="right"),
        ],
    )

    result = frame.remove_channel("left")

    assert result is not frame
    assert calls == []
    assert frame._data.shape == (2, 2)
    assert result._data is result._xr.data
    assert result._data.shape == (1, 2)
    assert list(result._xr.coords["channel"].values) == ["c1"]
    assert list(result._xr.coords["channel_label"].values) == ["right"]


def test_private_xarray_storage_retains_frame_attrs() -> None:
    frame = ChannelFrame(
        data=da.from_array(np.array([[1.0, 2.0]]), chunks=(1, -1)),
        sampling_rate=2.0,
        label="exported",
        metadata={"source": "unit-test"},
        lineage=source_lineage([{"operation": "wandas.audio.normalize", "version": 1, "params": {}}]),
    )

    assert isinstance(frame._xr, xr.DataArray)
    assert frame._xr.data is frame._data
    assert frame._xr.attrs["sampling_rate"] == 2.0
    assert frame._xr.attrs["label"] == "exported"
    assert frame._xr.attrs["metadata"] == {"source": "unit-test"}
    assert "operation_history" not in frame._xr.attrs


def test_private_storage_uses_attrs_backed_label() -> None:
    frame = ChannelFrame.from_numpy(
        np.array([1.0, 2.0]),
        sampling_rate=2.0,
        label="original",
    )
    frame._xr.attrs["label"] = "mutated"

    assert frame.label == "mutated"
    assert frame._xr.attrs["label"] == "mutated"


def test_metadata_setter_deep_copies_input_dict() -> None:
    metadata = {"nested": {"x": 1}, "tags": ["raw"]}
    frame = ChannelFrame.from_numpy(
        np.array([1.0, 2.0]),
        sampling_rate=2.0,
        metadata=metadata,
    )

    metadata["nested"]["x"] = 99
    metadata["tags"].append("mutated")

    assert frame.metadata == {"nested": {"x": 1}, "tags": ["raw"]}


def test_private_storage_omits_operation_history() -> None:
    frame = ChannelFrame(
        data=da.from_array(np.array([[1.0, 2.0]]), chunks=(1, -1)),
        sampling_rate=2.0,
        lineage=source_lineage([{"operation": "wandas.audio.normalize", "version": 1, "params": {}}]),
    )

    assert "operation_history" not in frame._xr.attrs
    assert frame.operation_history[0]["operation"] == "wandas.audio.normalize"


def test_frame_state_properties_are_backed_by_xarray_attrs() -> None:
    frame = ChannelFrame(
        data=da.from_array(np.array([[1.0, 2.0, 3.0]]), chunks=(1, -1)),
        sampling_rate=3.0,
        label="stateful",
        metadata={"owner": "attrs"},
        lineage=source_lineage([{"operation": "wandas.audio.normalize", "version": 1, "params": {}}]),
    )

    assert frame._xr.attrs["sampling_rate"] == 3.0
    assert frame._xr.attrs["label"] == "stateful"
    assert frame._xr.attrs["metadata"] == {"owner": "attrs"}
    assert "operation_history" not in frame._xr.attrs

    frame._xr.attrs["sampling_rate"] = 8.0
    frame._xr.attrs["label"] = "from-attrs"
    frame._xr.attrs["metadata"] = {"owner": "mutated"}

    assert frame.sampling_rate == 8.0
    assert frame.label == "from-attrs"
    assert frame.metadata == {"owner": "mutated"}
    assert frame.operation_history[0]["operation"] == "wandas.audio.normalize"


def test_frame_state_property_setters_update_xarray_attrs() -> None:
    frame = ChannelFrame.from_numpy(np.array([[1.0, 2.0, 3.0]]), sampling_rate=3.0)

    frame.sampling_rate = 6
    frame.label = "updated"
    frame.metadata = {"nested": {"x": 1}}

    assert frame._xr.attrs["sampling_rate"] == 6.0
    assert frame._xr.attrs["label"] == "updated"
    assert frame._xr.name == "updated"
    assert frame._xr.attrs["metadata"] == {"nested": {"x": 1}}
    assert "operation_history" not in frame._xr.attrs


def test_frame_state_property_setters_validate_inputs() -> None:
    frame = ChannelFrame.from_numpy(np.array([[1.0, 2.0, 3.0]]), sampling_rate=3.0)

    with pytest.raises(ValueError, match="Invalid sampling_rate"):
        frame.sampling_rate = 0

    with pytest.raises(TypeError, match="Frame label must be a string or None"):
        frame.label = 123  # ty: ignore[invalid-assignment]

    with pytest.raises(TypeError, match="Frame metadata must be a mapping"):
        frame.metadata = "invalid"  # ty: ignore[invalid-assignment]

    with pytest.raises(AttributeError):
        frame.operation_history = {"operation": "bad"}  # ty: ignore[invalid-assignment]


def test_removed_backend_api_is_absent() -> None:
    frame = ChannelFrame.from_numpy(np.array([1.0, 2.0]), sampling_rate=2.0)

    representatives = (
        frame,
        frame.fft(n_fft=2),
        frame.stft(n_fft=2, hop_length=1),
        frame.cepstrum(n_fft=2),
        frame.stft(n_fft=2, hop_length=1).cepstrum(),
    )
    for representative in representatives:
        for name in ("compute", "persist", "xr", "to_xarray"):
            assert not hasattr(representative, name)
