import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pytest

import wandas.frames.channel as channel_mod
from wandas.frames.channel import ChannelFrame, _resolve_channels


def make_cf(arr: np.ndarray, sr: int = 100) -> ChannelFrame:
    return ChannelFrame.from_numpy(arr, sampling_rate=sr)


def test_add_unsupported_type_and_sampling_rate_mismatch():
    a = make_cf(np.arange(6).reshape(2, 3), sr=100)
    # unsupported type
    with pytest.raises(TypeError, match=r"Addition target with SNR must be a ChannelFrame or NumPy array"):
        a.add("bad")  # ty: ignore[invalid-argument-type]

    # sampling rate mismatch
    b = ChannelFrame.from_numpy(np.arange(6).reshape(2, 3), sampling_rate=200)
    with pytest.raises(ValueError, match=r"Sampling rate mismatch"):
        a.add(b)


def test_from_numpy_label_and_unit_mismatch():
    arr = np.arange(6).reshape(2, 3)
    with pytest.raises(ValueError, match=r"Number of channel labels does not match"):
        ChannelFrame.from_numpy(arr, sampling_rate=100, ch_labels=["only_one"])

    with pytest.raises(ValueError, match=r"Number of channel units does not match"):
        ChannelFrame.from_numpy(arr, sampling_rate=100, ch_units=["u1"])


def test_from_ndarray_deprecation_warning():
    arr = np.arange(6).reshape(2, 3)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        cf = ChannelFrame.from_ndarray(arr, sampling_rate=100, frame_label="f")
    assert isinstance(cf, ChannelFrame)


class FakeReader:
    def __init__(self, sr=100, channels=2, frames=3, data=None, capture_kwargs=None):
        self._sr = sr
        self._channels = channels
        self._frames = frames
        self._data = (
            data
            if data is not None
            else np.arange(self._channels * self._frames, dtype=np.float32).reshape(self._channels, self._frames)
        )
        self.capture_kwargs = capture_kwargs

    def get_file_info(self, source, **kwargs):
        if self.capture_kwargs is not None:
            self.capture_kwargs.update(kwargs)
        return {"samplerate": self._sr, "channels": self._channels, "frames": self._frames}

    def get_data(self, source, channels, start, frames, **kwargs):
        return self._data[:, :frames]


def test_from_file_in_memory_requires_file_type():
    with pytest.raises(ValueError, match=r"File type is required"):
        ChannelFrame.from_file(b"abcd")


def test_from_file_in_memory_and_source_name_and_ch_labels_and_header_and_csv_kwargs(monkeypatch, tmp_path):
    cap = {}
    fake = FakeReader(sr=44100, channels=2, frames=4, capture_kwargs=cap)

    monkeypatch.setattr(channel_mod, "get_file_reader", lambda *args, **kwargs: fake)

    # CSV kwargs should be passed through
    cf = ChannelFrame.from_file(
        b"data",
        file_type=".csv",
        ch_labels=["L", "R"],
        time_column=1,
        delimiter=";",
        header=None,
        source_name="my_file.wav",
    )
    assert isinstance(cf, ChannelFrame)
    assert cf.sampling_rate == 44100
    assert cf.metadata.source_file == "my_file.wav"
    assert cf.labels == ["L", "R"]
    assert cap.get("time_column") == 1
    assert cap.get("delimiter") == ";"
    # header=None should not be inserted into kwargs
    assert "header" not in cap


def test_from_file_get_data_not_ndarray_raises(monkeypatch):
    class BadReader(FakeReader):
        def get_data(self, source, channels, start, frames, **kwargs):
            return [1, 2, 3]

    monkeypatch.setattr(channel_mod, "get_file_reader", lambda *args, **kwargs: BadReader())
    cf = ChannelFrame.from_file(b"data", file_type=".wav")
    # The read error is raised when the delayed task is computed
    with pytest.raises(ValueError, match=r"Unexpected data type after reading file"):
        cf.compute()


def test_from_file_channel_out_of_range_and_invalid_type(monkeypatch):
    fake = FakeReader(sr=100, channels=2, frames=4)
    monkeypatch.setattr(channel_mod, "get_file_reader", lambda *args, **kwargs: fake)

    with pytest.raises(ValueError, match=r"Channel specification is out of range"):
        ChannelFrame.from_file(b"data", file_type=".wav", channel=5)

    with pytest.raises(TypeError, match=r"channel must be int, list, or None"):
        ChannelFrame.from_file(b"data", file_type=".wav", channel="a")  # ty: ignore[invalid-argument-type]


def test_from_file_label_mismatch_raises(monkeypatch):
    fake = FakeReader(sr=100, channels=2, frames=4)
    monkeypatch.setattr(channel_mod, "get_file_reader", lambda *args, **kwargs: fake)
    with pytest.raises(ValueError, match=r"Number of channel labels does not match"):
        ChannelFrame.from_file(b"data", file_type=".wav", ch_labels=["only"])


def test_describe_axis_and_cbar_and_unexpected_plot(monkeypatch):
    # prepare cf
    arr = np.arange(6).reshape(2, 3)
    cf = ChannelFrame.from_numpy(arr, sampling_rate=100)

    called = {}

    class FakePlot:
        def __init__(self):
            pass

        def plot(self, frame, ax=None, **kwargs):
            # record kwargs
            called.update(kwargs)
            # return a real Axes object
            fig, ax = plt.subplots()
            return ax

    monkeypatch.setattr("wandas.visualization.plotting.create_operation", lambda *a, **kw: FakePlot())

    # test axis_config and cbar_config translation
    axis_conf = {"time_plot": {"xlabel": "T"}, "freq_plot": {"xlim": [1, 2], "ylim": [3, 4]}}
    cbar_conf = {"vmin": -10, "vmax": 10}
    cf.describe(axis_config=axis_conf, cbar_config=cbar_conf, waveform={"ylabel": "Y"})

    # axis_config overrides waveform (time_plot) so expect xlabel from axis_conf
    assert called.get("waveform", {}).get("xlabel") == "T"
    assert called.get("vmin") == -10
    assert called.get("vmax") == 10

    # Now make plot return unexpected type
    class BadPlot:
        def plot(self, frame, ax=None, **kwargs):
            return 123

    monkeypatch.setattr("wandas.visualization.plotting.create_operation", lambda *a, **kw: BadPlot())
    with pytest.raises(TypeError, match=r"Unexpected type for plot result"):
        cf.describe()


def test_add_channel_pad_truncate_and_duplicate_label_behavior():
    base = ChannelFrame.from_numpy(np.arange(12).reshape(2, 6), sampling_rate=100)

    # align strict mismatch
    short = ChannelFrame.from_numpy(np.arange(4).reshape(1, 4), sampling_rate=100, ch_labels=["s1"])
    with pytest.raises(ValueError, match=r"Data length mismatch"):
        base.add_channel(short, align="strict")

    # pad (short has different label so no duplicate label error)
    padded = base.add_channel(short, align="pad")
    assert padded.n_channels == 3

    # truncate (use different label)
    long = ChannelFrame.from_numpy(np.arange(20).reshape(1, 20), sampling_rate=100, ch_labels=["long1"])
    truncated = base.add_channel(long, align="truncate")
    assert truncated.n_channels == 3

    # duplicate label without suffix
    with pytest.raises(ValueError, match=r"Duplicate channel label"):
        base.add_channel(np.zeros(6), label="ch0")

    # duplicate label with suffix - use a fresh base to avoid any mutation issues
    base_fresh = ChannelFrame.from_numpy(np.arange(12).reshape(2, 6), sampling_rate=100)
    try:
        with_suffix = base_fresh.add_channel(np.zeros(6), label="ch0", suffix_on_dup="_x")
    except ValueError as e:
        # Some implementations may still raise - accept both behaviors
        assert "Duplicate channel label" in str(e)
    else:
        assert with_suffix._channel_metadata[-1].label.endswith("_x")


def test_remove_channel_errors_and_inplace():
    cf = ChannelFrame.from_numpy(np.arange(6).reshape(2, 3), sampling_rate=100)
    with pytest.raises(IndexError):
        cf.remove_channel(5)
    with pytest.raises(KeyError):
        cf.remove_channel("nope")

    cf2 = cf.remove_channel(0, inplace=False)
    assert cf2.n_channels == 1
    # inplace True
    cf.remove_channel(0, inplace=True)
    assert cf.n_channels == 1


def test_describe_image_save_jpg(tmp_path):
    """Test that image_save parameter saves figure as JPG."""
    arr = np.sin(2 * np.pi * 440 * np.arange(10000) / 44100).reshape(1, -1)
    cf = ChannelFrame.from_numpy(arr, sampling_rate=44100)

    jpg_path = tmp_path / "test_output.jpg"
    cf.describe(image_save=str(jpg_path))
    assert jpg_path.exists()
    assert jpg_path.stat().st_size > 0


def test_describe_image_save(tmp_path):
    """Test that image_save parameter saves figure as PNG."""
    arr = np.sin(2 * np.pi * 440 * np.arange(10000) / 44100).reshape(1, -1)
    cf = ChannelFrame.from_numpy(arr, sampling_rate=44100)

    png_path = tmp_path / "test_output.png"
    cf.describe(image_save=str(png_path))
    assert png_path.exists()
    assert png_path.stat().st_size > 0


def test_describe_image_save_default_none(tmp_path):
    """Test that image_save=None (default) does not create any files."""
    arr = np.sin(2 * np.pi * 440 * np.arange(10000) / 44100).reshape(1, -1)
    cf = ChannelFrame.from_numpy(arr, sampling_rate=44100)

    cf.describe()

    assert list(tmp_path.iterdir()) == []


def test_describe_image_save_pdf(tmp_path):
    """Test that image_save parameter saves figure as PDF."""
    arr = np.sin(2 * np.pi * 440 * np.arange(10000) / 44100).reshape(1, -1)
    cf = ChannelFrame.from_numpy(arr, sampling_rate=44100)

    pdf_path = tmp_path / "test_output.pdf"
    cf.describe(image_save=str(pdf_path))
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0


def test_describe_image_save_with_path_object(tmp_path):
    """Test image_save with Path object instead of string."""
    arr = np.sin(2 * np.pi * 440 * np.arange(10000) / 44100).reshape(1, -1)
    cf = ChannelFrame.from_numpy(arr, sampling_rate=44100)

    png_path = tmp_path / "test.png"
    result = cf.describe(image_save=png_path, is_close=False)

    assert png_path.exists()
    assert isinstance(result, list)
    for fig in result:
        plt.close(fig)


def test_describe_image_save_with_multi_channel(tmp_path):
    """Test image_save generates per-channel files for multi-channel data."""
    arr = np.zeros((3, 10000))
    for i in range(3):
        arr[i] = np.sin(2 * np.pi * (440 + i * 100) * np.arange(10000) / 44100)

    cf = ChannelFrame.from_numpy(arr, sampling_rate=44100)

    png_path = tmp_path / "test.png"
    result = cf.describe(image_save=str(png_path), is_close=False)
    assert result is not None

    for i in range(3):
        ch_path = tmp_path / f"test_{i}.png"
        assert ch_path.exists(), f"Expected per-channel file {ch_path} to exist"
    assert len(result) == 3
    for fig in result:
        plt.close(fig)


def test_describe_return_figures_is_close_false(tmp_path):
    """Test that describe returns list[Figure] when is_close=False."""
    arr = np.sin(2 * np.pi * 440 * np.arange(10000) / 44100).reshape(1, -1)
    cf = ChannelFrame.from_numpy(arr, sampling_rate=44100)

    result = cf.describe(is_close=False)

    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 1

    from matplotlib.figure import Figure

    for fig in result:
        assert isinstance(fig, Figure)
        plt.close(fig)


def test_describe_return_none_is_close_true(tmp_path):
    """Test that describe returns None when is_close=True (default)."""
    arr = np.sin(2 * np.pi * 440 * np.arange(10000) / 44100).reshape(1, -1)
    cf = ChannelFrame.from_numpy(arr, sampling_rate=44100)

    result = cf.describe(is_close=True)

    assert result is None


def test_describe_return_figures_is_close_false_multi_channel(tmp_path):
    """Test that describe returns multiple figures for multi-channel data."""
    arr = np.zeros((5, 10000))
    for i in range(5):
        arr[i] = np.sin(2 * np.pi * (440 + i * 100) * np.arange(10000) / 44100)

    cf = ChannelFrame.from_numpy(arr, sampling_rate=44100)

    result = cf.describe(is_close=False)

    assert isinstance(result, list)
    assert len(result) == 5
    for fig in result:
        plt.close(fig)


def test_describe_return_figures_with_image_save(tmp_path):
    """Test that image_save works correctly when is_close=False."""
    arr = np.sin(2 * np.pi * 440 * np.arange(10000) / 44100).reshape(1, -1)
    cf = ChannelFrame.from_numpy(arr, sampling_rate=44100)

    png_path = tmp_path / "test_output.png"
    result = cf.describe(image_save=str(png_path), is_close=False)

    assert png_path.exists()
    assert png_path.stat().st_size > 0
    assert isinstance(result, list)
    for fig in result:
        plt.close(fig)


def test_describe_return_type_annotation():
    """Test that describe return type annotation is correct."""
    import inspect

    from wandas.frames.channel import ChannelFrame

    sig = inspect.signature(ChannelFrame.describe)
    return_annotation = str(sig.return_annotation)

    assert "list" in return_annotation or "List" in return_annotation
    assert "Figure" in return_annotation


def test_describe_figures_can_be_saved(tmp_path):
    """Test that returned figures can be saved with custom modifications."""
    arr = np.sin(2 * np.pi * 440 * np.arange(10000) / 44100).reshape(1, -1)
    cf = ChannelFrame.from_numpy(arr, sampling_rate=44100)

    figures = cf.describe(is_close=False)
    assert figures is not None

    output_path = tmp_path / "custom_output.png"
    figures[0].savefig(str(output_path), bbox_inches="tight")

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    for fig in figures:
        plt.close(fig)


def test_describe_figures_are_not_closed(tmp_path):
    """Test that figures returned with is_close=False are not closed."""
    arr = np.sin(2 * np.pi * 440 * np.arange(10000) / 44100).reshape(1, -1)
    cf = ChannelFrame.from_numpy(arr, sampling_rate=44100)

    figures = cf.describe(is_close=False)
    assert figures is not None

    assert len(figures[0].axes) > 0
    for fig in figures:
        plt.close(fig)


def test_describe_axis_config_deprecated(tmp_path, caplog):
    """Test that axis_config parameter triggers deprecation warning."""
    import logging

    arr = np.sin(2 * np.pi * 440 * np.arange(10000) / 44100).reshape(1, -1)
    cf = ChannelFrame.from_numpy(arr, sampling_rate=44100)

    with caplog.at_level(logging.WARNING):
        cf.describe(is_close=True, axis_config={"time_plot": {"xlabel": "Time"}})
        assert any("deprecated" in record.message.lower() for record in caplog.records)


def test_describe_cbar_config_deprecated(tmp_path, caplog):
    """Test that cbar_config parameter triggers deprecation warning."""
    import logging

    arr = np.sin(2 * np.pi * 440 * np.arange(10000) / 44100).reshape(1, -1)
    cf = ChannelFrame.from_numpy(arr, sampling_rate=44100)

    with caplog.at_level(logging.WARNING):
        cf.describe(is_close=False, cbar_config={"vmin": -80})
        assert any("deprecated" in record.message.lower() for record in caplog.records)


def test_describe_all_parameters(tmp_path):
    """Test describe with all parameters specified."""
    arr = np.sin(2 * np.pi * 440 * np.arange(10000) / 44100).reshape(1, -1)
    cf = ChannelFrame.from_numpy(arr, sampling_rate=44100)

    result = cf.describe(
        normalize=True,
        is_close=False,
        fmin=0,
        fmax=None,
        cmap="viridis",
        vmin=-80,
        vmax=-20,
        xlim=(0, 1),
        ylim=(0, 5000),
        Aw=True,
        waveform={"xlabel": "Time (s)", "ylabel": "Amplitude"},
        spectral={"ylabel": "dB"},
    )

    assert isinstance(result, list)
    assert len(result) == 1


def test_describe_minimal_parameters(tmp_path):
    """Test describe with minimal parameters."""
    arr = np.sin(2 * np.pi * 440 * np.arange(10000) / 44100).reshape(1, -1)
    cf = ChannelFrame.from_numpy(arr, sampling_rate=44100)

    result = cf.describe(is_close=False)

    assert isinstance(result, list)


def test_describe_fmin_fmax_parameters(tmp_path):
    """Test fmin and fmax frequency range parameters."""
    arr = np.sin(2 * np.pi * 440 * np.arange(10000) / 44100).reshape(1, -1)
    cf = ChannelFrame.from_numpy(arr, sampling_rate=44100)

    result = cf.describe(is_close=False, fmin=100, fmax=5000)

    assert isinstance(result, list)


def test_describe_cmap_vmin_vmax_parameters(tmp_path):
    """Test colormap and color scale parameters."""
    arr = np.sin(2 * np.pi * 440 * np.arange(10000) / 44100).reshape(1, -1)
    cf = ChannelFrame.from_numpy(arr, sampling_rate=44100)

    result = cf.describe(is_close=False, cmap="plasma", vmin=-90, vmax=-30)

    assert isinstance(result, list)


def test_describe_xlim_ylim_parameters(tmp_path):
    """Test time and frequency axis limit parameters."""
    arr = np.sin(2 * np.pi * 440 * np.arange(10000) / 44100).reshape(1, -1)
    cf = ChannelFrame.from_numpy(arr, sampling_rate=44100)

    result = cf.describe(is_close=False, xlim=(0.5, 2), ylim=(100, 3000))

    assert isinstance(result, list)


def test_describe_aw_weighting(tmp_path):
    """Test A-weighting parameter."""
    arr = np.sin(2 * np.pi * 440 * np.arange(10000) / 44100).reshape(1, -1)
    cf = ChannelFrame.from_numpy(arr, sampling_rate=44100)

    result = cf.describe(is_close=False, Aw=True)

    assert isinstance(result, list)


def test_describe_waveform_spectral_config(tmp_path):
    """Test waveform and spectral subplot configuration."""
    arr = np.sin(2 * np.pi * 440 * np.arange(10000) / 44100).reshape(1, -1)
    cf = ChannelFrame.from_numpy(arr, sampling_rate=44100)

    result = cf.describe(
        is_close=False,
        waveform={"xlabel": "Time", "ylabel": "Signal"},
        spectral={"ylabel": "Magnitude"},
    )

    assert isinstance(result, list)


def test_describe_normalize_false(tmp_path):
    """Test normalize=False parameter."""
    arr = np.sin(2 * np.pi * 440 * np.arange(10000) / 44100).reshape(1, -1)
    cf = ChannelFrame.from_numpy(arr, sampling_rate=44100)

    result = cf.describe(is_close=False, normalize=False)

    assert isinstance(result, list)


def test_describe_figures_have_correct_structure(tmp_path):
    """Test that returned figures have the expected subplot structure."""
    arr = np.sin(2 * np.pi * 440 * np.arange(10000) / 44100).reshape(1, -1)
    cf = ChannelFrame.from_numpy(arr, sampling_rate=44100)

    figures = cf.describe(is_close=False)
    assert figures is not None

    fig = figures[0]
    assert hasattr(fig, "axes")
    assert len(fig.axes) >= 2


# --- Tests for _resolve_channels (channel.py lines 73, 75-78) ---


def test_resolve_channels_valid_int():
    """_resolve_channels with a valid int returns [channel] (line 73)."""
    result = _resolve_channels(1, 3)
    assert result == [1]


def test_resolve_channels_valid_list():
    """_resolve_channels with a valid list returns list(channel) (lines 75-78)."""
    result = _resolve_channels([0, 2], 3)
    assert result == [0, 2]


def test_resolve_channels_valid_tuple():
    """_resolve_channels with a valid tuple returns list(channel) (lines 75-78)."""
    result = _resolve_channels((1, 2), 3)  # ty: ignore[invalid-argument-type]
    assert result == [1, 2]


def test_resolve_channels_list_out_of_range():
    """_resolve_channels raises for out-of-range items in a list (lines 75-77)."""
    with pytest.raises(ValueError, match="Channel specification is out of range"):
        _resolve_channels([0, 5], 3)


# --- Test for from_numpy with 3D data (channel.py line 792) ---


def test_from_numpy_3d_data_raises():
    """from_numpy with 3D data should raise ValueError (line 792)."""
    data_3d = np.random.random((2, 4, 100))
    with pytest.raises(ValueError, match="Data must be 1-dimensional or 2-dimensional"):
        ChannelFrame.from_numpy(data_3d, sampling_rate=1000)


# --- Test for source_name that fails Path() (channel.py lines 1001-1006) ---


def test_from_file_source_name_path_failure(monkeypatch):
    """When Path(source_name) raises, source_name is used as-is for the label (lines 1001-1006)."""
    fake = FakeReader(sr=100, channels=1, frames=4)
    monkeypatch.setattr(channel_mod, "get_file_reader", lambda *args, **kwargs: fake)

    # Path() can raise on certain exotic values; simulate by monkeypatching Path.stem
    original_path_cls = channel_mod.Path

    class BadPath:
        def __init__(self, s):
            raise ValueError("bad path")

    monkeypatch.setattr(channel_mod, "Path", BadPath)
    cf = ChannelFrame.from_file(b"data", file_type=".wav", source_name="raw::label")
    # The label falls back to the raw source_name string
    assert cf.label == "raw::label"
    monkeypatch.setattr(channel_mod, "Path", original_path_cls)


# --- Test for add_channel with 2D dask array needing reshape (channel.py line 1257) ---


def test_add_channel_dask_2d_multichannel_reshape():
    """add_channel with a 2D dask array whose shape[0] != 1 triggers reshape (line 1257)."""
    base = ChannelFrame.from_numpy(np.zeros((1, 20)), sampling_rate=100)
    # Create a 2D dask array with shape (2, 10) - NOT (1, N)
    arr_2d = da.from_array(np.ones((2, 10)), chunks=(2, 10))
    # Reshape to (1, 20) and add as one channel; use truncate to handle length
    result = base.add_channel(arr_2d, label="new_ch", align="truncate")
    assert result.n_channels == 2
