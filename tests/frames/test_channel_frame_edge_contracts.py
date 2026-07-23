"""ChannelFrame edge contracts spanning construction, I/O, and display."""

from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

import wandas.frames.channel as channel_mod
from tests.frame_helpers import channel_first_values
from wandas.frames.channel import ChannelFrame, _resolve_channels
from wandas.pipeline import RecipePlan


def make_cf(arr: np.ndarray, sr: int = 100) -> ChannelFrame:
    return ChannelFrame.from_numpy(arr, sampling_rate=sr)


def test_mix_rejects_sampling_rate_mismatch():
    a = make_cf(np.arange(6).reshape(2, 3), sr=100)
    b = ChannelFrame.from_numpy(np.arange(6).reshape(2, 3), sampling_rate=200)

    with pytest.raises(ValueError, match=r"Sampling rate mismatch"):
        a.mix(b)


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
    assert cf.metadata["_source_file"] == "my_file.wav"
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
        channel_first_values(cf)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (np.ones((1, 3)), "unexpected channel-first shape"),
        (np.ones((2, 3), dtype=np.complex128), "real channel-first numeric array"),
    ],
)
def test_from_file_validates_custom_reader_boundary(monkeypatch, data, message):
    monkeypatch.setattr(channel_mod, "get_file_reader", lambda *args, **kwargs: FakeReader(data=data))
    frame = ChannelFrame.from_file(b"data", file_type=".wav")

    with pytest.raises((TypeError, ValueError), match=message):
        channel_first_values(frame)


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


def test_from_file_non_string_channel_label_raises(monkeypatch):
    fake = FakeReader(sr=100, channels=1, frames=4)
    monkeypatch.setattr(channel_mod, "get_file_reader", lambda *args, **kwargs: fake)

    with pytest.raises(TypeError, match="ChannelMetadata label must be a string"):
        ChannelFrame.from_file(b"data", file_type=".wav", ch_labels=cast(Any, [1]))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"ch_labels": cast(Any, [1])}, "ChannelMetadata label must be a string"),
        ({"ch_units": cast(Any, [1])}, "Invalid channel calibration unit"),
    ],
)
def test_from_numpy_non_string_channel_metadata_raises(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(TypeError, match=message):
        ChannelFrame.from_numpy(np.arange(4.0), sampling_rate=4, **kwargs)


def test_describe_translates_axis_and_colorbar_configuration(monkeypatch):
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
    figures = cf.describe(
        is_close=False,
        axis_config=axis_conf,
        cbar_config=cbar_conf,
        waveform={"ylabel": "Y"},
    )

    # axis_config overrides waveform (time_plot) so expect xlabel from axis_conf
    assert called.get("waveform", {}).get("xlabel") == "T"
    assert called.get("vmin") == -10
    assert called.get("vmax") == 10
    assert isinstance(figures, list)
    _close_figures(figures)


def test_add_channel_duplicate_label_uses_requested_suffix():
    base = ChannelFrame.from_numpy(np.arange(12).reshape(2, 6), sampling_rate=100)

    with_suffix = base.add_channel(np.zeros(6), label="ch0", suffix_on_dup="_x")

    assert with_suffix.labels[-1] == "ch0_x"


def test_remove_channel_returns_new_frame_with_history():
    cf = ChannelFrame.from_numpy(np.arange(6).reshape(2, 3), sampling_rate=100)

    cf2 = cf.remove_channel(0)

    assert cf2 is not cf
    assert cf2.n_channels == 1
    assert cf.n_channels == 2
    assert cf2.operation_history == [{"operation": "wandas.channel.remove_channel", "version": 1, "params": {"key": 0}}]


_DESCRIBE_SAMPLE_RATE = 44_100
_DESCRIBE_N_SAMPLES = 10_000


def _describe_frame(n_channels: int = 1) -> ChannelFrame:
    sample_index = np.arange(_DESCRIBE_N_SAMPLES)
    channels = [
        np.sin(2 * np.pi * (440 + channel_index * 100) * sample_index / _DESCRIBE_SAMPLE_RATE)
        for channel_index in range(n_channels)
    ]
    return ChannelFrame.from_numpy(np.stack(channels), sampling_rate=_DESCRIBE_SAMPLE_RATE)


def _close_figures(figures: list[Figure]) -> None:
    for figure in figures:
        plt.close(figure)


@pytest.mark.parametrize("suffix", ["jpg", "png", "pdf"])
def test_describe_image_save_writes_nonempty_file(tmp_path: Path, suffix: str) -> None:
    output_path = tmp_path / f"describe.{suffix}"

    _describe_frame().describe(image_save=output_path)

    assert output_path.stat().st_size > 0


def test_describe_image_save_accepts_path_and_returns_open_figure(tmp_path: Path) -> None:
    output_path = tmp_path / "describe.png"

    figures = _describe_frame().describe(image_save=output_path, is_close=False)

    assert isinstance(figures, list)
    assert output_path.stat().st_size > 0
    assert all(isinstance(figure, Figure) for figure in figures)
    _close_figures(figures)


def test_describe_image_save_uses_channel_suffixes_for_multichannel_frame(tmp_path: Path) -> None:
    output_path = tmp_path / "describe.png"

    figures = _describe_frame(n_channels=3).describe(image_save=output_path, is_close=False)

    assert isinstance(figures, list)
    assert len(figures) == 3
    assert [path.name for path in sorted(tmp_path.iterdir())] == [
        "describe_0.png",
        "describe_1.png",
        "describe_2.png",
    ]
    _close_figures(figures)


def test_describe_returns_open_figures_when_requested() -> None:
    figures = _describe_frame(n_channels=2).describe(is_close=False)

    assert isinstance(figures, list)
    assert len(figures) == 2
    assert all(len(figure.axes) >= 2 for figure in figures)
    _close_figures(figures)


def test_describe_closed_mode_displays_and_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    interactive_display = MagicMock()
    audio = MagicMock(return_value=object())
    monkeypatch.setattr(
        "wandas.visualization.notebook.require_ipython_display",
        lambda _feature: (interactive_display, audio),
    )

    assert _describe_frame().describe(is_close=True) is None
    assert interactive_display.call_count == 2
    audio.assert_called_once()


@pytest.mark.parametrize(
    ("legacy_config", "expected_warning"),
    [
        pytest.param(
            {"axis_config": {"time_plot": {"xlabel": "Time"}}},
            "axis_config is deprecated and will be removed in a future release",
            id="axis-config",
        ),
        pytest.param(
            {"cbar_config": {"vmin": -80}},
            "cbar_config is deprecated and will be removed in a future release",
            id="colorbar-config",
        ),
    ],
)
def test_describe_legacy_config_logs_deprecation_warning(
    legacy_config: dict[str, object],
    expected_warning: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        figures = _describe_frame().describe(
            is_close=False,
            **legacy_config,  # ty: ignore[invalid-argument-type]
        )

    assert isinstance(figures, list)
    _close_figures(figures)
    assert any(expected_warning in record.message for record in caplog.records)


def test_describe_combined_configuration_returns_structured_figure() -> None:
    figures = _describe_frame().describe(
        normalize=True,
        is_close=False,
        fmin=0,
        fmax=None,
        cmap="viridis",
        vmin=-80,
        vmax=-20,
        xlim=(0, 1),
        ylim=(0, 5_000),
        Aw=True,
        waveform={"xlabel": "Time (s)", "ylabel": "Amplitude"},
        spectral={"ylabel": "dB"},
    )

    assert isinstance(figures, list)
    assert len(figures) == 1
    assert len(figures[0].axes) >= 2
    _close_figures(figures)


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
    data_3d = np.random.default_rng(42).random((2, 4, 100))
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


def test_rename_channels_capture_rejects_non_mapping_and_non_string_or_integer_keys() -> None:
    frame = make_cf(np.arange(6).reshape(2, 3))

    with pytest.raises(TypeError, match="mapping must be a mapping"):
        frame.rename_channels(cast(Any, []))
    with pytest.raises(TypeError, match="keys must be integers or strings"):
        frame.rename_channels(cast(Any, {True: "invalid"}))


def test_rename_channels_recipe_decodes_integer_and_label_keys() -> None:
    source = ChannelFrame.from_numpy(
        np.arange(6).reshape(2, 3),
        sampling_rate=100,
        ch_labels=["left", "right"],
    )
    renamed = source.rename_channels({0: "front", "right": "rear"})
    plan = RecipePlan.from_frame(renamed, input_names=("signal",))

    replayed = plan.apply({"signal": source})

    assert replayed.labels == ["front", "rear"]


def test_download_url_normalizes_explicit_extension(monkeypatch, tmp_path: Path) -> None:
    owner = MagicMock()
    owner.path = tmp_path / "download.wav"
    download = MagicMock(return_value=owner)
    monkeypatch.setattr(channel_mod, "download_url_to_temporary_file", download)

    path, returned_owner, file_type, source_name = channel_mod._download_url(
        "https://example.com/audio",
        "wav",
        None,
        2.0,
    )

    assert path == owner.path
    assert returned_owner is owner
    assert file_type == ".wav"
    assert source_name == "https://example.com/audio"
    download.assert_called_once_with(
        "https://example.com/audio",
        timeout=2.0,
        suffix=".wav",
        resource_name="audio",
    )


@pytest.mark.parametrize(
    ("other", "kwargs", "error", "message"),
    [
        (np.ones(3), {"align": "invalid"}, ValueError, "align must be"),
        (np.ones(3), {"snr_db": "loud"}, TypeError, "snr_db must be numeric"),
        (np.ones((1, 1, 3)), {}, ValueError, "array input must be 1-D or channel-first 2-D"),
        (da.ones((1, 1, 3)), {}, ValueError, "array input must be 1-D or channel-first 2-D"),
        (1.0, {}, TypeError, "scalars are invalid"),
    ],
)
def test_mix_runtime_handler_rejects_invalid_inputs(
    other: object,
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    frame = make_cf(np.arange(6).reshape(2, 3))
    undecorated_mix = cast(Any, ChannelFrame.mix).__wrapped__

    with pytest.raises(error, match=message):
        undecorated_mix(frame, other, **kwargs)


def _url_download_owner(tmp_path: Path) -> MagicMock:
    owner = MagicMock()
    owner.path = tmp_path / "download.wav"
    return owner


def test_from_file_url_cleans_download_when_source_resolution_fails(monkeypatch, tmp_path: Path) -> None:
    owner = _url_download_owner(tmp_path)
    monkeypatch.setattr(
        channel_mod,
        "_download_url",
        lambda *_args: (owner.path, owner, ".wav", "https://example.com/audio.wav"),
    )

    def fail_resolution(*_args: object) -> object:
        raise ValueError("bad source")

    monkeypatch.setattr(channel_mod, "_resolve_source", fail_resolution)

    with pytest.raises(ValueError, match="bad source"):
        ChannelFrame.from_file("https://example.com/audio.wav")

    owner.cleanup.assert_called_once_with()


def test_from_file_url_cleans_download_when_file_info_fails(monkeypatch, tmp_path: Path) -> None:
    owner = _url_download_owner(tmp_path)

    class BadInfoReader(FakeReader):
        def get_file_info(self, source, **kwargs):
            raise ValueError("bad info")

    monkeypatch.setattr(
        channel_mod,
        "_download_url",
        lambda *_args: (owner.path, owner, ".wav", "https://example.com/audio.wav"),
    )
    monkeypatch.setattr(
        channel_mod,
        "_resolve_source",
        lambda *_args: (owner.path, owner.path, BadInfoReader(), ".wav"),
    )

    with pytest.raises(ValueError, match="bad info"):
        ChannelFrame.from_file("https://example.com/audio.wav")

    owner.cleanup.assert_called_once_with()


def test_from_file_url_cleans_download_when_dask_setup_fails(monkeypatch, tmp_path: Path) -> None:
    owner = _url_download_owner(tmp_path)
    monkeypatch.setattr(
        channel_mod,
        "_download_url",
        lambda *_args: (owner.path, owner, ".wav", "https://example.com/audio.wav"),
    )
    monkeypatch.setattr(
        channel_mod,
        "_resolve_source",
        lambda *_args: (owner.path, owner.path, FakeReader(), ".wav"),
    )

    def fail_dask_setup(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("bad dask setup")

    monkeypatch.setattr(channel_mod, "da_from_delayed", fail_dask_setup)

    with pytest.raises(RuntimeError, match="bad dask setup"):
        ChannelFrame.from_file("https://example.com/audio.wav")

    owner.cleanup.assert_called_once_with()


def test_from_file_url_cleans_download_when_frame_construction_fails(monkeypatch, tmp_path: Path) -> None:
    owner = _url_download_owner(tmp_path)
    from_file = ChannelFrame.from_file
    monkeypatch.setattr(
        channel_mod,
        "_download_url",
        lambda *_args: (owner.path, owner, ".wav", "https://example.com/audio.wav"),
    )
    monkeypatch.setattr(
        channel_mod,
        "_resolve_source",
        lambda *_args: (owner.path, owner.path, FakeReader(), ".wav"),
    )
    monkeypatch.setattr(channel_mod, "ChannelFrame", MagicMock(side_effect=RuntimeError("bad frame")))

    with pytest.raises(RuntimeError, match="bad frame"):
        from_file("https://example.com/audio.wav")

    owner.cleanup.assert_called_once_with()


# --- Test for add_channel with 2D dask array needing reshape (channel.py line 1257) ---


def test_add_channel_rejects_multichannel_raw_dask_array():
    """Raw add_channel input represents exactly one channel."""
    base = ChannelFrame.from_numpy(np.zeros((1, 20)), sampling_rate=100)
    # Create a 2D dask array with shape (2, 10) - NOT (1, N)
    arr_2d = da.from_array(np.ones((2, 10)), chunks=(2, 10))
    with pytest.raises(ValueError, match="Raw add_channel input"):
        base.add_channel(arr_2d, label="new_ch", align="truncate")
