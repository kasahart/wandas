"""Contracts for showing Frame figures and audio in a PyScript page."""

from __future__ import annotations

import base64
import io
import sys
import types
import wave
from typing import Any

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from wandas.frames.channel import ChannelFrame
from wandas.frames.roughness import RoughnessFrame
from wandas.visualization import notebook


class _FakeHTML:
    def __init__(self, html: str) -> None:
        self.html = html


class _FakePyScript:
    """Record pyscript.display() calls for a page whose output target is fixed."""

    def __init__(self, target: str) -> None:
        self.target = target
        self.calls: list[tuple[object, str, bool]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = types.ModuleType("pyscript")
        module.current_target = lambda: self.target  # ty: ignore[unresolved-attribute]
        module.HTML = _FakeHTML  # ty: ignore[unresolved-attribute]
        module.display = self._display  # ty: ignore[unresolved-attribute]
        monkeypatch.setitem(sys.modules, "pyscript", module)

    def _display(self, value: object, *, target: str, append: bool) -> None:
        self.calls.append((value, target, append))


def _install(monkeypatch: pytest.MonkeyPatch, target: str = "py-0-target") -> _FakePyScript:
    fake = _FakePyScript(target)
    fake.install(monkeypatch)
    return fake


def _decode_audio(html: str) -> tuple[int, np.ndarray]:
    prefix = 'src="data:audio/wav;base64,'
    encoded = html.split(prefix, 1)[1].split('"', 1)[0]
    with wave.open(io.BytesIO(base64.b64decode(encoded))) as reader:
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        rate = reader.getframerate()
        samples = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
    return rate, samples


@pytest.fixture(autouse=True)
def _no_pyscript(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "pyscript", raising=False)
    plt.close("all")


@pytest.fixture
def show_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    monkeypatch.setattr(plt, "show", lambda *_args, **_kwargs: calls.append(1))
    return calls


def test_target_is_none_without_pyscript() -> None:
    assert notebook.pyscript_display_target() is None


def test_empty_pyscript_target_counts_as_no_target(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, target="")

    assert notebook.pyscript_display_target() is None


def test_pyscript_target_is_the_current_target(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, target="py-3-target")

    assert notebook.pyscript_display_target() == "py-3-target"


def test_describe_displays_each_figure_then_its_audio_in_pyscript(
    monkeypatch: pytest.MonkeyPatch, show_calls: list[int]
) -> None:
    fake = _install(monkeypatch)
    samples = np.array([[0.0, 0.25, -0.5, 0.25] * 1024, [0.0, 0.1, -0.1, 0.05] * 1024])
    frame = ChannelFrame.from_numpy(samples, sampling_rate=8000.0)

    result = frame.describe()

    assert result is None
    assert show_calls == []
    assert [type(value) for value, _, _ in fake.calls] == [Figure, _FakeHTML, Figure, _FakeHTML]
    assert all(target == "py-0-target" and append for _, target, append in fake.calls)
    for (audio, _, _), channel in zip(fake.calls[1::2], samples, strict=True):
        assert isinstance(audio, _FakeHTML)
        rate, decoded = _decode_audio(audio.html)
        assert rate == 8000
        np.testing.assert_array_equal(decoded, np.round(channel / np.max(np.abs(channel)) * 32767))
    assert plt.get_fignums() == []


def test_describe_without_a_pyscript_target_keeps_pyplot_display(
    monkeypatch: pytest.MonkeyPatch, show_calls: list[int]
) -> None:
    fake = _install(monkeypatch, target="")
    monkeypatch.setattr(notebook, "in_notebook_frontend", lambda: False)
    frame = ChannelFrame.from_numpy(np.linspace(-0.5, 0.5, 4096).reshape(1, 4096), sampling_rate=8000.0)

    frame.describe()

    assert fake.calls == []
    assert show_calls == [1]


def test_plot_displays_and_closes_its_figure_in_pyscript(
    monkeypatch: pytest.MonkeyPatch, show_calls: list[int]
) -> None:
    fake = _install(monkeypatch)
    frame = ChannelFrame.from_numpy(np.linspace(-0.5, 0.5, 8192).reshape(2, 4096), sampling_rate=8000.0)

    frame.plot()

    assert show_calls == []
    assert len(fake.calls) == 1
    assert isinstance(fake.calls[0][0], Figure)
    assert plt.get_fignums() == []


def test_overlay_plot_displays_the_figure_it_created_in_pyscript(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install(monkeypatch)
    frame = ChannelFrame.from_numpy(np.linspace(-0.5, 0.5, 8192).reshape(2, 4096), sampling_rate=8000.0)

    axes = frame.welch().plot(overlay=True)

    assert len(fake.calls) == 1
    assert fake.calls[0][0] is getattr(axes, "figure", None)
    assert plt.get_fignums() == []


def test_plot_into_given_axes_is_left_to_the_caller_in_pyscript(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install(monkeypatch)
    frame = ChannelFrame.from_numpy(np.linspace(-0.5, 0.5, 4096).reshape(1, 4096), sampling_rate=8000.0)
    figure, axes = plt.subplots()

    frame.plot(ax=axes)
    frame.welch().plot(ax=axes)

    assert fake.calls == []
    assert plt.fignum_exists(figure.number)


def test_overlay_plot_outside_pyscript_does_not_show(show_calls: list[int]) -> None:
    frame = ChannelFrame.from_numpy(np.linspace(-0.5, 0.5, 8192).reshape(2, 4096), sampling_rate=8000.0)

    frame.welch().plot(overlay=True)

    assert show_calls == []


def test_unnormalized_audio_outside_unit_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="between -1 and 1"):
        notebook._wav_data_uri(np.array([0.0, 1.5]), sampling_rate=8000.0, normalize=False)


def test_unnormalized_audio_keeps_its_scale() -> None:
    uri = notebook._wav_data_uri(np.array([0.0, 0.5, -0.5]), sampling_rate=8000.0, normalize=False)

    _, decoded = _decode_audio(f'src="{uri}"')

    np.testing.assert_array_equal(decoded, [0, 16384, -16384])


def _cepstrum() -> Any:
    return ChannelFrame.from_numpy(np.sin(np.arange(4096) / 7.0).reshape(1, 4096), sampling_rate=8000.0).cepstrum()


def _cepstrogram() -> Any:
    signal = ChannelFrame.from_numpy(np.sin(np.arange(4096) / 7.0).reshape(1, 4096), sampling_rate=8000.0)
    return signal.stft(n_fft=256).cepstrum()


def _roughness() -> Any:
    return RoughnessFrame(
        data=da.from_array(np.ones((47, 2)), chunks=(47, 2)),
        sampling_rate=10.0,
        bark_axis=np.linspace(0.5, 23.5, 47),
        overlap=0.5,
    )


@pytest.mark.parametrize("make_frame", [_cepstrum, _cepstrogram, _roughness])
def test_frame_specific_plots_display_their_new_figure_in_pyscript(
    monkeypatch: pytest.MonkeyPatch, make_frame: Any
) -> None:
    fake = _install(monkeypatch)

    make_frame().plot()

    assert len(fake.calls) == 1
    assert isinstance(fake.calls[0][0], Figure)
    assert plt.get_fignums() == []
