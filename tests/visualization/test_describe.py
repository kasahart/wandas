"""Contracts for the ChannelFrame.describe presentation boundary."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame
from wandas.visualization import describe as describe_module
from wandas.visualization.notebook import NotebookDisplay


def test_channel_describe_is_thin_visualization_facade(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = ChannelFrame.from_numpy(np.arange(8, dtype=float).reshape(1, 8), 8.0)
    captured: dict[str, Any] = {}

    def fake_describe(receiver: ChannelFrame, **kwargs: Any) -> list[object]:
        captured["receiver"] = receiver
        captured.update(kwargs)
        return []

    monkeypatch.setattr(describe_module, "describe_frame", fake_describe)

    result = frame.describe(
        normalize=False,
        is_close=False,
        fmin=1.0,
        fmax=3.0,
        cmap="magma",
        waveform={"ylabel": "Amplitude"},
    )

    assert result == []
    assert captured["receiver"] is frame
    assert captured["normalize"] is False
    assert captured["is_close"] is False
    assert captured["fmin"] == 1.0
    assert captured["fmax"] == 3.0
    assert captured["cmap"] == "magma"
    assert captured["waveform"] == {"ylabel": "Amplitude"}


def test_describe_skips_axes_without_a_figure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A backend result without a figure must not enter the display lifecycle."""

    class AxesWithoutFigure:
        figure = None

    def fake_plot(_frame: ChannelFrame, *_args: Any, **_kwargs: Any) -> AxesWithoutFigure:
        return AxesWithoutFigure()

    monkeypatch.setattr(describe_module, "require_matplotlib_axes_type", lambda _operation: AxesWithoutFigure)
    monkeypatch.setattr(describe_module, "require_matplotlib_pyplot", lambda _operation: object())
    monkeypatch.setattr(ChannelFrame, "plot", fake_plot)

    frame = ChannelFrame.from_numpy(np.arange(8, dtype=float).reshape(1, 8), 8.0)

    assert describe_module.describe_frame(frame, is_close=False) == []


@pytest.mark.parametrize("failure_stage", ["audio-construction", "audio-display"])
def test_describe_closed_mode_closes_figure_when_notebook_audio_fails(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    """Closed mode must release the Figure even when notebook audio presentation fails."""
    figure, axes = plt.subplots()
    audio_value = object()

    def audio(*_args: Any, **_kwargs: Any) -> object:
        if failure_stage == "audio-construction":
            raise RuntimeError("audio construction failed")
        return audio_value

    def display(value: object) -> None:
        if failure_stage == "audio-display" and value is audio_value:
            raise RuntimeError("audio display failed")

    monkeypatch.setattr(describe_module, "require_matplotlib_axes_type", lambda _operation: type(axes))
    monkeypatch.setattr(describe_module, "require_matplotlib_pyplot", lambda _operation: plt)
    monkeypatch.setattr(
        describe_module.notebook,
        "resolve_notebook_display",
        lambda _feature: NotebookDisplay(display=display, audio=audio),
    )
    monkeypatch.setattr(ChannelFrame, "plot", lambda *_args, **_kwargs: axes)

    frame = ChannelFrame.from_numpy(np.arange(8, dtype=float).reshape(1, 8), 8.0)
    figure_number = figure.number

    with pytest.raises(RuntimeError, match=r"audio (construction|display) failed"):
        describe_module.describe_frame(frame, is_close=True)

    assert not plt.fignum_exists(figure_number)
    assert figure.axes == []
