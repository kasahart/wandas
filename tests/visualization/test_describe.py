"""Contracts for the ChannelFrame.describe presentation boundary."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from wandas.frames.channel import ChannelFrame
from wandas.visualization import describe as describe_module


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
