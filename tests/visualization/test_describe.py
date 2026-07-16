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
