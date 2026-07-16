"""Optional notebook presentation helpers for static Frame descriptions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from wandas.utils.optional_imports import require_ipython_display

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class NotebookDisplay:
    """Resolved optional IPython display functions for one presentation call."""

    display: Callable[..., Any]
    audio: Callable[..., Any]


def resolve_notebook_display(feature: str = "describe") -> NotebookDisplay:
    """Resolve IPython display support before expensive plot generation."""
    display, audio = require_ipython_display(feature)
    return NotebookDisplay(display=display, audio=audio)


def display_figure_and_audio(
    session: NotebookDisplay,
    figure: Figure,
    data: Any,
    *,
    sampling_rate: float,
    normalize: bool,
) -> None:
    """Present one static Figure followed by its matching audio channel."""
    session.display(figure)
    session.display(session.audio(data, rate=sampling_rate, normalize=normalize))


__all__ = ["NotebookDisplay", "display_figure_and_audio", "resolve_notebook_display"]
