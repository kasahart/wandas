"""Optional notebook presentation helpers for static Frame descriptions."""

from __future__ import annotations

import sys
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


def in_notebook_frontend() -> bool:
    """Return whether rich output reaches a notebook front end.

    True inside marimo or an IPython kernel (Jupyter, VS Code, Colab). A plain
    script or the terminal IPython shell has nowhere to render
    ``IPython.display`` output, so it gets False even when IPython is
    installed. Only modules the session has already imported are inspected.
    """
    marimo = sys.modules.get("marimo")
    if marimo is not None and marimo.running_in_notebook():
        return True
    ipython = sys.modules.get("IPython")
    shell = ipython.get_ipython() if ipython is not None else None
    if shell is None:
        return False
    terminal = sys.modules.get("IPython.terminal.interactiveshell")
    return terminal is None or not isinstance(shell, terminal.TerminalInteractiveShell)


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


__all__ = ["NotebookDisplay", "display_figure_and_audio", "in_notebook_frontend", "resolve_notebook_display"]
