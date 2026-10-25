"""Optional notebook presentation helpers for static Frame descriptions."""

from __future__ import annotations

import base64
import io
import sys
import wave
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from wandas.utils.optional_imports import require_ipython_display, require_matplotlib_pyplot

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


def pyscript_display_target() -> str | None:
    """Return the PyScript element id that ``pyscript.display`` writes to.

    None outside PyScript, and inside PyScript when the running code has no
    output element (``current_target()`` is empty, as in ``py-editor``). Only
    an already imported ``pyscript`` module is inspected.
    """
    pyscript = sys.modules.get("pyscript")
    current_target = getattr(pyscript, "current_target", None)
    if not callable(current_target):
        return None
    target = current_target()
    return target if isinstance(target, str) and target else None


def _wav_data_uri(data: Any, *, sampling_rate: float, normalize: bool) -> str:
    """Encode one channel as a 16-bit PCM WAV data URI for an HTML audio player."""
    samples = np.asarray(data, dtype=np.float64).reshape(-1)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if normalize:
        if peak > 0:
            samples = samples / peak
    elif peak > 1:
        raise ValueError("Audio data must be between -1 and 1 when normalize=False.")
    pcm = np.round(samples * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(round(sampling_rate))
        writer.writeframes(pcm.tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def display_in_pyscript(
    figure: Figure,
    target: str,
    *,
    audio: Any = None,
    sampling_rate: float | None = None,
    normalize: bool = True,
) -> None:
    """Append a Figure, and optionally an audio player, to a PyScript element."""
    pyscript = sys.modules["pyscript"]
    pyscript.display(figure, target=target, append=True)
    if audio is not None and sampling_rate is not None:
        source = _wav_data_uri(audio, sampling_rate=sampling_rate, normalize=normalize)
        pyscript.display(pyscript.HTML(f'<audio controls src="{source}"></audio>'), target=target, append=True)


def display_new_figure(figure: Figure) -> bool:
    """Display and close a Figure a plot call created, when PyScript has a target.

    PyScript has no end-of-cell figure rendering, so a plot that returns its
    Axes without ``pyplot.show()`` would otherwise never reach the page.
    Returns whether the Figure was displayed.
    """
    target = pyscript_display_target()
    if target is None:
        return False
    display_in_pyscript(figure, target)
    require_matplotlib_pyplot("plot").close(figure)
    return True


def show_figure(figure: Figure) -> None:
    """Show a Figure a plot call created: in PyScript when it has a target, else with pyplot."""
    if not display_new_figure(figure):
        require_matplotlib_pyplot("plot").show()


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


__all__ = [
    "NotebookDisplay",
    "display_figure_and_audio",
    "display_in_pyscript",
    "display_new_figure",
    "in_notebook_frontend",
    "pyscript_display_target",
    "resolve_notebook_display",
    "show_figure",
]
