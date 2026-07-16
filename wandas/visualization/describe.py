"""Composite static visualization workflow behind ``ChannelFrame.describe``."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from wandas.utils.optional_imports import (
    require_matplotlib_axes_type,
    require_matplotlib_pyplot,
)
from wandas.visualization import notebook

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from wandas.frames.channel import ChannelFrame

logger = logging.getLogger(__name__)


def _apply_deprecated_describe_kwargs(plot_kwargs: dict[str, Any]) -> None:
    """Migrate deprecated ``axis_config`` and ``cbar_config`` settings."""
    if "axis_config" in plot_kwargs:
        logger.warning(
            "axis_config is deprecated and will be removed in a future release; use waveform and spectral instead."
        )
        axis_config = plot_kwargs["axis_config"]
        if "time_plot" in axis_config:
            plot_kwargs["waveform"] = axis_config["time_plot"]
        if "freq_plot" in axis_config:
            if "xlim" in axis_config["freq_plot"]:
                vlim = axis_config["freq_plot"]["xlim"]
                plot_kwargs["vmin"] = vlim[0]
                plot_kwargs["vmax"] = vlim[1]
            if "ylim" in axis_config["freq_plot"]:
                plot_kwargs["ylim"] = axis_config["freq_plot"]["ylim"]

    if "cbar_config" in plot_kwargs:
        logger.warning("cbar_config is deprecated and will be removed in a future release; use vmin and vmax instead.")
        cbar_config = plot_kwargs["cbar_config"]
        if "vmin" in cbar_config:
            plot_kwargs["vmin"] = cbar_config["vmin"]
        if "vmax" in cbar_config:
            plot_kwargs["vmax"] = cbar_config["vmax"]


def _save_figure(figure: Figure, image_save: str | Path, *, channel: int, channel_count: int) -> None:
    """Save one channel figure using the established suffix contract."""
    if channel_count > 1:
        save_path = Path(image_save)
        target = save_path.parent / f"{save_path.stem}_{channel}{save_path.suffix}"
    else:
        target = image_save
    figure.savefig(target, bbox_inches="tight")


def describe_frame(
    frame: ChannelFrame,
    normalize: bool = True,
    is_close: bool = True,
    *,
    fmin: float = 0,
    fmax: float | None = None,
    cmap: str = "jet",
    vmin: float | None = None,
    vmax: float | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    Aw: bool = False,  # noqa: N803
    waveform: dict[str, Any] | None = None,
    spectral: dict[str, Any] | None = None,
    image_save: str | Path | None = None,
    **kwargs: Any,
) -> list[Figure] | None:
    """Create, save, optionally display, and close per-channel summaries."""
    plot_kwargs: dict[str, Any] = {
        "fmin": fmin,
        "fmax": fmax,
        "cmap": cmap,
        "vmin": vmin,
        "vmax": vmax,
        "xlim": xlim,
        "ylim": ylim,
        "Aw": Aw,
        "waveform": waveform or {},
        "spectral": spectral or {},
    }
    plot_kwargs.update(kwargs)
    _apply_deprecated_describe_kwargs(plot_kwargs)

    axes_cls = require_matplotlib_axes_type("describe")
    pyplot = require_matplotlib_pyplot("describe")
    display_session = notebook.resolve_notebook_display("describe") if image_save is None and is_close else None
    figures: list[Figure] = []

    for channel_index, channel in enumerate(frame):
        plotted = channel.plot("describe", title=f"{channel.label} {channel.labels[0]}", **plot_kwargs)
        if isinstance(plotted, axes_cls):
            axes = plotted
        elif isinstance(plotted, Iterator):
            axes = cast("Axes", next(plotted))
        else:
            raise TypeError(f"Unexpected type for plot result: {type(plotted)}. Expected Axes or Iterator[Axes].")
        figure = cast("Figure | None", getattr(axes, "figure", None))
        if figure is None:
            continue

        try:
            if not is_close:
                figures.append(figure)
            if image_save is not None:
                _save_figure(figure, image_save, channel=channel_index, channel_count=frame.n_channels)
            if display_session is not None:
                notebook.display_figure_and_audio(
                    display_session,
                    figure,
                    channel.data,
                    sampling_rate=channel.sampling_rate,
                    normalize=normalize,
                )
        finally:
            if is_close:
                figure.clf()
                pyplot.close(figure)

    return None if is_close else figures


__all__ = ["describe_frame"]
