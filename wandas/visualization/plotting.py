from __future__ import annotations

import abc
import inspect
import logging
from collections.abc import Callable, Iterator, Sequence
from functools import cache
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast

import numpy as np

from wandas.utils.introspection import filter_kwargs
from wandas.utils.optional_imports import (
    require_matplotlib_artist_type as _matplotlib_artist_type,
)
from wandas.utils.optional_imports import (
    require_matplotlib_axes_type as _matplotlib_axes_type,
)
from wandas.utils.optional_imports import (
    require_matplotlib_figure_type as _matplotlib_figure_type,
)
from wandas.utils.optional_imports import (
    require_matplotlib_gridspec as _matplotlib_gridspec,
)
from wandas.utils.optional_imports import (
    require_matplotlib_line2d_type as _matplotlib_line2d_type,
)
from wandas.utils.optional_imports import (
    require_matplotlib_pyplot as _matplotlib_pyplot,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from wandas.core.base_frame import BaseFrame
    from wandas.core.metadata import ChannelMetadata
    from wandas.frames.channel import ChannelFrame
    from wandas.frames.noct import NOctFrame
    from wandas.frames.spectral import SpectralFrame
    from wandas.frames.spectrogram import SpectrogramFrame

logger = logging.getLogger(__name__)

TFrame = TypeVar("TFrame", bound="BaseFrame[Any]")
PlotLabel = str | Sequence[str] | None


def _typed_hook(frame: Any, name: str) -> Callable[..., Any] | None:
    """Resolve a typed plotting hook from the Frame class, not its instance."""
    descriptor = inspect.getattr_static(type(frame), name, None)
    if descriptor is None:
        return None
    bind = getattr(descriptor, "__get__", None)
    hook = bind(frame, type(frame)) if bind is not None else descriptor
    if not callable(hook):
        raise TypeError(f"{type(frame).__name__}.{name} must be callable")
    return cast("Callable[..., Any]", hook)


def _typed_frequency_values(frame: Any, *, view: str | None, aw: bool) -> tuple[Any, str] | None:
    """Return values from a Frame's quantity-specific plotting contract.

    Dedicated pairwise Frames expose this hook on their class.  Looking up the
    attribute on the class (rather than on the instance) keeps arbitrary
    metadata-like instance attributes from becoming an implicit quantity
    dispatch mechanism.
    """
    hook = _typed_hook(frame, "_plot_frequency_values")
    if hook is None:
        return None
    values, ylabel = hook(view=view, Aw=aw)
    return values, str(ylabel)


def _typed_matrix_entries(
    frame: Any,
    *,
    view: str | None,
    aw: bool,
) -> tuple[tuple[int, int, Any, str], ...] | None:
    """Return typed output/input matrix entries when the Frame provides them."""
    hook = _typed_hook(frame, "_matrix_plot_entries")
    if hook is None:
        return None
    return tuple(hook(view=view, Aw=aw))


def _typed_plot_ylabel(frame: Any, *, view: str | None, aw: bool) -> str | None:
    """Return a typed plot label without materializing the plotted values."""
    hook = _typed_hook(frame, "_plot_ylabel")
    if hook is None:
        return None
    return str(hook(view=view, Aw=aw))


def _spectrogram_axis_values(frame: Any, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.fft.rfftfreq(frame.n_fft, 1.0 / frame.sampling_rate)
    times = np.arange(data.shape[-1]) * frame.hop_length / frame.sampling_rate
    return times, freqs


class PlotStrategy(abc.ABC, Generic[TFrame]):
    """Base class for plotting strategies"""

    name: ClassVar[str]

    @abc.abstractmethod
    def channel_plot(
        self,
        x: Any,
        y: Any,
        ax: Axes,
        label: PlotLabel = None,
        alpha: float = 1.0,
    ) -> None:
        """Implementation of channel plotting"""

    @abc.abstractmethod
    def plot(
        self,
        bf: TFrame,
        ax: Axes | None = None,
        title: str | None = None,
        overlay: bool = False,
        **kwargs: Any,
    ) -> Axes | Iterator[Axes]:
        """Implementation of plotting"""


# Helper function for return type
def _return_axes_iterator(axes_list: Any) -> Iterator[Axes]:
    """Helper to convert fig.axes to Iterator[Axes] with proper typing"""
    return iter(axes_list)


def _resolve_channel_label(
    label: str | Sequence[str] | None,
    ch_meta: ChannelMetadata,
    channel_index: int,
    n_channels: int,
) -> str:
    """Resolve the label for a single channel in the non-overlay (per-subplot) path.

    Args:
        label: str | Sequence[str] | None. User-supplied label override.  When a sequence is given its element at
            *channel_index* is used; when a plain string is given that string is
            used for every channel; when ``None`` the channel metadata label is
            used.
        ch_meta: ChannelMetadata. Metadata for the current channel (provides the default label).
        channel_index: int. Zero-based index of the current channel in the frame.
        n_channels: int. Total number of channels in the frame. Used to validate per-channel
            label sequences before indexing.

    Returns:
        str: The resolved label string.
    """
    if label is None:
        return str(ch_meta.label)
    if isinstance(label, str):
        return label
    if isinstance(label, Sequence):
        if len(label) != n_channels:
            raise ValueError(
                "Channel label count mismatch\n"
                f"  Got: {len(label)} labels for {n_channels} channels\n"
                "  Expected: One label per channel in non-overlay mode\n"
                "Provide label as a single string for all channels or a sequence matching the channel count."
            )
        return str(label[channel_index])
    return str(label)


def _reshape_to_2d(data: Any) -> Any:
    """
    Reshape 1D data to 2D for consistent processing across plot strategies.

    This function ensures that data has at least 2 dimensions for plotting operations.
    If the input data is 1D, it will be reshaped to (1, -1).

    Args:
        data: array-like. Input data that may be 1D or already 2D+

    Returns:
        array-like: Data reshaped to ensure at least 2 dimensions
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def _reshape_spectrogram_data(data: Any) -> Any:
    """
    Reshape spectrogram data to 3D for consistent processing.

    This function ensures that spectrogram data has 3 dimensions:
    (channels, freqs, time). Handles both 1D and 2D input data.

    Args:
        data: array-like. Input spectrogram data that may be 1D, 2D, or already 3D

    Returns:
        array-like: Data reshaped to ensure 3 dimensions for spectrogram plotting
    """
    if data.ndim == 1:
        # 1D data: reshape to (1, freqs, 1) - single channel, single time frame
        data = data.reshape((1, data.shape[0], 1))
    elif data.ndim == 2:
        # 2D data: reshape to (1, freqs, time) - single channel spectrogram
        data = data.reshape((1, *data.shape))
    return data


def _plot_line_layout(
    strategy: PlotStrategy[Any],
    bf: BaseFrame[Any],
    x_data: Any,
    data_2d: Any,
    *,
    ax: Axes | None,
    title: str | None,
    overlay: bool,
    ylabel: str,
    xlabel: str,
    default_title: str,
    alpha: float = 1,
    label: Any = None,
    plot_kwargs: dict[str, Any] | None = None,
    ax_set: dict[str, Any] | None = None,
    per_channel_ylabel_fn: Callable[[str, Any], str] | None = None,
    lines_drawn: Callable[[Axes, list[Any], list[int]], None] | None = None,
) -> Axes | Iterator[Axes]:
    """Shared layout for line-based plot strategies (waveform, frequency, noct).

    Handles overlay (single axes) and multi-subplot (per-channel) modes.

    Args:
        strategy: PlotStrategy. The strategy whose ``channel_plot`` method will be called.
        bf: BaseFrame. The frame being plotted (provides ``n_channels``, ``channels``, ``labels``).
        x_data: array-like. X-axis values (e.g. ``bf.time`` or ``bf.freqs``), or a 2-D array with one
            row of x values per channel.
        data_2d: array-like. 2-D array of shape ``(n_channels, n_samples)`` already reshaped via
            ``_reshape_to_2d``.
        per_channel_ylabel_fn: callable, optional. ``(ylabel, ch_meta) -> str`` override for per-channel y-labels
            (e.g. to append a unit suffix).
        lines_drawn: callable, optional. ``(ax, lines, channel_indices)`` called after each axes
            receives its channel lines and before axis limits are applied.
    """
    plot_kwargs = plot_kwargs or {}
    ax_set = ax_set or {}

    if ax is not None:
        overlay = True

    plt = _matplotlib_pyplot("plot")

    if overlay:
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        existing_lines = len(ax.lines)
        strategy.channel_plot(
            x_data.T if np.ndim(x_data) == 2 else x_data,
            data_2d.T,
            ax,
            label=label if label is not None else bf.labels,
            alpha=alpha,
            **plot_kwargs,
        )
        if lines_drawn is not None:
            lines_drawn(ax, list(ax.lines[existing_lines:]), list(range(bf.n_channels)))
        ax.set(
            ylabel=ylabel,
            xlabel=xlabel,
            title=title or default_title,
            **ax_set,
        )
        return ax

    num_channels = bf.n_channels
    fig, axs = plt.subplots(num_channels, 1, figsize=(10, 4 * num_channels), sharex=True)
    if not isinstance(axs, list | np.ndarray):
        axs = [axs]
    axes_list = list(axs)

    for ch_idx, (ax_i, channel_data, ch_meta) in enumerate(zip(axes_list, data_2d, bf.channels, strict=True)):
        ch_ylabel = per_channel_ylabel_fn(ylabel, ch_meta) if per_channel_ylabel_fn else ylabel
        strategy.channel_plot(
            x_data[ch_idx] if np.ndim(x_data) == 2 else x_data,
            channel_data,
            ax_i,
            label=_resolve_channel_label(label, ch_meta, ch_idx, num_channels),
            alpha=alpha,
            **plot_kwargs,
        )
        if lines_drawn is not None:
            lines_drawn(ax_i, list(ax_i.lines[-1:]), [ch_idx])
        ax_i.set(ylabel=ch_ylabel, title=ch_meta.label, **ax_set)

    axes_list[-1].set(xlabel=xlabel)
    fig.suptitle(title or default_title)
    plt.tight_layout()
    plt.show()
    return _return_axes_iterator(fig.axes)


_AUTO_MAX_POINTS = "auto"
# A bin is drawn as its minimum, its middle sample and its maximum. The middle sample
# keeps vertices inside the band, so a legend at loc="best" still avoids the waveform.
_POINTS_PER_BIN = 3
# The first and last sample of the span are drawn besides the bins.
_ENDPOINTS = 2


def _draws_along_every_sample(line: Any) -> bool:
    """Return whether ``line`` draws markers or dashes, which follow every drawn point."""
    return line.get_marker() not in (None, "", " ", "None", "none") or line.get_linestyle() not in ("-", "None")


def _has_linear_scales(ax: Axes) -> bool:
    """Return whether both axes are linear: equal sample bins match pixels, and log limits need every sample."""
    return ax.get_xscale() == "linear" and ax.get_yscale() == "linear"


def _validate_max_points(max_points: Any) -> int | str | None:
    """Return a waveform ``max_points`` option: a budget, ``"auto"``, or ``None`` to draw every sample."""
    if max_points is None or (isinstance(max_points, str) and max_points == _AUTO_MAX_POINTS):
        return max_points
    min_points = _POINTS_PER_BIN + _ENDPOINTS
    if isinstance(max_points, bool) or not isinstance(max_points, int | np.integer) or max_points < min_points:
        raise ValueError(
            "Invalid max_points for waveform plot\n"
            f"  Got: {max_points!r}\n"
            f'  Expected: "auto", an integer of at least {min_points}, or None to draw every sample\n'
            "Omit it to fit the axes width, pass a point budget such as max_points=4000, "
            "or pass None to draw every sample."
        )
    return int(max_points)


class _MinMaxEnvelope:
    """Draw a min-max envelope of long waveforms and re-reduce it whenever the axes is drawn.

    Spans longer than the point budget are split into equal sample bins; each bin is
    drawn as its minimum, its middle sample and its maximum at the times of those
    samples, in time order, between the first and the last sample of the span, so
    peaks stay visible where they occur and the line spans the whole view. Spans
    within the budget are drawn sample by sample. ``max_points="auto"`` gives one bin
    per horizontal pixel of the axes, so the envelope looks like every sample at any
    figure size or dpi. Each tracked axes re-reduces its own lines from the full
    samples when its x-limits or budget change, so zooming in shows the original
    samples again.
    """

    def __init__(self, time: np.ndarray, data: np.ndarray, max_points: int | str) -> None:
        self._time = time
        # Bins are reduced along each channel's samples, which is fastest in row-major memory.
        self._data = np.ascontiguousarray(data)
        self._max_points = max_points

    def budget(self, ax: Axes | None) -> int:
        """Return the point budget for ``ax`` (the figure default width when ``ax`` is ``None``)."""
        if not isinstance(self._max_points, str):
            return self._max_points
        if ax is None:
            plt = _matplotlib_pyplot("waveform plot")
            width = plt.rcParams["figure.figsize"][0] * plt.rcParams["figure.dpi"]
        else:
            width = ax.bbox.width
        return _POINTS_PER_BIN * max(1, int(np.ceil(width))) + _ENDPOINTS

    def reduce(
        self, span: tuple[float, float] | None, budget: int, channels: list[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(time, data)`` for ``channels`` (all if ``None``) inside ``span`` (all samples if ``None``).

        ``time`` is the shared 1-D sample times when every sample fits the budget, and
        otherwise a 2-D array with one row of envelope times per channel.
        """
        start, stop = 0, self._time.shape[0]
        if span is not None:
            low, high = sorted(span)
            # Keep one sample beyond each edge so the line reaches the axes border.
            start = max(int(np.searchsorted(self._time, low, side="left")) - 1, 0)
            stop = min(int(np.searchsorted(self._time, high, side="right")) + 1, stop)
        time = self._time[start:stop]
        rows: slice | list[int] = slice(None) if channels is None else channels
        if channels and channels == list(range(channels[0], channels[-1] + 1)):
            # Consecutive channels are a view; a list index would copy every sample of the span.
            rows = slice(channels[0], channels[-1] + 1)
        data = self._data[rows, start:stop]
        count = time.shape[0]
        if count <= budget:
            return time, data
        width = -(-count // ((budget - _ENDPOINTS) // _POINTS_PER_BIN))
        whole = count // width
        # Whole bins are a view of the samples; a shorter last bin is reduced on its own.
        groups = [data[:, : whole * width].reshape(data.shape[0], whole, width)]
        if whole * width < count:
            groups.append(data[:, whole * width :, None].transpose(0, 2, 1))
        extremes = [_bin_extreme_indices(grouped) for grouped in groups]
        lowest_at = np.concatenate([lowest for lowest, _ in extremes], axis=1)
        highest_at = np.concatenate([highest for _, highest in extremes], axis=1)
        offsets = np.arange(lowest_at.shape[1]) * width
        middle = np.broadcast_to(np.minimum(offsets + width // 2, count - 1), lowest_at.shape)
        # A missing middle sample would break the line inside its bin; repeat the minimum instead.
        middle = np.where(np.isnan(np.take_along_axis(data, middle, axis=1)), lowest_at + offsets, middle)
        # Each bin's three samples in time order.
        picked = np.sort(np.stack([lowest_at + offsets, middle, highest_at + offsets], axis=2), axis=2)
        channel_rows = np.arange(data.shape[0])[:, None]
        indices = np.empty((data.shape[0], picked.shape[1] * _POINTS_PER_BIN + _ENDPOINTS), dtype=np.intp)
        indices[:, 0] = 0
        indices[:, 1:-1] = picked.reshape(data.shape[0], -1)
        indices[:, -1] = count - 1
        # Each channel's points sit at the times of its own samples.
        values = data[channel_rows, indices]
        times = time[indices]
        return times, values

    def track(self, ax: Axes, lines: list[Any], channels: list[int]) -> None:
        """Re-reduce ``lines`` (one per channel index) before ``ax`` draws them."""
        drawn: dict[str, Any] = {}

        def refresh() -> None:
            # The effective style includes rcParams and the property cycle, so read it from the lines.
            linear = _has_linear_scales(ax)
            every = tuple(
                self._max_points == _AUTO_MAX_POINTS and (not linear or _draws_along_every_sample(line))
                for line in lines
            )
            key = (ax.get_xlim(), self.budget(ax), every)
            if drawn.get("key") == key:
                return
            drawn["key"] = key
            # Markers, dashes and nonlinear axes keep every sample under the automatic budget.
            # Those lines get every sample, not only the visible span, so markevery indices keep their meaning.
            for i in (i for i, full in enumerate(every) if full):
                lines[i].set_data(self._time, self._data[channels[i]])
            tracked = [i for i, full in enumerate(every) if not full]
            if tracked:
                # Reduce only this axes' channels: shared-x subplots each refresh their own.
                time, data = self.reduce(key[0], key[1], [channels[i] for i in tracked])
                for i, times, values in zip(tracked, np.broadcast_to(time, data.shape), data, strict=True):
                    lines[i].set_data(times, values)

        ax.add_artist(_envelope_refresher_type()(refresh))


def _bin_extreme_indices(grouped: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the in-bin indices of each bin's first minimum and maximum, skipping NaN samples.

    ``grouped`` has shape ``(channels, bins, width)``. An all-NaN bin points to its first sample.
    """
    lowest_at = np.argmin(grouped, axis=2)
    highest_at = np.argmax(grouped, axis=2)
    for found, extreme in ((lowest_at, np.fmin), (highest_at, np.fmax)):
        # argmin/argmax stop at the first NaN; look again only in bins that also hold a valid sample.
        picked = np.take_along_axis(grouped, found[..., None], axis=2)[..., 0]
        missing = np.isnan(picked)
        if missing.any():
            rows = grouped[missing]
            values = extreme.reduce(rows, axis=1)
            found[missing] = np.where(np.isnan(values), 0, np.argmax(rows == values[:, None], axis=1))
    return lowest_at, highest_at


@cache
def _envelope_refresher_type() -> Any:
    """Return an invisible artist type that calls its ``refresh`` before the axes draws its lines."""
    artist_cls = _matplotlib_artist_type("waveform plot")

    class _EnvelopeRefresher(artist_cls):
        # Axes draws children in zorder, so this runs before any line.
        zorder = -np.inf

        def __init__(self, refresh: Callable[[], None]) -> None:
            super().__init__()
            self._refresh = refresh
            # It draws nothing, so it must not stretch a tight layout or bbox.
            self.set_in_layout(False)

        def draw(self, renderer: Any) -> None:
            self._refresh()
            self.stale = False

        def __reduce__(self) -> tuple[Callable[[], Any], tuple[()]]:
            # A pickled figure keeps the lines as last drawn; the full samples stay behind.
            return _inert_artist, ()

    return _EnvelopeRefresher


def _inert_artist() -> Any:
    """Return an artist that draws nothing and takes no space, standing in for an unpickled refresher."""
    artist = _matplotlib_artist_type("waveform plot")()
    artist.set_in_layout(False)
    return artist


class WaveformPlotStrategy(PlotStrategy["ChannelFrame"]):
    """Strategy for waveform plotting"""

    name = "waveform"

    def channel_plot(
        self,
        x: Any,
        y: Any,
        ax: Axes,
        label: PlotLabel = None,
        alpha: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Implementation of channel plotting"""
        if label is not None:
            kwargs["label"] = label
        if alpha is not None:
            kwargs["alpha"] = alpha
        ax.plot(x, y, **kwargs)
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        if label is not None:
            ax.legend()

    def plot(
        self,
        bf: ChannelFrame,
        ax: Axes | None = None,
        title: str | None = None,
        overlay: bool = False,
        **kwargs: Any,
    ) -> Axes | Iterator[Axes]:
        """Waveform plotting"""
        kwargs = kwargs or {}
        axes_cls = _matplotlib_axes_type("waveform plot")
        line2d_cls = _matplotlib_line2d_type("waveform plot")
        explicit_ylabel = "ylabel" in kwargs
        append_channel_units = kwargs.pop("_append_channel_units", not explicit_ylabel)
        ylabel = kwargs.pop("ylabel", "Amplitude")
        xlabel = kwargs.pop("xlabel", "Time [s]")
        alpha = kwargs.pop("alpha", 1)
        label = kwargs.pop("label", None)
        max_points = _validate_max_points(kwargs.pop("max_points", _AUTO_MAX_POINTS))
        plot_kwargs = filter_kwargs(line2d_cls, kwargs, strict_mode=True)
        ax_set = filter_kwargs(axes_cls.set, kwargs, strict_mode=True)
        data = _reshape_to_2d(bf.data)
        time = bf.time
        lines_drawn = None
        if max_points is not None:
            envelope = _MinMaxEnvelope(np.asarray(time), np.asarray(data), max_points)
            # A nonlinear axes keeps every sample, and autoscales from the smallest positive sample.
            if max_points != _AUTO_MAX_POINTS or ax is None or _has_linear_scales(ax):
                time, data = envelope.reduce(None, envelope.budget(ax))
            lines_drawn = envelope.track
        channel_units = [ch_meta.unit for ch_meta in bf.channels]
        all_level_units = bool(channel_units) and all(unit.startswith("dB ") for unit in channel_units)
        if not explicit_ylabel and all_level_units:
            ylabel = "Level"

        def _waveform_ylabel(ylabel: str, ch_meta: Any) -> str:
            if not append_channel_units:
                return ylabel
            unit_suffix = f" [{ch_meta.unit}]" if ch_meta.unit else ""
            return f"{ylabel}{unit_suffix}"

        if (overlay or ax is not None) and append_channel_units:
            if channel_units and all(unit and unit == channel_units[0] for unit in channel_units):
                ylabel = f"{ylabel} [{channel_units[0]}]"
            elif all_level_units:
                ylabel = f"{ylabel} [dB re channel reference]"

        return _plot_line_layout(
            self,
            bf,
            time,
            data,
            ax=ax,
            title=title,
            overlay=overlay,
            ylabel=ylabel,
            xlabel=xlabel,
            default_title=bf.label or "Channel Data",
            alpha=alpha,
            label=label,
            plot_kwargs=plot_kwargs,
            ax_set=ax_set,
            per_channel_ylabel_fn=_waveform_ylabel,
            lines_drawn=lines_drawn,
        )


class FrequencyPlotStrategy(PlotStrategy["SpectralFrame"]):
    """Strategy for frequency domain plotting"""

    name = "frequency"

    def channel_plot(
        self,
        x: Any,
        y: Any,
        ax: Axes,
        label: PlotLabel = None,
        alpha: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Implementation of channel plotting"""
        if label is not None:
            kwargs["label"] = label
        if alpha is not None and alpha != 1.0:
            kwargs["alpha"] = alpha
        ax.plot(x, y, **kwargs)
        ax.grid(True)
        if label is not None:
            ax.legend()

    def plot(
        self,
        bf: SpectralFrame,
        ax: Axes | None = None,
        title: str | None = None,
        overlay: bool = False,
        **kwargs: Any,
    ) -> Axes | Iterator[Axes]:
        """Frequency domain plotting"""
        kwargs = kwargs or {}
        axes_cls = _matplotlib_axes_type("frequency plot")
        line2d_cls = _matplotlib_line2d_type("frequency plot")
        is_aw = kwargs.pop("Aw", False)
        view = kwargs.pop("view", None)
        typed_values = _typed_frequency_values(bf, view=view, aw=is_aw)
        if typed_values is not None:
            data, default_ylabel = typed_values
            ylabel = kwargs.pop("ylabel", default_ylabel)
        else:
            if view is not None:
                raise ValueError(
                    "The 'view' argument is supported only by a typed quantity Frame; "
                    "ordinary SpectralFrame plotting uses its amplitude level."
                )
            data = bf.dBA if is_aw else bf.dB
            default_ylabel = (
                "A-weighted amplitude level [dB re channel ref]" if is_aw else "Amplitude level [dB re channel ref]"
            )
            ylabel = kwargs.pop("ylabel", default_ylabel)
        data = _reshape_to_2d(data)
        xlabel = kwargs.pop("xlabel", "Frequency [Hz]")
        alpha = kwargs.pop("alpha", 1)
        label = kwargs.pop("label", None)
        plot_kwargs = filter_kwargs(line2d_cls, kwargs, strict_mode=True)
        ax_set = filter_kwargs(axes_cls.set, kwargs, strict_mode=True)

        return _plot_line_layout(
            self,
            bf,
            bf.freqs,
            data,
            ax=ax,
            title=title,
            overlay=overlay,
            ylabel=ylabel,
            xlabel=xlabel,
            default_title=bf.label or "Channel Data",
            alpha=alpha,
            label=label,
            plot_kwargs=plot_kwargs,
            ax_set=ax_set,
        )


class NOctPlotStrategy(PlotStrategy["NOctFrame"]):
    """Strategy for N-octave band analysis plotting"""

    name = "noct"

    def channel_plot(
        self,
        x: Any,
        y: Any,
        ax: Axes,
        label: PlotLabel = None,
        alpha: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Implementation of channel plotting"""
        if label is not None:
            kwargs["label"] = label
        if alpha is not None and alpha != 1.0:
            kwargs["alpha"] = alpha
        ax.step(x, y, **kwargs)
        ax.grid(True)
        if label is not None:
            ax.legend()

    def plot(
        self,
        bf: NOctFrame,
        ax: Axes | None = None,
        title: str | None = None,
        overlay: bool = False,
        **kwargs: Any,
    ) -> Axes | Iterator[Axes]:
        """N-octave band analysis plotting"""
        kwargs = kwargs or {}
        axes_cls = _matplotlib_axes_type("noct plot")
        line2d_cls = _matplotlib_line2d_type("noct plot")
        is_aw = kwargs.pop("Aw", False)

        if is_aw:
            data = bf.dBA
            default_ylabel = "A-weighted band RMS level [dB re channel ref]"
        else:
            data = bf.dB
            default_ylabel = "Band RMS level [dB re channel ref]"
        data = _reshape_to_2d(data)
        ylabel = kwargs.pop("ylabel", default_ylabel)
        xlabel = kwargs.pop("xlabel", "Center frequency [Hz]")
        alpha = kwargs.pop("alpha", 1)
        label = kwargs.pop("label", None)
        plot_kwargs = filter_kwargs(line2d_cls, kwargs, strict_mode=True)
        ax_set = filter_kwargs(axes_cls.set, kwargs, strict_mode=True)

        default_title = bf.label or f"1/{bf.n!s}-Octave Spectrum"
        return _plot_line_layout(
            self,
            bf,
            bf.freqs,
            data,
            ax=ax,
            title=title,
            overlay=overlay,
            ylabel=ylabel,
            xlabel=xlabel,
            default_title=default_title,
            alpha=alpha,
            label=label,
            plot_kwargs=plot_kwargs,
            ax_set=ax_set,
        )


class SpectrogramPlotStrategy(PlotStrategy["SpectrogramFrame"]):
    """Strategy for spectrogram plotting"""

    name = "spectrogram"

    def channel_plot(
        self,
        x: Any,
        y: Any,
        ax: Axes,
        label: PlotLabel = None,
        alpha: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Implementation of channel plotting"""

    def plot(
        self,
        bf: SpectrogramFrame,
        ax: Axes | None = None,
        title: str | None = None,
        overlay: bool = False,
        **kwargs: Any,
    ) -> Axes | Iterator[Axes]:
        """Spectrogram plotting"""
        # Explicit overlay mode is not supported for spectrograms
        if overlay:
            raise ValueError("Overlay is not supported for SpectrogramPlotStrategy.")

        # If an Axes is provided, allow drawing into it only for single-channel frames
        if ax is not None and bf.n_channels > 1:
            raise ValueError("ax must be None when n_channels > 1.")

        kwargs = kwargs or {}
        plt = _matplotlib_pyplot("spectrogram plot")
        axes_cls = _matplotlib_axes_type("spectrogram plot")
        figure_cls = _matplotlib_figure_type("spectrogram plot")

        is_aw = kwargs.pop("Aw", False)
        if is_aw:
            unit = "dBA"
            data = bf.dBA
        else:
            unit = "dB"
            data = bf.dB
        data = _reshape_spectrogram_data(data)

        cmap = kwargs.pop("cmap", "jet")
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)
        fmin = kwargs.pop("fmin", 0)
        fmax = kwargs.pop("fmax", None)
        xlim = kwargs.pop("xlim", None)
        ylim = kwargs.pop("ylim", None)
        shading = kwargs.pop("shading", "auto")
        ax_set_kwargs = filter_kwargs(axes_cls.set, kwargs, strict_mode=True)

        def draw_spectrogram(target_ax: Axes, channel_data: np.ndarray, plot_title: str | None) -> Any:
            times, freqs = _spectrogram_axis_values(bf, channel_data)
            image_kwargs = filter_kwargs(target_ax.pcolormesh, kwargs, strict_mode=True)
            img = target_ax.pcolormesh(
                times,
                freqs,
                channel_data,
                shading=shading,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                **image_kwargs,
            )
            spectrogram_ylim = ylim if ylim is not None else (fmin, fmax) if fmin != 0 or fmax is not None else None
            target_ax.set(
                title=plot_title,
                ylabel="Frequency [Hz]",
                xlabel="Time [s]",
                xlim=xlim,
                ylim=spectrogram_ylim,
                **ax_set_kwargs,
            )
            return img

        if ax is not None:
            img = draw_spectrogram(ax, data[0], title or bf.label or "Spectrogram")

            fig = ax.figure
            if fig is not None:
                try:
                    cbar = fig.colorbar(img, ax=ax)
                    cbar.set_label(f"Spectrum level [{unit}]")
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Failed to create colorbar for spectrogram: {type(e).__name__}: {e}")
            return ax

        # Create a new figure if ax is None
        num_channels = bf.n_channels
        fig, axs = plt.subplots(num_channels, 1, figsize=(10, 5 * num_channels), sharex=True)
        if not isinstance(fig, figure_cls):
            raise ValueError("fig must be a matplotlib Figure object.")
        # Convert axs to array if it is a single Axes object
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])

        for ax_i, channel_data, ch_meta in zip(axs.flatten(), data, bf.channels, strict=True):
            img = draw_spectrogram(ax_i, channel_data, ch_meta.label)
            try:
                cbar = ax_i.figure.colorbar(img, ax=ax_i)
                cbar.set_label(f"Spectrum level [{unit}]")
            except (ValueError, AttributeError) as e:
                logger.warning(f"Failed to create colorbar for spectrogram: {type(e).__name__}: {e}")
            fig.suptitle(title or "Spectrogram Data")
        plt.tight_layout()
        plt.show()

        return _return_axes_iterator(fig.axes)


class DescribePlotStrategy(PlotStrategy["ChannelFrame"]):
    """Strategy for visualizing ChannelFrame data with describe plot"""

    name = "describe"

    def channel_plot(
        self,
        x: Any,
        y: Any,
        ax: Axes,
        label: PlotLabel = None,
        alpha: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Implementation of channel plotting"""
        # This method is not used for describe plot

    def plot(
        self,
        bf: ChannelFrame,
        ax: Axes | None = None,
        title: str | None = None,
        overlay: bool = False,
        **kwargs: Any,
    ) -> Axes | Iterator[Axes]:
        """Implementation of describe method for visualizing ChannelFrame data"""

        plt = _matplotlib_pyplot("describe plot")
        gridspec = _matplotlib_gridspec("describe plot")

        fmin = kwargs.pop("fmin", 0)
        fmax = kwargs.pop("fmax", None)
        cmap = kwargs.pop("cmap", "jet")
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)
        xlim = kwargs.pop("xlim", None)
        ylim = kwargs.pop("ylim", None)
        is_aw = kwargs.pop("Aw", False)
        waveform = kwargs.pop("waveform", {})
        spectral = kwargs.pop("spectral", {"xlim": (vmin, vmax)})

        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 3], width_ratios=[3, 1, 0.1])
        gs.update(wspace=0.2)

        fig = plt.figure(figsize=(12, 6))
        fig.subplots_adjust(wspace=0.0001)

        # First subplot (Time Plot)
        ax_1 = fig.add_subplot(gs[0])
        bf.plot(plot_type="waveform", ax=ax_1, overlay=True)
        ax_1.set(**waveform)
        ax_1.legend().set_visible(False)
        ax_1.set(xlabel="", title="")

        # Second subplot (STFT Plot)
        ax_2 = fig.add_subplot(gs[3], sharex=ax_1)
        stft_ch = bf.stft()
        if is_aw:
            unit = "dBA"
            channel_data = stft_ch.dBA
        else:
            unit = "dB"
            channel_data = stft_ch.dB
        if channel_data.ndim == 3:
            channel_data = channel_data[0]
        # Get the maximum value of the data and round it to a convenient value
        if vmax is None:
            data_max = np.nanmax(channel_data)
            # Round to a convenient number with increments of 10, 5, or 2
            for step in [10, 5, 2]:
                rounded_max = np.ceil(data_max / step) * step
                if rounded_max >= data_max:
                    vmax = rounded_max
                    vmin = vmax - 180
                    break
        times, freqs = _spectrogram_axis_values(stft_ch, channel_data)
        img = ax_2.pcolormesh(times, freqs, channel_data, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        spectrogram_ylim = ylim if ylim is not None else (fmin, fmax) if fmin != 0 or fmax is not None else None
        ax_2.set(xlabel="Time [s]", ylabel="Frequency [Hz]", xlim=xlim, ylim=spectrogram_ylim)

        # Third subplot
        ax_3 = fig.add_subplot(gs[1])
        ax_3.axis("off")

        # Fourth subplot (Welch Plot)
        ax_4 = fig.add_subplot(gs[4], sharey=ax_2)
        welch_ch = bf.welch()
        if is_aw:
            unit = "dBA"
            data_db = welch_ch.dBA
        else:
            unit = "dB"
            data_db = welch_ch.dB
        ax_4.plot(data_db.T, welch_ch.freqs.T)
        ax_4.grid(True)
        ax_4.set(xlabel=f"Spectrum level [{unit}]", **spectral)

        cbar = fig.colorbar(img, ax=ax_4, format="%+2.0f")
        cbar.set_label(unit)
        fig.suptitle(title or bf.label or "Channel Data")

        return _return_axes_iterator(fig.axes)


class MatrixPlotStrategy(PlotStrategy["SpectralFrame"]):
    """Strategy for displaying relationships between channels in matrix format"""

    name = "matrix"

    def channel_plot(
        self,
        x: Any,
        y: Any,
        ax: Axes,
        label: PlotLabel = None,
        alpha: float = 1.0,
        **kwargs: Any,
    ) -> None:
        # Work on a local copy to avoid mutating caller-provided kwargs
        plot_kwargs = dict(kwargs)

        # Extract axes settings from kwargs copy
        title = plot_kwargs.pop("title", None)
        ylabel = plot_kwargs.pop("ylabel", None)
        xlabel = plot_kwargs.pop("xlabel", "Frequency [Hz]")

        if label is not None:
            plot_kwargs["label"] = label
        if alpha is not None and alpha != 1.0:
            plot_kwargs["alpha"] = alpha
        ax.plot(x, y, **plot_kwargs)
        ax.grid(True)
        if title is not None:
            ax.set_title(title)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        if label is not None:
            ax.legend()

    def plot(
        self,
        bf: SpectralFrame,
        ax: Axes | None = None,
        title: str | None = None,
        overlay: bool = False,
        **kwargs: Any,
    ) -> Axes | Iterator[Axes]:
        kwargs = kwargs or {}
        plt = _matplotlib_pyplot("matrix plot")
        axes_cls = _matplotlib_axes_type("matrix plot")
        line2d_cls = _matplotlib_line2d_type("matrix plot")
        is_aw = kwargs.pop("Aw", False)
        view = kwargs.pop("view", None)
        typed_entries = _typed_matrix_entries(bf, view=view, aw=is_aw)
        if typed_entries is not None:
            typed_ylabel = _typed_plot_ylabel(bf, view=view, aw=is_aw)
            if typed_ylabel is None:
                typed_values = _typed_frequency_values(bf, view=view, aw=is_aw)
                if typed_values is None:
                    raise TypeError(f"{type(bf).__name__} exposes matrix entries without a frequency plotting contract")
                _, default_ylabel = typed_values
            else:
                default_ylabel = typed_ylabel
            ylabel = kwargs.pop("ylabel", default_ylabel)
        else:
            if view is not None:
                raise ValueError(
                    "The 'view' argument is supported only by a typed quantity Frame; "
                    "ordinary SpectralFrame plotting uses its amplitude level."
                )
            if is_aw:
                data = bf.dBA
                default_ylabel = "A-weighted amplitude level [dB re channel ref]"
            else:
                data = bf.dB
                default_ylabel = "Amplitude level [dB re channel ref]"
            data = _reshape_to_2d(data)
            ylabel = kwargs.pop("ylabel", default_ylabel)

        xlabel = kwargs.pop("xlabel", "Frequency [Hz]")
        alpha = kwargs.pop("alpha", 1)
        plot_kwargs = filter_kwargs(line2d_cls, kwargs, strict_mode=True)
        ax_set = filter_kwargs(axes_cls.set, kwargs, strict_mode=True)

        # If an Axes is provided, prefer drawing into it (treat as overlay)
        if ax is not None:
            overlay = True
        if overlay:
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            else:
                fig = ax.figure
            if ax is None:
                raise RuntimeError(  # pragma: no cover
                    "Matplotlib did not provide an Axes for matrix plotting"
                )
            if typed_entries is None:
                self.channel_plot(
                    bf.freqs,
                    data.T,
                    ax,  # Always Axes type here
                    title=title or bf.label or "Spectral Data",
                    ylabel=ylabel,
                    xlabel=xlabel,
                    alpha=alpha,
                    **plot_kwargs,
                )
            else:
                for output_index, input_index, channel_data, pair_label in typed_entries:
                    del output_index, input_index
                    self.channel_plot(
                        bf.freqs,
                        channel_data,
                        ax,
                        label=pair_label,
                        title=title or bf.label or "Spectral Data",
                        ylabel=ylabel,
                        xlabel=xlabel,
                        alpha=alpha,
                        **plot_kwargs,
                    )
            ax.set(**ax_set)
            if fig is not None:
                fig.suptitle(title or bf.label or "Spectral Data")
            if ax.figure != fig:  # Only show if we created the figure
                plt.tight_layout()
                plt.show()
            return cast("Axes", ax)

        if typed_entries is not None:
            source_count = int(getattr(bf, "n_source_channels", 0))
            if source_count <= 0:
                raise ValueError("A typed pairwise matrix requires at least one source channel")
            fig, axs = plt.subplots(
                source_count,
                source_count,
                figsize=(3 * source_count, 3 * source_count),
                sharex=True,
                sharey=True,
            )
            axes_grid = np.asarray(axs, dtype=object)
            if axes_grid.ndim == 0:
                axes_grid = axes_grid.reshape(1, 1)
            elif axes_grid.ndim != 2:
                axes_grid = axes_grid.reshape(source_count, source_count)  # pragma: no cover
            entry_by_position = {
                (int(output_index), int(input_index)): (channel_data, pair_label)
                for output_index, input_index, channel_data, pair_label in typed_entries
            }
            for output_index in range(source_count):
                for input_index in range(source_count):
                    ax_i = axes_grid[output_index, input_index]
                    entry = entry_by_position.get((output_index, input_index))
                    if entry is not None:
                        channel_data, pair_label = entry
                        self.channel_plot(
                            bf.freqs,
                            channel_data,
                            ax_i,
                            title=pair_label,
                            ylabel=ylabel,
                            xlabel=xlabel,
                            alpha=alpha,
                            **plot_kwargs,
                        )
                    ax_i.set(**ax_set)
                    ax_i.set_xlabel(xlabel)
            fig.suptitle(title or bf.label or "Spectral Data")
            plt.tight_layout()
            plt.show()
            return _return_axes_iterator(fig.axes)

        num_channels = bf.n_channels
        num_rows = int(np.ceil(np.sqrt(num_channels)))
        fig, axs = plt.subplots(
            num_rows,
            num_rows,
            figsize=(3 * num_rows, 3 * num_rows),
            sharex=True,
            sharey=True,
        )
        if isinstance(axs, np.ndarray):
            axes_list = axs.flatten().tolist()
        elif isinstance(axs, list):
            import itertools

            axes_list = list(itertools.chain.from_iterable(axs))
        else:
            axes_list = [axs]
        for ax_i, channel_data, ch_meta in zip(axes_list, data, bf.channels, strict=False):
            self.channel_plot(
                bf.freqs,
                channel_data,
                ax_i,
                title=ch_meta.label,
                ylabel=ylabel,
                xlabel=xlabel,
                alpha=alpha,
                **plot_kwargs,
            )
            ax_i.set(**ax_set)
            ax_i.set_xlabel(xlabel)
        fig.suptitle(title or bf.label or "Spectral Data")
        plt.tight_layout()
        plt.show()
        return _return_axes_iterator(fig.axes)


# Maintain mapping of plot types to corresponding classes
_plot_strategies: dict[str, type[PlotStrategy[Any]]] = {}


def register_plot_strategy(strategy_cls: type) -> None:
    """Register a new plot strategy from a class"""
    if not issubclass(strategy_cls, PlotStrategy):
        raise TypeError("Strategy class must inherit from PlotStrategy.")
    if inspect.isabstract(strategy_cls):
        raise TypeError("Cannot register abstract PlotStrategy class.")
    _plot_strategies[strategy_cls.name] = strategy_cls


# Modified to auto-register only non-abstract subclasses
for strategy_cls in PlotStrategy.__subclasses__():
    if not inspect.isabstract(strategy_cls):
        register_plot_strategy(strategy_cls)


def get_plot_strategy(name: str) -> type[PlotStrategy[Any]]:
    """Get plot strategy by name"""
    if name not in _plot_strategies:
        raise ValueError(f"Unknown plot type: {name}")
    return _plot_strategies[name]


def create_operation(name: str, **params: Any) -> PlotStrategy[Any]:
    """Create operation instance from operation name and parameters"""
    operation_class = get_plot_strategy(name)
    return operation_class(**params)
