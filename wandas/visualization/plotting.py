import abc
import inspect
import logging
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

# Import librosa (including display)
import librosa

from wandas.utils.introspection import filter_kwargs

try:
    # Avoid error due to librosa.display not being explicitly exported
    from librosa import display
except ImportError:
    # fallback
    display = librosa.display

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from wandas.core.base_frame import BaseFrame
    from wandas.core.metadata import ChannelMetadata
    from wandas.frames.channel import ChannelFrame
    from wandas.frames.noct import NOctFrame
    from wandas.frames.spectral import SpectralFrame
    from wandas.frames.spectrogram import SpectrogramFrame

logger = logging.getLogger(__name__)

TFrame = TypeVar("TFrame", bound="BaseFrame[Any]")


class PlotStrategy(abc.ABC, Generic[TFrame]):
    """Base class for plotting strategies"""

    name: ClassVar[str]

    @abc.abstractmethod
    def channel_plot(
        self,
        x: Any,
        y: Any,
        ax: "Axes",
        label: str | None = None,
        alpha: float = 1.0,
    ) -> None:
        """Implementation of channel plotting"""

    @abc.abstractmethod
    def plot(
        self,
        bf: TFrame,
        ax: "Axes | None" = None,
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
    label: "str | Sequence[str] | None",
    ch_meta: "ChannelMetadata",
    channel_index: int,
    n_channels: int,
) -> str:
    """Resolve the label for a single channel in the non-overlay (per-subplot) path.

    Parameters
    ----------
    label : str | Sequence[str] | None
        User-supplied label override.  When a sequence is given its element at
        *channel_index* is used; when a plain string is given that string is
        used for every channel; when ``None`` the channel metadata label is
        used.
    ch_meta : ChannelMetadata
        Metadata for the current channel (provides the default label).
    channel_index : int
        Zero-based index of the current channel in the frame.
    n_channels : int
        Total number of channels in the frame. Used to validate per-channel
        label sequences before indexing.

    Returns
    -------
    str
        The resolved label string.
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

    Parameters
    ----------
    data : array-like
        Input data that may be 1D or already 2D+

    Returns
    -------
    array-like
        Data reshaped to ensure at least 2 dimensions
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def _reshape_spectrogram_data(data: Any) -> Any:
    """
    Reshape spectrogram data to 3D for consistent processing.

    This function ensures that spectrogram data has 3 dimensions:
    (channels, freqs, time). Handles both 1D and 2D input data.

    Parameters
    ----------
    data : array-like
        Input spectrogram data that may be 1D, 2D, or already 3D

    Returns
    -------
    array-like
        Data reshaped to ensure 3 dimensions for spectrogram plotting
    """
    if data.ndim == 1:
        # 1D data: reshape to (1, freqs, 1) - single channel, single time frame
        data = data.reshape((1, data.shape[0], 1))
    elif data.ndim == 2:
        # 2D data: reshape to (1, freqs, time) - single channel spectrogram
        data = data.reshape((1, *data.shape))
    return data


def _plot_line_layout(
    strategy: "PlotStrategy[Any]",
    bf: "BaseFrame[Any]",
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
) -> Axes | Iterator[Axes]:
    """Shared layout for line-based plot strategies (waveform, frequency, noct).

    Handles overlay (single axes) and multi-subplot (per-channel) modes.

    Parameters
    ----------
    strategy : PlotStrategy
        The strategy whose ``channel_plot`` method will be called.
    bf : BaseFrame
        The frame being plotted (provides ``n_channels``, ``channels``, ``labels``).
    x_data : array-like
        X-axis values (e.g. ``bf.time`` or ``bf.freqs``).
    data_2d : array-like
        2-D array of shape ``(n_channels, n_samples)`` already reshaped via
        ``_reshape_to_2d``.
    per_channel_ylabel_fn : callable, optional
        ``(ylabel, ch_meta) -> str`` override for per-channel y-labels
        (e.g. to append a unit suffix).
    """
    plot_kwargs = plot_kwargs or {}
    ax_set = ax_set or {}

    if ax is not None:
        overlay = True

    if overlay:
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        strategy.channel_plot(
            x_data,
            data_2d.T,
            ax,
            label=label if label is not None else (bf.labels[0] if isinstance(bf.labels, list) else bf.labels),
            alpha=alpha,
            **plot_kwargs,
        )
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
            x_data,
            channel_data,
            ax_i,
            label=_resolve_channel_label(label, ch_meta, ch_idx, num_channels),
            alpha=alpha,
            **plot_kwargs,
        )
        ax_i.set(ylabel=ch_ylabel, title=ch_meta.label, **ax_set)

    axes_list[-1].set(xlabel=xlabel)
    fig.suptitle(title or default_title)
    plt.tight_layout()
    plt.show()
    return _return_axes_iterator(fig.axes)


class WaveformPlotStrategy(PlotStrategy["ChannelFrame"]):
    """Strategy for waveform plotting"""

    name = "waveform"

    def channel_plot(
        self,
        x: Any,
        y: Any,
        ax: "Axes",
        label: str | None = None,
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
        bf: "ChannelFrame",
        ax: "Axes | None" = None,
        title: str | None = None,
        overlay: bool = False,
        **kwargs: Any,
    ) -> Axes | Iterator[Axes]:
        """Waveform plotting"""
        kwargs = kwargs or {}
        ylabel = kwargs.pop("ylabel", "Amplitude")
        xlabel = kwargs.pop("xlabel", "Time [s]")
        alpha = kwargs.pop("alpha", 1)
        label = kwargs.pop("label", None)
        plot_kwargs = filter_kwargs(Line2D, kwargs, strict_mode=True)
        ax_set = filter_kwargs(Axes.set, kwargs, strict_mode=True)
        data = _reshape_to_2d(bf.data)

        def _waveform_ylabel(ylabel: str, ch_meta: Any) -> str:
            unit_suffix = f" [{ch_meta.unit}]" if ch_meta.unit else ""
            return f"{ylabel}{unit_suffix}"

        return _plot_line_layout(
            self,
            bf,
            bf.time,
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
        )


class FrequencyPlotStrategy(PlotStrategy["SpectralFrame"]):
    """Strategy for frequency domain plotting"""

    name = "frequency"

    def channel_plot(
        self,
        x: Any,
        y: Any,
        ax: "Axes",
        label: str | None = None,
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
        bf: "SpectralFrame",
        ax: "Axes | None" = None,
        title: str | None = None,
        overlay: bool = False,
        **kwargs: Any,
    ) -> Axes | Iterator[Axes]:
        """Frequency domain plotting"""
        kwargs = kwargs or {}
        is_aw = kwargs.pop("Aw", False)
        if len(bf.operation_history) > 0 and bf.operation_history[-1]["operation"] == "coherence":
            data = bf.magnitude
            ylabel = kwargs.pop("ylabel", "coherence")
        else:
            if is_aw:
                unit = "dBA"
                data = bf.dBA
            else:
                unit = "dB"
                data = bf.dB
            ylabel = kwargs.pop("ylabel", f"Spectrum level [{unit}]")
        data = _reshape_to_2d(data)
        xlabel = kwargs.pop("xlabel", "Frequency [Hz]")
        alpha = kwargs.pop("alpha", 1)
        label = kwargs.pop("label", None)
        plot_kwargs = filter_kwargs(Line2D, kwargs, strict_mode=True)
        ax_set = filter_kwargs(Axes.set, kwargs, strict_mode=True)

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
        ax: "Axes",
        label: str | None = None,
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
        bf: "NOctFrame",
        ax: "Axes | None" = None,
        title: str | None = None,
        overlay: bool = False,
        **kwargs: Any,
    ) -> Axes | Iterator[Axes]:
        """N-octave band analysis plotting"""
        kwargs = kwargs or {}
        is_aw = kwargs.pop("Aw", False)

        if is_aw:
            unit = "dBrA"
            data = bf.dBA
        else:
            unit = "dBr"
            data = bf.dB
        data = _reshape_to_2d(data)
        ylabel = kwargs.pop("ylabel", f"Spectrum level [{unit}]")
        xlabel = kwargs.pop("xlabel", "Center frequency [Hz]")
        alpha = kwargs.pop("alpha", 1)
        label = kwargs.pop("label", None)
        plot_kwargs = filter_kwargs(Line2D, kwargs, strict_mode=True)
        ax_set = filter_kwargs(Axes.set, kwargs, strict_mode=True)

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
        ax: "Axes",
        label: str | None = None,
        alpha: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Implementation of channel plotting"""

    def plot(
        self,
        bf: "SpectrogramFrame",
        ax: "Axes | None" = None,
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

        is_aw = kwargs.pop("Aw", False)
        if is_aw:
            unit = "dBA"
            data = bf.dBA
        else:
            unit = "dB"
            data = bf.dB
        data = _reshape_spectrogram_data(data)
        specshow_kwargs = filter_kwargs(display.specshow, kwargs, strict_mode=True)
        ax_set_kwargs = filter_kwargs(Axes.set, kwargs, strict_mode=True)

        cmap = kwargs.pop("cmap", "jet")
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)

        if ax is not None:
            img = display.specshow(
                data=data[0],
                sr=bf.sampling_rate,
                hop_length=bf.hop_length,
                n_fft=bf.n_fft,
                win_length=bf.win_length,
                x_axis="time",
                y_axis="linear",
                cmap=cmap,
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                **specshow_kwargs,
            )
            ax.set(
                title=title or bf.label or "Spectrogram",
                ylabel="Frequency [Hz]",
                xlabel="Time [s]",
                **ax_set_kwargs,
            )

            fig = ax.figure
            if fig is not None:
                try:
                    cbar = fig.colorbar(img, ax=ax)
                    cbar.set_label(f"Spectrum level [{unit}]")
                except (ValueError, AttributeError) as e:
                    # Handle case where img doesn't have proper colorbar properties
                    logger.warning(f"Failed to create colorbar for spectrogram: {type(e).__name__}: {e}")
            return ax

        # Create a new figure if ax is None
        num_channels = bf.n_channels
        fig, axs = plt.subplots(num_channels, 1, figsize=(10, 5 * num_channels), sharex=True)
        if not isinstance(fig, Figure):
            raise ValueError("fig must be a matplotlib Figure object.")
        # Convert axs to array if it is a single Axes object
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])

        for ax_i, channel_data, ch_meta in zip(axs.flatten(), data, bf.channels, strict=True):
            img = display.specshow(
                data=channel_data,
                sr=bf.sampling_rate,
                hop_length=bf.hop_length,
                n_fft=bf.n_fft,
                win_length=bf.win_length,
                x_axis="time",
                y_axis="linear",
                ax=ax_i,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                **specshow_kwargs,
            )
            ax_i.set(
                title=ch_meta.label,
                ylabel="Frequency [Hz]",
                xlabel="Time [s]",
                **ax_set_kwargs,
            )
            try:
                cbar = ax_i.figure.colorbar(img, ax=ax_i)
                cbar.set_label(f"Spectrum level [{unit}]")
            except (ValueError, AttributeError) as e:
                # Handle case where img doesn't have proper colorbar properties
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
        ax: "Axes",
        label: str | None = None,
        alpha: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Implementation of channel plotting"""
        # This method is not used for describe plot

    def plot(
        self,
        bf: "ChannelFrame",
        ax: "Axes | None" = None,
        title: str | None = None,
        overlay: bool = False,
        **kwargs: Any,
    ) -> Axes | Iterator[Axes]:
        """Implementation of describe method for visualizing ChannelFrame data"""

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
        img = display.specshow(
            data=channel_data,
            sr=bf.sampling_rate,
            hop_length=stft_ch.hop_length,
            n_fft=stft_ch.n_fft,
            win_length=stft_ch.win_length,
            x_axis="time",
            y_axis="linear",
            ax=ax_2,
            fmin=fmin,
            fmax=fmax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax_2.set(xlim=xlim, ylim=ylim)

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
        ax: "Axes",
        label: str | None = None,
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
        bf: "SpectralFrame",
        ax: "Axes | None" = None,
        title: str | None = None,
        overlay: bool = False,
        **kwargs: Any,
    ) -> Axes | Iterator[Axes]:
        kwargs = kwargs or {}
        is_aw = kwargs.pop("Aw", False)
        if len(bf.operation_history) > 0 and bf.operation_history[-1]["operation"] == "coherence":
            unit = ""
            data = bf.magnitude
            ylabel = kwargs.pop("ylabel", "coherence")
        else:
            if is_aw:
                unit = "dBA"
                data = bf.dBA
            else:
                unit = "dB"
                data = bf.dB
            ylabel = kwargs.pop("ylabel", f"Spectrum level [{unit}]")

        data = _reshape_to_2d(data)

        xlabel = kwargs.pop("xlabel", "Frequency [Hz]")
        alpha = kwargs.pop("alpha", 1)
        plot_kwargs = filter_kwargs(Line2D, kwargs, strict_mode=True)
        ax_set = filter_kwargs(Axes.set, kwargs, strict_mode=True)
        num_channels = bf.n_channels
        # If an Axes is provided, prefer drawing into it (treat as overlay)
        if ax is not None:
            overlay = True
        if overlay:
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            else:
                fig = ax.figure
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
            ax.set(**ax_set)
            if fig is not None:
                fig.suptitle(title or bf.label or "Spectral Data")
            if ax.figure != fig:  # Only show if we created the figure
                plt.tight_layout()
                plt.show()
            return ax
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
