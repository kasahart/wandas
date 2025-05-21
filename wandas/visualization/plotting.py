import abc
import inspect
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Optional, TypeVar, Union

# Import librosa (including display)
import librosa

from wandas.utils.introspection import filter_kwargs

try:
    # Avoid error due to librosa.display not being explicitly exported
    from librosa import display  # type: ignore
except ImportError:
    # fallback
    display = librosa.display  # type: ignore

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from wandas.core.base_frame import BaseFrame
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
    def channel_plot(self, x: Any, y: Any, ax: "Axes") -> None:
        """Implementation of channel plotting"""
        pass

    @abc.abstractmethod
    def plot(
        self,
        bf: TFrame,
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        overlay: bool = False,
        **kwargs: Any,
    ) -> Union["Axes", Iterator["Axes"]]:
        """Implementation of plotting"""
        pass


# Helper function for return type
def _return_axes_iterator(axes_list: Any) -> Iterator[Axes]:
    """Helper to convert fig.axes to Iterator[Axes] with proper typing"""
    return iter(axes_list)


class WaveformPlotStrategy(PlotStrategy["ChannelFrame"]):
    """Strategy for waveform plotting"""

    name = "waveform"

    def channel_plot(
        self,
        x: Any,
        y: Any,
        ax: "Axes",
        **kwargs: Any,
    ) -> None:
        """Implementation of channel plotting"""
        ax.plot(x, y, **kwargs)
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        if "label" in kwargs:
            ax.legend()

    def plot(
        self,
        bf: "ChannelFrame",
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        overlay: bool = False,
        **kwargs: Any,
    ) -> Union["Axes", Iterator["Axes"]]:
        """Waveform plotting"""
        kwargs = kwargs or {}
        ylabel = kwargs.pop("ylabel", "Amplitude")
        xlabel = kwargs.pop("xlabel", "Time [s]")
        alpha = kwargs.pop("alpha", 1)
        plot_kwargs = filter_kwargs(
            Line2D,
            kwargs,
            strict_mode=True,
        )
        ax_set = filter_kwargs(
            Axes.set,
            kwargs,
            strict_mode=True,
        )
        data = bf.data
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if overlay:
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 4))

            self.channel_plot(
                bf.time, data.T, ax, label=bf.labels, alpha=alpha, **plot_kwargs
            )
            ax.set(
                ylabel=ylabel,
                title=title or bf.label or "Channel Data",
                xlabel=xlabel,
                **ax_set,
            )
            if ax is None:
                fig.suptitle(title or bf.label or None)
                plt.tight_layout()
                plt.show()
            return ax
        else:
            num_channels = bf.n_channels
            fig, axs = plt.subplots(
                num_channels, 1, figsize=(10, 4 * num_channels), sharex=True
            )
            # Convert axs to list if it is a single Axes object
            if not isinstance(axs, (list, np.ndarray)):
                axs = [axs]

            axes_list = list(axs)
            for ax_i, channel_data, ch_meta in zip(axes_list, data, bf.channels):
                self.channel_plot(
                    bf.time, channel_data, ax_i, alpha=alpha, **plot_kwargs
                )
                ax_i.set(
                    ylabel=ylabel,
                    title=ch_meta.label,
                    **ax_set,
                )

            axes_list[-1].set(
                ylabel=ylabel,
                xlabel="Time [s]",
            )
            fig.suptitle(title or bf.label or "Channel Data")

            if ax is None:
                plt.tight_layout()
                plt.show()

            return _return_axes_iterator(fig.axes)


class FrequencyPlotStrategy(PlotStrategy["SpectralFrame"]):
    """Strategy for frequency domain plotting"""

    name = "frequency"

    def channel_plot(
        self,
        x: Any,
        y: Any,
        ax: "Axes",
        **kwargs: Any,
    ) -> None:
        """Implementation of channel plotting"""
        ax.plot(x, y, **kwargs)
        ax.grid(True)
        if "label" in kwargs:
            ax.legend()

    def plot(
        self,
        bf: "SpectralFrame",
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        overlay: bool = False,
        **kwargs: Any,
    ) -> Union["Axes", Iterator["Axes"]]:
        """Frequency domain plotting"""
        kwargs = kwargs or {}
        is_aw = kwargs.pop("Aw", False)
        if bf.operation_history[-1]["operation"] == "coherence":
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
        if data.ndim == 1:
            data = data.reshape(1, -1)
        xlabel = kwargs.pop("xlabel", "Frequency [Hz]")
        alpha = kwargs.pop("alpha", 1)
        plot_kwargs = filter_kwargs(Line2D, kwargs, strict_mode=True)
        ax_set = filter_kwargs(Axes.set, kwargs, strict_mode=True)
        if overlay:
            if ax is None:
                _, ax = plt.subplots(figsize=(10, 4))
            self.channel_plot(
                bf.freqs,
                data.T,
                ax,
                label=bf.labels,
                alpha=alpha,
                **plot_kwargs,
            )
            ax.set(
                ylabel=ylabel,
                xlabel=xlabel,
                title=title or bf.label or "Channel Data",
                **ax_set,
            )
            if ax is None:
                plt.tight_layout()
                plt.show()
            return ax
        else:
            num_channels = bf.n_channels
            fig, axs = plt.subplots(
                num_channels, 1, figsize=(10, 4 * num_channels), sharex=True
            )
            # Convert axs to list if it is a single Axes object
            if not isinstance(axs, (list, np.ndarray)):
                axs = [axs]

            axes_list = list(axs)
            for ax_i, channel_data, ch_meta in zip(axes_list, data, bf.channels):
                self.channel_plot(
                    bf.freqs,
                    channel_data,
                    ax_i,
                    label=ch_meta.label,
                    alpha=alpha,
                    **plot_kwargs,
                )
                ax_i.set(
                    ylabel=ylabel,
                    title=ch_meta.label,
                    xlabel=xlabel,
                    **ax_set,
                )
            axes_list[-1].set(ylabel=ylabel, xlabel=xlabel)
            fig.suptitle(title or bf.label or "Channel Data")
            if ax is None:
                plt.tight_layout()
                plt.show()
            return _return_axes_iterator(fig.axes)


class NOctPlotStrategy(PlotStrategy["NOctFrame"]):
    """Strategy for N-octave band analysis plotting"""

    name = "noct"

    def channel_plot(
        self,
        x: Any,
        y: Any,
        ax: "Axes",
        **kwargs: Any,
    ) -> None:
        """Implementation of channel plotting"""
        ax.step(x, y, **kwargs)
        ax.grid(True)
        if "label" in kwargs:
            ax.legend()

    def plot(
        self,
        bf: "NOctFrame",
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        overlay: bool = False,
        **kwargs: Any,
    ) -> Union["Axes", Iterator["Axes"]]:
        """N-octave band analysis plotting"""
        kwargs = kwargs or {}
        is_aw = kwargs.pop("Aw", False)

        if is_aw:
            unit = "dBrA"
            data = bf.dBA
        else:
            unit = "dBr"
            data = bf.dB
        if data.ndim == 1:
            data = data.reshape(1, -1)
        ylabel = kwargs.pop("ylabel", f"Spectrum level [{unit}]")
        xlabel = kwargs.pop("xlabel", "Center frequency [Hz]")
        alpha = kwargs.pop("alpha", 1)
        plot_kwargs = filter_kwargs(Line2D, kwargs, strict_mode=True)
        ax_set = filter_kwargs(Axes.set, kwargs, strict_mode=True)
        if overlay:
            if ax is None:
                _, ax = plt.subplots(figsize=(10, 4))
            self.channel_plot(
                bf.freqs,
                data.T,
                ax,
                label=bf.labels,
                alpha=alpha,
                **plot_kwargs,
            )
            ax.set(
                ylabel=ylabel,
                xlabel=xlabel,
                title=title or bf.label or f"1/{str(bf.n)}-Octave Spectrum",
                **ax_set,
            )
            if ax is None:
                plt.tight_layout()
                plt.show()
            return ax
        else:
            num_channels = bf.n_channels
            fig, axs = plt.subplots(
                num_channels, 1, figsize=(10, 4 * num_channels), sharex=True
            )
            # Convert axs to list if it is a single Axes object
            if not isinstance(axs, (list, np.ndarray)):
                axs = [axs]

            axes_list = list(axs)
            for ax_i, channel_data, ch_meta in zip(axes_list, data, bf.channels):
                self.channel_plot(
                    bf.freqs,
                    channel_data,
                    ax_i,
                    label=ch_meta.label,
                    alpha=alpha,
                    **plot_kwargs,
                )
                ax_i.set(
                    ylabel=ylabel,
                    title=ch_meta.label,
                    xlabel=xlabel,
                    **ax_set,
                )
            axes_list[-1].set(ylabel=ylabel, xlabel=xlabel)
            fig.suptitle(title or bf.label or f"1/{str(bf.n)}-Octave Spectrum")
            if ax is None:
                plt.tight_layout()
                plt.show()
            return _return_axes_iterator(fig.axes)


class SpectrogramPlotStrategy(PlotStrategy["SpectrogramFrame"]):
    """Strategy for spectrogram plotting"""

    name = "spectrogram"

    def channel_plot(
        self,
        x: Any,
        y: Any,
        ax: "Axes",
        **kwargs: Any,
    ) -> None:
        """Implementation of channel plotting"""
        pass

    def plot(
        self,
        bf: "SpectrogramFrame",
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        overlay: bool = False,
        **kwargs: Any,
    ) -> Union["Axes", Iterator["Axes"]]:
        """Spectrogram plotting"""
        if overlay:
            raise ValueError("Overlay is not supported for SpectrogramPlotStrategy.")

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
        if data.ndim == 1:
            data = data.reshape(1, -1)
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
                cbar = fig.colorbar(img, ax=ax)
                cbar.set_label(f"Spectrum level [{unit}]")
            return ax

        else:
            # Create a new figure if ax is None
            num_channels = bf.n_channels
            fig, axs = plt.subplots(
                num_channels, 1, figsize=(10, 5 * num_channels), sharex=True
            )
            if not isinstance(fig, plt.Figure):
                raise ValueError("fig must be a matplotlib Figure object.")
            # Convert axs to array if it is a single Axes object
            if not isinstance(axs, np.ndarray):
                axs = np.array([axs])

            for ax_i, channel_data, ch_meta in zip(axs.flatten(), data, bf.channels):
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
                cbar = ax_i.figure.colorbar(img, ax=ax_i)
                cbar.set_label(f"Spectrum level [{unit}]")
                fig.suptitle(title or "Spectrogram Data")
            plt.tight_layout()
            plt.show()

            return _return_axes_iterator(fig.axes)


class DescribePlotStrategy(PlotStrategy["ChannelFrame"]):
    """Strategy for visualizing ChannelFrame data with describe plot"""

    name = "describe"

    def channel_plot(self, x: Any, y: Any, ax: "Axes", **kwargs: Any) -> None:
        """Implementation of channel plotting"""
        pass  # This method is not used for describe plot

    def plot(
        self,
        bf: "ChannelFrame",
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        overlay: bool = False,
        **kwargs: Any,
    ) -> Union["Axes", Iterator["Axes"]]:
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
        spectral = kwargs.pop("spectral", dict(xlim=(vmin, vmax)))

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


class MatrixPlotStrategy(PlotStrategy[Union["SpectralFrame"]]):
    """Strategy for displaying relationships between channels in matrix format"""

    name = "matrix"

    def channel_plot(
        self,
        x: Any,
        y: Any,
        ax: "Axes",
        title: Optional[str] = None,
        ylabel: str = "",
        xlabel: str = "Frequency [Hz]",
        alpha: float = 0,
        **kwargs: Any,
    ) -> None:
        ax.plot(x, y, **kwargs)
        ax.grid(True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title or "")

    def plot(
        self,
        bf: "SpectralFrame",
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        overlay: bool = False,
        **kwargs: Any,
    ) -> Union["Axes", Iterator["Axes"]]:
        kwargs = kwargs or {}
        is_aw = kwargs.pop("Aw", False)
        if (
            len(bf.operation_history) > 0
            and bf.operation_history[-1]["operation"] == "coherence"
        ):
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

        if data.ndim == 1:
            data = data.reshape(1, -1)

        xlabel = kwargs.pop("xlabel", "Frequency [Hz]")
        alpha = kwargs.pop("alpha", 1)
        plot_kwargs = filter_kwargs(Line2D, kwargs, strict_mode=True)
        ax_set = filter_kwargs(Axes.set, kwargs, strict_mode=True)
        num_channels = bf.n_channels
        if overlay:
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            self.channel_plot(
                bf.freqs,
                data.T,
                ax,  # ここで必ずAxes型
                title=title or bf.label or "Spectral Data",
                ylabel=ylabel,
                xlabel=xlabel,
                alpha=alpha,
                **plot_kwargs,
            )
            if ax is not None:
                ax.set(**ax_set)
                fig.suptitle(title or bf.label or "Spectral Data")
                plt.tight_layout()
                plt.show()
                return ax
            # axがNoneのケースはここで発生しない
        else:
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
            for ax_i, channel_data, ch_meta in zip(axes_list, data, bf.channels):
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

        raise NotImplementedError()


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
