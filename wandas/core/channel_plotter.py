from typing import TYPE_CHECKING, Any, Optional

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.axes import Axes

from wandas.core import util

if TYPE_CHECKING:
    from .channel import Channel


class ChannelPlotter:
    def __init__(self, channel: "Channel") -> None:
        self.channel = channel

    def plot_time(
        self,
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        plot_kwargs: Optional[dict[str, Any]] = None,
    ) -> "Axes":
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        plot_kwargs = plot_kwargs or {}
        ax.plot(
            self.channel.time,
            self.channel.data,
            label=self.channel.label or "Channel",
            **plot_kwargs,
        )

        ax.set_xlabel("Time [s]")
        ylabel = (
            f"Amplitude [{self.channel.unit}]" if self.channel.unit else "Amplitude"
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title or self.channel.label or "Channel Data")
        ax.grid(True)
        ax.legend()

        if ax is None:
            plt.tight_layout()
            plt.show()

        return ax

    def rms_plot(
        self,
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        Aw: bool = False,  # noqa: N803
        plot_kwargs: Optional[dict[str, Any]] = None,
    ) -> "Axes":
        _ax = ax
        if _ax is None:
            _, _ax = plt.subplots(figsize=(10, 4))

        plot_kwargs = plot_kwargs or {}

        rms_channel: Channel = self.channel.rms_trend(Aw=Aw)
        num_samples = len(rms_channel)
        t = np.arange(num_samples) / rms_channel.sampling_rate

        _ax.plot(
            t,
            util.amplitude_to_db(rms_channel.data, ref=self.channel.ref),
            label=rms_channel.label or "Channel",
            **plot_kwargs,
        )

        _ax.set_xlabel("Time [s]")
        ylabel = f"RMS [{rms_channel.unit}]" if rms_channel.unit else "RMS"
        _ax.set_ylabel(ylabel)
        _ax.set_title(title or rms_channel.label or "Channel Data")
        _ax.grid(True)
        _ax.legend()

        if ax is None:
            plt.tight_layout()

        return _ax

    def describe(
        self,
        axis_config: Optional[dict[str, dict[str, Any]]] = None,
        cbar_config: Optional[dict[str, Any]] = None,
    ) -> widgets.VBox:
        axis_config = axis_config or {}
        cbar_config = cbar_config or {}

        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 3], width_ratios=[3, 1, 0.1])
        gs.update(wspace=0.2)

        fig = plt.figure(figsize=(12, 6))

        # 最初のサブプロット (Time Plot)
        ax_1 = fig.add_subplot(gs[0])
        self.plot_time(ax=ax_1)
        if "time_plot" in axis_config:
            conf = axis_config["time_plot"]
            ax_1.set(**conf)
        ax_1.legend().set_visible(False)
        ax_1.set(xlabel="", title="")

        # 2番目のサブプロット (STFT Plot)
        ax_2 = fig.add_subplot(gs[3], sharex=ax_1)
        stft_ch = self.channel.stft()
        # Pass vmin and vmax from cbar_config to stft_ch._plot
        img, _ = stft_ch._plot(
            ax=ax_2, vmin=cbar_config.get("vmin"), vmax=cbar_config.get("vmax")
        )
        ax_2.set(title="")

        # 3番目のサブプロット
        ax_3 = fig.add_subplot(gs[1])
        ax_3.axis("off")

        # 4番目のサブプロット (Welch Plot)
        ax_4 = fig.add_subplot(gs[4], sharey=ax_2)
        welch_ch = self.channel.welch()
        data_db = util.amplitude_to_db(np.abs(welch_ch.data), ref=welch_ch.ref)
        ax_4.plot(data_db, welch_ch.freqs)
        ax_4.grid(True)
        ax_4.set(xlabel="Spectrum level [dB]")
        if "freq_plot" in axis_config:
            conf = axis_config["freq_plot"]
            ax_4.set(**conf)

        fig.subplots_adjust(wspace=0.0001)
        cbar = fig.colorbar(img, ax=ax_4, format="%+2.0f")
        cbar.set_label("dB")
        fig.suptitle(self.channel.label or "Channel Data")

        output = widgets.Output()
        with output:
            plt.show()

        container = widgets.VBox([output, self.channel.to_audio(label=False)])
        # container.add_class("white-bg")
        return container
