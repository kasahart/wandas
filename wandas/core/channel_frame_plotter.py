from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

if TYPE_CHECKING:
    from .channel_frame import ChannelFrame


class ChannelFramePlotter:
    def __init__(self, cf: "ChannelFrame") -> None:
        self.cf = cf

    def plot_time(
        self,
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        overlay: bool = True,
        plot_kwargs: Optional[dict[str, Any]] = None,
    ) -> Union["Axes", Iterable["Axes"]]:
        """
        すべてのチャンネルをプロットします。

        Parameters:
            title (str, optional): プロットのタイトル。
            overlay (bool, optional): True の場合、すべてのチャンネルを同じプロットに
                                      重ねて描画します。False の場合、各チャンネルを
                                      個別のプロットに描画します。
        """
        if ax is not None and not overlay:
            raise ValueError("ax must be None when overlay is False.")

        plot_kwargs = plot_kwargs or {}
        suptitle = title or self.cf.label or "Signal"

        if overlay:
            tmp = ax
            if tmp is None:
                fig, tmp = plt.subplots(figsize=(10, 4))

            for channel in self.cf:
                channel.plot(ax=tmp, plot_kwargs=plot_kwargs)

            tmp.grid(True)
            tmp.legend()
            tmp.set_title(suptitle)

            if ax is None:
                plt.tight_layout()
                plt.show()

            return tmp

        else:
            num_channels = len(self.cf)
            fig, axs = plt.subplots(
                num_channels, 1, figsize=(10, 4 * num_channels), sharex=True
            )
            axes_list: list[Axes] = []
            if isinstance(axs, Axes):
                axes_list = [
                    axs
                ]  # Ensure axs is iterable when there's only one channel
            else:
                axes_list = list(axs)

            for channel, ax_i in zip(self.cf, axes_list):
                channel.plot(ax=ax_i, plot_kwargs=plot_kwargs)
                leg = ax_i.get_legend()
                if leg:
                    leg.remove()
            axes_list[-1].set_xlabel("Time [s]")

            fig.suptitle(suptitle)
            plt.tight_layout()
            plt.show()

            if isinstance(axs, Axes):
                return axs

            return axes_list

    def rms_plot(
        self,
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        overlay: bool = True,
        Aw: bool = False,  # noqa: N803
        plot_kwargs: Optional[dict[str, Any]] = None,
    ) -> Union["Axes", Iterable["Axes"]]:
        if ax is None:
            plt.tight_layout()

        if overlay:
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 4))

            for channel in self.cf:
                channel.rms_plot(ax=ax, title=title, Aw=Aw, plot_kwargs=plot_kwargs)

            ax.set_title(title or self.cf.label or "Signal RMS")
            ax.grid(True)
            ax.legend()

            if ax is None:
                plt.tight_layout()
                plt.show()
            return ax
        else:
            num_channels = len(self.cf)
            fig, axs = plt.subplots(
                num_channels, 1, figsize=(10, 4 * num_channels), sharex=True
            )
            axes_list: list[Axes] = []
            if isinstance(axs, Axes):
                axes_list = [
                    axs
                ]  # Ensure axs is iterable when there's only one channel
            else:
                axes_list = list(axs)

            for channel, ax_i in zip(self.cf, axes_list):
                channel.rms_plot(
                    ax=ax_i, title=channel.label, Aw=Aw, plot_kwargs=plot_kwargs
                )

            axes_list[-1].set_xlabel("Time [s]")

            fig.suptitle(title or self.cf.label or "Signal")
            plt.tight_layout()
            plt.show()

            if isinstance(axs, Axes):
                return axs

            return axes_list
