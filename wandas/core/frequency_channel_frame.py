# wandas/core/spectrums.py

from typing import TYPE_CHECKING, Any, Optional

import matplotlib.pyplot as plt
import numpy as np

from wandas.core.frequency_channel import FrequencyChannel

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class FrequencyChannelFrame:
    def __init__(self, channels: list["FrequencyChannel"], label: Optional[str] = None):
        """
        FrequencyChannelFrame オブジェクトを初期化します。

        Parameters:
            channels (list of FrequencyChannel): FrequencyChannelオブジェクトのリスト。
            label (str, optional): スペクトルのラベル。
        """
        self.channels = channels
        self.label = label

    def plot(
        self,
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        Aw: bool = False,  # noqa: N803
        overlay: bool = True,
        plot_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        スペクトルデータをプロットします。

        Parameters:
            ax (matplotlib.axes.Axes, optional): プロットに使用する Axes オブジェクト。
            title (str, optional): プロットのタイトル。
        """
        if ax is not None and not overlay:
            raise ValueError("ax must be None when overlay is False.")

        suptitle = title or self.label or "Spectrum"

        if not overlay:
            num_channels = len(self.channels)
            fig, axs = plt.subplots(
                num_channels, 1, figsize=(10, 4 * num_channels), sharex=True
            )
            if num_channels == 1:
                axs = [axs]  # Ensure axs is iterable when there's only one channel

            for i, channel in enumerate(self.channels):
                tmp = axs[i]
                channel.plot(ax=tmp, Aw=Aw, plot_kwargs=plot_kwargs)
                leg = tmp.get_legend()
                if leg:
                    leg.remove()

            fig.suptitle(suptitle)
            plt.tight_layout()
            plt.show()
            return

        if ax is None:
            fig, tmp = plt.subplots(figsize=(10, 4))
        else:
            tmp = ax

        for channel in self.channels:
            channel.plot(ax=tmp, Aw=Aw, plot_kwargs=plot_kwargs)

        tmp.grid(True)
        tmp.legend()
        tmp.set_title(suptitle)

        if ax is None:
            plt.tight_layout()
            plt.show()

    def plot_matrix(
        self,
        title: Optional[str] = None,
        Aw: bool = False,  # noqa: N803
    ) -> tuple["Figure", "Axes"]:
        """
        チャンネル間をプロットします。

        Parameters:
            ax (matplotlib.axes.Axes, optional): プロット先の軸。
            title (str, optional): プロットのタイトル。
            cmap (str, optional): カラーマップ。
        """

        num_channels = len(self.channels)
        num_rows = int(np.ceil(np.sqrt(num_channels)))

        fig, axes = plt.subplots(
            num_rows,
            num_rows,
            figsize=(3 * num_rows, 3 * num_rows),
            sharex=True,
            sharey=True,
        )
        axes = axes.flatten()

        for ch, ax in zip(self.channels, axes):
            ch.plot(ax=ax, title=title, Aw=Aw)
            ax.set_title(title or self.label)

        if title:
            fig.suptitle(title)

        plt.tight_layout()
        return fig, axes
