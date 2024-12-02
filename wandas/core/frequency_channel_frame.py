# wandas/core/spectrums.py

from typing import Optional, Any, List
from wandas.core.frequency_channel import FrequencyChannel
import matplotlib.pyplot as plt
import numpy as np


class FrequencyChannelFrame:
    def __init__(self, channels: List["FrequencyChannel"], label: Optional[str] = None):
        """
        FrequencyChannelFrame オブジェクトを初期化します。

        Parameters:
            channels (list of FrequencyChannel): FrequencyChannel   w オブジェクトのリスト。
            label (str, optional): スペクトルのラベル。
        """
        self.channels = channels
        self.label = label

    def plot(
        self, ax: Optional[Any] = None, title: Optional[str] = None, Aw: bool = False
    ):
        """
        スペクトルデータをプロットします。

        Parameters:
            ax (matplotlib.axes.Axes, optional): プロットに使用する Axes オブジェクト。
            title (str, optional): プロットのタイトル。
        """
        _ax = ax
        if _ax is None:
            _, _ax = plt.subplots(figsize=(10, 4))

        for channel in self.channels:
            channel.plot(ax=_ax, Aw=Aw)

        _ax.set_title(title or self.label or "Spectrum")
        _ax.grid(True)
        _ax.legend()

        if ax is None:
            plt.tight_layout()
            plt.show()

    def plot_matrix(self, title: Optional[str] = None, Aw: bool = False):
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
