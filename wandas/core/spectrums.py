# wandas/core/spectrums.py

from typing import Optional, Any, List
from wandas.core.frequency_channel import FrequencyChannel
import matplotlib.pyplot as plt


class Spectrums:
    def __init__(self, channels: List["FrequencyChannel"], label: Optional[str] = None):
        """
        Spectrum オブジェクトを初期化します。

        Parameters:
            channels (list of FrequencyChannel): FrequencyChannel   w オブジェクトのリスト。
            label (str, optional): スペクトルのラベル。
        """
        self.channels = channels
        self.label = label

    def plot(self, ax: Optional[Any] = None, title: Optional[str] = None):
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
            channel.plot(ax=_ax)

        _ax.set_title(title or self.label or "Spectrum")
        _ax.grid(True)
        _ax.legend()

        if ax is None:
            plt.tight_layout()
            plt.show()
