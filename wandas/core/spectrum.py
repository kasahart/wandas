# wandas/core/spectrum.py

from typing import Optional, Any, List
from .frequency_channel import FrequencyChannel
import matplotlib.pyplot as plt


class Spectrum:
    def __init__(self, channels: List[FrequencyChannel], label: Optional[str] = None):
        """
        Spectrum オブジェクトを初期化します。

        Parameters:
            channels (list of FrequencyChannel): FrequencyChannel オブジェクトのリスト。
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

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        for channel in self.channels:
            channel.plot(ax=ax)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.set_title(title or self.label or "Spectrum")
        ax.grid(True)
        ax.legend()

        if ax is None:
            plt.tight_layout()
            plt.show()
