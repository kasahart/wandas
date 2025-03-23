import abc
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

if TYPE_CHECKING:
    from .channel_frame import ChannelFrame

logger = logging.getLogger(__name__)


class PlotStrategy(abc.ABC):
    """プロット戦略の基底クラス"""

    @abc.abstractmethod
    def plot(
        self,
        cf: "ChannelFrame",
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Union["Axes", Iterator["Axes"]]:
        """プロットの実装"""
        pass


class WaveformPlotStrategy(PlotStrategy):
    """波形プロットの戦略"""

    def plot(
        self,
        cf: "ChannelFrame",
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Union["Axes", Iterator["Axes"]]:
        """波形プロット"""
        kwargs = kwargs or {}

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        ax.plot(
            cf.time,
            cf.data.T,
            label=cf.labels,
            **kwargs,
        )

        ax.set_xlabel("Time [s]")
        ylabel = (
            f"Amplitude [{cf.channel[0].unit}]" if cf.channel[0].unit else "Amplitude"
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title or cf.label or "Channel Data")
        ax.grid(True)
        ax.legend()

        if ax is None:
            plt.tight_layout()
            plt.show()

        return ax


# class SpectrogramPlotStrategy(PlotStrategy):
#     """スペクトログラムプロットの戦略"""

#     def plot(
#         self,
#         cf: "ChannelFrame",
#         ax: Optional["Axes"],
#         **kwargs: Any,
#     ) -> Union["Axes", Iterator["Axes"]]:
#         """スペクトログラム"""
#         from matplotlib import pyplot as plt

#         channel = kwargs.get("channel", 0)
#         if channel >= data.shape[0]:
#             raise ValueError(f"Channel index out of range: {channel}")

#         n_fft = kwargs.get("n_fft", 2048)
#         hop_length = kwargs.get("hop_length", n_fft // 4)

#         # 単一チャネルのデータを取得
#         audio_data = data[channel]

#         # スペクトログラム計算 (scipy.signalを使用)
#         from scipy import signal

#         f, t, Sxx = signal.spectrogram(
#             audio_data,
#             sampling_rate,
#             window="hann",
#             nperseg=n_fft,
#             noverlap=n_fft - hop_length,
#         )

#         img = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading="gouraud")
#         ax.set_ylabel("Frequency [Hz]")
#         ax.set_xlabel("Time [s]")
#         plt.colorbar(img, ax=ax, label="Power/Frequency [dB/Hz]")

#         return ax


# プロット戦略のレジストリ
_plot_strategies: dict[str, PlotStrategy] = {
    "waveform": WaveformPlotStrategy(),
    # "spectrogram": SpectrogramPlotStrategy(),
}


def register_plot_strategy(name: str, strategy: PlotStrategy) -> None:
    """新しいプロット戦略を登録"""
    _plot_strategies[name] = strategy


def get_plot_strategy(name: str) -> PlotStrategy:
    """名前からプロット戦略を取得"""
    if name not in _plot_strategies:
        raise ValueError(f"Unknown plot type: {name}")
    return _plot_strategies[name]
