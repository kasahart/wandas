import abc
import inspect
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Optional, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

if TYPE_CHECKING:
    from .base_frame import BaseFrame
    from .channel_frame import ChannelFrame
    from .spectral_frame import SpectralFrame

logger = logging.getLogger(__name__)

TFrame = TypeVar("TFrame", bound="BaseFrame")


class PlotStrategy(abc.ABC, Generic[TFrame]):
    """プロット戦略の基底クラス"""

    name: ClassVar[str]

    @abc.abstractmethod
    def plot(
        self,
        bf: TFrame,
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Union["Axes", Iterator["Axes"]]:
        """プロットの実装"""
        pass


class WaveformPlotStrategy(PlotStrategy["ChannelFrame"]):
    """波形プロットの戦略"""

    name = "waveform"

    def plot(
        self,
        bf: "ChannelFrame",
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Union["Axes", Iterator["Axes"]]:
        """波形プロット"""
        kwargs = kwargs or {}
        metadata = bf.channels
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        ax.plot(
            bf.time,
            bf.data.T,
            label=bf.labels,
            **kwargs,
        )

        ax.set_xlabel("Time [s]")
        ylabel = f"Amplitude [{metadata[0].unit}]" if metadata[0].unit else "Amplitude"
        ax.set_ylabel(ylabel)
        ax.set_title(title or bf.label or "Channel Data")
        ax.grid(True)
        ax.legend()

        if ax is None:
            plt.tight_layout()
            plt.show()

        return ax


class FrequencyPlotStrategy(PlotStrategy["SpectralFrame"]):
    """周波数プロットの戦略"""

    name = "frequency"

    def plot(
        self,
        bf: "SpectralFrame",
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Union["Axes", Iterator["Axes"]]:
        """周波数プロット"""

        kwargs = kwargs or {}
        metadata = bf.channels
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        data = 20 * np.log10(bf.data)  # dBに変換
        ax.plot(
            bf.frequencies,
            data.T,
            label=bf.labels,
            **kwargs,
        )

        ax.set_xlabel("Frequency [Hz]")
        ylabel = f"Amplitude [{metadata[0].unit}]" if metadata[0].unit else "Amplitude"
        ax.set_ylabel(ylabel)
        ax.set_title(title or bf.label or "Channel Data")
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


# プロットタイプと対応するクラスのマッピングを保持
_plot_strategies: dict[str, type[PlotStrategy[Any]]] = {}


def register_plot_strategy(strategy_cls: type) -> None:
    """新しいプロット戦略をクラスから登録"""
    if not issubclass(strategy_cls, PlotStrategy):
        raise TypeError("Strategy class must inherit from PlotStrategy.")
    if inspect.isabstract(strategy_cls):
        raise TypeError("Cannot register abstract PlotStrategy class.")
    _plot_strategies[strategy_cls.name] = strategy_cls


# 抽象でないサブクラスのみを自動登録するように修正
for strategy_cls in PlotStrategy.__subclasses__():
    if not inspect.isabstract(strategy_cls):
        register_plot_strategy(strategy_cls)


def get_plot_strategy(name: str) -> type[PlotStrategy[Any]]:
    """名前からプロット戦略を取得"""
    if name not in _plot_strategies:
        raise ValueError(f"Unknown plot type: {name}")
    return _plot_strategies[name]


def create_operation(name: str, **params: Any) -> PlotStrategy[Any]:
    """操作名とパラメータから操作インスタンスを作成"""
    operation_class = get_plot_strategy(name)
    return operation_class(**params)
