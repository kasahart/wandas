import abc
import inspect
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Optional, TypeVar, Union

import librosa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

if TYPE_CHECKING:
    from .base_frame import BaseFrame
    from .channel_frame import ChannelFrame
    from .spectral_frame import SpectralFrame
    from .spectrogram_frame import SpectrogramFrame

logger = logging.getLogger(__name__)

TFrame = TypeVar("TFrame", bound="BaseFrame[Any]")


class PlotStrategy(abc.ABC, Generic[TFrame]):
    """プロット戦略の基底クラス"""

    name: ClassVar[str]

    @abc.abstractmethod
    def channel_plot(self, x: Any, y: Any, ax: "Axes") -> None:
        """チャンネルプロットの実装"""
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
        """プロットの実装"""
        pass


class WaveformPlotStrategy(PlotStrategy["ChannelFrame"]):
    """波形プロットの戦略"""

    name = "waveform"

    def channel_plot(
        self,
        x: Any,
        y: Any,
        ax: "Axes",
        **kwargs: Any,
    ) -> None:
        """チャンネルプロットの実装"""
        ax.plot(x, y, **kwargs)
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        ax.legend()

    def plot(
        self,
        bf: "ChannelFrame",
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        overlay: bool = False,
        **kwargs: Any,
    ) -> Union["Axes", Iterator["Axes"]]:
        """波形プロット"""
        kwargs = kwargs or {}

        if overlay:
            if ax is None:
                _, ax = plt.subplots(figsize=(10, 4))
            self.channel_plot(bf.time, bf.data.T, ax, label=bf.labels)
            ax.set_xlabel("Time [s]")
            ax.set_title(title or bf.label or "Channel Data")
            if ax is None:
                plt.tight_layout()
                plt.show()
            return ax
        else:
            num_channels = bf.n_channels
            fig, axs = plt.subplots(
                num_channels, 1, figsize=(10, 4 * num_channels), sharex=True
            )
            axes_list = list(axs)
            data = bf.data
            for ax_i, channel_data, ch_meta in zip(axes_list, data, bf.channels):
                self.channel_plot(bf.time, channel_data, ax_i, label=ch_meta.label)

            axes_list[-1].set_xlabel("Time [s]")
            fig.suptitle(title or bf.label or "Channel Data")

            if ax is None:
                plt.tight_layout()
                plt.show()

            return iter(axes_list)


class FrequencyPlotStrategy(PlotStrategy["SpectralFrame"]):
    """周波数プロットの戦略"""

    name = "frequency"

    def channel_plot(
        self,
        x: Any,
        y: Any,
        ax: "Axes",
        **kwargs: Any,
    ) -> None:
        """チャンネルプロットの実装"""
        ax.plot(x, y, **kwargs)
        ax.set_ylabel("Amplitude [dB]")
        ax.grid(True)
        ax.legend()

    def plot(
        self,
        bf: "SpectralFrame",
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        overlay: bool = False,
        **kwargs: Any,
    ) -> Union["Axes", Iterator["Axes"]]:
        """周波数プロット"""
        kwargs = kwargs or {}

        if overlay:
            if ax is None:
                _, ax = plt.subplots(figsize=(10, 4))
            self.channel_plot(bf.freqs, 20 * np.log10(bf.data.T), ax, label=bf.labels)
            ax.set_xlabel("Frequency [Hz]")
            ax.set_title(title or bf.label or "Channel Data")
            if ax is None:
                plt.tight_layout()
                plt.show()
            return ax
        else:
            num_channels = bf.n_channels
            fig, axs = plt.subplots(
                num_channels, 1, figsize=(10, 4 * num_channels), sharex=True
            )
            axes_list = list(axs)
            data = bf.data
            for ax_i, channel_data, ch_meta in zip(axes_list, data, bf.channels):
                self.channel_plot(
                    bf.freqs, 20 * np.log10(channel_data), ax_i, label=ch_meta.label
                )

            axes_list[-1].set_xlabel("Frequency [Hz]")
            fig.suptitle(title or bf.label or "Channel Data")

            if ax is None:
                plt.tight_layout()
                plt.show()

            return iter(axes_list)


class SpectrogramPlotStrategy(PlotStrategy["SpectrogramFrame"]):
    """スペクトログラムプロットの戦略"""

    name = "spectrogram"

    def channel_plot(
        self,
        x: Any,
        y: Any,
        ax: "Axes",
        **kwargs: Any,
    ) -> None:
        """チャンネルプロットの実装"""
        pass

    def plot(
        self,
        bf: "SpectrogramFrame",
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        overlay: bool = False,
        **kwargs: Any,
    ) -> Union["Axes", Iterator["Axes"]]:
        """スペクトログラムプロット"""
        kwargs = kwargs or {}
        if overlay:
            raise ValueError("Overlay is not supported for SpectrogramPlotStrategy.")

        if ax is not None and bf.n_channels > 1:
            raise ValueError("ax must be None when n_channels > 1.")

        num_channels = bf.n_channels
        fig, axs = plt.subplots(
            num_channels, 1, figsize=(10, 4 * num_channels), sharex=True
        )
        axes_list = list(axs)

        is_aw = kwargs.pop("Aw", False)
        if is_aw:
            unit = "dBA"
            data = bf.dBA
        else:
            unit = "dB"
            data = bf.dB

        fmin = kwargs.pop("fmin", 0)
        fmax = kwargs.pop("fmax", None)
        cmap = kwargs.pop("cmap", "jet")
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)
        xlim = kwargs.pop("xlim", None)
        ylim = kwargs.pop("ylim", None)
        for ax_i, channel_data, ch_meta in zip(axes_list, data, bf.channels):
            img = librosa.display.specshow(
                data=channel_data,
                sr=bf.sampling_rate,
                hop_length=bf.hop_length,
                n_fft=bf.n_fft,
                win_length=bf.win_length,
                x_axis="time",
                y_axis="linear",
                ax=ax_i,
                fmin=fmin,
                fmax=fmax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                **kwargs,
            )
            ax_i.set(xlim=xlim, ylim=ylim)
            cbar = ax_i.figure.colorbar(img, ax=ax_i)
            cbar.set_label(f"Spectrum level [{unit}]")
            ax_i.set_title(ch_meta.label)
            ax_i.set_xlabel("Frequency [Hz]")
            ax_i.set_ylabel(f"Spectrum level [{unit}]")

        fig.suptitle(title or bf.label or "Spectrogram Data")

        if ax is None:
            plt.tight_layout()
            plt.show()

        return iter(axes_list)


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
