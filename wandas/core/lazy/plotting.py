import abc
import inspect
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Optional, TypeVar, Union

import librosa
import matplotlib.gridspec as gridspec
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
            # axs が単一の Axes オブジェクトの場合、リストに変換
            if not isinstance(axs, (list, np.ndarray)):
                axs = [axs]

            axes_list = list(axs)
            data = bf.data
            for ax_i, channel_data, ch_meta in zip(axes_list, data, bf.channels):
                self.channel_plot(bf.time, channel_data, ax_i)
                ax_i.set_title(ch_meta.label)

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
        is_aw = kwargs.pop("Aw", False)

        if is_aw:
            unit = "dBA"
            data = bf.dBA
        else:
            unit = "dB"
            data = bf.dB
        if overlay:
            if ax is None:
                _, ax = plt.subplots(figsize=(10, 4))
            self.channel_plot(bf.freqs, data.T, ax, label=bf.labels)
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel(f"Spectrum level [{unit}]")
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
            # axs が単一の Axes オブジェクトの場合、リストに変換
            if not isinstance(axs, (list, np.ndarray)):
                axs = [axs]

            axes_list = list(axs)
            for ax_i, channel_data, ch_meta in zip(axes_list, data, bf.channels):
                self.channel_plot(bf.freqs, channel_data, ax_i, label=ch_meta.label)

            axes_list[-1].set_xlabel("Frequency [Hz]")
            axes_list[-1].set_ylabel(f"Spectrum level [{unit}]")
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

        # axが与えられていて、チャンネル数が1の場合は指定されたaxを使用
        if ax is not None:
            if bf.n_channels > 1:
                raise ValueError("ax must be None when n_channels > 1.")
            axes_list = [ax]
            create_new_figure = False
        else:
            # axがNoneの場合は新しい図を作成
            num_channels = bf.n_channels
            fig, axs = plt.subplots(
                num_channels, 1, figsize=(10, 4 * num_channels), sharex=True
            )
            # axs が単一の Axes オブジェクトの場合、リストに変換
            if not isinstance(axs, (list, np.ndarray)):
                axs = [axs]
            axes_list = list(axs)
            create_new_figure = True

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
            # タイトルの設定：外部からaxが与えられた場合はその指定を優先
            if create_new_figure or not title:
                ax_i.set_title(ch_meta.label)
            else:
                ax_i.set_title(title)
            ax_i.set_xlabel("Frequency [Hz]")
            ax_i.set_ylabel(f"Spectrum level [{unit}]")

        # 新しい図を作成した場合のみ、suptitleとtight_layout、showを行う
        if create_new_figure:
            axes_list[0].figure.suptitle(title or bf.label or "Spectrogram Data")
            plt.tight_layout()
            plt.show()
            return iter(axes_list)
        else:
            # axが与えられていて、チャンネル数が1の場合はaxを返す
            return axes_list[0]


class DescribePlotStrategy(PlotStrategy["ChannelFrame"]):
    """ChannelFrameのデータを可視化するdescribeプロット戦略"""

    name = "describe"

    def channel_plot(self, x: Any, y: Any, ax: "Axes", **kwargs: Any) -> None:
        """チャンネルプロットの実装"""
        pass  # このメソッドはdescribeでは使用しないため未実装

    def plot(
        self,
        bf: "ChannelFrame",
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        overlay: bool = False,
        **kwargs: Any,
    ) -> Union["Axes", Iterator["Axes"]]:
        """ChannelFrameのデータを可視化するdescribeメソッドの実装"""

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

        # 最初のサブプロット (Time Plot)
        ax_1 = fig.add_subplot(gs[0])
        bf.plot(plot_type="waveform", ax=ax_1, overlay=True)
        ax_1.set(**waveform)
        ax_1.legend().set_visible(False)
        ax_1.set(xlabel="", title="")

        # 2番目のサブプロット (STFT Plot)
        ax_2 = fig.add_subplot(gs[3], sharex=ax_1)
        stft_ch = bf.stft()
        if is_aw:
            unit = "dBA"
            channel_data = stft_ch.dBA[0]
        else:
            unit = "dB"
            channel_data = stft_ch.dB[0]
        # データの最大値を取得し、切りのいい値に丸める
        if vmax is None:
            data_max = np.nanmax(channel_data)
            # 10, 5, 2のいずれかの刻みで切りのいい数に丸める
            for step in [10, 5, 2]:
                rounded_max = np.ceil(data_max / step) * step
                if rounded_max >= data_max:
                    vmax = rounded_max
                    vmin = vmax - 180
                    break
        img = librosa.display.specshow(
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

        # 3番目のサブプロット
        ax_3 = fig.add_subplot(gs[1])
        ax_3.axis("off")

        # 4番目のサブプロット (Welch Plot)
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

        return iter(fig.axes)


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
