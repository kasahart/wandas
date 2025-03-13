# wandas/core/signal.py

import numbers
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Optional, Union

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wandas.core import util
from wandas.core.channel import Channel
from wandas.io import wav_io
from wandas.utils.types import NDArrayReal

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from wandas.core.frequency_channel_frame import FrequencyChannelFrame


class ChannelFrame:
    def __init__(self, channels: list["Channel"], label: Optional[str] = None):
        """

        ChannelFrame オブジェクトを初期化します。

        Parameters:
            channels (list of Channel): Channel オブジェクトのリスト。
            label (str, optional): 信号のラベル。
        """
        self.channels = channels
        self.label = label

        # サンプリングレートの一貫性をチェック
        sampling_rates = set(ch.sampling_rate for ch in channels)
        if len(sampling_rates) > 1:
            raise ValueError("All channels must have the same sampling_rate.")

        self.sampling_rate = channels[0].sampling_rate

        # チャンネル名で辞書のようにアクセスできるようにするための辞書を構築
        self.channel_dict = {ch.label: ch for ch in channels}
        if len(self.channel_dict) != len(channels):
            raise ValueError("Channel labels must be unique.")

    @classmethod
    def from_ndarray(
        cls,
        array: NDArrayReal,
        sampling_rate: int,
        labels: Optional[list[str]] = None,
        unit: str = "Pa",
    ) -> "ChannelFrame":
        """
        numpy の ndarray から ChannelFrame インスタンスを生成します。

        Parameters:
            array (np.ndarray): 信号データ。各行がチャンネルに対応します。
            sampling_rate (int): サンプリングレート（Hz）。
            labels (list[str], optional): 各チャンネルのラベル。
            unit (str): 信号の単位。

        Returns:
            ChannelFrame: ndarray から生成された ChannelFrame オブジェクト。
        """
        channels = []
        num_channels = array.shape[0]

        if labels is None:
            labels = [f"Channel {i + 1}" for i in range(num_channels)]

        for i in range(num_channels):
            channel = Channel(
                data=array[i], sampling_rate=sampling_rate, label=labels[i], unit=unit
            )
            channels.append(channel)

        return cls(channels=channels)

    @classmethod
    def read_wav(
        cls, filename: str, labels: Optional[list[str]] = None
    ) -> "ChannelFrame":
        """
        WAV ファイルを読み込み、ChannelFrame オブジェクトを作成します。

        Parameters:
            filename (str): WAV ファイルのパス。
            labels (list of str, optional): 各チャンネルのラベル。

        Returns:
            ChannelFrame: オーディオデータを含む ChannelFrame オブジェクト。
        """
        return wav_io.read_wav(filename, labels)

    def to_wav(self, filename: str) -> None:
        """
        ChannelFrame オブジェクトを WAV ファイルに書き出します。

        Parameters:
            filename (str): 出力する WAV ファイルのパス。
        """
        wav_io.write_wav(filename, self)

    @classmethod
    def read_csv(
        cls,
        filename: str,
        time_column: Union[int, str] = 0,
        labels: Optional[list[str]] = None,
        delimiter: str = ",",
        header: Optional[int] = 0,
    ) -> "ChannelFrame":
        """
        CSV ファイルを読み込み、ChannelFrame オブジェクトを作成します。

        Parameters:
            filename (str): CSV ファイルのパス。
            labels (list of str, optional): 各チャンネルのラベル。
            delimiter (str, optional): 区切り文字。デフォルトはカンマ。
            header (int or None, optional): ヘッダー行の位置。
                None の場合はヘッダーなし。
            time_column (int or str, optional): 時間列のインデックスまたは列名。
                デフォルトは最初の列。

        Returns:
            ChannelFrame: データを含む ChannelFrame オブジェクト。
        """
        # pandas を使用して CSV ファイルを読み込む
        df = pd.read_csv(filename, delimiter=delimiter, header=header)

        # サンプリングレートを計算
        try:
            time_values = (
                df[time_column].values
                if isinstance(time_column, str)
                else df.iloc[:, time_column].values
            )
        except KeyError:
            raise KeyError(f"Time column '{time_column}' not found in the CSV file.")
        except IndexError:
            raise IndexError(f"Time column index {time_column} is out of range.")
        if len(time_values) < 2:
            raise ValueError("Not enough time points to calculate sampling rate.")
        time_values = np.array(time_values)
        sampling_rate: int = int(1 / np.mean(np.diff(time_values)))

        # 時間列を削除
        df = df.drop(
            columns=[time_column]
            if isinstance(time_column, str)
            else df.columns[time_column]
        )

        # データを NumPy 配列に変換
        data = df.values  # shape: (サンプル数, チャンネル数)

        # 転置してチャンネルを最初の次元に持ってくる
        data = data.T  # shape: (チャンネル数, サンプル数)

        num_channels = data.shape[0]

        # ラベルの処理
        if labels is not None:
            if len(labels) != num_channels:
                raise ValueError("Length of labels must match number of channels.")
        elif header is not None:
            labels = df.columns.tolist()
        else:
            labels = [f"Ch{i}" for i in range(num_channels)]

        # 各チャンネルの Channel オブジェクトを作成
        channels = []
        for i in range(num_channels):
            ch_data = data[i]
            ch_label = labels[i]
            channel = Channel(
                data=ch_data,
                sampling_rate=sampling_rate,
                label=ch_label,
            )
            channels.append(channel)

        return cls(channels=channels)

    def to_audio(self, normalize: bool = True) -> widgets.VBox:
        return widgets.VBox([ch.to_audio(normalize) for ch in self.channels])

    def describe(
        self,
        axis_config: Optional[dict[str, dict[str, Any]]] = None,
        cbar_config: Optional[dict[str, Any]] = None,
    ) -> widgets.VBox:
        """
        チャンネルの情報を表示します。
        Parameters:
            axis_config (dict): 各サブプロットの軸設定を格納する辞書。
                {
                    "time_plot": {"xlim": (0, 1)},
                    "freq_plot": {"ylim": (0, 20000)}
                }
            cbar_config (dict): カラーバーの設定を格納する辞書
                （例: {"vmin": -80, "vmax": 0}）。
        """
        content = [
            widgets.HTML(
                f"<span style='font-size:20px; font-weight:normal;'>"
                f"{self.label}, {self.sampling_rate} Hz</span>"
            )
        ]
        content += [
            ch.describe(axis_config=axis_config, cbar_config=cbar_config)
            for ch in self.channels
        ]
        # 中央寄せのレイアウトを設定
        layout = widgets.Layout(
            display="flex", justify_content="center", align_items="center"
        )
        return widgets.VBox(content, layout=layout)

    def plot(
        self,
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        overlay: bool = True,
        plot_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
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

        suptitle = title or self.label or "Signal"

        if not overlay:
            num_channels = len(self.channels)
            fig, axs = plt.subplots(
                num_channels, 1, figsize=(10, 4 * num_channels), sharex=True
            )
            if num_channels == 1:
                axs = [axs]  # Ensure axs is iterable when there's only one channel

            for i, channel in enumerate(self.channels):
                tmp = axs[i]
                channel.plot(ax=tmp, plot_kwargs=plot_kwargs)
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
            channel.plot(ax=tmp, plot_kwargs=plot_kwargs)

        tmp.grid(True)
        tmp.legend()
        tmp.set_title(suptitle)

        if ax is None:
            plt.tight_layout()
            plt.show()

    def rms_plot(
        self,
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        overlay: bool = True,
    ) -> None:
        """
        すべてのチャンネルの RMS データをプロットします。

        Parameters:
            title (str, optional): プロットのタイトル。
        """
        if ax is None:
            plt.tight_layout()

        if overlay:
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 4))

            for channel in self.channels:
                channel.rms_plot(ax=ax)

            ax.set_title(title or self.label or "Signal RMS")
            ax.grid(True)
            ax.legend()

            if ax is None:
                plt.tight_layout()
                plt.show()
        else:
            num_channels = len(self.channels)
            fig, axs = plt.subplots(
                num_channels, 1, figsize=(10, 4 * num_channels), sharex=True
            )
            if num_channels == 1:
                axs = [axs]  # Ensure axs is iterable when there's only one channel

            for i, channel in enumerate(self.channels):
                channel.rms_plot(ax=axs[i])

            axs[-1].set_xlabel("Time [s]")

            fig.suptitle(title or self.label or "Signal")
            plt.tight_layout()
            plt.show()

    def high_pass_filter(self, cutoff: float, order: int = 5) -> "ChannelFrame":
        """
        ハイパスフィルタをすべてのチャンネルに適用します。

        Parameters:
            cutoff (float): カットオフ周波数（Hz）。
            order (int): フィルタの次数。

        Returns:
            ChannelFrame: フィルタリングされた新しい ChannelFrame オブジェクト。
        """
        filtered_channels = [ch.high_pass_filter(cutoff, order) for ch in self.channels]
        return ChannelFrame(filtered_channels, label=self.label)

    def low_pass_filter(self, cutoff: float, order: int = 5) -> "ChannelFrame":
        """
        ローパスフィルタをすべてのチャンネルに適用します。

        Parameters:
            cutoff (float): カットオフ周波数（Hz）。
            order (int): フィルタの次数。

        Returns:
            ChannelFrame: フィルタリングされた新しい ChannelFrame オブジェクト。
        """
        filtered_channels = [ch.low_pass_filter(cutoff, order) for ch in self.channels]
        return ChannelFrame(filtered_channels, label=self.label)

    def fft(
        self,
        n_fft: Optional[int] = None,
        window: Optional[str] = None,
    ) -> "FrequencyChannelFrame":
        """
        フーリエ変換をすべてのチャンネルに適用します。

        Returns:
            Spectrum: 周波数と振幅データを含む Spectrum オブジェクト。
        """
        from wandas.core.frequency_channel_frame import FrequencyChannelFrame

        chs = [ch.fft(n_fft=n_fft, window=window) for ch in self.channels]

        return FrequencyChannelFrame(
            channels=chs,
            label=self.label,
        )

    def welch(
        self,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        win_length: int = 2048,
        window: str = "hann",
        average: str = "mean",
    ) -> "FrequencyChannelFrame":
        """
        Welch 法を用いたパワースペクトル密度推定を実行します。

        Returns:
            FrequencyChannelFrame: 周波数と振幅データを含む Spectrum オブジェクト。
        """
        from wandas.core.frequency_channel_frame import FrequencyChannelFrame

        chs = [
            ch.welch(
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                average=average,
            )
            for ch in self.channels
        ]

        return FrequencyChannelFrame(
            channels=chs,
            label=self.label,
        )

    # forでループを回すためのメソッド
    def __iter__(self) -> Iterator["Channel"]:
        return iter(self.channels)

    def __getitem__(self, key: Union[str, int]) -> "Channel":
        """
        チャンネル名またはインデックスでチャンネルを取得するためのメソッド。

        Parameters:
            key (str or int): チャンネルの名前（label）またはインデックス番号。

        Returns:
            Channel: 対応するチャンネル。
        """
        if isinstance(key, str):
            # チャンネル名でアクセス
            if key not in self.channel_dict:
                raise KeyError(f"Channel '{key}' not found.")
            return self.channel_dict[key]
        elif isinstance(key, numbers.Integral):
            # インデックス番号でアクセス
            if key < 0 or key >= len(self.channels):
                raise IndexError(f"Channel index {key} out of range.")
            return self.channels[key]
        else:
            raise TypeError(
                "Key must be either a string (channel name) or an integer "
                "(channel index)."
            )

    def __len__(self) -> int:
        """
        チャンネルのデータ長を返します。
        """
        return len(self.channels)

    # 演算子オーバーロードの実装
    def __add__(self, other: "ChannelFrame") -> "ChannelFrame":
        """
        シグナル間の加算。
        """
        assert len(self.channels) == len(other.channels), (
            "ChannelFrame must have the same number of channels."
        )
        channels = [
            self.channels[i] + other.channels[i] for i in range(len(self.channels))
        ]
        return ChannelFrame(channels=channels, label=f"({self.label} + {other.label})")

    def __sub__(self, other: "ChannelFrame") -> "ChannelFrame":
        """
        シグナル間の減算。
        """
        assert len(self.channels) == len(other.channels), (
            "ChannelFrame must have the same number of channels."
        )
        channels = [
            self.channels[i] - other.channels[i] for i in range(len(self.channels))
        ]
        return ChannelFrame(channels=channels, label=f"({self.label} - {other.label})")

    def __mul__(self, other: "ChannelFrame") -> "ChannelFrame":
        """
        シグナル間の乗算。
        """
        assert len(self.channels) == len(other.channels), (
            "ChannelFrame must have the same number of channels."
        )
        channels = [
            self.channels[i] * other.channels[i] for i in range(len(self.channels))
        ]
        return ChannelFrame(channels=channels, label=f"({self.label} * {other.label})")

    def __truediv__(self, other: "ChannelFrame") -> "ChannelFrame":
        """
        シグナル間の除算。
        """
        assert len(self.channels) == len(other.channels), (
            "ChannelFrame must have the same number of channels."
        )
        channels = [
            self.channels[i] / other.channels[i] for i in range(len(self.channels))
        ]
        return ChannelFrame(channels=channels, label=f"({self.label} / {other.label})")

    def sum(self) -> "Channel":
        """
        すべてのチャンネルを合計します。

        Returns:
            Channel: 合計されたチャンネル。
        """
        data = np.stack([ch.data for ch in self.channels]).sum(axis=0)
        result = dict(
            data=data.squeeze(),
        )
        return util.transform_channel(self.channels[0], Channel, **result)

    def mean(self) -> "Channel":
        """
        すべてのチャンネルの平均を計算します。

        Returns:
            Channel: 平均されたチャンネル。
        """
        data = np.stack([ch.data for ch in self.channels]).mean(axis=0)
        result = dict(
            data=data.squeeze(),
        )
        return util.transform_channel(self.channels[0], Channel, **result)

    def channel_difference(self, other_channel: int = 0) -> "ChannelFrame":
        """
        チャンネル間の差分を計算します。

        Returns:
            ChannelFrame: 差分を計算した新しい ChannelFrame オブジェクト。
        """
        channels = [ch - self.channels[other_channel] for ch in self.channels]
        return ChannelFrame(channels=channels, label=f"(ch[*] - ch[{other_channel}])")
