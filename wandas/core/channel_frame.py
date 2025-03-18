# wandas/core/signal.py
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import ipywidgets as widgets
import numpy as np
import pandas as pd

from wandas.core.channel import Channel
from wandas.core.channel_access_mixin import ChannelAccessMixin
from wandas.io import wav_io
from wandas.utils.types import NDArrayReal

from . import channel_frame_processing as cfp
from .channel_frame_plotter import ChannelFramePlotter

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from wandas.core.frequency_channel_frame import FrequencyChannelFrame
    from wandas.core.matrix_frame import MatrixFrame


class ChannelFrame(ChannelAccessMixin["Channel"]):
    def __init__(self, channels: list["Channel"], label: Optional[str] = None):
        """

        ChannelFrame オブジェクトを初期化します。

        Parameters:
            channels (list of Channel): Channel オブジェクトのリスト。
            label (str, optional): 信号のラベル。
        """
        self._channels = channels
        self.label = label

        # サンプリングレートの一貫性をチェック
        sampling_rates = set(ch.sampling_rate for ch in channels)
        if len(sampling_rates) > 1:
            raise ValueError("All channels must have the same sampling_rate.")

        self.sampling_rate = channels[0].sampling_rate
        self._channel_dict = {ch.label: ch for ch in self.channels}
        if len(self._channel_dict) != len(self):
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
        return widgets.VBox([ch.to_audio(normalize) for ch in self._channels])

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
            for ch in self._channels
        ]
        # 中央寄せのレイアウトを設定
        layout = widgets.Layout(
            display="flex", justify_content="center", align_items="center"
        )
        return widgets.VBox(content, layout=layout)

    def trim(self, start: float, end: float) -> "ChannelFrame":
        """
        指定された時間範囲でチャンネルをトリムします。

        Parameters:
            start (float): トリムの開始時間（秒）。
            end (float): トリムの終了時間（秒）。

        Returns:
            ChannelFrame: トリムされた新しい ChannelFrame オブジェクト。
        """
        return cfp.trim_channel_frame(self, start, end)

    def cut(
        self,
        point_list: Union[list[int], list[float]],
        cut_len: Union[int, float],
        taper_rate: float = 0,
        dc_cut: bool = False,
    ) -> list["MatrixFrame"]:
        """
        チャンネルを指定された時間点でカットします。

        Parameters:
            point_list (list[int]): カットポイントのリスト。
            cut_len (int): カットするデータ長。
            taper_rate (float): テーパー率。
            dc_cut (bool): DC カット。

        Returns:
            Channel: カットされた新しい Channel オブジェクト。
        """

        return cfp.cut_channel_frame(
            cf=self,
            point_list=point_list,
            cut_len=cut_len,
            taper_rate=taper_rate,
            dc_cut=dc_cut,
        )

    def to_matrix_frame(self) -> "MatrixFrame":
        """
        ChannelFrame オブジェクトを MatrixFrame オブジェクトに変換します。

        Returns:
            MatrixFrame: チャンネルデータを含む MatrixFrame オブジェクト。
        """
        from wandas.core.matrix_frame import MatrixFrame

        return MatrixFrame.from_channel_frame(self)

    def plot(
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
        plotter = ChannelFramePlotter(self)

        return plotter.plot_time(
            ax=ax, title=title, overlay=overlay, plot_kwargs=plot_kwargs
        )

    def rms_plot(
        self,
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        overlay: bool = True,
        plot_kwargs: Optional[dict[str, Any]] = None,
    ) -> Union["Axes", Iterable["Axes"]]:
        """
        すべてのチャンネルの RMS データをプロットします。

        Parameters:
            title (str, optional): プロットのタイトル。
        """
        plotter = ChannelFramePlotter(self)

        return plotter.rms_plot(
            ax=ax, title=title, overlay=overlay, plot_kwargs=plot_kwargs
        )

    def high_pass_filter(self, cutoff: float, order: int = 5) -> "ChannelFrame":
        """
        ハイパスフィルタをすべてのチャンネルに適用します。

        Parameters:
            cutoff (float): カットオフ周波数（Hz）。
            order (int): フィルタの次数。

        Returns:
            ChannelFrame: フィルタリングされた新しい ChannelFrame オブジェクト。
        """
        filtered_channels = [ch.high_pass_filter(cutoff, order) for ch in self]
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
        filtered_channels = [ch.low_pass_filter(cutoff, order) for ch in self]
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

        chs = [ch.fft(n_fft=n_fft, window=window) for ch in self]

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
            for ch in self
        ]

        return FrequencyChannelFrame(
            channels=chs,
            label=self.label,
        )

    def _op(
        self,
        other: "ChannelFrame",
        op: Callable[["Channel", "Channel"], "Channel"],
        symbol: str,
    ) -> "ChannelFrame":
        assert len(self) == len(other), (
            "ChannelFrame must have the same number of channels."
        )

        channels: list[Channel] = [op(self[i], other[i]) for i in range(len(self))]

        return ChannelFrame(
            channels=channels, label=f"({self.label} {symbol} {other.label})"
        )

    # 演算子オーバーロードの実装
    def __add__(self, other: "ChannelFrame") -> "ChannelFrame":
        """
        シグナル間の加算。
        """
        return self._op(other, lambda a, b: a + b, "+")

    def __sub__(self, other: "ChannelFrame") -> "ChannelFrame":
        """
        シグナル間の減算。
        """
        return self._op(other, lambda a, b: a - b, "-")

    def __mul__(self, other: "ChannelFrame") -> "ChannelFrame":
        """
        シグナル間の乗算。
        """
        return self._op(other, lambda a, b: a * b, "*")

    def __truediv__(self, other: "ChannelFrame") -> "ChannelFrame":
        """
        シグナル間の除算。
        """
        return self._op(other, lambda a, b: a / b, "/")

    def sum(self) -> "Channel":
        """
        すべてのチャンネルを合計します。

        Returns:
            Channel: 合計されたチャンネル。
        """
        data = np.stack([ch.data for ch in self._channels]).sum(axis=0)
        return Channel.from_channel(self._channels[0], data=data.squeeze())

    def mean(self) -> "Channel":
        """
        すべてのチャンネルの平均を計算します。

        Returns:
            Channel: 平均されたチャンネル。
        """
        data = np.stack([ch.data for ch in self._channels]).mean(axis=0)
        return Channel.from_channel(self._channels[0], data=data.squeeze())

    def channel_difference(self, other_channel: int = 0) -> "ChannelFrame":
        """
        チャンネル間の差分を計算します。

        Returns:
            ChannelFrame: 差分を計算した新しい ChannelFrame オブジェクト。
        """
        channels = [ch - self._channels[other_channel] for ch in self._channels]
        return ChannelFrame(channels=channels, label=f"(ch[*] - ch[{other_channel}])")
