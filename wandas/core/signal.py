# wandas/core/signal.py

from typing import Optional, Any, List, Union, TYPE_CHECKING
import numpy as np

from scipy.io import wavfile
import matplotlib.pyplot as plt
from wandas.io import wav_io
from wandas.core.channel import Channel
import ipywidgets as widgets

if TYPE_CHECKING:
    from wandas.core.spectrums import Spectrums


class ChannelFrame:
    def __init__(self, channels: List["Channel"], label: Optional[str] = None):
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
        array: np.ndarray,
        sampling_rate: int,
        labels: Optional[List[str]] = None,
        unit: str = "Pa",
    ) -> "ChannelFrame":
        """
        numpy の ndarray から ChannelFrame インスタンスを生成します。

        Parameters:
            array (np.ndarray): 信号データ。各行がチャンネルに対応します。
            sampling_rate (int): サンプリングレート（Hz）。
            labels (List[str], optional): 各チャンネルのラベル。
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
        cls, filename: str, labels: Optional[List[str]] = None
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
        # チャンネルデータを結合して配列にする
        data = np.column_stack([ch.data for ch in self.channels])

        # 16ビット整数にスケーリング
        max_int16 = np.iinfo(np.int16).max
        scaled_data = np.int16(data / np.max(np.abs(data)) * max_int16)
        wavfile.write(filename, self.sampling_rate, scaled_data)

    def to_Audio(self, normalize: bool = True):
        return widgets.VBox([ch.to_Audio(normalize) for ch in self.channels])

    def describe(self):
        """
        チャンネルの情報を表示します。
        """
        content = [
            widgets.HTML(
                f"<span style='font-size:20px; font-weight:normal;'>{self.label}, {self.sampling_rate} Hz</span>"
            )
        ]
        content += [ch.describe() for ch in self.channels]
        # 中央寄せのレイアウトを設定
        layout = widgets.Layout(
            display="flex", justify_content="center", align_items="center"
        )
        return widgets.VBox(content, layout=layout)

    def plot(self, ax: Optional[Any] = None, title: Optional[str] = None):
        """
        すべてのチャンネルをプロットします。

        Parameters:
            title (str, optional): プロットのタイトル。
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        for channel in self.channels:
            channel.plot(ax=ax)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(title or self.label or "Signal")
        ax.grid(True)
        ax.legend()

        if ax is None:
            plt.tight_layout()
            plt.show()

    def rms_plot(self, ax: Optional[Any] = None, title: Optional[str] = None):
        """
        すべてのチャンネルの RMS データをプロットします。

        Parameters:
            title (str, optional): プロットのタイトル。
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        for channel in self.channels:
            channel.rms_plot(ax=ax)

        # ax.set_xlabel("Time [s)")
        # ax.set_ylabel("RMS")
        ax.set_title(title or self.label or "Signal RMS")
        ax.grid(True)
        ax.legend()

        if ax is None:
            plt.tight_layout()

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
    ) -> "Spectrums":
        """
        フーリエ変換をすべてのチャンネルに適用します。

        Returns:
            Spectrum: 周波数と振幅データを含む Spectrum オブジェクト。
        """
        from wandas.core.spectrums import Spectrums

        chs = [ch.fft(n_fft=n_fft, window=window) for ch in self.channels]

        return Spectrums(
            channels=chs,
            label=self.label,
        )

    # forでループを回すためのメソッド
    def __iter__(self):
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
        elif isinstance(key, int):
            # インデックス番号でアクセス
            if key < 0 or key >= len(self.channels):
                raise IndexError(f"Channel index {key} out of range.")
            return self.channels[key]
        else:
            raise TypeError(
                "Key must be either a string (channel name) or an integer (channel index)."
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
        assert len(self.channels) == len(
            other.channels
        ), "ChannelFrame must have the same number of channels."
        channels = [
            self.channels[i] + other.channels[i] for i in range(len(self.channels))
        ]
        return ChannelFrame(channels=channels, label=f"({self.label} + {other.label})")

    def __sub__(self, other: "ChannelFrame") -> "ChannelFrame":
        """
        シグナル間の減算。
        """
        assert len(self.channels) == len(
            other.channels
        ), "ChannelFrame must have the same number of channels."
        channels = [
            self.channels[i] - other.channels[i] for i in range(len(self.channels))
        ]
        return ChannelFrame(channels=channels, label=f"({self.label} - {other.label})")

    def __mul__(self, other: "ChannelFrame") -> "ChannelFrame":
        """
        シグナル間の乗算。
        """
        assert len(self.channels) == len(
            other.channels
        ), "ChannelFrame must have the same number of channels."
        channels = [
            self.channels[i] * other.channels[i] for i in range(len(self.channels))
        ]
        return ChannelFrame(channels=channels, label=f"({self.label} * {other.label})")

    def __truediv__(self, other: "ChannelFrame") -> "ChannelFrame":
        """
        シグナル間の除算。
        """
        assert len(self.channels) == len(
            other.channels
        ), "ChannelFrame must have the same number of channels."
        channels = [
            self.channels[i] / other.channels[i] for i in range(len(self.channels))
        ]
        return ChannelFrame(channels=channels, label=f"({self.label} / {other.label})")
