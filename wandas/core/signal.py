# wandas/core/signal.py

from typing import Optional, Any, List, Dict
import numpy as np
from .channel import Channel
from scipy.io import wavfile
from .spectrum import Spectrum
import matplotlib.pyplot as plt
from ..io import wav_io


class Signal:
    def __init__(self, channels: List[Channel], label: Optional[str] = None):
        """
        Signal オブジェクトを初期化します。

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

    @classmethod
    def read_wav(cls, filename: str, labels: Optional[List[str]] = None) -> "Signal":
        """
        WAV ファイルを読み込み、Signal オブジェクトを作成します。

        Parameters:
            filename (str): WAV ファイルのパス。
            labels (list of str, optional): 各チャンネルのラベル。

        Returns:
            Signal: オーディオデータを含む Signal オブジェクト。
        """
        return wav_io.read_wav(filename, labels)

    def to_wav(self, filename: str) -> None:
        """
        Signal オブジェクトを WAV ファイルに書き出します。

        Parameters:
            filename (str): 出力する WAV ファイルのパス。
        """
        # チャンネルデータを結合して配列にする
        data = np.column_stack([ch.data for ch in self.channels])

        # 16ビット整数にスケーリング
        max_int16 = np.iinfo(np.int16).max
        scaled_data = np.int16(data / np.max(np.abs(data)) * max_int16)
        wavfile.write(filename, self.sampling_rate, scaled_data)

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

    def low_pass_filter(self, cutoff: float, order: int = 5) -> "Signal":
        """
        ローパスフィルタをすべてのチャンネルに適用します。

        Parameters:
            cutoff (float): カットオフ周波数（Hz）。
            order (int): フィルタの次数。

        Returns:
            Signal: フィルタリングされた新しい Signal オブジェクト。
        """
        filtered_channels = [ch.low_pass_filter(cutoff, order) for ch in self.channels]
        return Signal(filtered_channels, label=self.label)

    def fft(
        self,
        n_fft: Optional[int] = None,
        window: Optional[str] = None,
        fft_params: Optional[Dict[str, Any]] = None,
    ) -> Spectrum:
        """
        フーリエ変換をすべてのチャンネルに適用します。

        Returns:
            Spectrum: 周波数と振幅データを含む Spectrum オブジェクト。
        """
        spectrums = [
            ch.fft(n_fft=n_fft, window=window, fft_params=fft_params)
            for ch in self.channels
        ]

        return Spectrum(
            channels=spectrums,
            label=self.label,
        )

    def get_channel_by_index(self, index: int) -> Channel:
        """
        指定されたインデックスのチャンネルを取得します。

        Parameters:
            index (int): チャンネルのインデックス。

        Returns:
            Channel: 対応するチャンネル。
        """
        if index < 0 or index >= len(self.channels):
            raise IndexError("Invalid channel index.")
        return self.channels[index]

    def apply_operation(self, ch1_idx: int, ch2_idx: int, operation: str) -> Channel:
        """
        指定されたチャンネル間に演算を適用します。

        Parameters:
            ch1_idx (int): 最初のチャンネルのインデックス。
            ch2_idx (int): 2番目のチャンネルのインデックス。
            operation (str): 実行する演算。'add', 'sub', 'mul', 'div' のいずれか。

        Returns:
            Channel: 新しいチャンネル（演算結果）。
        """
        ch1 = self.get_channel_by_index(ch1_idx)
        ch2 = self.get_channel_by_index(ch2_idx)

        if operation == "add":
            return ch1 + ch2
        elif operation == "sub":
            return ch1 - ch2
        elif operation == "mul":
            return ch1 * ch2
        elif operation == "div":
            return ch1 / ch2
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    # 演算子オーバーロードの実装
    def __add__(self, other: "Signal") -> "Signal":
        """
        シグナル間の加算。
        """
        assert len(self.channels) == len(
            other.channels
        ), "Signals must have the same number of channels."
        channels = [
            self.channels[i] + other.channels[i] for i in range(len(self.channels))
        ]
        return Signal(channels=channels, label=f"({self.label} + {other.label})")

    def __sub__(self, other: "Signal") -> "Signal":
        """
        シグナル間の減算。
        """
        assert len(self.channels) == len(
            other.channels
        ), "Signals must have the same number of channels."
        channels = [
            self.channels[i] - other.channels[i] for i in range(len(self.channels))
        ]
        return Signal(channels=channels, label=f"({self.label} - {other.label})")

    def __mul__(self, other: "Signal") -> "Signal":
        """
        シグナル間の乗算。
        """
        assert len(self.channels) == len(
            other.channels
        ), "Signals must have the same number of channels."
        channels = [
            self.channels[i] * other.channels[i] for i in range(len(self.channels))
        ]
        return Signal(channels=channels, label=f"({self.label} * {other.label})")

    def __truediv__(self, other: "Signal") -> "Signal":
        """
        シグナル間の除算。
        """
        assert len(self.channels) == len(
            other.channels
        ), "Signals must have the same number of channels."
        channels = [
            self.channels[i] / other.channels[i] for i in range(len(self.channels))
        ]
        return Signal(channels=channels, label=f"({self.label} / {other.label})")
