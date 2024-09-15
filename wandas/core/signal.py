# wandas/core/signal.py

from typing import Optional, Any, List, Dict
import numpy as np
from .channel import Channel
from scipy.io import wavfile
from .spectrum import Spectrum
import matplotlib.pyplot as plt


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
        sampling_rate, data = wavfile.read(filename)

        # データ型の正規化
        if data.dtype != np.float32 and data.dtype != np.float64:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        else:
            data = data.astype(np.float32)

        # データを2次元配列に変換（num_samples, num_channels）
        if data.ndim == 1:
            data = data[:, np.newaxis]

        num_channels = data.shape[1]
        channels = []

        for i in range(num_channels):
            channel_data = data[:, i]
            channel_label = (
                labels[i] if labels and i < len(labels) else f"Channel {i+1}"
            )
            channels.append(
                Channel(
                    data=channel_data, sampling_rate=sampling_rate, label=channel_label
                )
            )

        return cls(channels, label=filename)

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
