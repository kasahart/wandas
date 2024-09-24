# wandas/core/matrix_frame.py

from typing import Optional, Any, List, Union, Dict
import numpy as np

import scipy.signal as ss
from wandas.core.channel import Channel
import wandas.core.util as util
from .frequency_channel import FrequencyChannel
from wandas.core.signal import ChannelFrame
from wandas.core.spectrums import Spectrums


class MatrixFrame:
    def __init__(
        self,
        data: np.ndarray,
        sampling_rate: int,
        channel_units: Optional[List[str]] = None,
        channel_labels: Optional[List[str]] = None,
        channel_metadata: Optional[List[Dict[str, Any]]] = None,
        label: Optional[str] = None,
    ):
        """
        ChannelFrame オブジェクトを初期化します。

        Parameters:
            data (numpy.ndarray): 形状が (チャンネル数, サンプル数) の多次元配列。
            sampling_rate (int): サンプリングレート（Hz）。
            labels (list of str, optional): 各チャンネルのラベル。
            metadata (list of dict, optional): 各チャンネルのメタデータ。
            label (str, optional): ChannelFrame のラベル。
        """
        if data.ndim != 2:
            raise ValueError(
                "Data must be a 2D NumPy array with shape (num_channels, num_samples)."
            )

        self.data = data  # 形状: (チャンネル数, サンプル数)
        self.sampling_rate = sampling_rate
        self.label = label

        num_channels = data.shape[0]

        # unitの処理
        if channel_units is not None:
            if len(channel_units) != num_channels:
                raise ValueError(
                    "Length of channel_units must match number of channels."
                )
        else:
            channel_units = ["" for i in range(num_channels)]

        # ラベルの処理
        if channel_labels is not None:
            if len(channel_labels) != num_channels:
                raise ValueError(
                    "Length of channel_labels must match number of channels."
                )
        else:
            channel_labels = [f"Ch{i}" for i in range(num_channels)]

        # メタデータの処理
        if channel_metadata is not None:
            if len(channel_metadata) != num_channels:
                raise ValueError(
                    "Length of channel_metadata must match number of channels."
                )
        else:
            channel_metadata = [{} for _ in range(num_channels)]

            # BaseChannel オブジェクトのリストを作成
        self.channels = [
            Channel(
                data=np.array([]),
                sampling_rate=sampling_rate,
                unit=unit,
                label=label,
                metadata=metadata,
            )
            for unit, label, metadata in zip(
                channel_units, channel_labels, channel_metadata
            )
        ]

        # ラベルからインデックスへのマッピングを作成
        self.label_to_index = {ch.label: idx for idx, ch in enumerate(self.channels)}

    def __len__(self) -> int:
        """
        チャンネルの数を返します。
        """
        return self.data.shape[0]

    # forでループを回すためのメソッド
    def __iter__(self):
        """
        チャンネルをイテレートします。
        """
        for idx in range(self.data.shape[0]):
            yield self[idx]

    def __getitem__(self, key: Union[int, str]) -> "Channel":
        """
        インデックスまたはラベルでチャンネルを取得します。

        Parameters:
            key (int or str): チャンネルのインデックスまたはラベル。

        Returns:
            Channel: 対応する Channel オブジェクト。
        """

        if isinstance(key, int):
            # インデックスでアクセス
            if key < 0 or key >= self.data.shape[0]:
                raise IndexError("Channel index out of range.")
            idx = key
        elif isinstance(key, str):
            # ラベルでアクセス
            if key not in self.label_to_index:
                raise KeyError(f"Channel label '{key}' not found.")
            idx = self.label_to_index[key]
        else:
            raise TypeError("Key must be an integer index or a string label.")

        # チャネルデータとメタデータを取得
        ch = self.channels[idx]

        # Channel オブジェクトを作成して返す
        return util.transform_channel(
            org=ch, target_class=Channel, data=self.data[idx].copy()
        )

    def toChannelFrame(self) -> "ChannelFrame":
        """
        ChannelFrame オブジェクトに変換します。

        Returns:
            ChannelFrame: 変換された ChannelFrame オブジェクト。
        """
        return ChannelFrame(
            channels=[ch for ch in self],
            label=self.label,
        )

    @classmethod
    def fromChannelFrame(cls, cf: "ChannelFrame") -> "MatrixFrame":
        """
        ChannelFrame オブジェクトから MatrixFrame オブジェクトに変換します。

        Parameters:
            cf (ChannelFrame): 変換元の ChannelFrame オブジェクト。

        Returns:
            MatrixFrame: 変換された MatrixFrame オブジェクト。
        """
        # チャンネルデータの長さが全て等しいか確認
        length = len(cf[0].data)
        if not all([len(ch.data) == length for ch in cf]):
            raise ValueError("All channels must have the same length.")

        return MatrixFrame(
            data=np.array([ch.data for ch in cf]),
            sampling_rate=cf.sampling_rate,
            channel_units=[ch.unit for ch in cf],
            channel_labels=[ch.label for ch in cf],  # type: ignore
            channel_metadata=[ch.metadata for ch in cf],
            label=cf.label,
        )

    def coherence(
        self,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        win_length: int = 2048,
        window: str = "hann",
        detrend: str = "constant",
    ) -> "Spectrums":
        """
        コヒーレンス推定を実行します。

        Parameters:
            n_fft (int, optional): FFT のサンプル数。
            hop_length (int, optional): オーバーラップのサンプル数。
            win_length (int, optional): 窓関数のサイズ。
            window (str, optional): 窓関数の種類。
            detrend (str, optional): トレンドの除去方法。

        Returns:
            Spectrums: コヒーレンスデータを含むオブジェクト。
        """
        if win_length is None:
            win_length = 2048
        if n_fft is None:
            n_fft = win_length
        if hop_length is None:
            hop_length = win_length // 2

        f, coh = ss.coherence(
            x=self.data[:, np.newaxis],
            y=self.data[np.newaxis],
            fs=self.sampling_rate,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            window=window,
            detrend=detrend,
        )
        coh = coh.reshape(-1, coh.shape[-1])
        channel_labels = [
            f"Coherence between {ich.label} and {jch.label}"
            for ich in self
            for jch in self
        ]
        label = "Coherence"

        freq_channels = [
            FrequencyChannel(
                data=data,
                sampling_rate=self.sampling_rate,
                window=window,
                label=label,
                n_fft=n_fft,
            )
            for data, label in zip(coh, channel_labels)
        ]

        return Spectrums(freq_channels, label=label)

    def csd(
        self,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        win_length: int = 2048,
        window: str = "hann",
        detrend: str = "constant",
        scaling: str = "spectrum",
        average: str = "mean",
    ) -> "Spectrums":
        """
        クロススペクトル推定を実行します。

        Parameters:
            n_fft (int, optional): FFT のサンプル数。
            hop_length (int, optional): オーバーラップのサンプル数。
            win_length (int, optional): 窓関数のサイズ。
            window (str, optional): 窓関数の種類。
            detrend (str, optional): トレンドの除去方法。

        Returns:
            Spectrums: クロススペクトル密度データを含むオブジェクト。
        """
        if win_length is None:
            win_length = 2048
        if n_fft is None:
            n_fft = win_length
        if hop_length is None:
            hop_length = win_length // 2

        f, coh = ss.csd(
            x=self.data[:, np.newaxis],
            y=self.data[np.newaxis],
            fs=self.sampling_rate,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            window=window,
            detrend=detrend,
            scaling=scaling,
            average=average,
        )
        coh = np.sqrt(coh.reshape(-1, coh.shape[-1]))
        channel_labels = [
            f"Cross power spectral between {ich.label} and {jch.label}"
            for ich in self
            for jch in self
        ]
        channel_units = [f"{ich.unit}*{jch.unit}" for ich in self for jch in self]
        label = "Cross power spectral"

        freq_channels = [
            FrequencyChannel(
                data=data,
                sampling_rate=self.sampling_rate,
                window=window,
                label=label,
                n_fft=n_fft,
                unit=unit,
            )
            for data, label, unit in zip(coh, channel_labels, channel_units)
        ]

        return Spectrums(freq_channels, label=label)

    def plot(self, ax: Optional[Any] = None, title: Optional[str] = None):
        """
        すべてのチャンネルをプロットします。

        Parameters:
            ax (matplotlib.axes.Axes, optional): プロット先の軸。
            title (str, optional): プロットのタイトル。
        """
        cf = self.toChannelFrame()
        cf.plot(ax=ax, title=title)
