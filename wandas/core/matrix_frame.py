# wandas/core/matrix_frame.py

from collections.abc import Iterator
from typing import Any, Optional, Union

import numpy as np
import scipy.signal as ss

from wandas.core.channel import Channel
from wandas.core.channel_frame import ChannelFrame
from wandas.core.frequency_channel_frame import FrequencyChannelFrame
from wandas.utils.types import NDArrayReal

from .frequency_channel import FrequencyChannel


class MatrixFrame:
    def __init__(
        self,
        data: NDArrayReal,
        sampling_rate: int,
        channel_units: Optional[list[str]] = None,
        channel_labels: Optional[list[str]] = None,
        channel_metadata: Optional[list[dict[str, Any]]] = None,
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
        self._channels = [
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
        self.label_to_index = {ch.label: idx for idx, ch in enumerate(self._channels)}

    def __len__(self) -> int:
        """
        チャンネルの数を返します。
        """
        return int(self.data.shape[0])

    # forでループを回すためのメソッド
    def __iter__(self) -> Iterator["Channel"]:
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
        ch = self._channels[idx]

        # Channel オブジェクトを作成して返す
        return Channel.from_channel(ch, data=self.data[idx].copy())

    def to_channel_frame(self) -> "ChannelFrame":
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
    def from_channel_frame(cls, cf: "ChannelFrame") -> "MatrixFrame":
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
            channel_labels=[ch.label for ch in cf],
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
    ) -> "FrequencyChannelFrame":
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
        channel_labels = [f"{ich.label} & {jch.label}" for ich in self for jch in self]
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

        return FrequencyChannelFrame(freq_channels, label=label)

    def csd(
        self,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        win_length: int = 2048,
        window: str = "hann",
        detrend: str = "constant",
        scaling: str = "spectrum",
        average: str = "mean",
    ) -> "FrequencyChannelFrame":
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

        f, csd = ss.csd(
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
        coh = np.sqrt(csd.reshape(-1, csd.shape[-1]))
        channel_labels = [f"{ich.label} & {jch.label}" for ich in self for jch in self]
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

        return FrequencyChannelFrame(freq_channels, label=label)

    def transfer_function(
        self,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        win_length: int = 2048,
        window: str = "hann",
        detrend: str = "constant",
        scaling: str = "spectrum",
        average: str = "mean",
    ) -> "FrequencyChannelFrame":
        """
        伝達関数を推定します。

        Parameters:
            n_fft (int, optional): FFT のサンプル数。
            hop_length (int, optional): オーバーラップのサンプル数。
            win_length (int, optional): 窓関数のサイズ。
            window (str, optional): 窓関数の種類。
            detrend (str, optional): トレンドの除去方法。

        Returns:
            FrequencyChannelFrame: 伝達関数データを含むオブジェクト。
        """
        if win_length is None:
            win_length = 2048
        if n_fft is None:
            n_fft = win_length
        if hop_length is None:
            hop_length = win_length // 2

        num_channels = self.data.shape[0]

        # クロススペクトル密度の計算（全チャンネル間）
        f, p_yx = ss.csd(
            x=self.data[:, np.newaxis, :],  # shape: (チャンネル数, 1, サンプル数)
            y=self.data[np.newaxis, :, :],  # shape: (1, チャンネル数, サンプル数)
            fs=self.sampling_rate,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            window=window,
            detrend=detrend,
            scaling=scaling,
            average=average,
            axis=-1,
        )
        # P_yx の形状: (チャンネル数, チャンネル数, 周波数数)

        # パワースペクトル密度の計算（各チャンネル）
        f, p_xx = ss.welch(
            x=self.data,
            fs=self.sampling_rate,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            window=window,
            detrend=detrend,
            scaling=scaling,
            average=average,
            axis=-1,
        )
        # P_xx の形状: (チャンネル数, 周波数数)

        # 伝達関数の計算 H(f) = P_yx / P_xx（P_xx をブロードキャスト）
        h_f = (
            p_yx / p_xx[np.newaxis, :, :]
        )  # P_xx を形状 (1, チャンネル数, 周波数数) に拡張

        # ラベルと単位の生成
        channel_labels = np.array(
            [
                [
                    f"{self._channels[i].label} / {self._channels[j].label}"
                    for j in range(num_channels)
                ]
                for i in range(num_channels)
            ]
        )
        channel_units = np.array(
            [
                [
                    f"{self._channels[i].unit} / {self._channels[j].unit}"
                    for j in range(num_channels)
                ]
                for i in range(num_channels)
            ]
        )

        # H_f, channel_labels, channel_units を一次元配列に変形
        h_f_flat = h_f.reshape(
            -1, h_f.shape[-1]
        )  # shape: (チャンネル数 * チャンネル数, 周波数数)
        channel_labels_flat = channel_labels.flatten()
        channel_units_flat = channel_units.flatten()

        # FrequencyChannel のリストを作成
        freq_channels = [
            FrequencyChannel(
                data=h_f_flat[k],
                sampling_rate=self.sampling_rate,
                window=window,
                label=channel_labels_flat[k],
                n_fft=n_fft,
                unit=channel_units_flat[k],
            )
            for k in range(h_f_flat.shape[0])
        ]

        return FrequencyChannelFrame(freq_channels, label="Transfer Function")

    def plot(
        self,
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        overlay: bool = True,
    ) -> None:
        """
        すべてのチャンネルをプロットします。

        Parameters:
            ax (matplotlib.axes.Axes, optional): プロット先の軸。
            title (str, optional): プロットのタイトル。
        """
        cf = self.to_channel_frame()
        cf.plot(ax=ax, title=title, overlay=overlay)
