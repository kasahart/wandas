# wandas/core/channel.py

from typing import Optional, Dict, Any
import librosa.feature
import numpy as np
from .base_channel import BaseChannel
from scipy.signal import butter, filtfilt
from .frequency_channel import FrequencyChannel, NOctChannel
import librosa
from .time_frequency_channel import TimeFrequencyChannel, TimeMelFrequencyChannel
import wandas.core.util as util
from IPython.display import Audio, display
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Channel(BaseChannel):
    def __init__(
        self,
        data: np.ndarray,
        sampling_rate: int,
        label: Optional[str] = None,
        unit: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Channel オブジェクトを初期化します。

        Parameters:
            data (numpy.ndarray): 時系列データ。
            sampling_rate (int): サンプリングレート（Hz）。
            その他のパラメータは BaseChannel を参照。
        """
        super().__init__(
            data=data,
            sampling_rate=sampling_rate,
            label=label,
            unit=unit,
            metadata=metadata,
        )

    @property
    def time(self):
        """
        時刻データを返します。
        """
        num_samples = len(self._data)
        return np.arange(num_samples) / self.sampling_rate

    def high_pass_filter(self, cutoff: float, order: int = 5):
        """
        ハイパスフィルタを適用します。

        Parameters:
            cutoff (float): カットオフ周波数（Hz）。
            order (int): フィルタの次数。

        Returns:
            Channel: フィルタリングされた新しい Channel オブジェクト。
        """

        nyq = 0.5 * self.sampling_rate
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype="highpass", analog=False)
        filtered_data = filtfilt(b, a, self._data)

        result = dict(
            data=filtered_data.squeeze(),
        )

        return util.transform_channel(self, self.__class__, **result)

    def low_pass_filter(self, cutoff: float, order: int = 5):
        """
        ローパスフィルタを適用します。

        Parameters:
            cutoff (float): カットオフ周波数（Hz）。
            order (int): フィルタの次数。

        Returns:
            Channel: フィルタリングされた新しい Channel オブジェクト。
        """

        nyq = 0.5 * self.sampling_rate
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        filtered_data = filtfilt(b, a, self._data)

        result = dict(
            data=filtered_data.squeeze(),
        )

        return util.transform_channel(self, self.__class__, **result)

    def fft(
        self,
        n_fft: Optional[int] = None,
        window: Optional[str] = None,
    ) -> "FrequencyChannel":
        """
        フーリエ変換を実行します。

        Parameters:
            n_fft (int, optional): FFT のサンプル数。
            window (str, optional): ウィンドウ関数の種類。
            fft_params (dict, optional): その他の FFT パラメータ。

        Returns:
            FrequencyChannel: スペクトルデータを含むオブジェクト。
        """
        result = FrequencyChannel.fft(data=self.data, n_fft=n_fft, window=window)

        return util.transform_channel(self, FrequencyChannel, **result)

    def welch(
        self,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        win_length: int = 2048,
        window: str = "hann",
        average: str = "mean",
        # pad_mode: str = "constant"
    ) -> "FrequencyChannel":
        """
        Welch 法を用いたパワースペクトル密度推定を実行します。

        Parameters:
            nperseg (int): セグメントのサイズ。
            noverlap (int, optional): オーバーラップのサイズ。

        Returns:
            FrequencyChannel: スペクトルデータを含むオブジェクト。
        """
        result = FrequencyChannel.welch(
            data=self.data,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            average=average,
        )
        return util.transform_channel(self, FrequencyChannel, **result)

    def noct_spectrum(
        self,
        n_octaves: int = 3,
        fmin: float = 20,
        fmax: float = 20000,
        G: int = 10,
        fr: int = 1000,
    ) -> "NOctChannel":
        """
        オクターブバンドのスペクトルを計算します。

        Parameters:
            n_octaves (int): オクターブの数。

        Returns:
            FrequencyChannel: オクターブバンドのスペクトルデータを含むオブジェクト。
        """
        freqs = np.geomspace(fmin, fmax, n_octaves * G)
        freqs = freqs[(freqs >= fmin) & (freqs <= fmax)]
        freqs = np.round(freqs / fr) * fr

        result = NOctChannel.noct_spectrum(
            data=self.data,
            sampling_rate=self.sampling_rate,
            fmin=fmin,
            fmax=fmax,
            G=G,
            fr=fr,
        )
        return util.transform_channel(self, NOctChannel, **result)

    def stft(
        self,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = "hann",
        center: bool = True,
        # pad_mode: str = "constant",
    ) -> "TimeFrequencyChannel":
        """
        STFT（短時間フーリエ変換）を実行します。

        Parameters:
            n_fft (int): FFT のサンプル数。デフォルトは 1024。
            hop_length (int): ホップサイズ（フレーム間の移動量）。デフォルトは 512。
            win_length (int, optional): ウィンドウの長さ。デフォルトは n_fft と同じ。

        Returns:
            FrequencyChannel: STFT の結果を格納した FrequencyChannel オブジェクト。
        """

        result = TimeFrequencyChannel.stft(
            data=self.data,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            # center=center,
            # pad_mode=pad_mode,
        )
        return util.transform_channel(self, TimeFrequencyChannel, **result)

    def melspectrogram(
        self,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: int = 2048,
        window: str = "hann",
        center: bool = True,
        # pad_mode: str = "constant",
    ) -> "TimeMelFrequencyChannel":
        tf_ch = self.stft(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            # center=center,
            # pad_mode=pad_mode,
        )

        return tf_ch.melspectrogram(n_mels=n_mels)

    def rms_trend(self, frame_length: int = 2048, hop_length: int = 512):
        """
        移動平均を計算します。

        Parameters:
            window_size (int): 移動平均のウィンドウサイズ。

        Returns:
            Channel: 移動平均データを含む新しい Channel オブジェクト。
        """
        rms_data = librosa.feature.rms(
            y=self.data, frame_length=frame_length, hop_length=hop_length
        )
        result = dict(
            data=rms_data.squeeze(),
            sampling_rate=int(self.sampling_rate / hop_length),
        )

        return util.transform_channel(self, self.__class__, **result)

    def plot(self, ax: Optional[Any] = None, title: Optional[str] = None):
        """
        時系列データをプロットします。
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(self.time, self.data, label=self.label or "Channel")

        ax.set_xlabel("Time [s]")
        ylabel = f"Amplitude [{self.unit}]" if self.unit else "Amplitude"
        ax.set_ylabel(ylabel)
        ax.set_title(title or self.label or "Channel Data")
        ax.grid(True)
        ax.legend()

        if ax is None:
            plt.tight_layout()
            plt.show()

    def rms_plot(
        self, ax: Optional[Any] = None, title: Optional[str] = None
    ) -> "Channel":
        """
        RMS データをプロットします。
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        rms_channel: Channel = self.rms_trend()  # type: ignore
        num_samples = len(rms_channel)
        t = np.arange(num_samples) / rms_channel.sampling_rate
        ax.plot(
            t,
            librosa.amplitude_to_db(rms_channel.data, ref=self.ref),
            label=rms_channel.label or "Channel",
        )

        ax.set_xlabel("Time [s]")
        ylabel = f"RMS [{rms_channel.unit}]" if rms_channel.unit else "RMS"
        ax.set_ylabel(ylabel)
        ax.set_title(title or rms_channel.label or "Channel Data")
        ax.grid(True)
        ax.legend()

        if ax is None:
            plt.tight_layout()

        return rms_channel

    def __len__(self) -> int:
        """
        チャンネルのデータ長を返します。
        """
        return self._data.shape[-1]

    # 演算子オーバーロードの実装
    def __add__(self, other: "Channel") -> "Channel":
        """
        チャンネル間の加算。
        """
        assert (
            self.sampling_rate == other.sampling_rate
        ), "Sampling rates must be the same for channel addition."
        result = dict(
            data=self.data + other.data,
            sampling_rate=self.sampling_rate,
            label=f"({self.label} + {other.label})",
        )
        return util.transform_channel(self, self.__class__, **result)

    def __sub__(self, other: "Channel") -> "Channel":
        """
        チャンネル間の減算。
        """
        assert (
            self.sampling_rate == other.sampling_rate
        ), "Sampling rates must be the same for channel subtraction."
        result = dict(
            data=self.data - other.data,
            sampling_rate=self.sampling_rate,
            label=f"({self.label} - {other.label})",
        )
        return util.transform_channel(self, self.__class__, **result)

    def __mul__(self, other: "Channel") -> "Channel":
        """
        チャンネル間の乗算。
        """
        assert (
            self.sampling_rate == other.sampling_rate
        ), "Sampling rates must be the same for channel multiplication."
        result = dict(
            data=self.data * other.data,
            sampling_rate=self.sampling_rate,
            label=f"({self.label} * {other.label})",
        )
        return util.transform_channel(self, self.__class__, **result)

    def __truediv__(self, other: "Channel") -> "Channel":
        """
        チャンネル間の除算。
        """
        assert (
            self.sampling_rate == other.sampling_rate
        ), "Sampling rates must be the same for channel division."
        result = dict(
            data=self.data / other.data,
            sampling_rate=self.sampling_rate,
            label=f"({self.label} / {other.label})",
        )
        return util.transform_channel(self, self.__class__, **result)

    def to_Audio(self, normalize: bool = True, label: bool = True):
        output = widgets.Output()
        with output:
            display(Audio(self.data, rate=self.sampling_rate, normalize=normalize))

        if label:
            vbov = widgets.VBox([widgets.Label(self.label) if label else None, output])
        else:
            vbov = widgets.VBox([output])
        return vbov

    def describe(self):
        """
        チャンネルの統計情報を表示します。
        """

        gs = gridspec.GridSpec(
            2, 3, height_ratios=[1, 3], width_ratios=[3, 1, 0.1]
        )  # カラーマップの横幅をスペクトルの平均値の横幅の3倍に設定
        gs.update(wspace=0.2)

        fig = plt.figure(figsize=(12, 6))

        # 最初のサブプロットを作成
        ax_1 = fig.add_subplot(gs[0])
        ax_1.plot(self.time, self.data)
        ax_1.set(
            ylabel=f"Amplitude [{self.unit}]" if self.unit else "Amplitude",
        )
        ax_1.grid(True)

        # 2番目のサブプロットを作成し、x軸をax1と連動
        ax_2 = fig.add_subplot(gs[3], sharex=ax_1)
        stft_ch = self.stft()
        img = librosa.display.specshow(
            data=stft_ch._to_db(),
            sr=stft_ch.sampling_rate,
            hop_length=stft_ch.hop_length,
            n_fft=stft_ch.n_fft,
            win_length=stft_ch.win_length,
            x_axis="time",
            y_axis="linear",
            ax=ax_2,
            cmap="magma",
        )

        # 3番目のサブプロットを作成し、y軸をax1と連動
        ax_3 = fig.add_subplot(gs[1])
        ax_3.axis("off")

        # 4番目のサブプロットを作成し、x軸とy軸をそれぞれax2とax3と連動
        ax_4 = fig.add_subplot(gs[4], sharey=ax_2)
        # f_ch = self.fft()
        welch_ch = self.welch()
        data_db = librosa.amplitude_to_db(np.abs(welch_ch.data), ref=welch_ch.ref)
        ax_4.plot(data_db, welch_ch.freqs)
        ax_4.grid(True)
        ax_4.set(xlabel="Spectrum level [dB]")

        # サブプロット間の隙間を調整
        fig.subplots_adjust(wspace=0.0001)
        cbar = fig.colorbar(img, ax=ax_4, format="%+2.0f")
        cbar.set_label("dB")
        fig.suptitle(self.label or "Channel Data")

        output = widgets.Output()
        with output:
            plt.show()

        return widgets.VBox([output, self.to_Audio(label=False)])
