# wandas/core/channel.py

from typing import Optional, Dict, Any
import librosa.feature
import numpy as np
from .base_channel import BaseChannel
from scipy.signal import butter, filtfilt
from .frequency_channel import FrequencyChannel, NOctChannel
import librosa
from .time_frequency_channel import TimeFrequencyChannel
import wandas.core.util as util


class Channel(BaseChannel):
    def __init__(
        self,
        data: np.ndarray,
        sampling_rate: int,
        label: Optional[str] = None,
        unit: Optional[str] = None,
        calibration_value: Optional[float] = None,
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
            calibration_value=calibration_value,
            metadata=metadata,
        )

    @util.transform_method()
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

        return dict(data=filtered_data)

    @util.transform_method(FrequencyChannel)
    def fft(
        self,
        n_fft: Optional[int] = None,
        window: Optional[str] = None,
    ):
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
        result["calibration_value"] = 1.0
        return result

    @util.transform_method(FrequencyChannel)
    def welch(
        self,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        win_length: int = 2048,
        window: str = "hann",
        average: str = "mean",
        # pad_mode: str = "constant"
    ):
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
        result["calibration_value"] = 1.0
        return result

    @util.transform_method(NOctChannel)
    def noct_spectrum(
        self,
        n_octaves: int = 3,
        fmin: float = 20,
        fmax: float = 20000,
        G: int = 10,
        fr: int = 1000,
    ):
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
        result["calibration_value"] = 1.0
        return result

    @util.transform_method(TimeFrequencyChannel)
    def stft(
        self,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = "hann",
        center: bool = True,
        # pad_mode: str = "constant",
    ):
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
        result["calibration_value"] = 1.0
        return result

    @util.transform_method()
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

        return dict(
            data=rms_data.squeeze(), sampling_rate=int(self.sampling_rate / hop_length)
        )

    def plot(self, ax: Optional[Any] = None, title: Optional[str] = None):
        """
        時系列データをプロットします。
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        num_samples = len(self._data)
        t = np.arange(num_samples) / self.sampling_rate
        ax.plot(t, self.data, label=self.label or "Channel")

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
    @util.transform_method()
    def __add__(self, other: "Channel"):
        """
        チャンネル間の加算。
        """
        assert (
            self.sampling_rate == other.sampling_rate
        ), "Sampling rates must be the same for channel addition."
        return dict(
            data=self.data + other.data,
            sampling_rate=self.sampling_rate,
            label=f"({self.label} + {other.label})",
            calibration_value=1,
        )

    @util.transform_method()
    def __sub__(self, other: "Channel"):
        """
        チャンネル間の減算。
        """
        assert (
            self.sampling_rate == other.sampling_rate
        ), "Sampling rates must be the same for channel subtraction."
        return dict(
            data=self.data - other.data,
            sampling_rate=self.sampling_rate,
            label=f"({self.label} - {other.label})",
            calibration_value=1,
        )

    @util.transform_method()
    def __mul__(self, other: "Channel"):
        """
        チャンネル間の乗算。
        """
        assert (
            self.sampling_rate == other.sampling_rate
        ), "Sampling rates must be the same for channel multiplication."
        return dict(
            data=self.data * other.data,
            sampling_rate=self.sampling_rate,
            label=f"({self.label} * {other.label})",
            calibration_value=1,
        )

    @util.transform_method()
    def __truediv__(self, other: "Channel"):
        """
        チャンネル間の除算。
        """
        assert (
            self.sampling_rate == other.sampling_rate
        ), "Sampling rates must be the same for channel division."
        return dict(
            data=self.data / other.data,
            sampling_rate=self.sampling_rate,
            label=f"({self.label} / {other.label})",
            calibration_value=1,
        )
