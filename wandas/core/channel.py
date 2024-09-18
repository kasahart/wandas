# wandas/core/channel.py

from typing import Optional, Dict, Any
import librosa.feature
import numpy as np
from .base_channel import BaseChannel
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq
from .frequency_channel import FrequencyChannel
import librosa


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
            label=label,
            unit=unit,
            calibration_value=calibration_value,
            metadata=metadata,
        )
        self._data = data
        self.sampling_rate = sampling_rate

    @property
    def data(self) -> np.ndarray:
        """
        校正値を適用したデータを返します。
        """
        if self.calibration_value is not None:
            return self._data * self.calibration_value
        return self._data

    def low_pass_filter(self, cutoff: float, order: int = 5) -> "Channel":
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
        return Channel(
            data=filtered_data,
            sampling_rate=self.sampling_rate,
            label=self.label,
            unit=self.unit,
            calibration_value=self.calibration_value,
            metadata=self.metadata.copy(),
        )

    def fft(
        self,
        n_fft: Optional[int] = None,
        window: Optional[str] = None,
        fft_params: Optional[Dict[str, Any]] = None,
    ) -> FrequencyChannel:
        """
        フーリエ変換を実行します。

        Parameters:
            n_fft (int, optional): FFT のサンプル数。
            window (str, optional): ウィンドウ関数の種類。
            fft_params (dict, optional): その他の FFT パラメータ。

        Returns:
            FrequencyChannel: スペクトルデータを含むオブジェクト。
        """
        from scipy.signal import get_window

        n = n_fft or len(self._data)
        fft_parameters = fft_params.copy() if fft_params else {}
        fft_parameters.update({"n_fft": n, "window": window})

        if window:
            window_values = get_window(window, len(self._data))
            data = self._data * window_values
        else:
            data = self._data

        yf = rfft(data, n=n)
        xf = rfftfreq(n, 1 / self.sampling_rate)
        amplitude = np.abs(yf) * 2 / n

        return FrequencyChannel(
            frequencies=xf,
            data=amplitude,
            label=self.label,
            unit=self.unit,
            calibration_value=self.calibration_value,
            fft_params=fft_parameters,
            metadata=self.metadata.copy(),
        )

    def rms_trend(self, frame_length: int = 2048, hop_length: int = 512) -> "Channel":
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

        return Channel(
            data=rms_data.squeeze(),
            sampling_rate=int(self.sampling_rate / hop_length),
            label=self.label,
            unit=self.unit,
            metadata=self.metadata.copy(),
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

        ax.set_xlabel("Time (s)")
        ylabel = f"Amplitude ({self.unit})" if self.unit else "Amplitude"
        ax.set_ylabel(ylabel)
        ax.set_title(title or self.label or "Channel Data")
        ax.grid(True)
        ax.legend()

        if ax is None:
            plt.tight_layout()
            plt.show()

    def rms_plot(self, ax: Optional[Any] = None, title: Optional[str] = None):
        """
        RMS データをプロットします。
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        rms_channel = self.rms_trend()
        num_samples = len(rms_channel)
        t = np.arange(num_samples) / rms_channel.sampling_rate
        ax.plot(
            t,
            librosa.amplitude_to_db(rms_channel.data),
            label=rms_channel.label or "Channel",
        )

        ax.set_xlabel("Time (s)")
        ylabel = f"RMS ({rms_channel.unit})" if rms_channel.unit else "RMS"
        ax.set_ylabel(ylabel)
        ax.set_title(title or rms_channel.label or "Channel Data")
        ax.grid(True)
        ax.legend()

        if ax is None:
            plt.tight_layout()

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
        return Channel(
            data=self.data + other.data,
            sampling_rate=self.sampling_rate,
            label=f"({self.label} + {other.label})",
        )

    def __sub__(self, other: "Channel") -> "Channel":
        """
        チャンネル間の減算。
        """
        assert (
            self.sampling_rate == other.sampling_rate
        ), "Sampling rates must be the same for channel subtraction."
        return Channel(
            data=self.data - other.data,
            sampling_rate=self.sampling_rate,
            label=f"({self.label} - {other.label})",
        )

    def __mul__(self, other: "Channel") -> "Channel":
        """
        チャンネル間の乗算。
        """
        assert (
            self.sampling_rate == other.sampling_rate
        ), "Sampling rates must be the same for channel multiplication."
        return Channel(
            data=self.data * other.data,
            sampling_rate=self.sampling_rate,
            label=f"({self.label} * {other.label})",
        )

    def __truediv__(self, other: "Channel") -> "Channel":
        """
        チャンネル間の除算。
        """
        assert (
            self.sampling_rate == other.sampling_rate
        ), "Sampling rates must be the same for channel division."
        return Channel(
            data=self.data / other.data,
            sampling_rate=self.sampling_rate,
            label=f"({self.label} / {other.label})",
        )
