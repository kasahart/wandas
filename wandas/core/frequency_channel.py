# wandas/core/frequency_channel.py

from typing import Optional, Dict, Any, TYPE_CHECKING, Union
import numpy as np
from .base_channel import BaseChannel
import matplotlib.pyplot as plt
import librosa
from . import channel
from scipy import signal as ss
from scipy import fft

if TYPE_CHECKING:
    from .channel import Channel


class FrequencyChannel(BaseChannel):
    def __init__(
        self,
        data: np.ndarray,
        sampling_rate: int,
        n_fft: int,
        window: Union[np.ndarray, str],
        label: Optional[str] = None,
        unit: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        FrequencyChannel オブジェクトを初期化します。

        Parameters:
            frequencies (numpy.ndarray): 周波数データ。
            data (numpy.ndarray): 振幅データ。
            fft_params (dict, optional): FFT パラメータ。
            その他のパラメータは BaseChannel を参照。
        """
        super().__init__(
            label=label,
            unit=unit,
            calibration_value=1,
            metadata=metadata,
        )
        self._data = data
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.window = window

    @classmethod
    def _fft(
        cls,
        data: np.ndarray,
        n_fft: Optional[int] = None,
        window: Optional[str] = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        length = data.shape[-1]
        if n_fft is None:
            n_fft = length

        if n_fft < length:
            raise ValueError(
                "n_fft must be greater than or equal to the length of the input data."
            )

        if window:
            window_values = ss.get_window(window, length)
        else:
            window_values = np.ones(length)

        data = data * window_values

        out = fft.rfft(data, n=n_fft, norm=None)
        out[1:-1] *= 2.0
        # 窓関数補正
        scaling_factor = np.sum(window_values)
        out /= scaling_factor

        return (out, window_values, n_fft)

    @classmethod
    def from_channel(
        cls,
        ch: "Channel",
        n_fft: Optional[int] = None,
        window: Optional[str] = None,
    ) -> "FrequencyChannel":
        """
        Channel オブジェクトから FrequencyChannel オブジェクトを作成します。

        Parameters:
            ch (Channel): Channel オブジェクト。
            n_fft (int, optional): FFT サイズ。
            window (str, optional): 窓関数。
            fft_params (dict, optional): FFT パラメータ。

        Returns:
            FrequencyChannel: FrequencyChannel オブジェクト。
        """
        out, window_values, n_fft = cls._fft(ch.data, n_fft=n_fft, window=window)

        return cls(
            data=out.squeeze(),  # type: ignore
            sampling_rate=ch.sampling_rate,
            n_fft=n_fft,  # type: ignore
            window=window_values,
            label=ch.label,
            unit=ch.unit,
            metadata=ch.metadata.copy(),
        )

    @classmethod
    def _welch(
        cls,
        data: np.ndarray,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = 2048,
        window: str = "hann",
        average: str = "mean",
        detrend: str = "constant",
    ) -> np.ndarray:
        if win_length is None:
            win_length = 2048
        if n_fft is None:
            n_fft = win_length
        if hop_length is None:
            hop_length = win_length // 2

        _, out = ss.welch(
            data,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            window=window,
            average=average,
            detrend=detrend,
            scaling="spectrum",
        )

        # out[..., 1:-1] *= 2.0

        return out

    @classmethod
    def from_channel_to_welch(
        cls,
        ch: "Channel",
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = "hann",
        average: str = "mean",
    ) -> "FrequencyChannel":
        """
        Channel オブジェクトから FrequencyChannel オブジェクトを作成します。

        Parameters:
            ch (Channel): Channel オブジェクト。
            n_fft (int, optional): FFT サイズ。
            window (str, optional): 窓関数。
            fft_params (dict, optional): FFT パラメータ。

        Returns:
            FrequencyChannel: FrequencyChannel オブジェクト。
        """
        if win_length is None:
            win_length = 2048
        if n_fft is None:
            n_fft = win_length
        if hop_length is None:
            hop_length = win_length // 2

        out = cls._welch(
            ch.data,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            average=average,
            window=window,
        )

        return cls(
            data=out.squeeze(),  # type: ignore
            sampling_rate=ch.sampling_rate,
            n_fft=n_fft,  # type: ignore
            window=window,
            label=ch.label,
            unit=ch.unit,
            metadata=ch.metadata.copy(),
        )

    @property
    def data(self) -> np.ndarray:
        """
        校正値を適用した振幅データを返します。
        """
        return self._data

    def data_Aw(self, to_dB=False) -> np.ndarray:
        """
        A特性を適用した振幅データを返します。
        """
        freqs = fft.rfftfreq(self.n_fft, 1 / self.sampling_rate)
        weighted = librosa.perceptual_weighting(
            np.abs(self._data[..., None]) ** 2, freqs, kind="A", ref=self.ref**2
        ).squeeze()

        if to_dB:
            return weighted

        return librosa.db_to_amplitude(weighted, ref=self.ref)

    def plot(self, ax: Optional[Any] = None, title: Optional[str] = None, Aw=False):
        """
        スペクトルデータをプロットします。
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        x = fft.rfftfreq(self.n_fft, 1 / self.sampling_rate)

        if Aw:
            unit = "dBA"
            data = self.data_Aw(to_dB=True)
        else:
            unit = "dB"
            data = librosa.amplitude_to_db(np.abs(self.data), ref=self.ref)

        ax.plot(
            x,
            data,
            label=self.label or "Spectrum",
        )

        ax.set_xlabel("Frequency [Hz]")
        ylabel = f"Spectrum level [{unit}]"
        ax.set_ylabel(ylabel)
        ax.set_title(title or self.label or "Spectrum")
        ax.grid(True)
        ax.legend()

        if ax is None:
            plt.tight_layout()
            plt.show()
