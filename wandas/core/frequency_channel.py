# wandas/core/frequency_channel.py

from typing import Optional, Dict, Any
import numpy as np
from .base_channel import BaseChannel
import matplotlib.pyplot as plt
import librosa
from . import channel
from scipy import signal as ss
from scipy import fft


class FrequencyChannel(BaseChannel):
    def __init__(
        self,
        data: np.ndarray,
        sampling_rate: int,
        n_fft: int,
        window: np.ndarray,
        norm: str = "forward",
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
        self.norm = norm

    @classmethod
    def from_channel(
        cls,
        ch: "channel.Channel",
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

        if n_fft is None:
            n_fft = len(ch)

        if n_fft < len(ch):
            raise ValueError(
                "n_fft must be greater than or equal to the length of the input data."
            )

        data = ch._data

        if window:
            window_values = ss.get_window(window, len(ch))
        else:
            window_values = np.ones(len(ch))

        data *= window_values
        norm = "forward"
        out = fft.rfft(data, n=n_fft, norm=norm)
        # type: ignore

        return cls(
            data=out,  # type: ignore
            sampling_rate=ch.sampling_rate,
            n_fft=n_fft,
            window=window_values,
            norm=norm,
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

    def plot(self, ax: Optional[Any] = None, title: Optional[str] = None):
        """
        スペクトルデータをプロットします。
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        x = fft.rfftfreq(self.n_fft, 1 / self.sampling_rate)
        ax.plot(
            x,
            librosa.amplitude_to_db(np.abs(self.data)),
            label=self.label or "Spectrum",
        )

        ax.set_xlabel("Frequency [Hz]")
        ylabel = f"Amplitude [{self.unit}]" if self.unit else "Amplitude [dB]"
        ax.set_ylabel(ylabel)
        ax.set_title(title or self.label or "Spectrum")
        ax.grid(True)
        ax.legend()

        if ax is None:
            plt.tight_layout()
            plt.show()
