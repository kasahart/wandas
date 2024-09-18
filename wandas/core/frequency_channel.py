# wandas/core/frequency_channel.py

from typing import Optional, Dict, Any
import numpy as np
from .base_channel import BaseChannel
import matplotlib.pyplot as plt
import librosa


class FrequencyChannel(BaseChannel):
    def __init__(
        self,
        frequencies: np.ndarray,
        data: np.ndarray,
        label: Optional[str] = None,
        unit: Optional[str] = None,
        calibration_value: Optional[float] = None,
        fft_params: Optional[Dict[str, Any]] = None,
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
            calibration_value=calibration_value,
            metadata=metadata,
        )
        self.frequencies = frequencies
        self._data = data
        self.fft_params = fft_params or {}

    @property
    def data(self) -> np.ndarray:
        """
        校正値を適用した振幅データを返します。
        """
        if self.calibration_value is not None:
            return self._data * self.calibration_value
        return self._data

    def plot(self, ax: Optional[Any] = None, title: Optional[str] = None):
        """
        スペクトルデータをプロットします。
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(
            self.frequencies,
            librosa.amplitude_to_db(self.data),
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
