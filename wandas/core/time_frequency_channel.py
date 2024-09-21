from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from . import channel
from .base_channel import BaseChannel


class TimeFrequencyChannel(BaseChannel):
    def __init__(
        self,
        data: np.ndarray,
        sampling_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: str,
        center: bool,
        # pad_mode: str,
        label: Optional[str] = None,
        unit: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        TimeFrequencyChannel オブジェクトを初期化します。

        Parameters:
            frequencies (numpy.ndarray): 周波数データ。
            times (numpy.ndarray): 時間データ。
            data (numpy.ndarray): スペクトルデータ（時間周波数成分）。
            label (str, optional): チャンネルのラベル。
            unit (str, optional):    データの単位。
            calibration_value (float, optional): 校正値。
            metadata (dict, optional): メタデータ。
        """
        super().__init__(
            label=label,
            unit=unit,
            calibration_value=1,
            metadata=metadata,
        )
        self.data = data
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        # self.pad_mode = pad_mode

    @classmethod
    def from_channel(
        cls,
        ch: "channel.Channel",
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = "hann",
        center: bool = True,
        # pad_mode: str = "constant",
    ) -> "TimeFrequencyChannel":
        if n_fft is None:
            n_fft = 2048
        if hop_length is None:
            hop_length = n_fft // 2
        if win_length is None:
            win_length = n_fft

        data = librosa.stft(
            ch.data,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            # pad_mode=pad_mode,
        )
        return cls(
            data=data,
            sampling_rate=ch.sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            # pad_mode=pad_mode,
            label=ch.label,
            unit=ch.unit,
            metadata=ch.metadata.copy(),
        )

    def plot(
        self,
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        db_scale: bool = True,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
    ):
        """
        時間周波数データをプロットします。

        Parameters:
            ax (matplotlib.axes.Axes, optional): 既存のプロット軸。
            title (str, optional): プロットのタイトル。
            db_scale (bool): dBスケールでプロットするかどうか。
        """
        _ax = ax
        if _ax is None:
            _, _ax = plt.subplots(figsize=(10, 6))

        # dBスケールでのプロットを選択可能
        data_to_plot = np.abs(self.data)

        if db_scale:
            data_to_plot = self._to_db()

        # 時間周波数データをプロット
        img = librosa.display.specshow(
            data=data_to_plot,
            sr=self.sampling_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            win_length=self.win_length,
            x_axis="time",
            y_axis="linear",
            ax=ax,
            fmin=fmin,
            fmax=fmax,
            cmap="magma",
        )

        # ラベルとタイトルを設定
        _ax.set_xlabel("Time [s]")
        _ax.set_ylabel("Frequency [Hz]")
        _ax.set_title(title or self.label or "Time-Frequency Representation")
        if db_scale:
            _ax.figure.colorbar(img, ax=ax, format="%+2.0f dB")
        else:
            _ax.figure.colorbar(img, ax=ax)

        if ax is None:
            plt.tight_layout()
            plt.show()

    def _to_db(self, ref=None) -> np.ndarray:
        """
        スペクトルデータを dB スケールに変換した新しい TimeFrequencyChannel を返す。

        Returns:
            TimeFrequencyChannel: dBスケールに変換された新しい TimeFrequencyChannel。
        """
        if ref is None:
            ref = np.max
        return librosa.amplitude_to_db(np.abs(self.data), ref=ref)
