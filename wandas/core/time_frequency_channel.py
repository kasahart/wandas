from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from .base_channel import BaseChannel
from scipy import fft
from scipy import signal as ss


class TimeFrequencyChannel(BaseChannel):
    def __init__(
        self,
        data: np.ndarray,
        sampling_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: str,
        # center: bool = None,
        # pad_mode: str,
        label: Optional[str] = None,
        unit: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        calibration_value: Optional[float] = None,
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
            data=data,
            sampling_rate=sampling_rate,
            label=label,
            unit=unit,
            calibration_value=calibration_value,
            metadata=metadata,
        )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        # self.center = center
        # self.pad_mode = pad_mode

    @classmethod
    def stft(
        cls,
        data: np.ndarray,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = "hann",
        # pad_mode: str = "constant",
    ):
        """
        STFT（短時間フーリエ変換）を実行します。

        Parameters:
            data (numpy.ndarray): 入力データ。
            n_fft (int): FFT のサンプル数。
            hop_length (int): ホップサイズ（フレーム間の移動量）。
            win_length (int): ウィンドウの長さ。
            window (str): ウィンドウ関数の種類。
            center (bool): フレームを中央に配置するかどうか。
            pad_mode (str): パディングモード。

        Returns:
            numpy.ndarray: STFT の結果。
        """
        if win_length is None:
            win_length = 2048
        if n_fft is None:
            n_fft = win_length
        if hop_length is None:
            hop_length = win_length // 2

        _, _, data = ss.stft(
            data,
            nfft=n_fft,
            noverlap=win_length - hop_length,
            nperseg=win_length,
            window=window,
            detrend="constant",
            # pad_mode=pad_mode,
        )
        data[..., 1:-1, :] *= 2.0
        return dict(
            data=data,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
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
            np.abs(self._data) ** 2, freqs, kind="A", ref=self.ref**2
        )

        if to_dB:
            return weighted

        return librosa.db_to_amplitude(weighted)

    def plot(
        self,
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        db_scale: bool = True,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        Aw: bool = False,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> tuple[Any, np.ndarray]:
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

        if Aw:
            data_to_plot = self.data_Aw(to_dB=True)
        elif db_scale:
            data_to_plot = self._to_db()
        else:
            data_to_plot = np.abs(self.data)

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
            vmin=vmin,
            vmax=vmax,
        )

        # ラベルとタイトルを設定
        _ax.set_xlabel("Time [s]")
        _ax.set_ylabel("Frequency [Hz]")
        _ax.set_title(title or self.label or "Time-Frequency Representation")

        if _ax.figure is not None:
            if db_scale:
                cbar = _ax.figure.colorbar(img, ax=ax, format="%+2.0f")
                if Aw:
                    unit = "dBA"
                else:
                    unit = "dB"
                cbar.set_label(f"Spectrum level [{unit}]")
            else:
                cbar = _ax.figure.colorbar(img, ax=ax)
                cbar.set_label(f"Amplitude [{self.unit}]")

        if ax is None:
            plt.tight_layout()
            plt.show()

        return _ax, data_to_plot

    def _to_db(self) -> np.ndarray:
        """
        スペクトルデータを dB スケールに変換した新しい TimeFrequencyChannel を返す。

        Returns:
            TimeFrequencyChannel: dBスケールに変換された新しい TimeFrequencyChannel。
        """
        return librosa.amplitude_to_db(np.abs(self.data), ref=self.ref)
