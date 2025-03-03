from typing import TYPE_CHECKING, Any, Optional

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft
from scipy import signal as ss

from wandas.core import util
from wandas.utils.types import NDArrayComplex, NDArrayReal

from .base_channel import BaseChannel

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh


class TimeFrequencyChannel(BaseChannel):
    def __init__(
        self,
        data: NDArrayReal,
        sampling_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: str,
        # center: bool = None,
        # pad_mode: str,
        label: Optional[str] = None,
        unit: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """
        TimeFrequencyChannel オブジェクトを初期化します。

        Parameters:
            frequencies (numpy.ndarray): 周波数データ。
            times (numpy.ndarray): 時間データ。
            data (numpy.ndarray): スペクトルデータ（時間周波数成分）。
            label (str, optional): チャンネルのラベル。
            unit (str, optional):    データの単位。
            metadata (dict, optional): メタデータ。
        """
        super().__init__(
            data=data,
            sampling_rate=sampling_rate,
            label=label,
            unit=unit,
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
        data: NDArrayReal,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = "hann",
        # pad_mode: str = "constant",
    ) -> dict[str, Any]:
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

        spec_data: NDArrayComplex
        _, _, spec_data = ss.stft(
            data,
            nfft=n_fft,
            noverlap=win_length - hop_length,
            nperseg=win_length,
            window=window,
            detrend="constant",  # type: ignore[unused-ignore]
            # pad_mode=pad_mode,
        )
        spec_data[..., 1:-1, :] *= 2.0
        return dict(
            data=spec_data,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
        )

    def melspectrogram(
        self,
        n_mels: int = 128,
        # pad_mode: str = "constant",
    ) -> "TimeMelFrequencyChannel":
        result = TimeMelFrequencyChannel.spec2melspec(
            sampling_rate=self.sampling_rate,
            data=np.abs(self.data),
            n_fft=self.n_fft,
            n_mels=n_mels,
        )
        result.update(
            dict(
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
            )
        )
        return util.transform_channel(self, TimeMelFrequencyChannel, **result)

    @property
    def data(self) -> NDArrayReal:
        """
        校正値を適用した振幅データを返します。
        """
        return self._data

    def data_Aw(self, to_dB: bool = False) -> NDArrayReal:  # noqa: N802, N803
        """
        A特性を適用した振幅データを返します。
        """
        freqs = fft.rfftfreq(self.n_fft, 1 / self.sampling_rate)
        weighted: NDArrayReal = librosa.perceptual_weighting(
            np.abs(self._data) ** 2, freqs, kind="A", ref=self.ref**2
        )

        if to_dB:
            return weighted.astype(np.float64)

        return np.asarray(librosa.db_to_amplitude(weighted), dtype=np.float64)

    def _plot(
        self,
        ax: "Axes",
        title: Optional[str] = None,
        db_scale: bool = True,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        Aw: bool = False,  # noqa: N803
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> tuple["QuadMesh", NDArrayReal]:
        """
        時間周波数データをプロットします。

        Parameters:
            ax (matplotlib.axes.Axes, optional): 既存のプロット軸。
            title (str, optional): プロットのタイトル。
            db_scale (bool): dBスケールでプロットするかどうか。
        """

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
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_title(title or self.label or "Time-Frequency Representation")

        return img, data_to_plot

    def plot(
        self,
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        db_scale: bool = True,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        Aw: bool = False,  # noqa: N803
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> tuple["Axes", NDArrayReal]:
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

        img, data_to_plot = self._plot(
            ax=_ax,
            title=title,
            db_scale=db_scale,
            fmin=fmin,
            fmax=fmax,
            Aw=Aw,
            vmin=vmin,
            vmax=vmax,
        )

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

    def _to_db(self) -> NDArrayReal:
        """
        スペクトルデータを dB スケールに変換した新しい TimeFrequencyChannel を返す。

        Returns:
            TimeFrequencyChannel: dBスケールに変換された新しい TimeFrequencyChannel。
        """
        return np.asarray(
            librosa.amplitude_to_db(np.abs(self.data), ref=self.ref), np.float64
        )


class TimeMelFrequencyChannel(TimeFrequencyChannel):
    def __init__(
        self,
        data: NDArrayReal,
        sampling_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: str,
        # center: bool = None,
        # pad_mode: str,
        n_mels: int = 128,
        label: Optional[str] = None,
        unit: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """
        TimeMelFrequencyChannel オブジェクトを初期化します。

        Parameters:
            frequencies (numpy.ndarray): 周波数データ。
            times (numpy.ndarray): 時間データ。
            data (numpy.ndarray): スペクトルデータ（時間周波数成分）。
            label (str, optional): チャンネルのラベル。
            unit (str, optional):    データの単位。
            metadata (dict, optional): メタデータ。

        """
        super().__init__(
            data=data,
            sampling_rate=sampling_rate,
            label=label,
            unit=unit,
            metadata=metadata,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
        )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        # self.center = center
        # self.pad_mode = pad_mode

    @classmethod
    def spec2melspec(
        cls,
        sampling_rate: int,
        data: NDArrayReal,
        n_fft: int = 2048,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        n_mels: int = 128,
        # pad_mode: str = "constant",
    ) -> dict[str, Any]:
        melfb = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            norm=None,
        )

        melspec: NDArrayReal = np.einsum("...ft,mf->...mt", data, melfb, optimize=True)  # type: ignore[arg-type, unused-ignore]

        return dict(
            sampling_rate=sampling_rate,
            data=melspec,
            n_fft=n_fft,
            n_mels=n_mels,
        )

    def plot(
        self,
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        db_scale: bool = True,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        Aw: bool = False,  # noqa: N803
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> tuple["Axes", NDArrayReal]:
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
            y_axis="mel",
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
