# wandas/core/frequency_channel.py

from typing import TYPE_CHECKING, Any, Optional, Union

import librosa
import matplotlib.pyplot as plt
import numpy as np
from mosqito.sound_level_meter import noct_spectrum, noct_synthesis
from scipy import fft
from scipy import signal as ss

from wandas.core import util
from wandas.utils.types import NDArrayReal

from .base_channel import BaseChannel

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class NOctChannel(BaseChannel):
    def __init__(
        self,
        data: NDArrayReal,
        sampling_rate: int,
        fpref: NDArrayReal,
        n: int = 3,
        G: int = 10,  # noqa: N803
        fr: int = 1000,
        label: Optional[str] = None,
        unit: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """
        NOctChannel オブジェクトを初期化します。


        """
        super().__init__(
            data=data,
            sampling_rate=sampling_rate,
            label=label,
            unit=unit,
            metadata=metadata,
        )

        self.n = n
        self.G = G
        self.fr = fr
        self.fpref = fpref

    @classmethod
    def noct_spectrum(
        cls,
        data: NDArrayReal,
        sampling_rate: int,
        fmin: float,
        fmax: float,
        n: int = 3,
        G: int = 10,  # noqa: N803
        fr: int = 1000,
    ) -> dict[str, Any]:
        """
        N-Octave Spectrum を計算します。
        """

        spec, fpref = noct_spectrum(
            sig=data, fs=sampling_rate, fmin=fmin, fmax=fmax, n=n, G=G, fr=fr
        )

        return dict(data=np.squeeze(spec), fpref=fpref, n=n, G=G, fr=fr)

    @classmethod
    def noct_synthesis(
        cls,
        data: NDArrayReal,
        freqs: NDArrayReal,
        fmin: float,
        fmax: float,
        n: int = 3,
        G: int = 10,  # noqa: N803
        fr: int = 1000,
    ) -> dict[str, Any]:
        """
        N-Octave Spectrum を計算します。
        """

        fs = freqs.max() * 2
        if round(fs) == 48000:
            spec, fpref = noct_synthesis(
                spectrum=data, freqs=freqs, fmin=fmin, fmax=fmax, n=n, G=G, fr=fr
            )
        # elif n == 3:
        #     if data.ndim == 1:
        #         data = data[..., None]
        #     spec, fpref = freq_band_synthesis(
        #         spectrum=np.abs(data),
        #         freqs=freqs,
        #         fmin=np.array([fmin]),
        #         fmax=np.array([fmax]),
        #     )
        #     spec = np.squeeze(spec)

        else:
            raise ValueError("fs must be 48000")
        return dict(data=spec, fpref=fpref, n=n, G=G, fr=fr)

    def data_Aw(self, to_dB: bool = False) -> NDArrayReal:  # noqa: N802, N803
        """
        A特性を適用した振幅データを返します。
        """

        weighted = librosa.perceptual_weighting(
            self._data[..., None] ** 2, self.fpref, kind="A", ref=self.ref**2
        ).squeeze()

        if to_dB:
            return weighted

        return np.asarray(librosa.db_to_amplitude(weighted, ref=self.ref))

    def plot(
        self,
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        Aw: bool = False,  # noqa: N803
    ) -> tuple["Axes", NDArrayReal]:
        """
        スペクトルデータをプロットします。
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        if Aw:
            unit = "dBrA"
            data = self.data_Aw(to_dB=True)
        else:
            unit = "dBr"
            data = librosa.amplitude_to_db(np.abs(self.data), ref=self.ref)

        ax.step(
            self.fpref,
            data,
            label=self.label or "Spectrum",
        )

        ax.set_xlabel("Center frequency [Hz]")
        ylabel = f"Spectrum level [{unit}]"
        ax.set_ylabel(ylabel)
        ax.set_title(title or self.label or f"1/{str(self.n)}-Octave Spectrum")
        ax.grid(True)
        ax.legend()

        if ax is None:
            plt.tight_layout()
            plt.show()

        return ax, data


class FrequencyChannel(BaseChannel):
    def __init__(
        self,
        data: NDArrayReal,
        sampling_rate: int,
        n_fft: int,
        window: Union[NDArrayReal, str],
        label: Optional[str] = None,
        unit: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
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
            data=data,
            sampling_rate=sampling_rate,
            label=label,
            unit=unit,
            metadata=metadata,
        )

        self.n_fft = n_fft
        self.window = window

    @classmethod
    def fft(
        cls,
        data: NDArrayReal,
        n_fft: Optional[int] = None,
        window: Optional[str] = None,
    ) -> dict[str, Any]:
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

        out = np.asarray(fft.rfft(data, n=n_fft))
        out[1:-1] *= 2.0
        # 窓関数補正
        scaling_factor = np.sum(window_values)
        out /= scaling_factor

        return dict(data=out.squeeze(), window=window_values, n_fft=n_fft)

    @classmethod
    def welch(
        cls,
        data: NDArrayReal,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = 2048,
        window: str = "hann",
        average: str = "mean",
        detrend: str = "constant",
    ) -> dict[str, Any]:
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
            nfft=n_fft,
            window=window,
            average=average,
            detrend=detrend,
            scaling="spectrum",
        )

        return dict(
            data=out,
            n_fft=n_fft,
            # hop_length=hop_length,
            # win_length=win_length,
            window=window,
            # average=average,
            # detrend=detrend,
        )

    @property
    def freqs(self) -> NDArrayReal:
        """
        フーリエ変換後の周波数データを返します。
        """
        return np.asarray(fft.rfftfreq(self.n_fft, 1 / self.sampling_rate))

    def noct_synthesis(
        self,
        fmin: float,
        fmax: float,
        n: int = 3,
        G: int = 10,  # noqa: N803
        fr: int = 1000,
    ) -> NOctChannel:
        """
        N-Octave Spectrum を計算します。
        """
        result = NOctChannel.noct_synthesis(
            data=self.data / np.sqrt(2),
            freqs=self.freqs,
            fmin=fmin,
            fmax=fmax,
            n=n,
            G=G,
            fr=fr,
        )

        return util.transform_channel(self, NOctChannel, **result)

    def data_Aw(self, to_dB: bool = False) -> NDArrayReal:  # noqa: N802, N803
        """
        A特性を適用した振幅データを返します。
        """
        freqs = self.freqs
        weighted = librosa.perceptual_weighting(
            np.abs(self._data[..., None]) ** 2, freqs, kind="A", ref=self.ref**2
        ).squeeze()

        if to_dB:
            return weighted

        return np.asarray(librosa.db_to_amplitude(weighted, ref=self.ref))

    def plot(
        self,
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        Aw: bool = False,  # noqa: N803
    ) -> tuple["Axes", NDArrayReal]:
        """
        スペクトルデータをプロットします。
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        if Aw:
            unit = "dBA"
            data = self.data_Aw(to_dB=True)
        else:
            unit = "dB"
            data = librosa.amplitude_to_db(np.abs(self.data), ref=self.ref)

        ax.plot(
            self.freqs,
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

        return ax, data
