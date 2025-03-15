# wandas/core/channel.py

from typing import TYPE_CHECKING, Any, Optional, Union

import ipywidgets as widgets
import librosa
import librosa.feature
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio, display
from matplotlib import gridspec
from scipy.signal import butter, filtfilt

from wandas.core import util
from wandas.io import wav_io
from wandas.utils.types import NDArrayReal

from .base_channel import BaseChannel
from .frequency_channel import FrequencyChannel, NOctChannel
from .time_frequency_channel import TimeFrequencyChannel, TimeMelFrequencyChannel

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class Channel(BaseChannel):
    def __init__(
        self,
        data: NDArrayReal,
        sampling_rate: int,
        label: Optional[str] = None,
        unit: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
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
    def time(self) -> NDArrayReal:
        """
        時刻データを返します。
        """
        num_samples = len(self._data)
        return np.arange(num_samples) / self.sampling_rate

    def trim(self, start: float, end: float) -> "Channel":
        """
        指定された範囲のデータを抽出します。

        Parameters:
            start (float): 抽出開始時刻（秒）。
            end (float): 抽出終了時刻（秒）。

        Returns:
            Channel: 抽出されたデータを含む新しい Channel オブジェクト。
        """
        start_idx = int(start * self.sampling_rate)
        end_idx = int(end * self.sampling_rate)
        data = self.data[start_idx:end_idx]

        result = dict(
            data=data,
        )

        return util.transform_channel(self, self.__class__, **result)

    def trigger(
        self,
        threshold: float,
        offset: int = 0,
        hold: int = 1,
        trigger_type: str = "level",
    ) -> list[int]:
        """
        トリガーを検出します。

        Parameters:
            threshold (float): トリガー閾値。
            offset (int): トリガー検出位置のオフセット。
            hold (int): トリガーホールド。
            trigger_type (str): トリガ
                - "level": レベルトリガー
        Returns:
            list[int]: トリガー位置のリスト。
        """
        if trigger_type == "level":
            return util.level_trigger(self.data, threshold, offset=offset, hold=hold)
        else:
            raise ValueError(f"Unsupported trigger type: {trigger_type}")

    def cut(
        self,
        point_list: Union[list[int], list[float]],
        cut_len: Union[int, float],
        taper_rate: float = 0,
        dc_cut: bool = False,
    ) -> list["Channel"]:
        """
        チャンネルデータをカットします。

        Parameters:
            point_list (list[int]): カットポイントのリスト。
            cut_len (int): カットするデータ長。
            taper_rate (float): テーパー率。
            dc_cut (bool): DC カット。

        Returns:
            Channel: カットされた新しい Channel オブジェクト。
        """
        # point_list がfloatの場合、サンプリングレートを考慮して整数に変換
        _point_list: list[int] = [
            int(p * self.sampling_rate) if isinstance(p, float) else p
            for p in point_list
        ]
        # cut_len がfloatの場合、サンプリングレートを考慮して整数に変換
        _cut_len = (
            int(cut_len * self.sampling_rate) if isinstance(cut_len, float) else cut_len
        )
        data = util.cut_sig(self.data, _point_list, _cut_len, taper_rate, dc_cut)
        return [util.transform_channel(self, self.__class__, data=d) for d in data]

    def high_pass_filter(self, cutoff: float, order: int = 5) -> "Channel":
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
        b, a = butter(order, normal_cutoff, btype="highpass", analog=False)  # type: ignore[unused-ignore]
        filtered_data = filtfilt(b, a, self.data)

        result = dict(
            data=filtered_data.squeeze(),
        )

        return util.transform_channel(self, self.__class__, **result)

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
        b, a = butter(order, normal_cutoff, btype="lowpass", analog=False)  # type: ignore[unused-ignore]
        filtered_data = filtfilt(b, a, self.data)

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
        G: int = 10,  # noqa: N803
        fr: int = 1000,
    ) -> "NOctChannel":
        """
        オクターブバンドのスペクトルを計算します。

        Parameters:
            n_octaves (int): オクターブの数。

        Returns:
            FrequencyChannel: オクターブバンドのスペクトルデータを含むオブジェクト。
        """

        result = NOctChannel.noct_spectrum(
            data=self.data,
            sampling_rate=self.sampling_rate,
            fmin=fmin,
            fmax=fmax,
            n=n_octaves,
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
        result = dict(
            data=rms_data.squeeze(),
            sampling_rate=int(self.sampling_rate / hop_length),
        )

        return util.transform_channel(self, self.__class__, **result)

    def plot(
        self,
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        plot_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple["Axes", NDArrayReal]:
        """
        時系列データをプロットします。
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        plot_kwargs = plot_kwargs or {}
        ax.plot(self.time, self.data, label=self.label or "Channel", **plot_kwargs)

        ax.set_xlabel("Time [s]")
        ylabel = f"Amplitude [{self.unit}]" if self.unit else "Amplitude"
        ax.set_ylabel(ylabel)
        ax.set_title(title or self.label or "Channel Data")
        ax.grid(True)
        ax.legend()

        if ax is None:
            plt.tight_layout()
            plt.show()

        return ax, self.data

    def rms_plot(
        self, ax: Optional[Any] = None, title: Optional[str] = None
    ) -> "Channel":
        """
        RMS データをプロットします。
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        rms_channel: Channel = self.rms_trend()
        num_samples = len(rms_channel)
        t = np.arange(num_samples) / rms_channel.sampling_rate
        ax.plot(
            t,
            util.amplitude_to_db(rms_channel.data, ref=self.ref),
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
        return int(self._data.shape[-1])

    def add(self, other: "Channel", snr: Optional[float] = None) -> "Channel":
        """_summary_

        Args:
            other (Channel): _description_
            snr (float): _description_

        Returns:
            Channel: _description_
        """
        if snr is None:
            return self + other

        clean_rms = util.calculate_rms(self.data)
        other_rms = util.calculate_rms(other.data)
        desired_noise_rms = util.calculate_desired_noise_rms(clean_rms, snr)
        gain = desired_noise_rms / other_rms

        return self + other * gain

    # 演算子オーバーロードの実装
    def __add__(self, other: Union["Channel", int, float, NDArrayReal]) -> "Channel":
        """
        チャンネル間の加算。
        """
        if isinstance(other, Channel):
            assert self.sampling_rate == other.sampling_rate, (
                "Sampling rates must be the same for channel addition."
            )
            data = self.data + other.data
            label = f"({self.label} + {other.label})"
        elif isinstance(other, (int, float, np.ndarray)):
            data = self.data + other
            label = f"({self.label} + {other})"
        else:
            raise TypeError("Unsupported type for addition with Channel")

        result = dict(
            data=data,
            sampling_rate=self.sampling_rate,
            label=label,
        )
        return util.transform_channel(self, self.__class__, **result)

    def __sub__(self, other: Union["Channel", int, float, NDArrayReal]) -> "Channel":
        """
        チャンネル間の減算。
        """
        if isinstance(other, Channel):
            assert self.sampling_rate == other.sampling_rate, (
                "Sampling rates must be the same for channel subtraction."
            )
            data = self.data - other.data
            label = f"({self.label} - {other.label})"
        elif isinstance(other, (int, float, np.ndarray)):
            data = self.data - other
            label = f"({self.label} - {other})"
        else:
            raise TypeError("Unsupported type for subtraction with Channel")

        result = dict(
            data=data,
            sampling_rate=self.sampling_rate,
            label=label,
        )
        return util.transform_channel(self, self.__class__, **result)

    def __mul__(self, other: Union["Channel", int, float, NDArrayReal]) -> "Channel":
        """
        チャンネル間の乗算。
        """
        if isinstance(other, Channel):
            assert self.sampling_rate == other.sampling_rate, (
                "Sampling rates must be the same for channel multiplication."
            )
            data = self.data * other.data
            label = f"({self.label} * {other.label})"
        elif isinstance(other, (int, float, np.ndarray)):
            data = self.data * other
            label = f"({self.label} * {other})"
        else:
            raise TypeError("Unsupported type for multiplication with Channel")

        result = dict(
            data=data,
            sampling_rate=self.sampling_rate,
            label=label,
        )
        return util.transform_channel(self, self.__class__, **result)

    def __truediv__(
        self, other: Union["Channel", int, float, NDArrayReal]
    ) -> "Channel":
        """
        チャンネル間の除算。
        """
        if isinstance(other, Channel):
            assert self.sampling_rate == other.sampling_rate, (
                "Sampling rates must be the same for channel division."
            )
            data = self.data / other.data
            label = f"({self.label} / {other.label})"
        elif isinstance(other, (int, float, np.ndarray)):
            data = self.data / other
            label = f"({self.label} / {other})"
        else:
            raise TypeError("Unsupported type for division with Channel")

        result = dict(
            data=data,
            sampling_rate=self.sampling_rate,
            label=label,
        )
        return util.transform_channel(self, self.__class__, **result)

    def to_wav(self, filename: str) -> None:
        """
        Channel オブジェクトを WAV ファイルに書き出します。

        Parameters:
            filename (str): 出力する WAV ファイルのパス。
        """
        wav_io.write_wav(filename, self)

    def to_audio(self, normalize: bool = True, label: bool = True) -> widgets.VBox:
        output = widgets.Output()
        with output:
            display(Audio(self.data, rate=self.sampling_rate, normalize=normalize))  # type: ignore [unused-ignore, no-untyped-call]

        if label:
            vbov = widgets.VBox([widgets.Label(self.label) if label else None, output])
        else:
            vbov = widgets.VBox([output])
        return vbov

    def describe(
        self,
        axis_config: Optional[dict[str, dict[str, Any]]] = None,
        cbar_config: Optional[dict[str, Any]] = None,
    ) -> widgets.VBox:
        """
        チャンネルの統計情報を表示します。軸設定およびカラーバー設定を受け付けます。

        Parameters:
            axis_config (dict): 各サブプロットの軸設定を格納する辞書。
                {
                    "time_plot": {"xlim": (0, 1)},
                    "freq_plot": {"ylim": (0, 20000)}
                }
            cbar_config (dict): カラーバーの設定を格納する辞書
                例: {"vmin": -80, "vmax": 0}
        """
        axis_config = axis_config or {}
        cbar_config = cbar_config or {}

        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 3], width_ratios=[3, 1, 0.1])
        gs.update(wspace=0.2)

        fig = plt.figure(figsize=(12, 6))

        # 最初のサブプロット (Time Plot)
        ax_1 = fig.add_subplot(gs[0])
        self.plot(ax=ax_1)
        if "time_plot" in axis_config:
            conf = axis_config["time_plot"]
            ax_1.set(**conf)
        ax_1.legend().set_visible(False)
        ax_1.set(xlabel="", title="")

        # 2番目のサブプロット (STFT Plot)
        ax_2 = fig.add_subplot(gs[3], sharex=ax_1)
        stft_ch = self.stft()
        # Pass vmin and vmax from cbar_config to stft_ch._plot
        img, _ = stft_ch._plot(
            ax=ax_2, vmin=cbar_config.get("vmin"), vmax=cbar_config.get("vmax")
        )
        ax_2.set(title="")

        # 3番目のサブプロット
        ax_3 = fig.add_subplot(gs[1])
        ax_3.axis("off")

        # 4番目のサブプロット (Welch Plot)
        ax_4 = fig.add_subplot(gs[4], sharey=ax_2)
        welch_ch = self.welch()
        data_db = util.amplitude_to_db(np.abs(welch_ch.data), ref=welch_ch.ref)
        ax_4.plot(data_db, welch_ch.freqs)
        ax_4.grid(True)
        ax_4.set(xlabel="Spectrum level [dB]")
        if "freq_plot" in axis_config:
            conf = axis_config["freq_plot"]
            ax_4.set(**conf)

        fig.subplots_adjust(wspace=0.0001)
        cbar = fig.colorbar(img, ax=ax_4, format="%+2.0f")
        cbar.set_label("dB")
        fig.suptitle(self.label or "Channel Data")

        output = widgets.Output()
        with output:
            plt.show()

        container = widgets.VBox([output, self.to_audio(label=False)])
        # container.add_class("white-bg")
        return container
