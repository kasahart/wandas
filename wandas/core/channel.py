# wandas/core/channel.py

from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Union

import ipywidgets as widgets
import numpy as np
from IPython.display import Audio, display

from wandas.core import channel_processing, util
from wandas.core.arithmetic import ArithmeticMixin
from wandas.io import wav_io
from wandas.utils.types import NDArrayReal

from .base_channel import BaseChannel
from .channel_plotter import ChannelPlotter
from .frequency_channel import FrequencyChannel, NOctChannel
from .time_frequency_channel import TimeFrequencyChannel, TimeMelFrequencyChannel

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class Channel(BaseChannel, ArithmeticMixin):
    def __init__(
        self,
        data: NDArrayReal,
        sampling_rate: int,
        label: Optional[str] = None,
        unit: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        previous: Optional["Channel"] = None,
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
            previous=previous,
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

        return Channel.from_channel(self, data=data)

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
        return [Channel.from_channel(self, data=d) for d in data]

    def high_pass_filter(self, cutoff: float, order: int = 5) -> "Channel":
        """
        ハイパスフィルタを適用します。

        Parameters:
            cutoff (float): カットオフ周波数（Hz）。
            order (int): フィルタの次数。

        Returns:
            Channel: フィルタリングされた新しい Channel オブジェクト。
        """
        result = channel_processing.apply_filter(
            ch=self,
            cutoff=cutoff,
            order=order,
            filter_type="highpass",
        )
        return Channel.from_channel(self, **result)

    def low_pass_filter(self, cutoff: float, order: int = 5) -> "Channel":
        """
        ローパスフィルタを適用します。

        Parameters:
            cutoff (float): カットオフ周波数（Hz）。
            order (int): フィルタの次数。

        Returns:
            Channel: フィルタリングされた新しい Channel オブジェクト。
        """
        result = channel_processing.apply_filter(
            ch=self,
            cutoff=cutoff,
            order=order,
            filter_type="lowpass",
        )
        return Channel.from_channel(self, **result)

    def hpss_harmonic(
        self,
        kernel_size: Union[int, tuple[int, int], list[int]] = 31,
        power: float = 2.0,
        mask: bool = False,
        margin: Union[float, tuple[float, float], list[float]] = 1.0,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: Union[str, NDArrayReal] = "hann",
        center: bool = True,
        pad_mode: Literal[
            "constant", "edge", "linear_ramp", "reflect", "symmetric", "empty"
        ]
        | Callable[..., Any] = "constant",
    ) -> "Channel":
        """
        HPSS（Harmonic-Percussive Source Separation）のうち、
        Harmonic 成分を取得します。
        """
        result = channel_processing.apply_hpss_harmonic(
            ch=self,
            kernel_size=kernel_size,
            power=power,
            mask=mask,
            margin=margin,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
        )
        return Channel.from_channel(self, **result)

    def hpss_percussive(
        self,
        kernel_size: Union[int, tuple[int, int], list[int]] = 31,
        power: float = 2.0,
        mask: bool = False,
        margin: Union[float, tuple[float, float], list[float]] = 1.0,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: Union[str, NDArrayReal] = "hann",
        center: bool = True,
        pad_mode: Literal[
            "constant", "edge", "linear_ramp", "reflect", "symmetric", "empty"
        ]
        | Callable[..., Any] = "constant",
    ) -> "Channel":
        """
        HPSS（Harmonic-Percussive Source Separation）のうち、
        Percussive 成分を取得します。
        """
        result = channel_processing.apply_hpss_percussive(
            ch=self,
            kernel_size=kernel_size,
            power=power,
            mask=mask,
            margin=margin,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
        )
        return Channel.from_channel(self, **result)

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
        result = channel_processing.compute_fft(
            ch=self,
            n_fft=n_fft,
            window=window,
        )

        return FrequencyChannel.from_channel(self, **result)

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
        result = channel_processing.compute_welch(
            ch=self,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            average=average,
        )
        return FrequencyChannel.from_channel(self, **result)

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

        result = channel_processing.compute_octave(
            ch=self,
            n_octaves=n_octaves,
            fmin=fmin,
            fmax=fmax,
            G=G,
            fr=fr,
        )
        return NOctChannel.from_channel(self, **result)

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

        result = channel_processing.compute_stft(
            ch=self,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            # pad_mode=pad_mode
        )
        return TimeFrequencyChannel.from_channel(self, **result)

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
        result = channel_processing.compute_rms_trend(
            ch=self,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        return Channel.from_channel(self, **result)

    def plot(
        self,
        ax: Optional["Axes"] = None,
        title: Optional[str] = None,
        plot_kwargs: Optional[dict[str, Any]] = None,
    ) -> "Axes":
        """
        時系列データをプロットします。
        """
        plotter = ChannelPlotter(self)

        return plotter.plot_time(ax=ax, title=title, plot_kwargs=plot_kwargs)

    def rms_plot(
        self,
        ax: Optional[Any] = None,
        title: Optional[str] = None,
        plot_kwargs: Optional[dict[str, Any]] = None,
    ) -> "Axes":
        """
        RMS データをプロットします。
        """
        plotter = ChannelPlotter(self)

        return plotter.rms_plot(ax=ax, title=title, plot_kwargs=plot_kwargs)

    def __len__(self) -> int:
        """
        チャンネルのデータ長を返します。
        """
        return int(self._data.shape[-1])

    def add(
        self, other: Union["Channel", NDArrayReal], snr: Optional[float] = None
    ) -> "Channel":
        """_summary_

        Args:
            other (Channel): _description_
            snr (float): _description_

        Returns:
            Channel: _description_
        """
        if isinstance(other, np.ndarray):
            other = Channel.from_channel(self, data=other, label="ndarray")

        if snr is None:
            return self + other

        return channel_processing.apply_add(self, other, snr)

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
        plotter = ChannelPlotter(self)
        return plotter.describe(axis_config=axis_config, cbar_config=cbar_config)
