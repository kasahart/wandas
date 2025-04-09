import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union, cast

import dask
import dask.array as da
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from IPython.display import Audio, display

if TYPE_CHECKING:
    from librosa._typing import (
        _FloatLike_co,
        _IntLike_co,
        _PadModeSTFT,
        _WindowSpec,
    )
    from matplotlib.axes import Axes

    from .spectral_frame import SpectralFrame
    from .spectrogram_frame import SpectrogramFrame


from dask.array.core import Array as DaArray

from wandas.utils.types import NDArrayReal

from .base_frame import BaseFrame
from .channel_metadata import ChannelMetadata
from .file_readers import get_file_reader
from .plotting import create_operation

logger = logging.getLogger(__name__)

dask_delayed = dask.delayed  # type: ignore [unused-ignore]
da_from_delayed = da.from_delayed  # type: ignore [unused-ignore]
da_from_array = da.from_array  # type: ignore [unused-ignore]


S = TypeVar("S", bound="BaseFrame[Any]")


class ChannelFrame(BaseFrame[NDArrayReal]):
    """
    音声チャネルのラッパークラス
    データ形状: (channels, samples) または単一チャネルの場合 (1, samples)
    """

    def __init__(
        self,
        data: DaArray,
        sampling_rate: float,
        label: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        operation_history: Optional[list[dict[str, Any]]] = None,
        channel_metadata: Optional[list[ChannelMetadata]] = None,
        previous: Optional["BaseFrame[Any]"] = None,
    ) -> None:
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim > 2:
            raise ValueError(
                f"データは1次元または2次元である必要があります。形状: {data.shape}"
            )
        super().__init__(
            data=data,
            sampling_rate=sampling_rate,
            label=label,
            metadata=metadata,
            operation_history=operation_history,
            channel_metadata=channel_metadata,
            previous=previous,
        )

    @property
    def _n_channels(self) -> int:
        """チャネル数を返します。"""
        return self.shape[-2]

    @property
    def time(self) -> NDArrayReal:
        """
        時刻データを返します。
        """
        return np.arange(self.n_samples) / self.sampling_rate

    @property
    def n_samples(self) -> int:
        n: int = self._data.shape[-1]
        return n

    @property
    def duration(self) -> float:
        return self.n_samples / self.sampling_rate

    def _apply_operation_impl(self: S, operation_name: str, **params: Any) -> S:
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")
        from .time_series_operation import create_operation

        # 操作インスタンスを作成
        operation = create_operation(operation_name, self.sampling_rate, **params)

        # データに処理を適用
        processed_data = operation.process(self._data)

        # メタデータ更新
        operation_metadata = {"operation": operation_name, "params": params}
        new_history = self.operation_history.copy()
        new_history.append(operation_metadata)
        new_metadata = {**self.metadata}
        new_metadata[operation_name] = params

        logger.debug(
            f"Created new ChannelFrame with operation {operation_name} added to graph"
        )
        return self._create_new_instance(
            data=processed_data,
            metadata=new_metadata,
            operation_history=new_history,
        )

    def _binary_op(
        self,
        other: Union["ChannelFrame", int, float, NDArrayReal, "DaArray"],
        op: Callable[["DaArray", Any], "DaArray"],
        symbol: str,
    ) -> "ChannelFrame":
        """
        二項演算の共通実装 - daskの遅延演算を活用

        Parameters
        ----------
        other : ChannelFrame, int, float, ndarray, dask.array
            演算の右オペランド
        op : callable
            演算を実行する関数 (例: lambda a, b: a + b)
        symbol : str
            演算のシンボル表現 (例: '+')

        Returns
        -------
        ChannelFrame
            演算結果を含む新しいチャネル（遅延実行）
        """
        from .channel_frame import ChannelFrame

        logger.debug(f"Setting up {symbol} operation (lazy)")

        # Handle potentially None metadata and operation_history
        metadata = {}
        if self.metadata is not None:
            metadata = self.metadata.copy()

        operation_history = []
        if self.operation_history is not None:
            operation_history = self.operation_history.copy()

        # Check if other is a ChannelFrame - improved type checking
        if isinstance(other, ChannelFrame):
            if self.sampling_rate != other.sampling_rate:
                raise ValueError(
                    "サンプリングレートが一致していません。演算できません。"
                )

            # dask arrayを直接演算（遅延実行を維持）
            result_data = op(self._data, other._data)

            # チャネルメタデータを結合
            merged_channel_metadata = []
            for self_ch, other_ch in zip(
                self._channel_metadata, other._channel_metadata
            ):
                ch = self_ch.copy(deep=True)
                ch["label"] = f"({self_ch['label']} {symbol} {other_ch['label']})"
                merged_channel_metadata.append(ch)

            operation_history.append({"operation": symbol, "with": other.label})

            return ChannelFrame(
                data=result_data,
                sampling_rate=self.sampling_rate,
                label=f"({self.label} {symbol} {other.label})",
                metadata=metadata,
                operation_history=operation_history,
                channel_metadata=merged_channel_metadata,
                previous=self,
            )

        # スカラー、NumPy配列、または他のタイプとの演算
        else:
            # dask arrayに直接演算を適用（遅延実行を維持）
            result_data = op(self._data, other)

            # オペランドの表示用文字列
            if isinstance(other, (int, float)):
                other_str = str(other)
            elif isinstance(other, np.ndarray):
                other_str = f"ndarray{other.shape}"
            elif hasattr(other, "shape"):  # dask.array.Arrayのチェック
                other_str = f"dask.array{other.shape}"
            else:
                other_str = str(type(other).__name__)

            # チャネルメタデータを更新
            updated_channel_metadata: list[ChannelMetadata] = []
            for self_ch in self._channel_metadata:
                ch = self_ch.copy(deep=True)
                ch["label"] = f"({self_ch.label} {symbol} {other_str})"
                updated_channel_metadata.append(ch)

            operation_history.append({"operation": symbol, "with": other_str})

            return ChannelFrame(
                data=result_data,
                sampling_rate=self.sampling_rate,
                label=f"({self.label} {symbol} {other_str})",
                metadata=metadata,
                operation_history=operation_history,
                channel_metadata=updated_channel_metadata,
                previous=self,
            )

    def plot(
        self, plot_type: str = "waveform", ax: Optional["Axes"] = None, **kwargs: Any
    ) -> Union["Axes", Iterator["Axes"]]:
        """
        様々な形式のプロット (Strategyパターンを使用)

        Parameters
        ----------
        plot_type : str
            'waveform', 'spectrogram'などのプロット種類
        ax : matplotlib.axes.Axes, optional
            プロット先の軸。Noneの場合は新しい軸を作成
        **kwargs : dict
            プロット固有のパラメータ
        """

        logger.debug(f"Plotting audio with plot_type={plot_type} (will compute now)")

        # プロット戦略を取得
        plot_strategy = create_operation(plot_type)

        # プロット実行
        _ax = plot_strategy.plot(self, ax=ax, **kwargs)

        logger.debug("Plot rendering complete")

        return _ax

    def describe(self, normalize: bool = True, **kwargs: Any) -> widgets.VBox:
        container = []
        for ch in self:
            output = widgets.Output()
            with output:
                ch.plot("describe", title=f"{ch.label} {ch.labels[0]}", **kwargs)
                plt.show()
            audio_output = widgets.Output()
            with audio_output:
                display(Audio(ch.data, rate=ch.sampling_rate, normalize=normalize))  # type: ignore [unused-ignore, no-untyped-call]
            container.append(widgets.VBox([output, audio_output]))

        return widgets.VBox(container)

    @classmethod
    def from_numpy(
        cls,
        data: NDArrayReal,
        sampling_rate: float,
        label: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        ch_labels: Optional[list[str]] = None,
        ch_units: Optional[list[str]] = None,
    ) -> "ChannelFrame":
        """
        NumPy配列からチャネルフレームを作成します。

        Parameters
        ----------
        data : numpy.ndarray
            音声データ。形状は
            (bach, channels, samples) または (channels, samples) または (samples,)
        sampling_rate : float
            サンプリングレート (Hz)
        label : str, optional
            チャネルのラベル

        Returns
        -------
        ChannelFrame
            データを含む新しいチャネルフレーム
        """

        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim > 2:
            raise ValueError(
                f"データは1次元または2次元である必要があります。形状: {data.shape}"
            )

        # NumPy配列をdask配列に変換
        dask_data = da_from_array(data)
        cf = ChannelFrame(
            data=dask_data,
            sampling_rate=sampling_rate,
            label=label or "numpy_data",
        )
        if metadata is not None:
            cf.metadata = metadata
        if ch_labels is not None:
            if len(ch_labels) != cf.n_channels:
                raise ValueError(
                    "チャネルラベルの数が指定されたチャネル数と一致しません"
                )
            for i in range(len(ch_labels)):
                cf._channel_metadata[i].label = ch_labels[i]
        if ch_units is not None:
            if len(ch_units) != cf.n_channels:
                raise ValueError("チャネル単位の数が指定されたチャネル数と一致しません")
            for i in range(len(ch_units)):
                cf._channel_metadata[i].unit = ch_units[i]

        return cf

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        channel: Optional[Union[int, list[int]]] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
        chunk_size: Optional[int] = None,
        ch_labels: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> "ChannelFrame":
        """
        ファイルから遅延読み込みでチャネルを作成します。
        様々なファイル形式（WAV、CSV など）を自動的に検出してサポートします。

        Parameters
        ----------
        path : str or Path
            読み込むファイルのパス
        channel : int or list of int, optional
            読み込むチャネル番号。None の場合はすべてのチャネル
        start : float, optional
            読み込み開始位置（秒）。None の場合は先頭から
        end : float, optional
            読み込み終了位置（秒）。None の場合はファイル末尾まで
        chunk_size : int, optional
            処理するチャンクサイズ。遅延処理の分割サイズを指定
        ch_labels : list of str, optional
            各チャネルに設定するラベル
        **kwargs :
            追加のファイル固有パラメータ

        Returns
        -------
        ChannelFrame
            読み込んだデータを含む新しいチャネルフレーム（遅延読み込み）
        """
        from .channel_frame import ChannelFrame

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {path}")

        # ファイルリーダー取得
        reader = get_file_reader(path)

        # ファイル情報取得
        info = reader.get_file_info(path, **kwargs)
        sr = info["samplerate"]
        n_channels = info["channels"]
        n_frames = info["frames"]

        logger.debug(f"File info: sr={sr}, channels={n_channels}, frames={n_frames}")

        # チャネル選択処理
        all_channels = list(range(n_channels))

        if channel is None:
            channels_to_load = all_channels
            logger.debug(f"Will load all channels: {channels_to_load}")
        elif isinstance(channel, int):
            if channel < 0 or channel >= n_channels:
                raise ValueError(
                    f"チャネル指定が範囲外です: {channel} (有効範囲: 0-{n_channels - 1})"  # noqa: E501
                )
            channels_to_load = [channel]
            logger.debug(f"Will load single channel: {channel}")
        elif isinstance(channel, (list, tuple)):
            for ch in channel:
                if ch < 0 or ch >= n_channels:
                    raise ValueError(
                        f"チャネル指定が範囲外です: {ch} (有効範囲: 0-{n_channels - 1})"
                    )
            channels_to_load = list(channel)
            logger.debug(f"Will load specific channels: {channels_to_load}")
        else:
            raise TypeError("channel は int, list, または None である必要があります")

        # インデックス計算
        start_idx = 0 if start is None else max(0, int(start * sr))
        end_idx = n_frames if end is None else min(n_frames, int(end * sr))
        frames_to_read = end_idx - start_idx

        logger.debug(
            f"Setting up lazy load from file={path}, frames={frames_to_read}, "
            f"start_idx={start_idx}, end_idx={end_idx}"
        )

        # 遅延読み込み用の設定
        expected_shape = (len(channels_to_load), frames_to_read)

        # Define the loading function using the file reader
        def _load_audio() -> NDArrayReal:
            logger.debug(">>> EXECUTING DELAYED LOAD <<<")
            # Use the reader to get audio data with parameters
            out = reader.get_data(path, channels_to_load, start_idx, frames_to_read)
            if not isinstance(out, np.ndarray):
                raise ValueError("Unexpected data type after reading file")
            return out

        logger.debug(
            f"Creating delayed dask task with expected shape: {expected_shape}"
        )

        # Create delayed operation
        delayed_data = dask_delayed(_load_audio)()
        logger.debug("Wrapping delayed function in dask array")

        # Create dask array from delayed computation
        dask_array = da_from_delayed(
            delayed_data, shape=expected_shape, dtype=np.float32
        )

        if chunk_size is not None:
            if chunk_size <= 0:
                raise ValueError("Chunk size must be a positive integer")
            logger.debug(f"Setting chunk size: {chunk_size} for sample axis")
            dask_array = dask_array.rechunk({0: -1, 1: chunk_size})

        logger.debug(
            "ChannelFrame setup complete - actual file reading will occur on compute()"  # noqa: E501
        )

        cf = ChannelFrame(
            data=dask_array,
            sampling_rate=sr,
            label=path.stem,
            metadata={
                "filename": str(path),
            },
        )
        if ch_labels is not None:
            if len(ch_labels) != len(cf):
                raise ValueError(
                    "チャネルラベルの数が指定されたチャネル数と一致しません"
                )
            for i in range(len(ch_labels)):
                cf._channel_metadata[i].label = ch_labels[i]
        return cf

    @classmethod
    def read_wav(
        cls, filename: str, ch_labels: Optional[list[str]] = None
    ) -> "ChannelFrame":
        """
        WAVファイルを読み込むユーティリティメソッド

        Parameters
        ----------
        filename : str
            WAVファイルのパス
        ch_labels : list of str, optional
            各チャネルに設定するラベル

        Returns
        -------
        ChannelFrame
            読み込んだデータを含む新しいチャネルフレーム（遅延読み込み）
        """
        from .channel_frame import ChannelFrame

        cf = ChannelFrame.from_file(filename, ch_labels=ch_labels)
        return cf

    @classmethod
    def read_csv(
        cls,
        filename: str,
        ch_labels: Optional[list[str]] = None,
        time_column: Union[int, str] = 0,
        delimiter: str = ",",
        header: Optional[int] = 0,
    ) -> "ChannelFrame":
        """
        CSVファイルを読み込むユーティリティメソッド

        Parameters
        ----------
        filename : str
            CSVファイルのパス
        ch_labels : list of str, optional
            各チャネルに設定するラベル
        time_column : int or str, optional
            時間列のインデックスまたは列名
        delimiter : str, optional
            区切り文字
        header : int, optional
            ヘッダー行の番号

        Returns
        -------
        ChannelFrame
            読み込んだデータを含む新しいチャネルフレーム（遅延読み込み）
        """
        from .channel_frame import ChannelFrame

        cf = ChannelFrame.from_file(
            filename,
            ch_labels=ch_labels,
            time_column=time_column,
            delimiter=delimiter,
            header=header,
        )
        return cf

    def save(self, path: Union[str, Path], format: Optional[str] = None) -> None:
        """
        音声データをファイルに保存します。

        Parameters
        ----------
        path : str or Path
            保存先のファイルパス
        format : str, optional
            ファイル形式。None の場合は拡張子から判断
        """
        logger.debug(f"Saving audio data to file: {path} (will compute now)")
        data = self.compute()
        data = data.T
        if data.shape[1] == 1:
            data = data.squeeze(axis=1)
        sf.write(str(path), data, int(self.sampling_rate), format=format)
        logger.debug(f"Save complete: {path}")

    def highpass_filter(self, cutoff: float, order: int = 4) -> "ChannelFrame":
        """ハイパスフィルターを適用します。"""
        logger.debug(
            f"Setting up highpass filter: cutoff={cutoff}, order={order} (lazy)"
        )
        return self.apply_operation("highpass_filter", cutoff=cutoff, order=order)

    def lowpass_filter(self, cutoff: float, order: int = 4) -> "ChannelFrame":
        """ローパスフィルターを適用します。"""
        logger.debug(
            f"Setting up lowpass filter: cutoff={cutoff}, order={order} (lazy)"
        )
        return self.apply_operation("lowpass_filter", cutoff=cutoff, order=order)

    def normalize(
        self, target_level: float = -20, channel_wise: bool = True
    ) -> "ChannelFrame":
        """信号レベルを正規化します。"""
        logger.debug(
            f"Setting up normalize: target_level={target_level}, channel_wise={channel_wise} (lazy)"  # noqa: E501
        )
        return self.apply_operation(
            "normalize", target_level=target_level, channel_wise=channel_wise
        )

    def a_weighting(self) -> "ChannelFrame":
        """A加重フィルタを適用します。"""
        return self.apply_operation("a_weighting")

    def hpss_harmonic(
        self,
        kernel_size: Union[
            "_IntLike_co", tuple["_IntLike_co", "_IntLike_co"], list["_IntLike_co"]
        ] = 31,
        power: float = 2,
        margin: Union[
            "_FloatLike_co",
            tuple["_FloatLike_co", "_FloatLike_co"],
            list["_FloatLike_co"],
        ] = 1,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: "_WindowSpec" = "hann",
        center: bool = True,
        pad_mode: "_PadModeSTFT" = "constant",
    ) -> "ChannelFrame":
        """HPSS（Harmonic-Percussive Source Separation）の調波成分を抽出します。"""
        return self.apply_operation(
            "hpss_harmonic",
            kernel_size=kernel_size,
            power=power,
            margin=margin,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
        )

    def hpss_percussive(
        self,
        kernel_size: Union[
            "_IntLike_co", tuple["_IntLike_co", "_IntLike_co"], list["_IntLike_co"]
        ] = 31,
        power: float = 2,
        margin: Union[
            "_FloatLike_co",
            tuple["_FloatLike_co", "_FloatLike_co"],
            list["_FloatLike_co"],
        ] = 1,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: "_WindowSpec" = "hann",
        center: bool = True,
        pad_mode: "_PadModeSTFT" = "constant",
    ) -> "ChannelFrame":
        """HPSS（Harmonic-Percussive Source Separation）の打撃音成分を抽出します。"""
        return self.apply_operation(
            "hpss_percussive",
            kernel_size=kernel_size,
            power=power,
            margin=margin,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
        )

    def abs(self) -> "ChannelFrame":
        """絶対値を計算します。"""
        return self.apply_operation("abs")

    def power(self, exponent: float) -> "ChannelFrame":
        """べき乗計算を行います。"""
        return self.apply_operation("power", exponent=exponent)

    def trim(
        self,
        start: float = 0,
        end: Optional[float] = None,
    ) -> "ChannelFrame":
        """
        チャネルをトリミングします。

        Parameters
        ----------
        start : float, optional
            トリミング開始位置（秒）。Noneの場合は先頭から
        end : float, optional
            トリミング終了位置（秒）。Noneの場合はファイル末尾まで

        Returns
        -------
        ChannelFrame
            トリミングされたチャネルフレーム
        """
        if end is None:
            end = self.duration
        if start > end:
            raise ValueError("start must be less than end")
        # トリミング操作を適用
        return self.apply_operation("trim", start=start, end=end)

    def rms_trend(
        self,
        frame_length: int = 2048,
        hop_length: int = 512,
        Aw: bool = False,  # noqa: N803
    ) -> "ChannelFrame":
        """RMSトレンドを計算します。"""
        cf = self.apply_operation(
            "rms_trend", frame_length=frame_length, hop_length=hop_length, Aw=Aw
        )
        cf.sampling_rate = self.sampling_rate / hop_length
        return cf

    def sum(self) -> "ChannelFrame":
        """合計値を計算します。"""
        return self.apply_operation("sum")

    def mean(self) -> "ChannelFrame":
        """平均値を計算します。"""
        return self.apply_operation("mean")

    def channel_difference(self, other_channel: Union[int, str] = 0) -> "ChannelFrame":
        """チャンネル間の差分を計算します。"""
        if isinstance(other_channel, str):
            return self.apply_operation(
                "channel_difference", other_channel=self.label2index(other_channel)
            )
        return self.apply_operation("channel_difference", other_channel=other_channel)

    def fft(self, n_fft: Optional[int] = None, window: str = "hann") -> "SpectralFrame":
        """時間領域データから周波数領域データへ変換（FFT）"""
        from .spectral_frame import SpectralFrame
        from .time_series_operation import FFT

        params = {"n_fft": n_fft, "window": window}
        operation_name = "fft"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")
        from .time_series_operation import create_operation

        # 操作インスタンスを作成
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("FFT", operation)
        # データに処理を適用
        spectrum_data = operation.process(self._data)

        logger.debug(
            f"Created new SpectralFrame with operation {operation_name} added to graph"
        )

        return SpectralFrame(
            data=spectrum_data,
            sampling_rate=self.sampling_rate,
            n_fft=operation.n_fft,
            window=operation.window,
            label=f"Spectrum of {self.label}",
            metadata={**self.metadata, "window": window, "n_fft": n_fft},
            operation_history=[
                *self.operation_history,
                {"operation": "fft", "params": {"n_fft": n_fft, "window": window}},
            ],
            channel_metadata=self._channel_metadata,
            previous=self,
        )

    def welch(
        self, n_fft: Optional[int] = None, window: str = "hann"
    ) -> "SpectralFrame":
        """時間領域データから周波数領域データへ変換（welch）"""
        from .spectral_frame import SpectralFrame
        from .time_series_operation import Welch

        params = {"n_fft": n_fft, "window": window}
        operation_name = "welch"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")
        from .time_series_operation import create_operation

        # 操作インスタンスを作成
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("Welch", operation)
        # データに処理を適用
        spectrum_data = operation.process(self._data)

        logger.debug(
            f"Created new SpectralFrame with operation {operation_name} added to graph"
        )

        return SpectralFrame(
            data=spectrum_data,
            sampling_rate=self.sampling_rate,
            n_fft=operation.n_fft,
            window=operation.window,
            label=f"Spectrum of {self.label}",
            metadata={**self.metadata, "window": window, "n_fft": n_fft},
            operation_history=[
                *self.operation_history,
                {"operation": "fft", "params": {"n_fft": n_fft, "window": window}},
            ],
            channel_metadata=self._channel_metadata,
            previous=self,
        )

    def stft(
        self,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = "hann",
        boundary: Optional[str] = "zeros",
    ) -> "SpectrogramFrame":
        """
        短時間フーリエ変換（STFT）を計算し、時間-周波数領域のスペクトログラムを返します。

        Parameters
        ----------
        n_fft : int, default=2048
            FFTのサンプル数
        hop_length : int, optional
            フレーム間のホップ長。指定がない場合は n_fft//4
        win_length : int, optional
            各フレームの窓長。指定がない場合は n_fft
        window : str, default="hann"
            窓関数の種類
        center : bool, default=True
            信号の中心化を行うかどうか

        Returns
        -------
        SpectrogramFrame
            スペクトログラムデータを含むSpectrogramFrameオブジェクト
        """
        from .spectrogram_frame import SpectrogramFrame
        from .time_series_operation import STFT, create_operation

        # ホップ長とウィンドウ長の設定
        _hop_length = hop_length if hop_length is not None else n_fft // 4
        _win_length = win_length if win_length is not None else n_fft

        params = {
            "n_fft": n_fft,
            "hop_length": _hop_length,
            "win_length": _win_length,
            "window": window,
            "boundary": boundary,
        }
        operation_name = "stft"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")

        # 操作インスタンスを作成
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("STFT", operation)

        # データに処理を適用
        spectrogram_data = operation.process(self._data)

        logger.debug(
            f"Created new SpectrogramFrame with operation {operation_name} added to graph"  # noqa: E501
        )

        # 新しいインスタンスを作成
        return SpectrogramFrame(
            data=spectrogram_data,
            sampling_rate=self.sampling_rate,
            n_fft=n_fft,
            hop_length=_hop_length,
            win_length=_win_length,
            window=window,
            label=f"stft({self.label})",
            metadata=self.metadata,
            operation_history=self.operation_history,
            channel_metadata=self._channel_metadata,
        )

    def _get_additional_init_kwargs(self) -> dict[str, Any]:
        """
        ChannelFrame に必要な追加の初期化引数を提供します。
        """
        return {}
