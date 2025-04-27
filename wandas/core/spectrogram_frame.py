import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union, cast

import dask.array as da
import librosa
import numpy as np
from dask.array.core import Array as DaArray

from wandas.utils.types import NDArrayComplex, NDArrayReal

from .base_frame import BaseFrame
from .channel_metadata import ChannelMetadata

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from wandas.core.plotting import PlotStrategy

    from .channel_frame import ChannelFrame
    from .spectral_frame import SpectralFrame

logger = logging.getLogger(__name__)

S = TypeVar("S", bound="BaseFrame[Any]")


class SpectrogramFrame(BaseFrame[NDArrayComplex]):
    """
    時間-周波数領域のデータ（スペクトログラム）を扱うクラス
    データ形状: (channels, frequency_bins, time_frames) または
             単一チャネルの場合 (1, frequency_bins, time_frames)
    """

    n_fft: int
    hop_length: int
    win_length: int
    window: str

    def __init__(
        self,
        data: DaArray,
        sampling_rate: float,
        n_fft: int,
        hop_length: int,
        win_length: Optional[int] = None,
        window: str = "hann",
        label: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        operation_history: Optional[list[dict[str, Any]]] = None,
        channel_metadata: Optional[list[ChannelMetadata]] = None,
        previous: Optional["BaseFrame[Any]"] = None,
    ) -> None:
        if data.ndim == 2:
            data = da.expand_dims(data, axis=0)  # type: ignore [unused-ignore]
        elif data.ndim != 3:
            raise ValueError(
                f"データは2次元または3次元である必要があります。形状: {data.shape}"
            )
        if not data.shape[-2] == n_fft // 2 + 1:
            raise ValueError(
                f"データの形状が無効です。周波数ビン数は {n_fft // 2 + 1} である必要があります。"  # noqa: E501
            )

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.window = window

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
    def magnitude(self) -> NDArrayReal:
        """振幅スペクトログラム"""
        return np.abs(self.data)

    @property
    def phase(self) -> NDArrayReal:
        """位相スペクトログラム"""
        return np.angle(self.data)

    @property
    def power(self) -> NDArrayReal:
        """パワースペクトログラム"""
        return np.abs(self.data) ** 2

    @property
    def dB(self) -> NDArrayReal:  # noqa: N802
        """デシベル単位のスペクトログラム"""
        # dB規定値を_channel_metadataから収集
        ref = np.array([ch.ref for ch in self._channel_metadata])
        # dB変換
        # 0除算を避けるために、最大値と1e-12のいずれかを使用
        level: NDArrayReal = 20 * np.log10(
            np.maximum(self.magnitude / ref[..., np.newaxis, np.newaxis], 1e-12)
        )
        return level

    @property
    def dBA(self) -> NDArrayReal:  # noqa: N802
        """A特性重み付けデシベル単位のスペクトログラム"""
        weighted: NDArrayReal = librosa.A_weighting(frequencies=self.freqs, min_db=None)
        return self.dB + weighted[:, np.newaxis]  # 周波数軸に沿ってブロードキャスト

    @property
    def _n_channels(self) -> int:
        """チャネル数を返します"""
        return self.shape[0]

    @property
    def n_frames(self) -> int:
        """時間フレーム数を返します"""
        return self.shape[-1]

    @property
    def n_freq_bins(self) -> int:
        """周波数ビン数を返します"""
        return self.shape[-2]

    @property
    def freqs(self) -> NDArrayReal:
        """周波数軸を返します"""
        return np.fft.rfftfreq(self.n_fft, 1.0 / self.sampling_rate)

    @property
    def times(self) -> NDArrayReal:
        """時間軸を返します"""
        return np.arange(self.n_frames) * self.hop_length / self.sampling_rate

    def _apply_operation_impl(self: S, operation_name: str, **params: Any) -> S:
        """
        操作を適用する内部実装。遅延評価を利用します。
        """
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")
        from .time_series_operation import create_operation

        operation = create_operation(operation_name, self.sampling_rate, **params)
        processed_data = operation.process(self._data)

        operation_metadata = {"operation": operation_name, "params": params}
        new_history = self.operation_history.copy()
        new_history.append(operation_metadata)
        new_metadata = {**self.metadata}
        new_metadata[operation_name] = params

        logger.debug(
            f"Created new SpectrogramFrame with operation {operation_name} added to graph"  # noqa: E501
        )
        return self._create_new_instance(
            data=processed_data,
            metadata=new_metadata,
            operation_history=new_history,
        )

    def _binary_op(
        self,
        other: Union[
            "SpectrogramFrame",
            int,
            float,
            complex,
            NDArrayComplex,
            NDArrayReal,
            "DaArray",
        ],
        op: Callable[["DaArray", Any], "DaArray"],
        symbol: str,
    ) -> "SpectrogramFrame":
        """
        二項演算の共通実装 - daskの遅延演算を活用
        """
        logger.debug(f"Setting up {symbol} operation (lazy)")

        metadata = {}
        if self.metadata is not None:
            metadata = self.metadata.copy()

        operation_history = []
        if self.operation_history is not None:
            operation_history = self.operation_history.copy()

        if isinstance(other, SpectrogramFrame):
            if self.sampling_rate != other.sampling_rate:
                raise ValueError(
                    "サンプリングレートが一致していません。演算できません。"
                )

            result_data = op(self._data, other._data)

            merged_channel_metadata = []
            for self_ch, other_ch in zip(
                self._channel_metadata, other._channel_metadata
            ):
                ch = self_ch.model_copy(deep=True)
                ch["label"] = f"({self_ch['label']} {symbol} {other_ch['label']})"
                merged_channel_metadata.append(ch)

            operation_history.append({"operation": symbol, "with": other.label})

            return SpectrogramFrame(
                data=result_data,
                sampling_rate=self.sampling_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                label=f"({self.label} {symbol} {other.label})",
                metadata=metadata,
                operation_history=operation_history,
                channel_metadata=merged_channel_metadata,
                previous=self,
            )
        else:
            result_data = op(self._data, other)

            if isinstance(other, (int, float)):
                other_str = str(other)
            elif isinstance(other, complex):
                other_str = f"complex({other.real}, {other.imag})"
            elif isinstance(other, np.ndarray):
                other_str = f"ndarray{other.shape}"
            elif hasattr(other, "shape"):
                other_str = f"dask.array{other.shape}"
            else:
                other_str = str(type(other).__name__)

            updated_channel_metadata: list[ChannelMetadata] = []
            for self_ch in self._channel_metadata:
                ch = self_ch.model_copy(deep=True)
                ch["label"] = f"({self_ch.label} {symbol} {other_str})"
                updated_channel_metadata.append(ch)

            operation_history.append({"operation": symbol, "with": other_str})

            return SpectrogramFrame(
                data=result_data,
                sampling_rate=self.sampling_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                label=f"({self.label} {symbol} {other_str})",
                metadata=metadata,
                operation_history=operation_history,
                channel_metadata=updated_channel_metadata,
            )

    def plot(
        self, plot_type: str = "spectrogram", ax: Optional["Axes"] = None, **kwargs: Any
    ) -> Union["Axes", Iterator["Axes"]]:
        """
        様々な形式のプロットを生成します (Strategyパターンを使用)

        Parameters
        ----------
        plot_type : str
            'spectrogram', 'phasegram' などのプロット種類
        ax : matplotlib.axes.Axes, optional
            プロット先の軸。Noneの場合は新しい軸を作成
        **kwargs : dict
            プロット固有のパラメータ
        """
        from .plotting import create_operation

        logger.debug(
            f"Plotting spectrogram with plot_type={plot_type} (will compute now)"
        )

        # プロット戦略を取得
        plot_strategy: PlotStrategy[SpectrogramFrame] = create_operation(plot_type)

        # プロット実行
        _ax = plot_strategy.plot(self, ax=ax, **kwargs)

        logger.debug("Plot rendering complete")

        return _ax

    def plot_Aw(  # noqa: N802
        self, plot_type: str = "spectrogram", ax: Optional["Axes"] = None, **kwargs: Any
    ) -> Union["Axes", Iterator["Axes"]]:
        return self.plot(plot_type=plot_type, ax=ax, Aw=True, **kwargs)

    def get_frame_at(self, time_idx: int) -> "SpectralFrame":
        """
        特定の時間フレームのスペクトルデータを取得します

        Parameters
        ----------
        time_idx : int
            取得する時間フレームのインデックス

        Returns
        -------
        SpectralFrame
            指定された時間フレームのスペクトルデータ
        """
        from .spectral_frame import SpectralFrame

        if time_idx < 0 or time_idx >= self.n_frames:
            raise IndexError(
                f"時間インデックス {time_idx} が範囲外です。有効範囲: 0-{self.n_frames - 1}"  # noqa: E501
            )

        frame_data = self._data[..., time_idx]

        return SpectralFrame(
            data=frame_data,
            sampling_rate=self.sampling_rate,
            n_fft=self.n_fft,
            window=self.window,
            label=f"{self.label} (Frame {time_idx}, Time {self.times[time_idx]:.3f}s)",
            metadata=self.metadata,
            operation_history=self.operation_history,
            channel_metadata=self._channel_metadata,
        )

    def to_channel_frame(self) -> "ChannelFrame":
        """
        スペクトログラムを逆STFTで時間領域のデータに変換します

        Returns
        -------
        ChannelFrame
            時間領域のデータを含むChannelFrame
        """
        from .channel_frame import ChannelFrame
        from .time_series_operation import ISTFT, create_operation

        params = {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "window": self.window,
        }
        operation_name = "istft"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")

        # 操作インスタンスを作成
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("ISTFT", operation)

        # データに処理を適用
        time_series = operation.process(self._data)

        logger.debug(
            f"Created new ChannelFrame with operation {operation_name} added to graph"
        )

        # 新しいインスタンスを作成
        return ChannelFrame(
            data=time_series,
            sampling_rate=self.sampling_rate,
            label=f"istft({self.label})",
            metadata=self.metadata,
            operation_history=self.operation_history,
            channel_metadata=self._channel_metadata,
        )

    def _get_additional_init_kwargs(self) -> dict[str, Any]:
        """
        SpectrogramFrame に必要な追加の初期化引数を提供します。
        """
        return {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "window": self.window,
        }
