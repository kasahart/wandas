# spectral_frame.py
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union

import dask
import dask.array as da
import librosa
import numpy as np
from dask.array.core import Array as DaArray

from wandas.utils.types import NDArrayComplex, NDArrayReal

from .base_frame import BaseFrame
from .channel_metadata import ChannelMetadata
from .plotting import create_operation

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from wandas.core.lazy.plotting import PlotStrategy

dask_delayed = dask.delayed  # type: ignore [unused-ignore]
da_from_delayed = da.from_delayed  # type: ignore [unused-ignore]
da_from_array = da.from_array  # type: ignore [unused-ignore]

logger = logging.getLogger(__name__)

S = TypeVar("S", bound="BaseFrame[Any]")


class SpectralFrame(BaseFrame[NDArrayComplex]):
    """
    周波数領域のデータを扱うクラス
    データ形状: (channels, frequency_bins)または単一チャネルの場合 (1, frequency_bins)
    """

    n_fft: int
    window: str

    def __init__(
        self,
        data: DaArray,
        sampling_rate: float,
        n_fft: int,
        window: str = "hann",
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
        self.n_fft = n_fft
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
        return np.abs(self.data)

    @property
    def phase(self) -> NDArrayReal:
        return np.angle(self.data)

    @property
    def power(self) -> NDArrayReal:
        return np.abs(self.data) ** 2

    @property
    def dB(self) -> NDArrayReal:  # noqa: N802
        # dB規定値を_channel_metadataから収集
        ref = np.array([ch.ref for ch in self._channel_metadata])
        # dB変換
        return 20 * np.log10(np.maximum(self.magnitude / ref, 1e-12))

    @property
    def dBA(self) -> NDArrayReal:  # noqa: N802
        # dB規定値を_channel_metadataから収集
        weighted = librosa.A_weighting(frequencies=self.freqs, min_db=None)
        return self.dB + weighted

    @property
    def _n_channels(self) -> int:
        """チャネル数を返します。"""
        return self.shape[-2]

    @property
    def freqs(self) -> NDArrayReal:
        """周波数軸を返します"""
        return np.fft.rfftfreq(self.n_fft, 1.0 / self.sampling_rate)

    # 四則演算のメソッドは BaseFrame または ChannelFrame から同様に実装

    def plot(
        self, plot_type: str = "frequency", ax: Optional["Axes"] = None, **kwargs: Any
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
        plot_strategy: PlotStrategy[SpectralFrame] = create_operation(plot_type)

        # プロット実行
        _ax = plot_strategy.plot(self, ax=ax, **kwargs)

        logger.debug("Plot rendering complete")

        return _ax

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
        other: Union["SpectralFrame", int, float, NDArrayReal, DaArray],
        op: Callable[[DaArray, Any], DaArray],
        symbol: str,
    ) -> "SpectralFrame":
        """二項演算の基本実装"""
        # 基本的なロジック
        # 実際の実装は派生クラスに任せる
        return self
