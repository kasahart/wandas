# spectral_frame.py
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union

import dask
import dask.array as da
import librosa
import numpy as np
from dask.array.core import Array as DaArray
from mosqito.sound_level_meter.noct_spectrum._center_freq import _center_freq

from wandas.utils.types import NDArrayReal

from .base_frame import BaseFrame
from .channel_metadata import ChannelMetadata

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from wandas.core.lazy.plotting import PlotStrategy


dask_delayed = dask.delayed  # type: ignore [unused-ignore]
da_from_delayed = da.from_delayed  # type: ignore [unused-ignore]
da_from_array = da.from_array  # type: ignore [unused-ignore]

logger = logging.getLogger(__name__)

S = TypeVar("S", bound="BaseFrame[Any]")


class NOctFrame(BaseFrame[NDArrayReal]):
    """
    Nオクターブデータを扱うクラス
    データ形状: (channels, frequency_bins)または単一チャネルの場合 (1, frequency_bins)
    """

    fmin: float
    fmax: float
    n: int
    G: int
    fr: int

    def __init__(
        self,
        data: DaArray,
        sampling_rate: float,
        fmin: float = 0,
        fmax: float = 0,
        n: int = 3,
        G: int = 10,  # noqa: N803
        fr: int = 1000,
        label: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        operation_history: Optional[list[dict[str, Any]]] = None,
        channel_metadata: Optional[list[ChannelMetadata]] = None,
        previous: Optional["BaseFrame[Any]"] = None,
    ) -> None:
        self.n = n
        self.G = G
        self.fr = fr
        self.fmin = fmin
        self.fmax = fmax
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
    def dB(self) -> NDArrayReal:  # noqa: N802
        # dB規定値を_channel_metadataから収集
        ref = np.array([ch.ref for ch in self._channel_metadata])
        # dB変換
        # 0除算を避けるために、最大値と1e-12のいずれかを使用
        level: NDArrayReal = 20 * np.log10(
            np.maximum(self.data / ref[..., np.newaxis], 1e-12)
        )
        return level

    @property
    def dBA(self) -> NDArrayReal:  # noqa: N802
        # dB規定値を_channel_metadataから収集
        weighted: NDArrayReal = librosa.A_weighting(frequencies=self.freqs, min_db=None)
        return self.dB + weighted

    @property
    def _n_channels(self) -> int:
        """チャネル数を返します。"""
        return self.shape[-2]

    @property
    def freqs(self) -> NDArrayReal:
        """周波数軸を返します"""
        _, freqs = _center_freq(
            fmax=self.fmax,
            fmin=self.fmin,
            n=self.n,
            G=self.G,
            fr=self.fr,
        )
        if isinstance(freqs, np.ndarray):
            return freqs
        else:
            raise ValueError("freqs is not numpy array.")

    def _binary_op(
        self: S,
        other: Union[S, int, float, NDArrayReal, DaArray],
        op: Callable[[DaArray, Any], DaArray],
        symbol: str,
    ) -> S:
        """二項演算の基本実装"""
        # 基本的なロジック
        # 実際の実装は派生クラスに任せる
        raise NotImplementedError(
            f"Operation {symbol} is not implemented for NOctFrame."
        )
        return self

    def _apply_operation_impl(self: S, operation_name: str, **params: Any) -> S:
        """
        遅延評価を使用して操作を適用します。
        """
        # 遅延評価を使用して操作を適用
        raise NotImplementedError(
            f"Operation {operation_name} is not implemented for NOctFrame."
        )
        return self

    def plot(
        self, plot_type: str = "noct", ax: Optional["Axes"] = None, **kwargs: Any
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
        from .plotting import create_operation

        logger.debug(f"Plotting audio with plot_type={plot_type} (will compute now)")

        # プロット戦略を取得
        plot_strategy: PlotStrategy[NOctFrame] = create_operation(plot_type)

        # プロット実行
        _ax = plot_strategy.plot(self, ax=ax, **kwargs)

        logger.debug("Plot rendering complete")

        return _ax

    def _get_additional_init_kwargs(self) -> dict[str, Any]:
        """
        SpectralFrame に必要な追加の初期化引数を提供します。
        """
        return {
            "n": self.n,
            "G": self.G,
            "fr": self.fr,
            "fmin": self.fmin,
            "fmax": self.fmax,
        }
