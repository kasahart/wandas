# spectral_frame.py
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union, cast

import dask
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
    from .noct_frame import NOctFrame


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
        # 0除算を避けるために、最大値と1e-12のいずれかを使用
        level: NDArrayReal = 20 * np.log10(
            np.maximum(self.magnitude / ref[..., np.newaxis], 1e-12)
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
        return np.fft.rfftfreq(self.n_fft, 1.0 / self.sampling_rate)

    # 四則演算のメソッドは BaseFrame または ChannelFrame から同様に実装

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
        other: Union[
            "SpectralFrame", int, float, complex, NDArrayComplex, NDArrayReal, "DaArray"
        ],
        op: Callable[["DaArray", Any], "DaArray"],
        symbol: str,
    ) -> "SpectralFrame":
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

        logger.debug(f"Setting up {symbol} operation (lazy)")

        # Handle potentially None metadata and operation_history
        metadata = {}
        if self.metadata is not None:
            metadata = self.metadata.copy()

        operation_history = []
        if self.operation_history is not None:
            operation_history = self.operation_history.copy()

        # Check if other is a ChannelFrame - improved type checking
        if isinstance(other, SpectralFrame):
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

            return SpectralFrame(
                data=result_data,
                sampling_rate=self.sampling_rate,
                n_fft=self.n_fft,
                window=self.window,
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
            elif isinstance(other, complex):
                other_str = f"complex({other.real}, {other.imag})"
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

            return SpectralFrame(
                data=result_data,
                sampling_rate=self.sampling_rate,
                n_fft=self.n_fft,
                window=self.window,
                label=f"({self.label} {symbol} {other_str})",
                metadata=metadata,
                operation_history=operation_history,
                channel_metadata=updated_channel_metadata,
            )

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
        from .plotting import create_operation

        logger.debug(f"Plotting audio with plot_type={plot_type} (will compute now)")

        # プロット戦略を取得
        plot_strategy: PlotStrategy[SpectralFrame] = create_operation(plot_type)

        # プロット実行
        _ax = plot_strategy.plot(self, ax=ax, **kwargs)

        logger.debug("Plot rendering complete")

        return _ax

    def ifft(self) -> "ChannelFrame":
        """
        逆FFTを計算し、時間領域のデータを返します。
        """
        from .channel_frame import ChannelFrame
        from .time_series_operation import IFFT, create_operation

        params = {"n_fft": self.n_fft, "window": self.window}
        operation_name = "ifft"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")

        # 操作インスタンスを作成
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("IFFT", operation)
        # データに処理を適用
        time_series = operation.process(self._data)

        logger.debug(
            f"Created new SpectralFrame with operation {operation_name} added to graph"
        )

        # 新しいインスタンスを作成
        return ChannelFrame(
            data=time_series,
            sampling_rate=self.sampling_rate,
            label=f"ifft({self.label})",
            metadata=self.metadata,
            operation_history=self.operation_history,
            channel_metadata=self._channel_metadata,
        )

    def _get_additional_init_kwargs(self) -> dict[str, Any]:
        """
        SpectralFrame に必要な追加の初期化引数を提供します。
        """
        return {
            "n_fft": self.n_fft,
            "window": self.window,
        }

    def noct_synthesis(
        self,
        fmin: float,
        fmax: float,
        n: int = 3,
        G: int = 10,  # noqa: N803
        fr: int = 1000,
    ) -> "NOctFrame":
        """
        N-Octave Spectrum を計算します。
        """
        if self.sampling_rate != 48000:
            raise ValueError(
                "noct_synthesisは48000Hzのサンプリングレートでのみ使用できます。"
            )
        from .noct_frame import NOctFrame
        from .time_series_operation import NOctSynthesis

        params = {"fmin": fmin, "fmax": fmax, "n": n, "G": G, "fr": fr}
        operation_name = "noct_synthesis"
        logger.debug(f"Applying operation={operation_name} with params={params} (lazy)")
        from .time_series_operation import create_operation

        # 操作インスタンスを作成
        operation = create_operation(operation_name, self.sampling_rate, **params)
        operation = cast("NOctSynthesis", operation)
        # データに処理を適用
        spectrum_data = operation.process(self._data)

        logger.debug(
            f"Created new SpectralFrame with operation {operation_name} added to graph"
        )

        return NOctFrame(
            data=spectrum_data,
            sampling_rate=self.sampling_rate,
            fmin=fmin,
            fmax=fmax,
            n=n,
            G=G,
            fr=fr,
            label=f"1/{n}Oct of {self.label}",
            metadata={**self.metadata, **params},
            operation_history=[
                *self.operation_history,
                {
                    "operation": "noct_synthesis",
                    "params": params,
                },
            ],
            channel_metadata=self._channel_metadata,
            previous=self,
        )
