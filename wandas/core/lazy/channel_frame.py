import logging
import uuid
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Optional, Union

import dask
import dask.array as da
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from dask.array.core import Array as DaArray

from wandas.utils.types import NDArrayReal

from .channel_io import ChannelIOMixin
from .channel_operations import ChannelOperationsMixin
from .metadata import ChannelMetadataCollection
from .plotting import get_plot_strategy

logger = logging.getLogger(__name__)

dask_delayed = dask.delayed  # type: ignore [unused-ignore]
da_from_delayed = da.from_delayed  # type: ignore [unused-ignore]


class ChannelFrame(ChannelOperationsMixin, ChannelIOMixin):
    """
    音声チャネルのラッパークラス
    データ形状: (channels, samples) または単一チャネルの場合 (1, samples)
    """

    _data: DaArray
    sampling_rate: float
    label: str
    metadata: dict[str, Any]
    operation_history: list[dict[str, Any]]
    _channel_metadata: dict[int, dict[str, Any]]

    def __init__(
        self,
        data: DaArray,
        sampling_rate: float,
        label: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        operation_history: Optional[list[dict[str, Any]]] = None,
        channel_metadata: Optional[dict[int, dict[str, Any]]] = None,
    ):
        """
        AudioChannelの初期化

        Parameters
        ----------
        data : dask.array.Array
            音声データ。形状は (channels, samples) または (1, samples)
        sampling_rate : float
            サンプリングレート (Hz)
        label : str, optional
            チャネルのラベル
        metadata : dict, optional
            メタデータ辞書
        operation_history : list, optional
            適用された操作の履歴
        channel_metadata : dict, optional
            チャネルごとのメタデータ。キーはチャネルインデックス、値はメタデータ辞書
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim > 2:
            raise ValueError(
                f"データは1次元または2次元である必要があります。形状: {data.shape}"
            )

        self._data = data
        self.sampling_rate = sampling_rate
        self.label = label or "unnamed_frame"
        self.metadata = metadata or {}
        self.operation_history = operation_history or []

        # チャネルごとのメタデータを初期化
        self._channel_metadata: dict[int, dict[str, Any]] = {}

        # メタデータアクセサーを初期化
        self._metadata_accessor = ChannelMetadataCollection(self)

        # 提供されたチャネルメタデータがあれば使用、
        # なければデフォルト値を各チャネルに設定
        if channel_metadata:
            self._channel_metadata = channel_metadata.copy()

        # 各チャネルにメタデータがあることを確認（なければデフォルト値を設定）
        for i in range(self.n_channels):
            if i not in self._channel_metadata:
                self._channel_metadata[i] = {}

            # ラベルがなければデフォルトを設定
            if "label" not in self._channel_metadata[i]:
                if self.n_channels == 1:
                    self._channel_metadata[i]["label"] = self.label
                else:
                    self._channel_metadata[i]["label"] = f"{self.label}_ch{i}"

            # 単位がなければデフォルトを設定
            if "unit" not in self._channel_metadata[i]:
                self._channel_metadata[i]["unit"] = ""

        try:
            # 新しいdaskバージョンでの情報表示
            logger.debug(f"Dask graph layers: {list(self._data.dask.layers.keys())}")
            logger.debug(
                f"Dask graph dependencies: {len(self._data.dask.dependencies)}"
            )
        except Exception as e:
            logger.debug(f"Dask graph visualization details unavailable: {e}")

    def _create_channel_frame(
        self,
        data: DaArray,
        label: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        operation_history: Optional[list[dict[str, Any]]] = None,
        channel_metadata: Optional[dict[int, dict[str, Any]]] = None,
    ) -> "ChannelFrame":
        """
        新しい ChannelFrame インスタンスを作成するヘルパーメソッド。
        現在のインスタンスの属性をデフォルトとして使用しつつ、必要に応じて上書きします。

        Parameters
        ----------
        data : dask.array.Array
            新しいインスタンスに使用するデータ
        label : str, optional
            新しいラベル（指定しない場合は現在のラベルを使用）
        metadata : dict, optional
            新しいメタデータ（指定しない場合は現在のメタデータのコピーを使用）
        operation_history : list, optional
            新しい操作履歴（指定しない場合は現在の操作履歴のコピーを使用）
        channel_metadata : dict, optional
            新しいチャネルメタデータ（指定しない場合は現在のチャネルメタデータを使用）

        Returns
        -------
        ChannelFrame
            新しい ChannelFrame インスタンス
        """
        return ChannelFrame(
            data=data,
            sampling_rate=self.sampling_rate,
            label=label if label is not None else self.label,
            metadata=metadata if metadata is not None else self.metadata.copy(),
            operation_history=operation_history
            if operation_history is not None
            else self.operation_history.copy(),
            channel_metadata=channel_metadata
            if channel_metadata is not None
            else self._metadata_accessor.get_all(),
        )

    @property
    def time(self) -> NDArrayReal:
        """
        時刻データを返します。
        """
        return np.arange(self.n_samples) / self.sampling_rate

    @property
    def channel(self) -> ChannelMetadataCollection:
        """チャネルのメタデータにアクセスするためのプロパティ。"""
        return self._metadata_accessor

    @property
    def data(self) -> NDArrayReal:
        """
        ユーザーから見ると普通の NumPy 配列ですが、内部では遅延計算されています。
        初めてアクセスしたときに計算が実行されます。
        """
        return self.compute()

    @property
    def n_channels(self) -> int:
        n: int = self._data.shape[-2]
        return n

    @property
    def n_samples(self) -> int:
        n: int = self._data.shape[-1]
        return n

    @property
    def duration(self) -> float:
        return self.n_samples / self.sampling_rate

    @property
    def shape(self) -> tuple[int, int]:
        _shape: tuple[int, int] = self._data.shape
        return _shape

    @property
    def labels(self) -> list[str]:
        """すべてのチャネルのラベルをリストとして取得します。"""
        return [self.channel[i].label for i in range(self.n_channels)]

    def compute(self) -> NDArrayReal:
        logger.debug(
            "COMPUTING DASK ARRAY - This will trigger file reading and all processing"
        )
        result = self._data.compute()

        if not isinstance(result, np.ndarray):
            raise ValueError(f"計算結果がnp.ndarrayではありません: {type(result)}")

        logger.debug(f"Computation complete, result shape: {result.shape}")
        return result

    def persist(self) -> "ChannelFrame":
        logger.debug("Persisting data in memory for reuse.")
        persisted_data = self._data.persist()
        return self._create_channel_frame(data=persisted_data)

    def get_channel(self, channel_idx: int) -> "ChannelFrame":
        if channel_idx < 0 or channel_idx >= self.n_channels:
            range_max = self.n_channels - 1
            raise ValueError(
                f"チャネル指定が範囲外です: {channel_idx} (有効範囲: 0-{range_max})"
            )
        logger.debug(f"Extracting channel index={channel_idx} (lazy operation).")
        channel_data = self._data[channel_idx : channel_idx + 1]

        # チャネル固有のメタデータを取得
        channel_label = self.channel[channel_idx].label
        channel_metadata = {**self.metadata, "channel_idx": channel_idx}

        # チャネル固有のメタデータを新しいChannelFrameに渡す
        new_channel_metadata = {0: self._channel_metadata[channel_idx].copy()}

        return self._create_channel_frame(
            data=channel_data,
            label=channel_label,
            metadata=channel_metadata,
            channel_metadata=new_channel_metadata,
        )

    def apply_operation(self, operation_name: str, **params: Any) -> "ChannelFrame":
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
        return self._create_channel_frame(
            data=processed_data,
            metadata=new_metadata,
            operation_history=new_history,
        )

    # forでループを回すためのメソッド
    def __iter__(self) -> Iterator["ChannelFrame"]:
        for idx in range(self.n_channels):
            yield self.get_channel(idx)

    def __array__(self, dtype: npt.DTypeLike = None) -> NDArrayReal:
        """
        NumPy の ndarray への暗黙的な変換をサポート
        ユーザーが np.array(channel) や他の NumPy 操作を実行すると、
        自動的に compute されます
        """
        result = self.compute()
        if dtype is not None:
            return result.astype(dtype)
        return result

    # NumPy 配列のように振る舞うための追加メソッド
    def __getitem__(self, key: Union[str, int]) -> Union["ChannelFrame", NDArrayReal]:
        """
        インデックス操作をサポート（遅延実行を維持）
        """
        # インデックス操作をサポート（遅延実行を維持）
        if isinstance(key, tuple) and len(key) > 2:
            raise IndexError("インデックスの次元が多すぎます")

        new_data = self._data[key]

        # インデックス操作の結果が単一の要素の場合は値を返す（自動計算）
        if isinstance(new_data, DaArray) and new_data.ndim == 0:
            new_data = new_data.compute()
            if isinstance(new_data, np.ndarray):
                return new_data
            else:
                raise ValueError(
                    f"計算結果がnp.ndarrayではありません: {type(new_data)}"
                )

        # それ以外は新しい LazyAudioChannel を返す
        if isinstance(new_data, DaArray):
            return self._create_channel_frame(
                data=new_data,
                label=f"{self.label}[{key}]",
            )

        # スカラー値など、配列でない場合はそのまま返す
        raise ValueError(f"インデックス操作の結果が不明: {type(new_data)}")

    def visualize_graph(self, filename: Optional[str] = None) -> Optional[str]:
        """計算グラフを可視化してファイルに保存"""
        try:
            # daskのvisualize機能を使用
            filename = filename or f"audio_graph_{uuid.uuid4().hex[:8]}.png"
            logger.debug(f"Visualizing computation graph to file: {filename}")
            self._data.visualize(filename=filename)
            logger.debug(f"Graph visualization saved to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to visualize graph: {str(e)}")
            return None

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
        plot_strategy = get_plot_strategy(plot_type)

        # プロット実行
        _ax = plot_strategy.plot(self, ax=ax, **kwargs)

        logger.debug("Plot rendering complete")

        return _ax

    def debug_info(self) -> None:
        """詳細なデバッグ情報を出力"""
        logger.debug("=== LazyAudioChannel Debug Info ===")
        logger.debug(f"Label: {self.label}")
        logger.debug(f"Shape: {self.shape}")
        logger.debug(f"Sampling rate: {self.sampling_rate} Hz")
        logger.debug(f"Duration: {self.duration:.2f} seconds")
        logger.debug(f"Channels: {self.n_channels}")
        logger.debug(f"Samples: {self.n_samples}")
        logger.debug(f"Dask chunks: {self._data.chunks}")
        logger.debug(f"Operation history: {len(self.operation_history)} operations")
        for i, op in enumerate(self.operation_history):
            logger.debug(f"  Operation {i + 1}: {op}")
        logger.debug(f"Metadata: {self.metadata}")
        logger.debug("=== End Debug Info ===")

    def _validate_channel_idx(self, channel_idx: int) -> None:
        """チャネルインデックスが有効かどうかを確認します。"""
        if not isinstance(channel_idx, int):
            raise TypeError(
                f"チャネルインデックスは整数である必要があります: {type(channel_idx)}"
            )
        if channel_idx < 0 or channel_idx >= self.n_channels:
            raise ValueError(
                f"チャネル指定が範囲外です: {channel_idx} (有効範囲: 0-{self.n_channels - 1})"  # noqa: E501
            )
