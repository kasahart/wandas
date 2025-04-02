import copy
import logging
import numbers
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Callable, Generic, Optional, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt
from dask.array.core import Array as DaArray
from matplotlib.axes import Axes

from wandas.utils.types import NDArrayComplex, NDArrayReal

from .channel_metadata import ChannelMetadata

logger = logging.getLogger(__name__)

T = TypeVar("T", NDArrayComplex, NDArrayReal)
S = TypeVar("S", bound="BaseFrame[Any]")


class BaseFrame(ABC, Generic[T]):
    """
    すべての信号フレーム型の抽象基底クラス
    """

    _data: DaArray
    sampling_rate: float
    label: str
    metadata: dict[str, Any]
    operation_history: list[dict[str, Any]]
    _channel_metadata: list[ChannelMetadata]
    _previous: Optional["BaseFrame[Any]"]

    def __init__(
        self,
        data: DaArray,
        sampling_rate: float,
        label: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        operation_history: Optional[list[dict[str, Any]]] = None,
        channel_metadata: Optional[list[ChannelMetadata]] = None,
        previous: Optional["BaseFrame[Any]"] = None,
    ):
        self._data = data.rechunk(chunks=-1)  # type: ignore [unused-ignore]
        self.sampling_rate = sampling_rate
        self.label = label or "unnamed_frame"
        self.metadata = metadata or {}
        self.operation_history = operation_history or []
        self._previous = previous

        if channel_metadata:
            self._channel_metadata = copy.deepcopy(channel_metadata)
        else:
            self._channel_metadata = [
                ChannelMetadata(label=f"ch{i}", unit="", extra={})
                for i in range(self._n_channels)
            ]

        try:
            # 新しいdaskバージョンでの情報表示
            logger.debug(f"Dask graph layers: {list(self._data.dask.layers.keys())}")
            logger.debug(
                f"Dask graph dependencies: {len(self._data.dask.dependencies)}"
            )
        except Exception as e:
            logger.debug(f"Dask graph visualization details unavailable: {e}")

    @property
    @abstractmethod
    def _n_channels(self) -> int:
        """チャネル数を返します。"""

    @property
    def n_channels(self) -> int:
        """チャネル数を返します。"""
        return len(self)

    @property
    def channels(self) -> list[ChannelMetadata]:
        """チャネルのメタデータにアクセスするためのプロパティ。"""
        return self._channel_metadata

    def get_channel(self: S, channel_idx: int) -> S:
        n_channels = len(self)
        if channel_idx < 0 or channel_idx >= n_channels:
            range_max = n_channels - 1
            raise ValueError(
                f"チャネル指定が範囲外です: {channel_idx} (有効範囲: 0-{range_max})"
            )
        logger.debug(f"Extracting channel index={channel_idx} (lazy operation).")
        channel_data = self._data[channel_idx : channel_idx + 1]

        return self._create_new_instance(
            data=channel_data,
            operation_history=self.operation_history,
            channel_metadata=[self._channel_metadata[channel_idx]],
        )

    def __len__(self) -> int:
        """
        チャンネルのデータ長を返します。
        """
        return len(self._channel_metadata)

    def __iter__(self: S) -> Iterator[S]:
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self: S, key: Union[str, int, slice, tuple[slice, ...]]) -> S:
        """
        チャンネル名またはインデックスでチャンネルを取得するためのメソッド。

        Parameters:
            key (str or int): チャンネルの名前（label）またはインデックス番号。

        Returns:
            Channel: 対応するチャンネル。
        """
        if isinstance(key, str):
            index = self.label2index(key)
            return self.get_channel(index)

        elif isinstance(key, tuple):
            # タプルの場合、最初の要素をインデックスとして扱う
            if len(key) > len(self.shape):
                raise ValueError(
                    f"Invalid key length: {len(key)} for shape {self.shape}"
                )
            new_data = self._data[key]
            new_channel_metadata = self._channel_metadata[key[0]]
            if isinstance(new_channel_metadata, ChannelMetadata):
                new_channel_metadata = [new_channel_metadata]
            return self._create_new_instance(
                data=new_data,
                operation_history=self.operation_history,
                channel_metadata=new_channel_metadata,
            )
        elif isinstance(key, slice):
            new_data = self._data[key]
            new_channel_metadata = self._channel_metadata[key]
            if isinstance(new_channel_metadata, ChannelMetadata):
                new_channel_metadata = [new_channel_metadata]
            return self._create_new_instance(
                data=new_data,
                operation_history=self.operation_history,
                channel_metadata=new_channel_metadata,
            )
        elif isinstance(key, numbers.Integral):
            # インデックス番号でアクセス
            if key < 0 or key >= len(self):
                raise IndexError(f"Channel index {key} out of range.")
            return self.get_channel(key)
        else:
            raise TypeError(
                f"Invalid key type: {type(key)}. Expected str, int, or tuple."
            )

    def label2index(self, label: str) -> int:
        """
        チャンネルラベルからインデックスを取得するメソッド。

        Parameters:
            label (str): チャンネルのラベル。

        Returns:
            int: 対応するインデックス。
        """
        for idx, ch in enumerate(self._channel_metadata):
            if ch.label == label:
                return idx
        raise KeyError(f"Channel label '{label}' not found.")

    @property
    def shape(self) -> tuple[int, ...]:
        _shape: tuple[int, ...] = self._data.shape
        return _shape

    @property
    def data(self) -> T:
        """
        計算済みデータを返します。初めてアクセスしたときに計算が実行されます。
        """
        return self.compute()

    @property
    def labels(self) -> list[str]:
        """すべてのチャネルのラベルをリストとして取得します。"""
        return [ch.label for ch in self._channel_metadata]

    def compute(self) -> T:
        """
        データを計算して返します。
        このメソッドは遅延計算されたデータを具体的なNumPy配列として実体化します。

        Returns
        -------
        NDArrayReal
            計算されたデータ
        """
        logger.debug(
            "COMPUTING DASK ARRAY - This will trigger file reading and all processing"
        )
        result = self._data.compute()

        if not isinstance(result, np.ndarray):
            raise ValueError(f"計算結果がnp.ndarrayではありません: {type(result)}")

        logger.debug(f"Computation complete, result shape: {result.shape}")
        return cast(T, result)

    @abstractmethod
    def plot(
        self, plot_type: str = "default", ax: Optional[Axes] = None, **kwargs: Any
    ) -> Union[Axes, Iterator[Axes]]:
        """データをプロットする"""
        pass

    def persist(self: S) -> S:
        """データをメモリに持続化する"""
        persisted_data = self._data.persist()
        return self._create_new_instance(data=persisted_data)

    def _create_new_instance(self: S, data: DaArray, **kwargs: Any) -> S:
        """
        Create a new channel instance based on an existing channel.
        Keyword arguments can override or extend the original attributes.
        """

        sampling_rate = kwargs.pop("sampling_rate", self.sampling_rate)
        if not isinstance(sampling_rate, int):
            raise TypeError("Sampling rate must be an integer")

        label = kwargs.pop("label", self.label)
        if not isinstance(label, str):
            raise TypeError("Label must be a string")

        metadata = kwargs.pop("metadata", copy.deepcopy(self.metadata))
        if not isinstance(metadata, dict):
            raise TypeError("Metadata must be a dictionary")

        return type(self)(
            data=data,
            sampling_rate=sampling_rate,
            label=label,
            metadata=metadata,
            previous=self,
            **kwargs,
        )

    def __array__(self, dtype: npt.DTypeLike = None) -> NDArrayReal:
        """NumPy配列への暗黙的な変換"""
        result = self.compute()
        if dtype is not None:
            return result.astype(dtype)
        return result

    def visualize_graph(self, filename: Optional[str] = None) -> Optional[str]:
        """計算グラフを可視化してファイルに保存"""
        try:
            filename = filename or f"graph_{uuid.uuid4().hex[:8]}.png"
            self._data.visualize(filename=filename)
            return filename
        except Exception as e:
            logger.warning(f"Failed to visualize the graph: {e}")
            return None

    @abstractmethod
    def _binary_op(
        self: S,
        other: Union[S, int, float, NDArrayReal, DaArray],
        op: Callable[[DaArray, Any], DaArray],
        symbol: str,
    ) -> S:
        """二項演算の基本実装"""
        # 基本的なロジック
        # 実際の実装は派生クラスに任せる
        pass

    def __add__(self: S, other: Union[S, int, float, NDArrayReal]) -> S:
        """加算演算子"""
        return self._binary_op(other, lambda x, y: x + y, "+")

    def __sub__(self: S, other: Union[S, int, float, NDArrayReal]) -> S:
        """減算演算子"""
        return self._binary_op(other, lambda x, y: x - y, "-")

    def __mul__(self: S, other: Union[S, int, float, NDArrayReal]) -> S:
        """乗算演算子"""
        return self._binary_op(other, lambda x, y: x * y, "*")

    def __truediv__(self: S, other: Union[S, int, float, NDArrayReal]) -> S:
        """除算演算子"""
        return self._binary_op(other, lambda x, y: x / y, "/")

    def apply_operation(self: S, operation_name: str, **params: Any) -> S:
        """
        名前付き操作を適用します。

        Parameters
        ----------
        operation_name : str
            適用する操作の名前
        **params : Any
            操作に渡すパラメータ

        Returns
        -------
        T
            操作を適用した新しいインスタンス
        """
        # 操作を適用する抽象メソッド
        return self._apply_operation_impl(operation_name, **params)

    @abstractmethod
    def _apply_operation_impl(self: S, operation_name: str, **params: Any) -> S:
        """操作適用の実装"""
        pass

    def debug_info(self) -> None:
        """詳細なデバッグ情報を出力"""
        logger.debug(f"=== {self.__class__.__name__} Debug Info ===")
        logger.debug(f"Label: {self.label}")
        logger.debug(f"Shape: {self.shape}")
        logger.debug(f"Sampling rate: {self.sampling_rate} Hz")
        logger.debug(f"Operation history: {len(self.operation_history)} operations")
        self._debug_info_impl()
        logger.debug("=== End Debug Info ===")

    def _debug_info_impl(self) -> None:
        """派生クラス固有のデバッグ情報を実装"""
        pass
