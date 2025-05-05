"""共通プロトコル定義モジュール。

このモジュールには、ミックスインクラスが使用する共通プロトコルが含まれています。
"""

import logging
from typing import Any, Protocol, TypeVar, runtime_checkable

from dask.array.core import Array as DaArray

from wandas.core.metadata import ChannelMetadata

logger = logging.getLogger(__name__)


@runtime_checkable
class BaseFrameProtocol(Protocol):
    """基本的なフレーム操作を定義するプロトコル。

    すべてのフレームクラスが提供する基本的なメソッドとプロパティを定義します。
    """

    _data: DaArray
    sampling_rate: float
    _channel_metadata: list[ChannelMetadata]
    metadata: dict[str, Any]
    operation_history: list[dict[str, Any]]
    label: str

    @property
    def duration(self) -> float:
        """Returns the duration in seconds."""
        ...

    def label2index(self, label: str) -> int:
        """
        Get the index from a channel label.
        """
        ...

    def apply_operation(
        self, operation_name: str, **params: Any
    ) -> "BaseFrameProtocol":
        """名前付き操作を適用する。

        Args:
            operation_name: 適用する操作の名前
            **params: 操作に渡すパラメータ

        Returns:
            操作を適用した新しいフレームインスタンス
        """
        ...


@runtime_checkable
class ProcessingFrameProtocol(BaseFrameProtocol, Protocol):
    """信号処理に関連する操作を定義するプロトコル。

    信号処理に関連するフレーム操作を提供するメソッドを定義します。
    """

    pass


@runtime_checkable
class TransformFrameProtocol(BaseFrameProtocol, Protocol):
    """変換操作に関連するプロトコル。

    周波数解析やスペクトル変換などの操作を提供するメソッドを定義します。
    """

    pass


# 型変数の定義
T_Base = TypeVar("T_Base", bound=BaseFrameProtocol)
T_Processing = TypeVar("T_Processing", bound=ProcessingFrameProtocol)
T_Transform = TypeVar("T_Transform", bound=TransformFrameProtocol)

__all__ = [
    "BaseFrameProtocol",
    "ProcessingFrameProtocol",
    "TransformFrameProtocol",
    "T_Base",
    "T_Processing",
    "T_Transform",
]
