# wandas/core/base_channel.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from wandas.core import util
from wandas.utils.types import NDArrayReal

if TYPE_CHECKING:
    from matplotlib.axes import Axes


@dataclass
class BaseChannel(ABC):
    _data: NDArrayReal
    _sampling_rate: int
    label: str
    unit: str
    metadata: dict[str, Any]
    ref: float = 1

    def __init__(
        self,
        data: NDArrayReal,
        sampling_rate: int,
        label: Optional[str] = None,
        unit: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """
        BaseChannel オブジェクトを初期化します。

        Parameters:
            label (str, optional): チャンネルのラベル。
            unit (str, optional): 単位。
            metadata (dict, optional): その他のメタデータ。
        """
        self._data = data
        self._sampling_rate = sampling_rate
        self.label = label or ""
        self.unit = unit or ""
        self.metadata = metadata or {}
        self.ref = util.unit_to_ref(self.unit)

    @property
    def data(self) -> NDArrayReal:
        """
        データを返します。
        """
        return self._data

    @property
    def sampling_rate(self) -> int:
        """
        サンプリング周波数を返します。
        """
        return self._sampling_rate

    @abstractmethod
    def plot(
        self, ax: Optional["Axes"] = None, title: Optional[str] = None
    ) -> tuple["Axes", NDArrayReal]:
        """
        データをプロットします。派生クラスで実装が必要です。
        """
        pass

    # 共通のメソッドやプロパティをここに追加できます
