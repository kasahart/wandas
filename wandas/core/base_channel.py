# wandas/core/base_channel.py

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from wandas.core import util

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class BaseChannel(ABC):
    def __init__(
        self,
        data: np.ndarray,
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
        self.label = label
        self.unit = unit if unit is not None else ""
        self.metadata = metadata or {}
        self.ref = util.unit_to_ref(self.unit)

    @property
    def data(self) -> np.ndarray:
        """
        データを返します。
        """
        return self._data

    @property
    def sampling_rate(self) -> int:
        """
        データを返します。派生クラスで実装が必要です。
        """
        return self._sampling_rate

    @abstractmethod
    def plot(
        self, ax: Optional["Axes"] = None, title: Optional[str] = None
    ) -> tuple["Axes", np.ndarray]:
        """
        データをプロットします。派生クラスで実装が必要です。
        """
        pass

    # 共通のメソッドやプロパティをここに追加できます
