# wandas/core/base_channel.py

from typing import Optional, Dict, Any, Callable, Type
from abc import ABC, abstractmethod
from wandas.core import util

import numpy as np


class BaseChannel(ABC):
    def __init__(
        self,
        data: np.ndarray,
        sampling_rate: int,
        label: Optional[str] = None,
        unit: Optional[str] = None,
        calibration_value: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        BaseChannel オブジェクトを初期化します。

        Parameters:
            label (str, optional): チャンネルのラベル。
            unit (str, optional): 単位。
            calibration_value (float, optional): 校正値。
            metadata (dict, optional): その他のメタデータ。
        """
        self._data = data
        self._sampling_rate = sampling_rate
        self.label = label
        self.unit = unit if unit is not None else ""
        self.calibration_value = calibration_value
        self.metadata = metadata or {}
        self.ref = util.unit_to_ref(self.unit)

    @property
    def data(self):
        """
        データを返します。
        """
        if self.calibration_value is not None:
            return self._data * self.calibration_value
        return self._data

    @property
    def sampling_rate(self):
        """
        データを返します。派生クラスで実装が必要です。
        """
        return self._sampling_rate

    @abstractmethod
    def plot(self, ax: Optional[Any] = None, title: Optional[str] = None):
        """
        データをプロットします。派生クラスで実装が必要です。
        """
        pass

    # 共通のメソッドやプロパティをここに追加できます
