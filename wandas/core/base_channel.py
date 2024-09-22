# wandas/core/base_channel.py

from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from wandas.core import util


class BaseChannel(ABC):
    def __init__(
        self,
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
        self.label = label
        self.unit = unit
        self.calibration_value = calibration_value
        self.metadata = metadata or {}
        self.ref = util.unit_to_ref(unit)

    @abstractmethod
    def plot(self, ax: Optional[Any] = None, title: Optional[str] = None):
        """
        データをプロットします。派生クラスで実装が必要です。
        """
        pass

    # 共通のメソッドやプロパティをここに追加できます
