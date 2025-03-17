# wandas/core/base_channel.py

import logging
import os
import tempfile
import threading
import weakref
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, TypeVar, Union

import dask.array as da
import h5py
import numpy as np

from wandas.core import util
from wandas.utils.types import NDArrayReal

if TYPE_CHECKING:
    from dask.array.core import Array

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="BaseChannel")


class BaseChannel(ABC):
    __slots__ = (
        "_data",
        "_data_path",
        "_owns_file",
        "_is_closed",
        "_sampling_rate",
        "_label",
        "_unit",
        "_metadata",
        "_ref",
        "_finalizer",
        "_lock",
        "previous",  # 追加：変換前の状態を保持する属性
    )

    def __init__(
        self,
        data: Union[NDArrayReal, np.memmap[Any, np.dtype[Any]]],
        sampling_rate: int,
        label: Optional[str] = None,
        unit: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        previous: Optional["BaseChannel"] = None,  # 新たな引数
    ):
        """
        Initializes the BaseChannel object.

        Parameters:
            data (NDArrayReal): チャンネルのデータ。
            sampling_rate (int): サンプリング周波数。
            label (str, optional): チャンネルのラベル。
            unit (str, optional): 単位。
            metadata (dict, optional): その他のメタデータ。
        """
        self._sampling_rate = sampling_rate
        self._label = label or ""
        self._unit = unit or ""
        self._metadata = metadata or {}
        self._ref = util.unit_to_ref(self.unit)
        self._is_closed = False
        self._lock = threading.Lock()
        self.previous = previous  # 変換前の状態を保持
        self._data_path = None

        if isinstance(data, np.memmap):
            # 既にメモリマップされたデータは numpy 配列に変換して dask.array でラップ
            self._data: Array = da.from_array(data, chunks="auto")  # type: ignore [unused-ignore, attr-defined, no-untyped-call]
            self._data_path = None
            self._owns_file = False
        else:
            temp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
            temp.close()
            self._data_path = temp.name
            try:
                with h5py.File(self._data_path, "w") as f:
                    f.create_dataset("data", data=data)
            except Exception as e:
                os.unlink(self._data_path)
                raise RuntimeError(f"HDF5 file creation failed: {e}")
            self._owns_file = True
            h5f = h5py.File(self._data_path, "r")
            self._data = da.from_array(h5f["data"], chunks="auto")  # type: ignore [unused-ignore, attr-defined, no-untyped-call]
        self._finalizer = weakref.finalize(
            self, BaseChannel._finalize_cleanup, self._owns_file, self._data_path
        )

    @property
    def dask_data(self) -> "Array":
        """
        内部で h5py の "data" データセットを dask.array としてラップして返します。
        このプロパティは内部処理用で、ユーザー向けではありません。
        """
        with self._lock:
            if self._is_closed:
                raise RuntimeError("Channel is closed")
            return self._data

    @property
    def data(self) -> NDArrayReal:
        """
        ユーザー向けのプロパティです。内部の dask.array を compute() して、
        常に numpy 配列として返します。
        """
        if self.dask_data is None:
            raise RuntimeError("No data source available")
        data: NDArrayReal = np.array(self.dask_data)
        return data

    @property
    def sampling_rate(self) -> int:
        """
        サンプリング周波数を返します。
        """
        return self._sampling_rate

    @property
    def label(self) -> str:
        """
        ラベルを返します。
        """
        return self._label

    @property
    def unit(self) -> str:
        """
        単位を返します。
        """
        return self._unit

    @unit.setter
    def unit(self, unit: str) -> None:
        """
        単位を設定します。
        """
        self._unit = unit
        self._ref = util.unit_to_ref(unit)

    @property
    def metadata(self) -> dict[str, Any]:
        """
        メタデータを返します。
        """
        return self._metadata

    @property
    def ref(self) -> float:
        """
        参照値を返します。
        """
        return self._ref

    @staticmethod
    def _finalize_cleanup(owns_file: bool, data_path: Union[str, Path, None]) -> None:
        if owns_file and data_path is not None:
            try:
                os.unlink(data_path)
                logger.debug(f"Temporary HDF5 file {data_path} deleted in finalizer.")
            except Exception as e:
                logger.warning(f"Failed to delete temporary HDF5 file {data_path}: {e}")

    def get_previous(self) -> Optional["BaseChannel"]:
        """
        処理前のオブジェクト（元の状態）を返します。
        """
        return self.previous

    def close(self) -> None:
        with self._lock:
            if self._is_closed:
                return
            del self._data
            if self._finalizer.alive:
                self._finalizer()
            self._is_closed = True
            self._data_path = None

    @classmethod
    def from_channel(cls: type[T], org: "BaseChannel", **kwargs: Any) -> T:
        """
        Create a new channel instance based on an existing channel.
        Keyword arguments can override or extend the original attributes.
        """
        data = kwargs.pop("data", org.data)
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array")
        sampling_rate = kwargs.pop("sampling_rate", org.sampling_rate)
        if not isinstance(sampling_rate, int):
            raise TypeError("Sampling rate must be an integer")

        label = kwargs.pop("label", org.label)
        if not isinstance(label, str):
            raise TypeError("Label must be a string")
        unit = kwargs.pop("unit", org.unit)
        if not isinstance(unit, str):
            raise TypeError("Unit must be a string")
        metadata = kwargs.pop("metadata", org.metadata.copy())
        if not isinstance(metadata, dict):
            raise TypeError("Metadata must be a dictionary")

        return cls(
            data=data,
            sampling_rate=sampling_rate,
            label=label,
            unit=unit,
            metadata=metadata,
            previous=org,
            **kwargs,
        )

    # @abstractmethod
    # def plot(
    #     self,
    #     ax: Optional["Axes"] = None,
    #     title: Optional[str] = None,
    #     plot_kwargs: Optional[dict[str, Any]] = None,
    # ) -> "Axes":
    #     """
    #     データをプロットします。派生クラスで実装が必要です。
    #     """
    #     pass

    def __repr__(self) -> str:
        state = "closed" if self._is_closed else "open"
        data_status = "loaded" if self._data is not None else "not loaded"
        return (
            f"<{self.__class__.__name__} label='{self.label}' unit='{self.unit}' "
            f"sampling_rate={self._sampling_rate} state={state}, data {data_status}>"
        )
