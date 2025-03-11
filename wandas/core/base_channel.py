# wandas/core/base_channel.py

import logging
import os
import tempfile
import threading
import weakref
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import dask.array as da
import h5py
import numpy as np

from wandas.core import util
from wandas.utils.types import NDArrayReal

if TYPE_CHECKING:
    from dask.array.core import Array
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)


class BaseChannel(ABC):
    __slots__ = (
        "_data",
        "_data_path",
        "_owns_file",
        "_is_closed",
        "_sampling_rate",
        "label",
        "unit",
        "metadata",
        "ref",
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
        self.label = label or ""
        self.unit = unit or ""
        self.metadata = metadata or {}
        self.ref = util.unit_to_ref(self.unit)
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

    @abstractmethod
    def plot(
        self, ax: Optional["Axes"] = None, title: Optional[str] = None
    ) -> tuple["Axes", NDArrayReal]:
        """
        データをプロットします。派生クラスで実装が必要です。
        """
        pass

    def __repr__(self) -> str:
        state = "closed" if self._is_closed else "open"
        data_status = "loaded" if self._data is not None else "not loaded"
        return (
            f"<{self.__class__.__name__} label='{self.label}' unit='{self.unit}' "
            f"sampling_rate={self._sampling_rate} state={state}, data {data_status}>"
        )
