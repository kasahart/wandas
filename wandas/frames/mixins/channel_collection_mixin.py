"""
ChannelCollectionMixin: ChannelFrame系のch追加・削除共通機能
"""

from typing import TYPE_CHECKING, Any, Literal, Optional, TypeVar, Union

import dask.array as da
import numpy as np

if TYPE_CHECKING:
    pass

T = TypeVar("T", bound="ChannelCollectionMixin")


class ChannelCollectionMixin:
    def add_channel(
        self: T,
        data: Union[np.ndarray[Any, Any], da.Array, T],
        label: Optional[str] = None,
        align: Literal["strict", "pad", "truncate"] = "strict",
        suffix_on_dup: Optional[str] = None,
        inplace: bool = False,
        **kwargs: Any,
    ) -> T:
        """
        チャンネルを追加する
        Args:
            data: 追加するチャンネル（1ch ndarray/dask/ChannelFrame）
            label: 追加チャンネルのラベル
            align: 長さ不一致時の挙動
            suffix_on_dup: ラベル重複時のsuffix
            inplace: Trueで自己書換え
        Returns:
            新しいFrame or self
        Raises:
            ValueError, TypeError
        """
        raise NotImplementedError("add_channel()はサブクラスで実装してください")

    def remove_channel(
        self: T,
        key: Union[int, str],
        inplace: bool = False,
    ) -> T:
        """
        チャンネルを削除する
        Args:
            key: 削除対象（index or label）
            inplace: Trueで自己書換え
        Returns:
            新しいFrame or self
        Raises:
            ValueError, KeyError, IndexError
        """
        raise NotImplementedError("remove_channel()はサブクラスで実装してください")
