import numbers
from collections.abc import Iterator
from typing import TYPE_CHECKING, Generic, TypeVar, Union

if TYPE_CHECKING:
    from wandas.core.base_channel import BaseChannel


# Generic type variable constrained to types that have a 'label' attribute
ChannelT = TypeVar("ChannelT", bound="BaseChannel")


class ChannelAccessMixin(Generic[ChannelT]):
    """
    A mixin class providing channel access functionality for container classes.

    This mixin provides standard implementation of __getitem__ and __setitem__
    to access channels by either name or index.

    Classes using this mixin must implement:
    - self._channels: list containing channel objects
    - self.channel_dict: dictionary mapping channel labels to channel objects

    Type parameter:
    - ChannelT: Type of channel objects, must have a 'label' attribute
    """

    _channels: list[ChannelT]
    _channel_dict: dict[str, ChannelT]

    @property
    def channels(self) -> list[ChannelT]:
        """
        チャンネルのリストを返します。
        """
        return self._channels

    @property
    def channel_dict(self) -> dict[str, ChannelT]:
        """
        チャンネルのラベルをキーとして、チャンネルオブジェクトを格納する辞書を返します。
        """
        return self._channel_dict

    # forでループを回すためのメソッド
    def __iter__(self) -> Iterator[ChannelT]:
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, key: Union[str, int]) -> ChannelT:
        """
        チャンネル名またはインデックスでチャンネルを取得するためのメソッド。

        Parameters:
            key (str or int): チャンネルの名前（label）またはインデックス番号。

        Returns:
            Channel: 対応するチャンネル。
        """
        if isinstance(key, str):
            # チャンネル名でアクセス
            if key not in self.channel_dict:
                raise KeyError(f"Channel '{key}' not found.")
            return self.channel_dict[key]
        elif isinstance(key, numbers.Integral):
            # インデックス番号でアクセス
            if key < 0 or key >= len(self._channels):
                raise IndexError(f"Channel index {key} out of range.")
            return self._channels[key]
        else:
            raise TypeError(
                "Key must be either a string (channel name) or an integer "
                "(channel index)."
            )

    def __setitem__(self, key: Union[str, int], value: ChannelT) -> None:
        """
        チャンネル名またはインデックスでチャンネルを設定するためのメソッド。

        Parameters:
            key (str or int): チャンネルの名前（label）またはインデックス番号。
            value (Channel): 設定するチャンネル。
        """
        if isinstance(key, str):
            # チャンネル名でアクセス
            if key not in self.channel_dict:
                raise KeyError(f"Channel '{key}' not found.")
            self._channels[self._channels.index(self.channel_dict[key])] = value
            self.channel_dict[key] = value
        elif isinstance(key, numbers.Integral):
            # インデックス番号でアクセス
            if key < 0 or key >= len(self._channels):
                raise IndexError(f"Channel index {key} out of range.")
            self._channels[key] = value
            self.channel_dict[value.label] = value
        else:
            raise TypeError(
                "Key must be either a string (channel name) or an integer "
                "(channel index)."
            )

    def __len__(self) -> int:
        """
        チャンネルのデータ長を返します。
        """
        return len(self._channels)
