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
        Return the list of channels.

        Returns
        -------
        list
            List of channel objects.
        """
        return self._channels

    @property
    def channel_dict(self) -> dict[str, ChannelT]:
        """
        Return a dictionary that maps channel labels to channel objects.

        Returns
        -------
        dict
            Dictionary with channel labels as keys and channel objects as values.
        """
        return self._channel_dict

    def __iter__(self) -> Iterator[ChannelT]:
        """
        Iterate through channels.

        Returns
        -------
        Iterator
            Iterator of channel objects.
        """
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, key: Union[str, int]) -> ChannelT:
        """
        Get a channel by name or index.

        Parameters
        ----------
        key : str or int
            Channel name (label) or index.

        Returns
        -------
        Channel
            Corresponding channel.

        Raises
        ------
        KeyError
            If the channel name is not found.
        IndexError
            If the channel index is out of range.
        TypeError
            If the key is neither a string nor an integer.
        """
        if isinstance(key, str):
            # Access by channel name
            if key not in self.channel_dict:
                raise KeyError(f"Channel '{key}' not found.")
            return self.channel_dict[key]
        elif isinstance(key, numbers.Integral):
            # Access by index
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
        Set a channel by name or index.

        Parameters
        ----------
        key : str or int
            Channel name (label) or index.
        value : Channel
            Channel to set.

        Raises
        ------
        KeyError
            If the channel name is not found.
        IndexError
            If the channel index is out of range.
        TypeError
            If the key is neither a string nor an integer.
        """
        if isinstance(key, str):
            # Access by channel name
            if key not in self.channel_dict:
                raise KeyError(f"Channel '{key}' not found.")
            self._channels[self._channels.index(self.channel_dict[key])] = value
            self.channel_dict[key] = value
        elif isinstance(key, numbers.Integral):
            # Access by index
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
        Return the number of channels.

        Returns
        -------
        int
            Number of channels.
        """
        return len(self._channels)

    def append(self, channel: ChannelT) -> None:
        """
        Add a channel.

        Parameters
        ----------
        channel : Channel
            Channel to add.

        Raises
        ------
        KeyError
            If a channel with the same label already exists.
        """
        if channel.label in self.channel_dict:
            raise KeyError(f"Channel '{channel.label}' already exists.")
        self._channels.append(channel)
        self.channel_dict[channel.label] = channel

    def remove(self, channel: ChannelT) -> None:
        """
        Remove a channel.

        Parameters
        ----------
        channel : Channel
            Channel to remove.

        Raises
        ------
        KeyError
            If the channel is not found.
        """
        if channel.label not in self.channel_dict:
            raise KeyError(f"Channel '{channel.label}' not found.")
        self._channels.remove(channel)
        del self.channel_dict[channel.label]
