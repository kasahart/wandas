import numpy as np
import pytest

from wandas.core.base_channel import BaseChannel
from wandas.core.channel_access_mixin import ChannelAccessMixin


class DummyChannel(BaseChannel):
    pass


class TestChannelAccessMixin:
    class TestableChannelAccess(ChannelAccessMixin[DummyChannel]):
        def __init__(self) -> None:
            self._channels: list[DummyChannel] = []
            self._channel_dict: dict[str, DummyChannel] = {}

    def test_append_valid_channel(self) -> None:
        test_signal = np.array([1, 2, 3])
        fs = 100
        channel = DummyChannel(test_signal, fs, "ch1")

        access = self.TestableChannelAccess()
        access.append(channel)
        assert access.channels == [channel]
        assert access.channel_dict["ch1"] == channel

    def test_append_duplicate_channel_raises_key_error(self) -> None:
        test_signal = np.array([1, 2, 3])
        fs = 100
        channel = DummyChannel(test_signal, fs, "ch1")
        access = self.TestableChannelAccess()
        access.append(channel)
        with pytest.raises(KeyError):
            access.append(channel)

    def test_remove_valid_channel(self) -> None:
        test_signal = np.array([1, 2, 3])
        fs = 100
        channel = DummyChannel(test_signal, fs, "ch1")

        access = self.TestableChannelAccess()
        access.append(channel)
        access.remove(channel)
        assert len(access.channels) == 0
        assert "ch1" not in access.channel_dict

    def test_remove_non_existent_channel_raises_key_error(self) -> None:
        test_signal = np.array([1, 2, 3])
        fs = 100
        channel = DummyChannel(test_signal, fs, "ch1")

        access = self.TestableChannelAccess()
        with pytest.raises(KeyError):
            access.remove(channel)
