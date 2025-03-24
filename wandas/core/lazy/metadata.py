import copy  # copyモジュールのインポートが欠けていました
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .channel_frame import ChannelFrame


class ChannelMetadata:
    """単一チャネルのメタデータにアクセスするためのクラス"""

    def __init__(self, owner: "ChannelFrame", channel_idx: int):
        self._owner = owner
        self._channel_idx = channel_idx

    @property
    def label(self) -> str:
        """チャネルのラベルを取得します。"""
        self._owner._validate_channel_idx(self._channel_idx)
        metadata = self._owner._channel_metadata.get(self._channel_idx, {})
        _label: str = metadata.get(
            "label", f"{self._owner.label}_ch{self._channel_idx}"
        )
        return _label

    @label.setter
    def label(self, value: str) -> None:
        """チャネルのラベルを設定します。"""
        self._owner._validate_channel_idx(self._channel_idx)
        if self._channel_idx not in self._owner._channel_metadata:
            self._owner._channel_metadata[self._channel_idx] = {}
        self._owner._channel_metadata[self._channel_idx]["label"] = value

    @property
    def unit(self) -> str:
        """チャネルの単位を取得します。"""
        self._owner._validate_channel_idx(self._channel_idx)
        metadata = self._owner._channel_metadata.get(self._channel_idx, {})
        _unit: str = metadata.get("unit", "")
        return _unit

    @unit.setter
    def unit(self, value: str) -> None:
        """チャネルの単位を設定します。"""
        self._owner._validate_channel_idx(self._channel_idx)
        if self._channel_idx not in self._owner._channel_metadata:
            self._owner._channel_metadata[self._channel_idx] = {}
        self._owner._channel_metadata[self._channel_idx]["unit"] = value

    def __getitem__(self, key: str) -> Any:
        """任意のメタデータ項目を取得します。"""
        self._owner._validate_channel_idx(self._channel_idx)
        metadata = self._owner._channel_metadata.get(self._channel_idx, {})
        return metadata.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """任意のメタデータ項目を設定します。"""
        self._owner._validate_channel_idx(self._channel_idx)
        if self._channel_idx not in self._owner._channel_metadata:
            self._owner._channel_metadata[self._channel_idx] = {}
        self._owner._channel_metadata[self._channel_idx][key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """指定したキーのメタデータを取得し、ない場合はデフォルト値を返します。"""
        self._owner._validate_channel_idx(self._channel_idx)
        metadata = self._owner._channel_metadata.get(self._channel_idx, {})
        return metadata.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """指定したキーのメタデータを設定します。"""
        self.__setitem__(key, value)

    def all(self) -> dict[str, Any]:
        """チャネルのすべてのメタデータを辞書として返します。"""
        self._owner._validate_channel_idx(self._channel_idx)
        return self._owner._channel_metadata.get(self._channel_idx, {}).copy()


class ChannelMetadataCollection:
    """すべてのチャネルのメタデータにアクセスするためのクラス"""

    def __init__(self, owner: "ChannelFrame"):
        self._owner = owner

    def __getitem__(self, channel_idx: int) -> ChannelMetadata:
        """指定したチャネルのメタデータにアクセスするためのオブジェクトを返します。"""
        self._owner._validate_channel_idx(channel_idx)
        return ChannelMetadata(self._owner, channel_idx)

    def get_all(self) -> dict[int, dict[str, Any]]:
        """すべてのチャネルのメタデータを辞書として返します。"""
        return copy.deepcopy(self._owner._channel_metadata)

    def set_all(self, metadata: dict[int, dict[str, Any]]) -> None:
        """すべてのチャネルのメタデータを設定します。"""
        valid_indices = {idx for idx in metadata if 0 <= idx < self._owner.n_channels}
        for idx in valid_indices:
            self._owner._channel_metadata[idx] = copy.deepcopy(metadata[idx])
