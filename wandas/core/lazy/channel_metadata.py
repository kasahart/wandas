from typing import Any

from pydantic import BaseModel, Field


class ChannelMetadata(BaseModel):  # type: ignore[misc]
    """
    チャネルのメタデータを格納するデータクラス
    """

    label: str = ""
    unit: str = ""
    # 拡張性のために追加のメタデータを保存
    extra: dict[str, Any] = Field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        """辞書のような振る舞いを提供"""
        if key == "label":
            return self.label
        elif key == "unit":
            return self.unit
        else:
            return self.extra.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """辞書のような振る舞いを提供"""
        if key == "label":
            self.label = value
        elif key == "unit":
            self.unit = value
        else:
            self.extra[key] = value

    def to_json(self) -> str:
        """JSON形式に変換"""
        json_data: str = self.model_dump_json(indent=4)
        return json_data

    @classmethod
    def from_json(cls, json_data: str) -> "ChannelMetadata":
        """JSON形式から変換"""
        root_model: ChannelMetadata = ChannelMetadata.model_validate_json(json_data)

        return root_model
