from typing import Any

from pydantic import BaseModel, Field  # Direct import from pydantic


class ChannelMetadata(BaseModel):
    """
    Data class for storing channel metadata
    """

    label: str = ""
    unit: str = ""
    ref: float = 1.0
    # Additional metadata for extensibility
    extra: dict[str, Any] = Field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        """Provide dictionary-like behavior"""
        if key == "label":
            return self.label
        elif key == "unit":
            return self.unit
        elif key == "ref":
            return self.ref
        else:
            return self.extra.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Provide dictionary-like behavior"""
        if key == "label":
            self.label = value
        elif key == "unit":
            self.unit = value
        elif key == "ref":
            self.ref = value
        else:
            self.extra[key] = value

    def to_json(self) -> str:
        """Convert to JSON format"""
        json_data: str = self.model_dump_json(indent=4)
        return json_data

    @classmethod
    def from_json(cls, json_data: str) -> "ChannelMetadata":
        """Convert from JSON format"""
        root_model: ChannelMetadata = ChannelMetadata.model_validate_json(json_data)

        return root_model
