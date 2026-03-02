from typing import Any

from pydantic import BaseModel, Field  # Direct import from pydantic

from wandas.utils.util import unit_to_ref


class ChannelMetadata(BaseModel):
    """
    Data class for storing channel metadata
    """

    label: str = ""
    unit: str = ""
    ref: float = 1.0
    # Additional metadata for extensibility
    extra: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data: Any):
        super().__init__(**data)
        # unitが指定されていてrefがデフォルト値ならunit_to_refで自動設定
        if self.unit and ("ref" not in data or data.get("ref", 1.0) == 1.0):
            self.ref = unit_to_ref(self.unit)

    def __setattr__(self, name: str, value: Any) -> None:
        """Override setattr to update ref when unit is changed directly"""
        super().__setattr__(name, value)
        # Only proceed if unit is being set to a non-empty value
        if name == "unit" and value and isinstance(value, str):
            super().__setattr__("ref", unit_to_ref(value))

    @property
    def label_value(self) -> str:
        """Get the label value"""
        return self.label

    @property
    def unit_value(self) -> str:
        """Get the unit value"""
        return self.unit

    @property
    def ref_value(self) -> float:
        """Get the ref value"""
        return self.ref

    @property
    def extra_data(self) -> dict[str, Any]:
        """Get the extra metadata dictionary"""
        return self.extra

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
            self.ref = unit_to_ref(value)
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
