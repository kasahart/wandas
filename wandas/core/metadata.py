from typing import Any

from pydantic import BaseModel, Field

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
        # Auto-set ref via unit_to_ref when unit is specified and ref is at default
        if self.unit and ("ref" not in data or data.get("ref", 1.0) == 1.0):
            self.ref = unit_to_ref(self.unit)

    def __setattr__(self, name: str, value: Any) -> None:
        """Override setattr to update ref when unit is changed directly"""
        super().__setattr__(name, value)
        # Only proceed if unit is being set to a non-empty value
        if name == "unit" and value and isinstance(value, str):
            super().__setattr__("ref", unit_to_ref(value))

    _MODEL_FIELDS = frozenset({"label", "unit", "ref", "extra"})

    def __getitem__(self, key: str) -> Any:
        """Provide dictionary-like behavior"""
        if key in self._MODEL_FIELDS:
            return getattr(self, key)
        return self.extra.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Provide dictionary-like behavior"""
        if key in ("label", "ref"):
            setattr(self, key, value)
        elif key == "unit":
            self.unit = value  # __setattr__ auto-updates ref via unit_to_ref
        else:
            self.extra[key] = value

    def matches_query(self, query: dict[str, Any]) -> bool:
        """Check whether this channel matches all key-value pairs in *query*.

        Values may be literals (compared with ``==``) or compiled regex
        patterns (matched via ``.search()`` against string attributes).
        """
        for key, expected in query.items():
            actual = getattr(self, key, None)
            if actual is None:
                # Fall back to extra dict
                actual = self.extra.get(key)
                if actual is None:
                    return False

            if hasattr(expected, "search") and callable(expected.search):
                if not (isinstance(actual, str) and expected.search(actual)):
                    return False
            elif actual != expected:
                return False
        return True

    def to_json(self) -> str:
        """Convert to JSON format"""
        return self.model_dump_json(indent=4)

    @classmethod
    def from_json(cls, json_data: str) -> "ChannelMetadata":
        """Convert from JSON format"""
        return cls.model_validate_json(json_data)
