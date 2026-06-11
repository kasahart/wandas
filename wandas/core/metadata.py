import json
from dataclasses import dataclass, field
from typing import Any

from wandas.utils.util import unit_to_ref


@dataclass
class ChannelMetadata:
    """Metadata for a single channel."""

    label: str = ""
    unit: str = ""
    ref: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)
    _initialized: bool = field(default=False, init=False, repr=False)

    _MODEL_FIELDS = frozenset({"label", "unit", "ref", "extra"})

    def __post_init__(self) -> None:
        if not isinstance(self.label, str):
            raise TypeError("ChannelMetadata label must be a string")
        if not isinstance(self.unit, str):
            raise TypeError("ChannelMetadata unit must be a string")
        if isinstance(self.ref, bool) or not isinstance(self.ref, (int, float)):
            raise TypeError("ChannelMetadata ref must be a number")
        self.ref = float(self.ref)
        if not isinstance(self.extra, dict):
            raise TypeError("ChannelMetadata extra must be a dictionary")
        if self.unit and self.ref == 1.0:
            self.ref = unit_to_ref(self.unit)
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if name == "unit" and getattr(self, "_initialized", False) and value and isinstance(value, str):
            object.__setattr__(self, "ref", unit_to_ref(value))

    def __getitem__(self, key: str) -> Any:
        """Provide dictionary-like behavior."""
        if key in self._MODEL_FIELDS:
            return getattr(self, key)
        return self.extra.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Provide dictionary-like behavior."""
        if key in ("label", "ref"):
            setattr(self, key, value)
        elif key == "unit":
            self.unit = value
        else:
            self.extra[key] = value

    def matches_query(self, query: dict[str, Any]) -> bool:
        """Check whether this channel matches all key-value pairs in query."""
        for key, expected in query.items():
            actual = getattr(self, key, None)
            if actual is None:
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
        """Convert to JSON format."""
        return json.dumps(
            {
                "label": self.label,
                "unit": self.unit,
                "ref": self.ref,
                "extra": self.extra,
            },
            indent=4,
        )

    @classmethod
    def from_json(cls, json_data: str) -> "ChannelMetadata":
        """Convert from JSON format."""
        try:
            data = json.loads(json_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid ChannelMetadata JSON: {e}") from e
        if not isinstance(data, dict):
            raise ValueError("ChannelMetadata JSON must decode to an object")
        try:
            return cls(**data)
        except TypeError as e:
            raise ValueError(f"Invalid ChannelMetadata JSON object: {e}") from e
