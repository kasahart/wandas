import copy as _copy
from typing import Any

from pydantic import BaseModel, Field  # Direct import from pydantic

from wandas.utils.util import unit_to_ref


class FrameMetadata(dict[str, Any]):
    """
    Frame-level metadata with explicit source file tracking.

    Behaves like a regular dict for backward compatibility while also
    storing the path or name of the file the frame was loaded from.

    Parameters
    ----------
    *args : Any
        Positional arguments forwarded to :class:`dict`.
    source_file : str | None, optional
        Path or name of the source file.
    **kwargs : Any
        Keyword arguments forwarded to :class:`dict`.

    Examples
    --------
    >>> meta = FrameMetadata({"gain": 0.5}, source_file="audio.wav")
    >>> meta.source_file
    'audio.wav'
    >>> meta["gain"]
    0.5
    """

    source_file: str | None

    def __init__(
        self,
        *args: Any,
        source_file: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.source_file = source_file

    def copy(self) -> "FrameMetadata":
        """Return a shallow copy that preserves *source_file*."""
        result = FrameMetadata(super().copy())
        result.source_file = self.source_file
        return result

    def merged(self, **kwargs: Any) -> "FrameMetadata":
        """Return a copy with additional key-value pairs merged in."""
        result = self.copy()
        result.update(kwargs)
        return result

    def __copy__(self) -> "FrameMetadata":
        return self.copy()

    def __deepcopy__(self, memo: dict[int, Any]) -> "FrameMetadata":
        result = FrameMetadata(_copy.deepcopy(dict(self), memo))
        result.source_file = _copy.deepcopy(self.source_file, memo)
        return result

    def __repr__(self) -> str:
        return f"FrameMetadata({dict.__repr__(self)}, source_file={self.source_file!r})"


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
