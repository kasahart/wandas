"""Private warning-aware containers for the v0.7 Frame mutation bridge."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Iterable
from typing import Any, SupportsIndex, cast, overload


def _warn(message: str) -> None:
    warnings.warn(message, DeprecationWarning, stacklevel=4)


def wrap_mutable(value: Any, message: str) -> Any:
    """Defensively wrap nested dictionaries and lists with one warning policy."""
    if isinstance(value, _DeprecatedDict | _DeprecatedList):
        return copy.deepcopy(value)
    if isinstance(value, dict):
        return _DeprecatedDict(value, message)
    if isinstance(value, list):
        return _DeprecatedList(value, message)
    return copy.deepcopy(value)


def is_wrapped_mutable(value: Any) -> bool:
    """Return whether *value* already belongs to a warning-aware Frame view."""
    return isinstance(value, _DeprecatedDict | _DeprecatedList)


class _DeprecatedDict(dict[str, Any]):
    def __init__(self, value: dict[str, Any], message: str) -> None:
        self._message = message
        dict.__init__(self, ((key, wrap_mutable(item, message)) for key, item in value.items()))

    def __deepcopy__(self, memo: dict[int, Any]) -> _DeprecatedDict:
        copied = _DeprecatedDict({}, self._message)
        memo[id(self)] = copied
        dict.update(copied, ((copy.deepcopy(key, memo), copy.deepcopy(value, memo)) for key, value in self.items()))
        return copied

    def __setitem__(self, key: str, value: Any) -> None:
        _warn(self._message)
        dict.__setitem__(self, key, wrap_mutable(value, self._message))

    def __delitem__(self, key: str) -> None:
        _warn(self._message)
        dict.__delitem__(self, key)

    def clear(self) -> None:
        _warn(self._message)
        dict.clear(self)

    def pop(self, key: str, default: Any = ...) -> Any:
        _warn(self._message)
        return dict.pop(self, key) if default is ... else dict.pop(self, key, default)

    def popitem(self) -> tuple[str, Any]:
        _warn(self._message)
        return cast(tuple[str, Any], dict.popitem(self))

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key in self:
            return dict.__getitem__(self, key)
        _warn(self._message)
        wrapped = wrap_mutable(default, self._message)
        dict.__setitem__(self, key, wrapped)
        return wrapped

    def update(self, *args: Any, **kwargs: Any) -> None:
        _warn(self._message)
        incoming = dict(*args, **kwargs)
        dict.update(self, ((key, wrap_mutable(value, self._message)) for key, value in incoming.items()))

    def __ior__(self, other: Any) -> _DeprecatedDict:
        self.update(other)
        return self


class _DeprecatedList(list[Any]):
    def __init__(self, value: list[Any], message: str) -> None:
        self._message = message
        list.__init__(self, (wrap_mutable(item, message) for item in value))

    def __deepcopy__(self, memo: dict[int, Any]) -> _DeprecatedList:
        copied = _DeprecatedList([], self._message)
        memo[id(self)] = copied
        list.extend(copied, (copy.deepcopy(value, memo) for value in self))
        return copied

    @overload
    def __setitem__(self, key: SupportsIndex, value: Any) -> None: ...

    @overload
    def __setitem__(self, key: slice, value: Any) -> None: ...

    def __setitem__(self, key: SupportsIndex | slice, value: Any) -> None:
        _warn(self._message)
        wrapped = (
            [wrap_mutable(item, self._message) for item in value]
            if isinstance(key, slice)
            else wrap_mutable(value, self._message)
        )
        list.__setitem__(self, key, wrapped)

    def __delitem__(self, key: SupportsIndex | slice) -> None:
        _warn(self._message)
        list.__delitem__(self, key)

    def append(self, value: Any) -> None:
        _warn(self._message)
        list.append(self, wrap_mutable(value, self._message))

    def extend(self, values: Iterable[Any]) -> None:
        _warn(self._message)
        snapshot = list(values)
        list.extend(self, (wrap_mutable(value, self._message) for value in snapshot))

    def __iadd__(self, values: Iterable[Any]) -> _DeprecatedList:
        self.extend(values)
        return self

    def __imul__(self, value: SupportsIndex) -> _DeprecatedList:
        _warn(self._message)
        list.__imul__(self, value)
        return self

    def insert(self, index: SupportsIndex, value: Any) -> None:
        _warn(self._message)
        list.insert(self, index, wrap_mutable(value, self._message))

    def pop(self, index: SupportsIndex = -1) -> Any:
        _warn(self._message)
        return list.pop(self, index)

    def remove(self, value: Any) -> None:
        _warn(self._message)
        list.remove(self, value)

    def clear(self) -> None:
        _warn(self._message)
        list.clear(self)

    def reverse(self) -> None:
        _warn(self._message)
        list.reverse(self)

    def sort(self, *args: Any, **kwargs: Any) -> None:
        _warn(self._message)
        list.sort(self, *args, **kwargs)
