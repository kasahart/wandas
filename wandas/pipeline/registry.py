"""Immutable operation registry shared by Recipe validation and execution."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from functools import cache
from types import MappingProxyType
from typing import Any

from wandas.processing.semantic import (
    FrozenList,
    FrozenMap,
    FrozenNumber,
    InputBinding,
    thaw_value,
)

RecipeHandler = Callable[[tuple[Any, ...], Mapping[str, Any]], Any]
ParamValidator = Callable[[Mapping[str, Any]], None]


def _no_param_validation(_params: Mapping[str, Any]) -> None:
    return None


def _immutable_value(value: Any) -> Any:
    if isinstance(value, FrozenList):
        return tuple(_immutable_value(item) for item in value.items)
    if isinstance(value, FrozenMap):
        return MappingProxyType({key: _immutable_value(item) for key, item in value.entries})
    if isinstance(value, FrozenNumber):
        return thaw_value(value)
    return value


def immutable_params(params: FrozenMap) -> Mapping[str, Any]:
    """Decode params to a recursively immutable runtime mapping."""
    return MappingProxyType({key: _immutable_value(value) for key, value in params.entries})


@dataclass(frozen=True)
class RecipeOperation:
    """One complete registered operation contract."""

    operation_id: str
    version: int
    binding_patterns: tuple[tuple[InputBinding, ...], ...]
    handler: RecipeHandler = field(compare=False, repr=False)
    validate_params: ParamValidator = field(default=_no_param_validation, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "binding_patterns", tuple(self.binding_patterns))
        if not isinstance(self.operation_id, str) or not self.operation_id.strip():
            raise ValueError("Recipe operation id must be a non-blank string")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("Recipe operation version must be a positive integer")
        if not self.binding_patterns:
            raise ValueError("Recipe operation requires at least one binding pattern")
        if not all(
            isinstance(pattern, tuple) and all(isinstance(binding, InputBinding) for binding in pattern)
            for pattern in self.binding_patterns
        ):
            raise TypeError("Recipe binding patterns must contain InputBinding tuples")
        if len(set(self.binding_patterns)) != len(self.binding_patterns):
            raise ValueError("Recipe binding patterns must be unique")
        kind_patterns = tuple(tuple(binding.kind for binding in pattern) for pattern in self.binding_patterns)
        if len(set(kind_patterns)) != len(kind_patterns):
            raise ValueError("Recipe binding patterns must have unique input kind signatures")
        if not callable(self.handler) or not callable(self.validate_params):
            raise TypeError("Recipe handler and parameter validator must be callable")

    def accepts(self, bindings: tuple[InputBinding, ...]) -> bool:
        return bindings in self.binding_patterns

    def invoke(self, inputs: tuple[Any, ...], params: FrozenMap) -> Any:
        """Invoke a handler after complete-plan validation."""
        decoded = immutable_params(params)
        return self.handler(inputs, decoded)


@dataclass(frozen=True)
class RecipeRegistry:
    """Persistent-style immutable registry; extensions return a new value."""

    operations: tuple[RecipeOperation, ...] = ()
    _by_key: Mapping[tuple[str, int], RecipeOperation] = field(init=False, repr=False, compare=False)

    def __init__(self, operations: Iterable[RecipeOperation] = ()) -> None:
        normalized = tuple(operations)
        by_key: dict[tuple[str, int], RecipeOperation] = {}
        for operation in normalized:
            if not isinstance(operation, RecipeOperation):
                raise TypeError("RecipeRegistry entries must be RecipeOperation values")
            key = (operation.operation_id, operation.version)
            if key in by_key:
                raise ValueError(f"Recipe operation is already registered: {key!r}")
            by_key[key] = operation
        object.__setattr__(self, "operations", normalized)
        object.__setattr__(self, "_by_key", MappingProxyType(by_key))

    def with_operation(self, operation: RecipeOperation) -> RecipeRegistry:
        """Return a registry containing one additional operation."""
        return RecipeRegistry((*self.operations, operation))

    def require(self, operation_id: str, version: int) -> RecipeOperation:
        try:
            return self._by_key[(operation_id, version)]
        except KeyError as exc:
            raise KeyError(f"Recipe operation is not registered: {operation_id!r} version {version}") from exc


@cache
def default_recipe_registry() -> RecipeRegistry:
    """Return the immutable built-in operation registry."""
    from wandas.pipeline.builtins import builtin_recipe_operations

    return RecipeRegistry(builtin_recipe_operations())
