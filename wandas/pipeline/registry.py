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
    FrozenTuple,
    ImmutableList,
    InputBinding,
    thaw_value,
)

RecipeHandler = Callable[[tuple[Any, ...], Mapping[str, Any]], Any]
ParamValidator = Callable[[Mapping[str, Any]], None]
BindingParamValidator = Callable[[tuple[InputBinding, ...], Mapping[str, Any]], None]


def _no_param_validation(_params: Mapping[str, Any]) -> None:
    """Accept parameters for operations without a stricter validator."""
    return None


def _no_binding_param_validation(
    _bindings: tuple[InputBinding, ...],
    _params: Mapping[str, Any],
) -> None:
    """Accept parameters that do not depend on the selected binding pattern."""
    return None


def _immutable_value(value: Any) -> Any:
    """Decode one canonical value without exposing mutable containers."""
    if isinstance(value, FrozenList):
        return ImmutableList(_immutable_value(item) for item in value.items)
    if isinstance(value, FrozenTuple):
        return tuple(_immutable_value(item) for item in value.items)
    if isinstance(value, FrozenMap):
        return MappingProxyType({key: _immutable_value(item) for key, item in value.entries})
    if isinstance(value, FrozenNumber):
        return thaw_value(value)
    return value


def immutable_params(params: FrozenMap) -> Mapping[str, Any]:
    """Decode canonical parameters to recursively immutable runtime values.

    Args:
        params: Canonical mapping stored by semantic lineage or a Recipe node.

    Returns:
        A read-only mapping whose nested maps and sequences are also immutable.
    """
    return MappingProxyType({key: _immutable_value(value) for key, value in params.entries})


@dataclass(frozen=True, eq=False)
class RecipeOperation:
    """Complete versioned contract for one portable public operation.

    Operation values use identity-based equality so independently declared handlers are
    never treated as interchangeable merely because their metadata matches.

    Args:
        operation_id: Stable identifier persisted in Recipe nodes.
        version: Positive version of this operation contract.
        binding_patterns: Accepted ordered input-role and input-kind patterns. Runtime
            kind signatures must be unique.
        handler: Replay callback receiving ordered runtime inputs and immutable params.
        validate_params: Callback that rejects invalid decoded parameters.
        validate_binding_params: Callback that rejects parameters invalid for the
            selected input-kind pattern.
    """

    operation_id: str
    version: int
    binding_patterns: tuple[tuple[InputBinding, ...], ...]
    handler: RecipeHandler = field(repr=False)
    validate_params: ParamValidator = field(default=_no_param_validation, repr=False)
    validate_binding_params: BindingParamValidator = field(default=_no_binding_param_validation, repr=False)

    def __post_init__(self) -> None:
        """Snapshot and validate the complete operation contract."""
        object.__setattr__(self, "binding_patterns", tuple(self.binding_patterns))
        if not isinstance(self.operation_id, str) or not self.operation_id.strip():
            raise ValueError("Recipe operation id must be a non-blank string")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("Recipe operation version must be a positive integer")
        if not self.binding_patterns:
            raise ValueError("Recipe operation requires at least one binding pattern")
        if any(not pattern for pattern in self.binding_patterns):
            raise ValueError("Recipe binding patterns must each contain at least one input")
        if not all(
            isinstance(pattern, tuple) and all(isinstance(binding, InputBinding) for binding in pattern)
            for pattern in self.binding_patterns
        ):
            raise TypeError("Recipe binding patterns must contain InputBinding tuples")
        if len(set(self.binding_patterns)) != len(self.binding_patterns):
            raise ValueError("Recipe binding patterns must be unique")
        if any(len({binding.role for binding in pattern}) != len(pattern) for pattern in self.binding_patterns):
            raise ValueError("Recipe binding roles must be unique within each pattern")
        kind_patterns = tuple(tuple(binding.kind for binding in pattern) for pattern in self.binding_patterns)
        if len(set(kind_patterns)) != len(kind_patterns):
            raise ValueError("Recipe binding patterns must have unique input kind signatures")
        if (
            not callable(self.handler)
            or not callable(self.validate_params)
            or not callable(self.validate_binding_params)
        ):
            raise TypeError("Recipe handler and parameter validators must be callable")

    def accepts(self, bindings: tuple[InputBinding, ...]) -> bool:
        """Return whether ``bindings`` exactly match a declared pattern."""
        return bindings in self.binding_patterns

    def invoke(self, inputs: tuple[Any, ...], params: FrozenMap) -> Any:
        """Invoke the replay handler after complete-plan validation.

        Args:
            inputs: Ordered runtime values matching one declared binding pattern.
            params: Canonical parameters already accepted by ``validate_params``.

        Returns:
            The handler's Frame result.
        """
        decoded = immutable_params(params)
        return self.handler(inputs, decoded)


@dataclass(frozen=True)
class RecipeRegistry:
    """Immutable collection of versioned Recipe operation contracts.

    Args:
        operations: Initial operation definitions. Each ``(operation_id, version)``
            pair must be unique.

    Raises:
        TypeError: If an entry is not a :class:`RecipeOperation`.
        ValueError: If two entries use the same identifier and version.
    """

    operations: tuple[RecipeOperation, ...] = ()
    _by_key: Mapping[tuple[str, int], RecipeOperation] = field(init=False, repr=False, compare=False)

    def __init__(self, operations: Iterable[RecipeOperation] = ()) -> None:
        """Snapshot definitions and build a read-only lookup index."""
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
        """Return a new registry containing one additional operation.

        Args:
            operation: Definition to append without changing this registry.

        Returns:
            A new immutable registry.

        Raises:
            ValueError: If the operation identifier and version already exist.
            TypeError: If ``operation`` is not a :class:`RecipeOperation`.
        """
        return RecipeRegistry((*self.operations, operation))

    def require(self, operation_id: str, version: int) -> RecipeOperation:
        """Return an exact registered operation definition.

        Args:
            operation_id: Stable operation identifier.
            version: Required contract version.

        Returns:
            The matching operation definition.

        Raises:
            KeyError: If no exact identifier-version pair is registered.
        """
        try:
            return self._by_key[(operation_id, version)]
        except KeyError as exc:
            raise KeyError(f"Recipe operation is not registered: {operation_id!r} version {version}") from exc


@cache
def default_recipe_registry() -> RecipeRegistry:
    """Return the cached immutable registry of built-in Frame operations.

    Returns:
        A process-wide immutable registry. Use :meth:`RecipeRegistry.with_operation`
        to derive an extension registry without mutating the built-in value.
    """
    from wandas.pipeline.builtins import builtin_recipe_operations

    return RecipeRegistry(builtin_recipe_operations())
