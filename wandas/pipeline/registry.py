"""Immutable operation registry shared by Recipe validation and execution."""

from __future__ import annotations

import copy
import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from functools import cache
from types import MappingProxyType, NoneType, UnionType
from typing import Any, Literal, TypeVar, Union, get_args, get_origin

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


def _no_param_validation(_params: Mapping[str, Any]) -> None:
    """Accept parameters for operations without a stricter validator."""
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


@dataclass(frozen=True)
class RecipeParameterDescription:
    """Machine-readable description of one operation method parameter.

    Args:
        name: Parameter name accepted by the public method.
        kind: :class:`inspect.Parameter` kind name, such as ``"KEYWORD_ONLY"``.
        required: Whether the call must supply a value.
        default: Deep copy of the default Python value, or ``None`` when ``required``
            is true. A default that cannot be deep-copied, such as a lock, is
            returned as the same object.
        annotation: Annotation text as written for postponed annotations, or the
            :func:`inspect.formatannotation` text of an evaluated annotation; ``None``
            when the parameter is unannotated.
        choices: Accepted values when the annotation is a ``Literal`` (optionally
            joined with ``None``); otherwise ``None``.
    """

    name: str
    kind: str
    required: bool
    default: Any
    annotation: str | None
    choices: tuple[Any, ...] | None


@dataclass(frozen=True)
class RecipeOperationDescription:
    """Machine-readable description of one registered Recipe operation.

    Args:
        operation_id: Stable identifier persisted in Recipe nodes.
        version: Operation contract version.
        method: Name of the decorated method, if known. Public calls record
            operations whose method name is public; contracts kept only to replay
            older Recipes use private names such as ``_fft_recipe_v2``.
        binding_patterns: Accepted ordered input-role and input-kind patterns.
        parameters: Parameters of that method after its Frame receiver, or ``None``
            when the operation was declared without a method.
        receiver: Frame classes the method is called on. Subclasses inherit it, so
            ``isinstance(frame, receiver)`` tells whether a Frame accepts this
            operation. Empty when no method is attached or no imported Frame class
            exposes it.
        returns: Frame class of the result; ``"receiver"`` when the result has the
            same type as the Frame the method is called on (such as filters and
            arithmetic); ``None`` when the method does not declare a Frame type.
        extras: Optional ``wandas`` extras the operation needs, using the extra
            names of ``pip install "wandas[<extra>]"``.
    """

    operation_id: str
    version: int
    method: str | None
    binding_patterns: tuple[tuple[InputBinding, ...], ...]
    parameters: tuple[RecipeParameterDescription, ...] | None
    receiver: tuple[type[Any], ...] = ()
    returns: type[Any] | Literal["receiver"] | None = None
    extras: tuple[str, ...] = ()


def _resolved_annotation(annotation: Any, owner: Callable[..., Any]) -> Any:
    """Evaluate a postponed annotation, returning ``None`` when it cannot resolve."""
    if not isinstance(annotation, str):
        return annotation
    try:
        return eval(annotation, getattr(owner, "__globals__", {}))  # noqa: S307
    except Exception:
        return None


def _frame_classes() -> tuple[type[Any], ...]:
    """Return ``BaseFrame`` and every imported subclass, parents first."""
    from wandas.core.base_frame import BaseFrame

    classes: dict[type[Any], None] = {}
    pending: list[type[Any]] = [BaseFrame]
    while pending:
        cls = pending.pop(0)
        if cls not in classes:
            classes[cls] = None
            pending.extend(cls.__subclasses__())
    return tuple(classes)


def _receiver_classes(operation: RecipeOperation, method_name: str) -> tuple[type[Any], ...]:
    """Return the most general Frame classes whose ``method_name`` records ``operation``."""
    owners = [
        cls
        for cls in _frame_classes()
        if getattr(getattr(cls, method_name, None), "__wandas_recipe_operation__", None) is operation
    ]
    return tuple(cls for cls in owners if not any(base in owners for base in cls.__mro__[1:]))


def _returned_frame_class(method: Callable[..., Any]) -> type[Any] | Literal["receiver"] | None:
    """Resolve a method's return annotation to a Frame class or the receiver's type."""
    annotation = inspect.signature(method).return_annotation
    if isinstance(annotation, TypeVar):
        return "receiver"
    frame_classes = _frame_classes()
    if isinstance(annotation, str):
        resolved = _resolved_annotation(annotation, method)
        if isinstance(resolved, TypeVar):
            return "receiver"
        if resolved in frame_classes:
            return resolved
        # Frame modules import other Frame types only for type checking, so match by name.
        matches = [cls for cls in frame_classes if cls.__name__ == annotation]
        return matches[0] if len(matches) == 1 else None
    return annotation if annotation in frame_classes else None


def _copied_default(value: Any) -> Any:
    """Return a deep copy of a default, or the value itself when it cannot be copied."""
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _literal_choices(annotation: Any) -> tuple[Any, ...] | None:
    """Return the closed value set of a ``Literal`` or ``Literal | None`` annotation."""
    origin = get_origin(annotation)
    if origin is Literal:
        return get_args(annotation)
    if origin not in (Union, UnionType):
        return None
    choices: list[Any] = []
    for member in get_args(annotation):
        if member is NoneType:
            choices.append(None)
        elif get_origin(member) is Literal:
            choices.extend(get_args(member))
        else:
            return None
    return tuple(choices)


def _describe_parameter(parameter: inspect.Parameter, owner: Callable[..., Any]) -> RecipeParameterDescription:
    """Describe one signature parameter without changing how calls bind."""
    annotation = parameter.annotation
    text = None
    if annotation is not inspect.Parameter.empty:
        text = annotation if isinstance(annotation, str) else inspect.formatannotation(annotation)
    required = parameter.default is inspect.Parameter.empty and parameter.kind not in {
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    }
    return RecipeParameterDescription(
        name=parameter.name,
        kind=parameter.kind.name,
        required=required,
        default=None if parameter.default is inspect.Parameter.empty else _copied_default(parameter.default),
        annotation=text,
        choices=None if text is None else _literal_choices(_resolved_annotation(annotation, owner)),
    )


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
        method: Decorated method whose calls record this operation. It is used only by
            :meth:`describe`; capture and replay never call it through this field.
        extras: Optional ``wandas`` extras the operation needs, such as
            ``("effects",)``. Reported by :meth:`describe`; not checked at replay.
    """

    operation_id: str
    version: int
    binding_patterns: tuple[tuple[InputBinding, ...], ...]
    handler: RecipeHandler = field(repr=False)
    validate_params: ParamValidator = field(default=_no_param_validation, repr=False)
    method: Callable[..., Any] | None = field(default=None, repr=False)
    extras: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Snapshot and validate the complete operation contract."""
        object.__setattr__(self, "binding_patterns", tuple(self.binding_patterns))
        object.__setattr__(self, "extras", tuple(self.extras))
        if any(not isinstance(extra, str) or not extra.strip() for extra in self.extras):
            raise ValueError("Recipe operation extras must be non-blank strings")
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
        if not callable(self.handler) or not callable(self.validate_params):
            raise TypeError("Recipe handler and parameter validator must be callable")
        if self.method is not None and not callable(self.method):
            raise TypeError("Recipe operation method must be callable")

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

    def describe(self) -> RecipeOperationDescription:
        """Describe this operation's inputs, method parameters, and Frame types.

        Parameters come from the decorated method signature after its Frame receiver,
        so they describe the call that Recipe extraction captures. ``Literal``
        annotations also report their accepted values as ``choices``. The receiver
        comes from the imported Frame classes that expose the method, and the result
        type from the method's return annotation.

        Returns:
            An immutable description; ``parameters`` is ``None`` when no method is
                attached to this operation.
        """
        parameters = None
        method_name = None
        receiver: tuple[type[Any], ...] = ()
        returns: type[Any] | Literal["receiver"] | None = None
        if self.method is not None:
            method_name = getattr(self.method, "__name__", type(self.method).__name__)
            signature = inspect.signature(self.method)
            parameters = tuple(
                _describe_parameter(parameter, self.method) for parameter in tuple(signature.parameters.values())[1:]
            )
            receiver = _receiver_classes(self, method_name)
            returns = _returned_frame_class(self.method)
        return RecipeOperationDescription(
            self.operation_id,
            self.version,
            method_name,
            self.binding_patterns,
            parameters,
            receiver,
            returns,
            self.extras,
        )


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

    def describe(self) -> tuple[RecipeOperationDescription, ...]:
        """Describe every registered operation in registration order.

        Use this catalog to build parameter forms, documentation, or tool schemas
        from the same declarations that Recipe validation and replay use. Released
        contracts kept only to replay older Recipes are included; their ``method``
        is private (one leading underscore, such as ``_fft_recipe_v2``). Public
        calls record the operations whose ``method`` is public, including operator
        methods such as ``__add__``.

        Returns:
            One :class:`RecipeOperationDescription` per registered operation.
        """
        return tuple(operation.describe() for operation in self.operations)


@cache
def default_recipe_registry() -> RecipeRegistry:
    """Return the cached immutable registry of built-in Frame operations.

    Returns:
        A process-wide immutable registry. Use :meth:`RecipeRegistry.with_operation`
            to derive an extension registry without mutating the built-in value.
    """
    from wandas.pipeline.builtins import builtin_recipe_operations

    return RecipeRegistry(builtin_recipe_operations())
