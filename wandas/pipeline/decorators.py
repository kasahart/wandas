"""Public-operation semantic capture backed by Recipe registry entries."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from typing import Any, ParamSpec, TypeVar, cast

from wandas.pipeline.registry import ParamValidator, RecipeHandler, RecipeOperation
from wandas.processing.semantic import (
    FrozenMap,
    InputBinding,
    LineageNode,
    active_semantic_lineage,
    freeze_params,
    freeze_value,
    invoke_semantic,
)

P = ParamSpec("P")
R = TypeVar("R")
_NONPORTABLE_HISTORY_VALUE = FrozenMap((("recipe_portable", False),))


def _freeze_display_params(params: Mapping[str, Any]) -> FrozenMap:
    """Preserve portable top-level values and mark only nonportable siblings."""
    if not all(isinstance(name, str) for name in params):
        raise TypeError("Semantic parameter names must be strings")
    entries = []
    for name in sorted(params):
        try:
            value = freeze_value(params[name])
        except (TypeError, ValueError):
            value = _NONPORTABLE_HISTORY_VALUE
        entries.append((name, value))
    return FrozenMap(tuple(entries))


@dataclass(frozen=True)
class OperationCapture:
    """Semantic inputs and parameters captured at a public method boundary.

    Args:
        bindings: Ordered input roles and runtime kinds for this invocation.
        parents: Lineage parent for each binding; array bindings use ``None``.
        params: Call intent to snapshot into semantic lineage. Nonportable top-level
            values become display-only markers and make Recipe extraction fail.
        recipe_error: Optional explanation that makes later Recipe extraction fail
            atomically while retaining display history for the public call.
    """

    bindings: tuple[InputBinding, ...]
    parents: tuple[LineageNode | None, ...]
    params: Mapping[str, Any]
    recipe_error: str | None = None


CaptureResolver = Callable[[tuple[Any, ...], Mapping[str, Any]], OperationCapture]


def _bound_arguments(signature: inspect.Signature, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Bind one public call and return parameters without its Frame receiver."""
    bound = signature.bind(*args, **kwargs)
    arguments = dict(bound.arguments)
    receiver_name = next(iter(signature.parameters))
    arguments.pop(receiver_name, None)
    for name, parameter in signature.parameters.items():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD and name in arguments:
            arguments.update(cast(Mapping[str, Any], arguments.pop(name)))
    return arguments


def _unary_capture(binding: InputBinding) -> CaptureResolver:
    """Build the default capture resolver for one Frame input."""

    def capture(args: tuple[Any, ...], params: Mapping[str, Any]) -> OperationCapture:
        """Capture the receiver lineage and already-bound call parameters."""
        if not args or not isinstance(getattr(args[0], "lineage", None), LineageNode):
            raise TypeError("Recipe semantic methods require a Frame receiver")
        return OperationCapture((binding,), (args[0].lineage,), params)

    return capture


def recipe_operation(
    operation_id: str,
    *,
    version: int = 1,
    bindings: tuple[InputBinding, ...] | None = None,
    binding_patterns: tuple[tuple[InputBinding, ...], ...] | None = None,
    capture: CaptureResolver | None = None,
    handler: RecipeHandler | None = None,
    validate_params: ParamValidator | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Declare and capture one portable public Frame operation.

    The returned decorator attaches a :class:`RecipeOperation` declaration to the
    wrapped method and records one authoritative semantic lineage node per public call.
    The default unary replay handler delegates to the wrapped public method. An
    extension that supplies an explicit handler is responsible for keeping that handler
    behaviorally equivalent to its public method.

    Args:
        operation_id: Stable serialized identifier for the public operation.
        version: Positive contract version for ``operation_id``.
        bindings: Ordered input contract for one invocation shape. Defaults to one
            Frame binding named ``frame``.
        binding_patterns: Alternative accepted input contracts for operations that
            support more than one shape, such as Frame-or-array operands. Mutually
            exclusive with ``bindings``.
        capture: Callback that derives actual bindings, lineage parents, and portable
            parameters from a public call. Required for non-unary contracts.
        handler: Replay callback receiving ordered runtime inputs and an immutable
            parameter mapping. Required for non-unary contracts; unary operations use
            the wrapped method by default.
        validate_params: Optional validator called on immutable decoded parameters
            during complete-plan validation.

    Returns:
        A method decorator that preserves the wrapped signature and exposes its
        Recipe declaration through :func:`recipe_definition`.

    Raises:
        TypeError: If ``capture`` is supplied but is not callable.
        ValueError: If declarations conflict, binding patterns are empty, a required
            capture or handler is missing, or the method signature is unsupported.
    """
    if bindings is not None and binding_patterns is not None:
        raise ValueError("Specify either bindings or binding_patterns, not both")
    if capture is not None and not callable(capture):
        raise TypeError("Recipe capture must be callable")
    declared_bindings = bindings if bindings is not None else (InputBinding("frame", "frame"),)
    patterns = binding_patterns if binding_patterns is not None else (declared_bindings,)
    if not patterns:
        raise ValueError("Recipe operation requires at least one binding pattern")
    unary_frame_contract = (
        len(patterns) == 1
        and isinstance(patterns[0], tuple)
        and len(patterns[0]) == 1
        and isinstance(patterns[0][0], InputBinding)
        and patterns[0][0].kind == "frame"
    )
    if handler is None and not unary_frame_contract:
        raise ValueError("Recipe operations with non-unary Frame bindings require an explicit handler")
    if capture is None and not unary_frame_contract:
        raise ValueError("Recipe operations with non-unary Frame bindings require an explicit capture")

    def decorate(method: Callable[P, R]) -> Callable[P, R]:
        """Attach semantic capture and a registry-ready operation definition."""
        signature = inspect.signature(method)
        parameters = tuple(signature.parameters.values())
        if not parameters or parameters[0].kind not in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            raise ValueError("Recipe operation methods require a positional Frame receiver")
        method_parameters = parameters[1:]
        if handler is None and any(
            parameter.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL}
            for parameter in method_parameters
        ):
            raise ValueError("Default Recipe handlers do not support positional-only or variadic positional parameters")
        capture_resolver = capture if capture is not None else _unary_capture(patterns[0][0])

        @wraps(method)
        def semantic_call(*args: P.args, **kwargs: P.kwargs) -> R:
            """Capture a public call once, then execute it under that lineage."""
            if active_semantic_lineage() is not None:
                return method(*args, **kwargs)
            params = _bound_arguments(signature, cast(tuple[Any, ...], args), kwargs)
            captured = capture_resolver(cast(tuple[Any, ...], args), params)
            try:
                frozen = freeze_params(captured.params)
                recipe_error = captured.recipe_error
            except (TypeError, ValueError) as exc:
                frozen = _freeze_display_params(captured.params)
                recipe_error = captured.recipe_error or f"Public arguments are not portable Recipe values: {exc}"
            operation = __operation_for_capture(captured.bindings, frozen)
            lineage = LineageNode(operation, captured.parents, recipe_error=recipe_error)
            return cast(R, invoke_semantic(method, lineage, *args, **kwargs))

        def default_handler(inputs: tuple[Any, ...], params: Mapping[str, Any]) -> Any:
            """Replay a unary operation through its decorated public method."""
            return semantic_call(inputs[0], **dict(params))

        definition = RecipeOperation(
            operation_id,
            version,
            patterns,
            handler if handler is not None else default_handler,
            validate_params if validate_params is not None else (lambda _params: None),
        )

        def __operation_for_capture(actual_bindings: tuple[InputBinding, ...], params: Any) -> Any:
            """Create semantic intent after checking captured binding agreement."""
            from wandas.processing.semantic import SemanticOperation

            if not definition.accepts(actual_bindings):
                raise RuntimeError(
                    f"Public Recipe capture produced undeclared bindings for {operation_id!r}: {actual_bindings!r}"
                )
            return SemanticOperation(operation_id, version, actual_bindings, params)

        setattr(semantic_call, "__wandas_recipe_operation__", definition)
        return semantic_call

    return decorate


def recipe_definition(value: object) -> RecipeOperation:
    """Return the operation declaration attached by :func:`recipe_operation`.

    Args:
        value: Decorated public method or an equivalent statically inspected member.

    Returns:
        The immutable operation contract ready to add to a :class:`RecipeRegistry`.

    Raises:
        TypeError: If ``value`` has no Recipe operation declaration.
    """
    definition = getattr(value, "__wandas_recipe_operation__", None)
    if not isinstance(definition, RecipeOperation):
        raise TypeError("Object has no Recipe operation declaration")
    return definition


__all__ = ["OperationCapture", "recipe_definition", "recipe_operation"]
