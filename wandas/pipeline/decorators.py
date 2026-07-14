"""Public-operation semantic capture backed by Recipe registry entries."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from typing import Any, ParamSpec, TypeVar, cast

from wandas.pipeline.registry import ParamValidator, RecipeHandler, RecipeOperation
from wandas.processing.semantic import (
    InputBinding,
    LineageNode,
    active_semantic_lineage,
    freeze_params,
    invoke_semantic,
)

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True)
class OperationCapture:
    """Actual bindings, parents, and params selected at public call entry."""

    bindings: tuple[InputBinding, ...]
    parents: tuple[LineageNode | None, ...]
    params: Mapping[str, Any]
    recipe_error: str | None = None


CaptureResolver = Callable[[tuple[Any, ...], Mapping[str, Any]], OperationCapture]


def _bound_arguments(signature: inspect.Signature, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    bound = signature.bind(*args, **kwargs)
    arguments = dict(bound.arguments)
    receiver_name = next(iter(signature.parameters))
    arguments.pop(receiver_name, None)
    for name, parameter in signature.parameters.items():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD and name in arguments:
            arguments.update(cast(Mapping[str, Any], arguments.pop(name)))
    return arguments


def _unary_capture(args: tuple[Any, ...], params: Mapping[str, Any]) -> OperationCapture:
    if not args or not isinstance(getattr(args[0], "lineage", None), LineageNode):
        raise TypeError("Recipe semantic methods require a Frame receiver")
    return OperationCapture(
        (InputBinding("frame", "frame"),),
        (args[0].lineage,),
        params,
    )


def recipe_operation(
    operation_id: str,
    *,
    version: int = 1,
    bindings: tuple[InputBinding, ...] = (InputBinding("frame", "frame"),),
    binding_patterns: tuple[tuple[InputBinding, ...], ...] | None = None,
    capture: CaptureResolver | None = None,
    handler: RecipeHandler | None = None,
    validate_params: ParamValidator | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Declare one Recipe contract and capture it at the public call boundary."""
    patterns = binding_patterns or (bindings,)

    def decorate(method: Callable[P, R]) -> Callable[P, R]:
        signature = inspect.signature(method)
        capture_resolver = capture or _unary_capture

        @wraps(method)
        def semantic_call(*args: P.args, **kwargs: P.kwargs) -> R:
            if active_semantic_lineage() is not None:
                return method(*args, **kwargs)
            params = _bound_arguments(signature, cast(tuple[Any, ...], args), kwargs)
            captured = capture_resolver(cast(tuple[Any, ...], args), params)
            try:
                frozen = freeze_params(captured.params)
                recipe_error = captured.recipe_error
            except (TypeError, ValueError) as exc:
                frozen = freeze_params({"unsupported_parameters": type(exc).__name__})
                recipe_error = f"Public arguments are not portable Recipe values: {exc}"
            operation = __operation_for_capture(captured.bindings, frozen)
            lineage = LineageNode(operation, captured.parents, recipe_error=recipe_error)
            return cast(R, invoke_semantic(method, lineage, *args, **kwargs))

        def default_handler(inputs: tuple[Any, ...], params: Mapping[str, Any]) -> Any:
            if len(inputs) != 1:
                raise TypeError(f"Default Recipe method handler requires one input, got {len(inputs)}")
            return semantic_call(inputs[0], **dict(params))

        definition = RecipeOperation(
            operation_id,
            version,
            patterns,
            handler or default_handler,
            validate_params or (lambda _params: None),
        )

        def __operation_for_capture(actual_bindings: tuple[InputBinding, ...], params: Any) -> Any:
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
    """Return the registry entry attached by ``@recipe_operation``."""
    definition = getattr(value, "__wandas_recipe_operation__", None)
    if not isinstance(definition, RecipeOperation):
        raise TypeError("Object has no Recipe operation declaration")
    return definition


__all__ = ["OperationCapture", "recipe_definition", "recipe_operation"]
