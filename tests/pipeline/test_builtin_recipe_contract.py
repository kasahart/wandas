"""Completeness contracts for built-in Recipe declaration owners."""

from __future__ import annotations

import inspect
from collections import Counter
from collections.abc import Iterable
from typing import Any

import wandas.frames as public_frames
from wandas.core.base_frame import BaseFrame
from wandas.pipeline.builtins import builtin_recipe_operations
from wandas.pipeline.registry import RecipeOperation


def _direct_recipe_operations(owner: type[Any]) -> tuple[RecipeOperation, ...]:
    return tuple(
        definition
        for member in vars(owner).values()
        if isinstance((definition := getattr(member, "__wandas_recipe_operation__", None)), RecipeOperation)
    )


def _reachable_recipe_owners() -> tuple[type[Any], ...]:
    owners: list[type[Any]] = []
    seen: set[type[Any]] = set()
    for export_name in public_frames.__all__:
        frame_type = getattr(public_frames, export_name)
        if not inspect.isclass(frame_type) or not issubclass(frame_type, BaseFrame) or inspect.isabstract(frame_type):
            continue
        for owner in frame_type.__mro__:
            if owner not in seen and _direct_recipe_operations(owner):
                seen.add(owner)
                owners.append(owner)
    return tuple(owners)


def _operation_key(operation: RecipeOperation) -> tuple[str, int]:
    return operation.operation_id, operation.version


def _expanded(counter: Counter[tuple[str, int]]) -> list[str]:
    return [f"{operation_id}@v{version}" for operation_id, version in sorted(counter.elements())]


def _definitions(counter: Counter[RecipeOperation]) -> list[str]:
    return sorted(f"{operation.operation_id}@v{operation.version}" for operation in counter.elements())


def _owner_operations(owners: Iterable[type[Any]]) -> tuple[RecipeOperation, ...]:
    return tuple(operation for owner in owners for operation in _direct_recipe_operations(owner))


def test_builtin_recipe_operations_cover_public_frame_mro_owners_exactly_once() -> None:
    owners = _reachable_recipe_owners()
    reachable = _owner_operations(owners)
    actual = builtin_recipe_operations()
    reachable_definitions = Counter(reachable)
    actual_definitions = Counter(actual)
    missing = reachable_definitions - actual_definitions
    unexpected = actual_definitions - reachable_definitions
    actual_keys = Counter(map(_operation_key, actual))
    duplicate_definitions = Counter(
        {definition: count for definition, count in actual_definitions.items() if count > 1}
    )
    duplicate_keys = Counter({key: count for key, count in actual_keys.items() if count > 1})

    assert not (missing or unexpected or duplicate_definitions or duplicate_keys), (
        "Built-in Recipe declaration drift:\n"
        f"  missing={_definitions(missing)}\n"
        f"  unexpected={_definitions(unexpected)}\n"
        f"  duplicate definitions={_definitions(duplicate_definitions)}\n"
        f"  duplicate IDs={_expanded(duplicate_keys)}\n"
        f"  reachable owners={[owner.__module__ + '.' + owner.__qualname__ for owner in owners]}"
    )


def test_builtin_recipe_operation_order_is_deterministic() -> None:
    first = builtin_recipe_operations()
    second = builtin_recipe_operations()

    assert tuple(map(_operation_key, first)) == tuple(map(_operation_key, second))
