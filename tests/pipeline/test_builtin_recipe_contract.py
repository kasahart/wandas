"""Completeness contracts for built-in Recipe declaration owners."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections import Counter
from collections.abc import Iterable
from typing import Any

import wandas.core
import wandas.frames
from wandas.core.base_frame import BaseFrame
from wandas.pipeline.builtins import builtin_recipe_operations, builtin_recipe_owners
from wandas.pipeline.registry import RecipeOperation


def _direct_recipe_operations(owner: type[Any]) -> tuple[RecipeOperation, ...]:
    return tuple(
        definition
        for member in vars(owner).values()
        if isinstance((definition := getattr(member, "__wandas_recipe_operation__", None)), RecipeOperation)
    )


def _reachable_recipe_owners() -> tuple[type[Any], ...]:
    """Find declaring classes by scanning the Frame packages, independent of ``__all__``."""
    owners: list[type[Any]] = []
    for package in (wandas.core, wandas.frames):
        for module_info in pkgutil.walk_packages(package.__path__, f"{package.__name__}."):
            module = importlib.import_module(module_info.name)
            for value in vars(module).values():
                if (
                    inspect.isclass(value)
                    and value.__module__ == module.__name__
                    and value not in owners
                    and _direct_recipe_operations(value)
                ):
                    owners.append(value)
    return tuple(owners)


def _operation_key(operation: RecipeOperation) -> tuple[str, int]:
    return operation.operation_id, operation.version


def _expanded(counter: Counter[tuple[str, int]]) -> list[str]:
    return [f"{operation_id}@v{version}" for operation_id, version in sorted(counter.elements())]


def _definitions(counter: Counter[RecipeOperation]) -> list[str]:
    return sorted(f"{operation.operation_id}@v{operation.version}" for operation in counter.elements())


def _owner_operations(owners: Iterable[type[Any]]) -> tuple[RecipeOperation, ...]:
    return tuple(operation for owner in owners for operation in _direct_recipe_operations(owner))


def test_builtin_recipe_operations_cover_every_declaring_frame_class_exactly_once() -> None:
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


def _owner_positions() -> dict[type[Any], int]:
    owners = builtin_recipe_owners()
    return {owner: index for index, owner in enumerate(owners)}


def test_builtin_recipe_owners_start_with_base_frame() -> None:
    assert builtin_recipe_owners()[0] is BaseFrame


def test_builtin_recipe_owners_list_base_classes_and_mixins_before_the_frame_using_them() -> None:
    positions = _owner_positions()
    misordered = []
    for owner, position in positions.items():
        for ancestor in owner.__mro__[1:]:
            if ancestor in positions and positions[ancestor] > position:
                misordered.append(f"{ancestor.__qualname__} after {owner.__qualname__}")

    assert not misordered, f"Recipe owners out of base-first order: {misordered}"
