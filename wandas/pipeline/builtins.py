"""Built-in Recipe declarations collected without mutable registration."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from wandas.pipeline.registry import RecipeOperation


def _declared_operations(owner: type[Any]) -> Iterable[RecipeOperation]:
    """Yield Recipe declarations attached directly to one public API owner."""
    for member in vars(owner).values():
        definition = getattr(member, "__wandas_recipe_operation__", None)
        if isinstance(definition, RecipeOperation):
            yield definition


def _frame_owners(frame_type: type[Any], base_frame: type[Any]) -> Iterator[type[Any]]:
    """Yield the classes of one Frame type, base classes before subclasses.

    Each Frame class in the ``BaseFrame`` chain is preceded by the mixins it lists
    directly as bases, in declaration order and each one base-first, so mixin
    operations sit right before the operations of the Frame that adopts them.
    """
    for frame_class in reversed(frame_type.__mro__):
        if not issubclass(frame_class, base_frame):
            continue
        for mixin in frame_class.__bases__:
            if not issubclass(mixin, base_frame):
                yield from (owner for owner in reversed(mixin.__mro__) if not issubclass(owner, base_frame))
        yield frame_class


def builtin_recipe_owners() -> tuple[type[Any], ...]:
    """Return the classes that declare built-in Recipe operations, in catalog order.

    The owners are found from the public Frame types exported by
    :mod:`wandas.frames`, walked in export order, so a new Frame type or mixin
    contributes its operations without being listed here.
    """
    import wandas.frames as public_frames
    from wandas.core.base_frame import BaseFrame

    owners: dict[type[Any], None] = {}
    for export_name in public_frames.__all__:
        for owner in _frame_owners(getattr(public_frames, export_name), BaseFrame):
            if owner not in owners and any(True for _ in _declared_operations(owner)):
                owners[owner] = None
    return tuple(owners)


def builtin_recipe_operations() -> tuple[RecipeOperation, ...]:
    """Return all explicitly declared built-in public Frame operations."""
    return tuple(operation for owner in builtin_recipe_owners() for operation in _declared_operations(owner))
