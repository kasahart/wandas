"""Every built-in Operation class must be reachable through the runtime registry."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any

import wandas.processing as processing
from wandas.processing.base import AudioOperation, get_operation


def _operation_classes_declaring_a_name() -> list[type[AudioOperation[Any, Any]]]:
    classes = []
    for module_info in pkgutil.iter_modules(processing.__path__, f"{processing.__name__}."):
        module = importlib.import_module(module_info.name)
        for candidate in vars(module).values():
            if (
                inspect.isclass(candidate)
                and issubclass(candidate, AudioOperation)
                and candidate.__module__ == module.__name__
                and "name" in vars(candidate)
            ):
                classes.append(candidate)
    return classes


def test_processing_modules_define_operation_classes() -> None:
    names = {operation_class.name for operation_class in _operation_classes_declaring_a_name()}

    assert {"highpass_filter", "astype", "custom", "envelope", "fft", "loudness_zwtv"} <= names


def test_every_operation_class_declaring_a_name_resolves_to_itself() -> None:
    unresolved = []
    for operation_class in _operation_classes_declaring_a_name():
        try:
            resolved: object = get_operation(operation_class.name)
        except ValueError as error:
            resolved = error
        if resolved is not operation_class:
            unresolved.append(f"{operation_class.__module__}.{operation_class.__qualname__}: {resolved!r}")

    assert not unresolved, (
        "Operation classes not reachable by name; eager modules call "
        "_register_module_operations(__name__), lazy modules need an entry in "
        "wandas.processing._LAZY_OPERATION_CLASSES or _PRIVATE_LAZY_OPERATION_PROVIDERS:\n" + "\n".join(unresolved)
    )
