"""Check that declared public exports are importable and documented."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Iterator
from types import ModuleType

import wandas


def _modules() -> Iterator[ModuleType]:
    """Yield Wandas modules so each module-level ``__all__`` is checked."""
    yield wandas
    for module_info in pkgutil.walk_packages(wandas.__path__, prefix="wandas."):
        yield importlib.import_module(module_info.name)


def _callable_docstring_errors(name: str, value: object) -> list[str]:
    """Return missing-docstring errors for an exported callable or class."""
    errors: list[str] = []
    if not (inspect.isclass(value) or inspect.isfunction(value) or inspect.ismethod(value)):
        return errors
    if not inspect.getdoc(value):
        errors.append(f"{name} has no docstring")
    if inspect.isclass(value):
        for member_name, member in value.__dict__.items():
            if member_name.startswith("_"):
                continue
            if isinstance(member, (staticmethod, classmethod)):
                member = member.__func__
            if callable(member) and not inspect.getdoc(member):
                errors.append(f"{name}.{member_name} has no docstring")
    return errors


def check_public_exports() -> list[str]:
    """Return importability and documentation errors for declared exports."""
    errors: list[str] = []
    for module in _modules():
        exported = getattr(module, "__all__", ())
        if not isinstance(exported, (list, tuple)):
            errors.append(f"{module.__name__}.__all__ must be a list or tuple")
            continue
        for name in exported:
            if not isinstance(name, str):
                errors.append(f"{module.__name__}.__all__ contains a non-string name")
                continue
            try:
                value = getattr(module, name)
            except AttributeError:
                errors.append(f"{module.__name__}.{name} is not importable")
                continue
            if not name.startswith("_"):
                errors.extend(_callable_docstring_errors(f"{module.__name__}.{name}", value))
    return errors


def main() -> int:
    """Run the public export check as a command-line program."""
    errors = check_public_exports()
    if errors:
        print("Public export documentation errors:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("Public exports are importable and documented.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
