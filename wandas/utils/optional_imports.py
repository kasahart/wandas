from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any


def require_optional_dependency(module_name: str, *, extra: str, feature: str) -> ModuleType:
    """Import an optional dependency or raise an actionable install error."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f'{feature} requires optional dependency {module_name!r}.\nInstall it with: pip install "wandas[{extra}]"'
        ) from exc


def require_optional_attr(module_name: str, attr_name: str, *, extra: str, feature: str) -> Any:
    """Import an attribute from an optional dependency with the same error style."""
    module = require_optional_dependency(module_name, extra=extra, feature=feature)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ImportError(
            f"{feature} requires {module_name!r} to provide attribute {attr_name!r}.\n"
            f'Install or update it with: pip install "wandas[{extra}]"'
        ) from exc
