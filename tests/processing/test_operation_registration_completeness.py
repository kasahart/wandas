"""Every built-in Operation class must be reachable through the runtime registry."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import pkgutil
import subprocess
import sys
from typing import Any

import wandas.processing as processing
from wandas.processing.base import AudioOperation

_RULE = (
    "Rule: every built-in Operation class with a `name` resolves through get_operation() "
    "right after `import wandas` (see docs/src/contributing/frame-operation-extensions.md, "
    "step 6 'Register the class')."
)
_FIX = (
    "Fix: in wandas/processing/__init__.py, import the module eagerly (the module ends with "
    "_register_module_operations(__name__)), or add the class to _LAZY_OPERATION_CLASSES or "
    "_PRIVATE_LAZY_OPERATION_PROVIDERS."
)

# Runs in a fresh interpreter, so only modules that ``import wandas`` loads, plus
# lazy providers loaded by get_operation() itself, can register anything.
_FRESH_IMPORT_CHECK = """
import json, sys
import wandas
import wandas.processing
from wandas.processing.base import get_operation

results = []
for name, module, qualname in json.loads(sys.argv[1]):
    try:
        resolved = get_operation(name)
        found = f"{resolved.__module__}.{resolved.__qualname__}"
    except ValueError as error:
        found = repr(error)
    if found != f"{module}.{qualname}":
        results.append(f"{module}.{qualname} (name={name!r}): {found}")
print(json.dumps({"package": wandas.processing.__file__, "unresolved": results}))
"""


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


def _lazy_provider_modules() -> set[str]:
    entries = (*processing._LAZY_OPERATION_CLASSES.values(), *processing._PRIVATE_LAZY_OPERATION_PROVIDERS)
    return {module_name for _, module_name, _ in entries}


def test_processing_modules_define_operation_classes() -> None:
    names = {operation_class.name for operation_class in _operation_classes_declaring_a_name()}

    assert {"highpass_filter", "astype", "custom", "envelope", "fft", "loudness_zwtv"} <= names


def test_every_operation_class_declaring_a_name_resolves_after_fresh_import() -> None:
    lazy_modules = _lazy_provider_modules()
    # Resolve eager names first so a lazy provider import cannot load an
    # unregistered eager module as a side effect and hide it.
    expected = sorted(
        ((cls.name, cls.__module__, cls.__qualname__) for cls in _operation_classes_declaring_a_name()),
        key=lambda entry: (entry[1] in lazy_modules, entry[1], entry[0]),
    )

    result = subprocess.run(
        [sys.executable, "-c", _FRESH_IMPORT_CHECK, json.dumps(expected)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    report = json.loads(result.stdout.strip().splitlines()[-1])
    # The child puts its cwd first on sys.path, so make sure it inspected the
    # same wandas package as this process.
    assert os.path.realpath(report["package"]) == os.path.realpath(processing.__file__), (
        f"Subprocess imported {report['package']}, but the test enumerated {processing.__file__}."
    )
    unresolved = report["unresolved"]

    assert not unresolved, (
        "Operation classes not reachable by name after a fresh `import wandas`:\n"
        + "\n".join(unresolved)
        + f"\n{_RULE}\n{_FIX}"
    )
