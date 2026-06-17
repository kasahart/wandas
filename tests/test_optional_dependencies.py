from __future__ import annotations

import importlib
import re

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from wandas.utils.optional_imports import DEPENDENCY_REGISTRY, require_dependency


def _dependency_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9_.-]+", requirement)
    if match is None:
        raise AssertionError(f"Could not parse requirement: {requirement!r}")
    return match.group(0).lower().replace("_", "-")


def _dependency_names(requirements: list[str]) -> set[str]:
    return {_dependency_name(requirement) for requirement in requirements}


def _pyproject() -> dict[str, object]:
    with open("pyproject.toml", "rb") as file:
        return tomllib.load(file)


def test_dependency_registry_matches_pyproject() -> None:
    pyproject = _pyproject()
    project = pyproject["project"]
    assert isinstance(project, dict)

    core_dependencies = _dependency_names(project["dependencies"])
    optional_dependencies = project.get("optional-dependencies", {})
    assert isinstance(optional_dependencies, dict)

    known_extras = set(optional_dependencies) | {"core"}
    for key, spec in DEPENDENCY_REGISTRY.items():
        package_name = spec.package_name.lower().replace("_", "-")
        assert spec.extra in known_extras, key
        if spec.extra == "core":
            assert package_name in core_dependencies, key
        else:
            extra_dependencies = _dependency_names(optional_dependencies[spec.extra])
            assert package_name in extra_dependencies, key


def test_require_dependency_core_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import(name: str):
        raise ImportError(f"missing {name}")

    monkeypatch.setattr(importlib, "import_module", fail_import)

    with pytest.raises(ImportError) as error:
        require_dependency("pandas", feature="dataframe export")

    message = str(error.value)
    assert "dataframe export" in message
    assert "Module: pandas" in message
    assert "Package: pandas" in message
    assert 'pip install "wandas"' in message


def test_require_dependency_optional_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import(name: str):
        raise ImportError(f"missing {name}")

    monkeypatch.setattr(importlib, "import_module", fail_import)

    with pytest.raises(ImportError) as error:
        require_dependency("h5py", feature="loading WDF files")

    message = str(error.value)
    assert "loading WDF files" in message
    assert "Module: h5py" in message
    assert "Package: h5py" in message
    assert 'pip install "wandas[io]"' in message
