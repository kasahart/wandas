from pathlib import Path

import pytest
import tomli

from wandas.utils.optional_imports import require_optional_attr, require_optional_dependency


def _pyproject() -> dict:
    return tomli.loads(Path("pyproject.toml").read_text())


def test_runtime_dependencies_are_balanced_core_only() -> None:
    dependencies = set(_pyproject()["project"]["dependencies"])
    names = {dep.split("[", 1)[0].split(">", 1)[0].split("=", 1)[0].split("<", 1)[0] for dep in dependencies}

    assert {"numpy", "scipy", "dask", "pydantic", "soundfile"}.issubset(names)
    assert "matplotlib" not in names
    assert "librosa" not in names
    assert "ipykernel" not in names
    assert "ipywidgets" not in names
    assert "ipympl" not in names
    assert "ipycytoscape" not in names
    assert "japanize-matplotlib" not in names
    assert "pandas" not in names
    assert "h5py" not in names
    assert "mosqito" not in names
    assert "torch" not in names
    assert "tensorflow" not in names
    assert "types-requests" not in names


def test_optional_dependency_groups_exist() -> None:
    optional = _pyproject()["project"]["optional-dependencies"]

    assert set(optional) >= {"io", "viz", "notebook", "psychoacoustic", "ml"}
    assert "pandas" in optional["io"]
    assert any(dep.startswith("h5py") for dep in optional["io"])
    assert any(dep.startswith("matplotlib") for dep in optional["viz"])
    assert "librosa" in optional["viz"]
    assert any(dep.startswith("japanize-matplotlib") for dep in optional["viz"])
    assert "ipykernel" in optional["notebook"]
    assert "ipywidgets" in optional["notebook"]
    assert any(dep.startswith("ipympl") for dep in optional["notebook"])
    assert any(dep.startswith("ipycytoscape") for dep in optional["notebook"])
    assert "mosqito" in optional["psychoacoustic"]
    assert any(dep.startswith("torch") for dep in optional["ml"])
    assert any(dep.startswith("tensorflow") for dep in optional["ml"])


def test_require_optional_dependency_imports_installed_module() -> None:
    module = require_optional_dependency("math", extra="core", feature="test feature")
    assert module.sqrt(4) == 2


def test_require_optional_dependency_error_message() -> None:
    with pytest.raises(ImportError) as exc_info:
        require_optional_dependency(
            "definitely_missing_wandas_dependency",
            extra="viz",
            feature="plot",
        )

    message = str(exc_info.value)
    assert "plot requires optional dependency 'definitely_missing_wandas_dependency'" in message
    assert 'pip install "wandas[viz]"' in message


def test_require_optional_dependency_wraps_missing_parent_package() -> None:
    with pytest.raises(ImportError) as exc_info:
        require_optional_dependency(
            "definitely_missing_wandas_dependency.submodule",
            extra="viz",
            feature="plot",
        )

    message = str(exc_info.value)
    assert "plot requires optional dependency 'definitely_missing_wandas_dependency.submodule'" in message
    assert 'pip install "wandas[viz]"' in message


def test_require_optional_dependency_reraises_transitive_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    original_error = ModuleNotFoundError(
        "No module named 'missing_transitive_dependency'",
        name="missing_transitive_dependency",
    )

    def raise_transitive_error(module_name: str) -> None:
        assert module_name == "installed_optional_package"
        raise original_error

    monkeypatch.setattr(
        "wandas.utils.optional_imports.importlib.import_module",
        raise_transitive_error,
    )

    with pytest.raises(ModuleNotFoundError) as exc_info:
        require_optional_dependency(
            "installed_optional_package",
            extra="viz",
            feature="plot",
        )

    assert exc_info.value is original_error


def test_require_optional_attr_returns_installed_attribute() -> None:
    sqrt = require_optional_attr("math", "sqrt", extra="core", feature="test feature")
    assert sqrt(9) == 3


def test_require_optional_attr_missing_attribute_error_message() -> None:
    with pytest.raises(ImportError) as exc_info:
        require_optional_attr(
            "math",
            "definitely_missing_wandas_attr",
            extra="viz",
            feature="plot",
        )

    message = str(exc_info.value)
    assert "plot requires 'math' to provide attribute 'definitely_missing_wandas_attr'" in message
    assert 'pip install "wandas[viz]"' in message
