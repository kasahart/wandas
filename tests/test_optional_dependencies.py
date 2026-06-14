import pytest

from wandas.utils.optional_imports import require_optional_attr, require_optional_dependency


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
