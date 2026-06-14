import pytest

from wandas.utils.optional_imports import require_optional_dependency


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
