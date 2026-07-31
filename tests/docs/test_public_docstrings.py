from pathlib import Path

import pytest
from griffe import Parser

from scripts.check_public_docstrings import (
    MKDOCS_CONFIG,
    _declared_sections,
    audit_public_docstrings,
    configured_docstring_style,
    public_docstrings,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_mkdocstrings_auto_parser_covers_both_established_styles() -> None:
    assert configured_docstring_style(MKDOCS_CONFIG) == Parser.auto.value

    result = audit_public_docstrings()

    assert result.errors == ()
    assert result.audited_docstrings >= 500
    assert result.checked_docstrings >= 200
    assert result.google_docstrings > 0
    assert result.numpy_docstrings > 0
    assert result.structured_sections >= result.checked_docstrings


def test_migration_representatives_remain_in_the_repository_wide_inventory() -> None:
    names = {docstring.qualified_name for docstring in public_docstrings()}

    assert {
        "wandas.frames.channel.ChannelFrame.add_channel",
        "wandas.frames.spectral.SpectralFrame.plot_matrix",
        "wandas.frames.mixins.channel_processing_mixin.ChannelProcessingMixin.sharpness_din",
        "wandas.frames.mixins.channel_transform_mixin.ChannelTransformMixin.cepstrum",
        "wandas.processing.spectral.FFT.calculate_output_shape",
        "wandas.processing.temporal.Trim.calculate_output_shape",
        "wandas.utils.frame_dataset.FrameDataset.get_by_label",
    } <= names


def test_public_docstring_scope_includes_modules_classes_functions_and_methods(tmp_path: Path) -> None:
    source = tmp_path / "scoped.py"
    source.write_text(
        '''"""A public module.

Examples:
    >>> module_value = 1
"""

class PublicApi:
    """A public class.

    Attributes:
        value: A value.
    """

    def method(self):
        """A public method.

        Returns:
            int: A value.
        """

def public_function():
    """A public function.

    Notes
    -----
    Additional context.
    """
''',
        encoding="utf-8",
    )

    names = {docstring.qualified_name for docstring in public_docstrings(tmp_path)}

    assert names == {
        f"{tmp_path.name}.scoped",
        f"{tmp_path.name}.scoped.PublicApi",
        f"{tmp_path.name}.scoped.PublicApi.method",
        f"{tmp_path.name}.scoped.public_function",
    }

    private_source = tmp_path / "_private.py"
    private_source.write_text('''"""A private module docstring."""\n''', encoding="utf-8")
    assert all(docstring.qualified_name != f"{tmp_path.name}._private" for docstring in public_docstrings(tmp_path))


def test_audit_rejects_mixed_style_outside_parameter_sections(tmp_path: Path) -> None:
    source = tmp_path / "mixed.py"
    source.write_text(
        '''class PublicApi:
    """An intentionally invalid mixed-style docstring.

    Args:
        value: A value.

    Examples
    --------
    >>> PublicApi()
    """
''',
        encoding="utf-8",
    )

    result = audit_public_docstrings(tmp_path)

    assert len(result.errors) == 1
    assert "mixes Google and NumPy structured sections (Args, Examples)" in result.errors[0]


def test_audit_rejects_mixed_style_with_only_non_core_sections(tmp_path: Path) -> None:
    source = tmp_path / "mixed.py"
    source.write_text(
        '''class PublicApi:
    """An intentionally invalid mixed-style docstring.

    Attributes:
        value: A value.

    Examples
    --------
    >>> PublicApi()
    """
''',
        encoding="utf-8",
    )

    result = audit_public_docstrings(tmp_path)

    assert len(result.errors) == 1
    assert "mixes Google and NumPy structured sections (Attributes, Examples)" in result.errors[0]


@pytest.mark.parametrize(
    ("value", "style", "kind"),
    [
        ("Args:\n    value: A value.", "google", "parameters"),
        ("Returns: Result value\n    int: A value.", "google", "returns"),
        ("Examples:\n    >>> PublicApi()", "google", "examples"),
        ("Note:\n    Additional context.", "google", "admonition"),
        ("Deprecated:\n    Use the replacement.", "google", "admonition"),
        ("Parameters\n----------\nvalue : int\n    A value.", "numpy", "parameters"),
        ("Parameters\n  ----------\nvalue : int\n    A value.", "numpy", "parameters"),
        ("Examples\n--------\n>>> PublicApi()", "numpy", "examples"),
        ("Notes\n-----\nAdditional context.", "numpy", "admonition"),
        ("Deprecated\n----------\n0.2.0\n    Use the replacement.", "numpy", "deprecated"),
    ],
)
def test_declared_section_matrix_tracks_style_and_griffe_kind(value: str, style: str, kind: str) -> None:
    expected, styles, headers, sphinx_fields = _declared_sections(value)

    assert expected == {kind: 1}
    assert styles == {style}
    assert len(headers) == 1
    assert sphinx_fields == []


def test_declared_sections_ignore_unknown_and_nested_headings() -> None:
    expected, styles, headers, sphinx_fields = _declared_sections(
        """Summary.

        Custom:
            This project-specific heading is ordinary prose.

        Examples:
            Warnings:
            >>> PublicApi()
        """
    )

    assert expected == {"examples": 1}
    assert styles == {"google"}
    assert headers == ["Examples"]
    assert sphinx_fields == []


@pytest.mark.parametrize(
    "literal",
    [
        "Args:\n    value: A value.",
        "Returns\n-------\nint\n    A value.",
        ":param value: A value.",
    ],
)
def test_declared_sections_ignore_fenced_literal_syntax(literal: str) -> None:
    expected, styles, headers, sphinx_fields = _declared_sections(f"Summary.\n\n```text\n{literal}\n```")

    assert expected == {}
    assert styles == set()
    assert headers == []
    assert sphinx_fields == []


def test_audit_ignores_fenced_literal_syntax_end_to_end(tmp_path: Path) -> None:
    source = tmp_path / "fenced.py"
    source.write_text(
        '''class GoogleApi:
    """Exercise the established Google parser.

    Args:
        value: A value.
    """

class NumpyApi:
    """Exercise the established NumPy parser.

    Parameters
    ----------
    value : int
        A value.
    """

def syntax_example():
    """Show syntax that must remain literal.

    ```text
    Returns:
        int: A value.
    Parameters
    ----------
    :param value: A value.
    ```
    """
''',
        encoding="utf-8",
    )

    result = audit_public_docstrings(tmp_path)

    assert result.errors == ()
    assert result.audited_docstrings == 3
    assert result.checked_docstrings == 2


def test_declared_sections_match_numpy_headings_case_insensitively() -> None:
    expected, styles, headers, sphinx_fields = _declared_sections("returns\n-------\nint\n    A value.")

    assert expected == {"returns": 1}
    assert styles == {"numpy"}
    assert headers == ["Returns"]
    assert sphinx_fields == []


def test_declared_sections_match_google_headings_case_insensitively() -> None:
    expected, styles, headers, sphinx_fields = _declared_sections("returns:\n    int: A value.")

    assert expected == {"returns": 1}
    assert styles == {"google"}
    assert headers == ["Returns"]
    assert sphinx_fields == []


def test_audit_rejects_google_with_lowercase_numpy_heading(tmp_path: Path) -> None:
    source = tmp_path / "mixed.py"
    source.write_text(
        '''class PublicApi:
    """An invalid mixed-style docstring.

    Args:
        value: A value.

    returns
    -------
    int
        A value.
    """
''',
        encoding="utf-8",
    )

    result = audit_public_docstrings(tmp_path)

    assert len(result.errors) == 1
    assert "mixes Google and NumPy structured sections (Args, Returns)" in result.errors[0]


def test_audit_rejects_sphinx_field_lists_explicitly(tmp_path: Path) -> None:
    source = tmp_path / "sphinx.py"
    source.write_text(
        '''class GoogleApi:
    """Exercise the established Google parser.

    Args:
        value: A value.
    """

class NumpyApi:
    """Exercise the established NumPy parser.

    Parameters
    ----------
    value : int
        A value.
    """

def public_function(value):
    """An unsupported Sphinx-style docstring.

    :param value: A value.
    :returns: The value.
    """
''',
        encoding="utf-8",
    )

    result = audit_public_docstrings(tmp_path)

    assert len(result.errors) == 1
    assert "uses unsupported Sphinx field-list sections (param, returns)" in result.errors[0]


def test_audit_applies_style_rules_to_public_module_docstrings(tmp_path: Path) -> None:
    source = tmp_path / "module_api.py"
    source.write_text(
        '''"""An invalid mixed-style public module.

Args:
    value: A value.

Returns
-------
int
    A value.
"""

class GoogleApi:
    """Exercise Google coverage.

    Args:
        value: A value.
    """

class NumpyApi:
    """Exercise NumPy coverage.

    Parameters
    ----------
    value : int
        A value.
    """
''',
        encoding="utf-8",
    )

    result = audit_public_docstrings(tmp_path)

    assert len(result.errors) == 1
    assert f"{tmp_path.name}.module_api" in result.errors[0]
    assert "mixes Google and NumPy structured sections (Args, Returns)" in result.errors[0]


def test_audit_accepts_auto_parsed_non_core_sections(tmp_path: Path) -> None:
    source = tmp_path / "valid.py"
    source.write_text(
        '''class GoogleApi:
    """A valid Google examples-only docstring.

    Examples:
        >>> GoogleApi()
    """

class NumpyApi:
    """A valid NumPy attributes-only docstring.

    Attributes
    ----------
    value : int
        A value.
    """
''',
        encoding="utf-8",
    )

    result = audit_public_docstrings(tmp_path)

    assert result.errors == ()
    assert result.checked_docstrings == 2
    assert result.structured_sections == 2


def test_audit_rejects_unparsed_numpy_only_examples(tmp_path: Path) -> None:
    source = tmp_path / "examples.py"
    source.write_text(
        '''class GoogleApi:
    """Exercise the established Google parser.

    Args:
        value: A value.
    """

class NumpyApi:
    """An examples-only NumPy docstring that auto-detection cannot select.

    Examples
    --------
    >>> NumpyApi()
    """
''',
        encoding="utf-8",
    )

    result = audit_public_docstrings(tmp_path)

    assert len(result.errors) == 1
    assert "Griffe auto did not parse examples" in result.errors[0]


def test_documentation_governance_records_translation_and_compatibility_scope() -> None:
    contributing = (REPO_ROOT / "docs/src/contributing.md").read_text(encoding="utf-8")
    stability = (REPO_ROOT / "docs/src/explanation/public-api-stability.md").read_text(encoding="utf-8")
    previous_release_notes = (REPO_ROOT / "docs/src/release-notes/v0.6.1.md").read_text(encoding="utf-8")
    release_notes = (REPO_ROOT / "docs/src/release-notes/v0.6.2.md").read_text(encoding="utf-8")
    release_template = (REPO_ROOT / "docs/src/release-notes/template.md").read_text(encoding="utf-8")
    release_template_flat = " ".join(release_template.split())
    mkdocs = (REPO_ROOT / "docs/mkdocs.yml").read_text(encoding="utf-8")

    assert "does not require every technical document" in contributing
    assert "`README.md` and `README.ja.md` are a maintained language pair" in contributing
    assert "one complete Google or NumPy style per docstring" in contributing

    required_release_fields = (
        "Classification",
        "Deprecation start",
        "Replacement or migration",
        "Removal/change version",
        "Exception reason and decision link",
    )
    assert "Patch releases do" in stability
    assert "not consume that window" in stability
    assert "An exception does not reclassify a stable surface as experimental" in stability
    assert "any release\nthat contains a compatibility change" in stability
    assert all(field in release_template for field in required_release_fields)
    assert "any release containing a\ncompatibility change" in release_template
    assert "Experimental removals may use `None` without an exception" in release_template_flat
    assert "Stable and supported serialized contracts may use `None` only" in release_template_flat

    assert "`ChannelFrame.add_channel(ChannelFrame)` | Stable user surface | None" in release_notes
    assert "Recipe version 1 replay | Serialized operation-version compatibility | None" in release_notes
    assert "`ChannelFrame.add_channel(ChannelFrame)`" not in previous_release_notes
    assert "Recipe version 1" not in previous_release_notes
    assert "Wandas 0.6.2: release-notes/v0.6.2.md" in mkdocs


def test_ci_and_deployment_run_the_focused_docstring_gate() -> None:
    command = "python scripts/check_public_docstrings.py"
    ci = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    deploy = (REPO_ROOT / ".github/workflows/deploy-docs.yml").read_text(encoding="utf-8")

    assert command in ci
    assert command in deploy
