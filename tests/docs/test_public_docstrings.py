from pathlib import Path

import pytest
from griffe import Parser

from scripts.check_public_docstrings import (
    MKDOCS_CONFIG,
    _mask_fenced_code,
    _parse_identities,
    _sphinx_fields,
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

    def __getitem__(self, index):
        """Return an item through the public indexing protocol.

        Args:
            index: Item position.

        Returns:
            int: A value.
        """

    def _helper(self):
        """Remain outside the public surface."""

    def __post_init__(self):
        """Remain an explicitly internal construction hook.

        Args:
            value: A value.

        Returns
        -------
        int
            A value.
        """

def public_function(value):
    """A public function.

    Parameters
    ----------
    value : int
        A value.
    """
''',
        encoding="utf-8",
    )

    names = {docstring.qualified_name for docstring in public_docstrings(tmp_path)}

    assert names == {
        f"{tmp_path.name}.scoped",
        f"{tmp_path.name}.scoped.PublicApi",
        f"{tmp_path.name}.scoped.PublicApi.method",
        f"{tmp_path.name}.scoped.PublicApi.__getitem__",
        f"{tmp_path.name}.scoped.public_function",
    }

    private_source = tmp_path / "_private.py"
    private_source.write_text(
        '''"""A private module docstring."""

class PublicApi:
    """A normally named definition that remains private with its module.

    Args:
        value: A value.

    Returns
    -------
    int
        A value.
    """
''',
        encoding="utf-8",
    )
    private_names = {
        f"{tmp_path.name}._private",
        f"{tmp_path.name}._private.PublicApi",
    }
    assert private_names.isdisjoint(docstring.qualified_name for docstring in public_docstrings(tmp_path))
    assert audit_public_docstrings(tmp_path).errors == ()


def test_audit_treats_inline_google_labels_as_prose(tmp_path: Path) -> None:
    source = tmp_path / "inline_label.py"
    source.write_text(
        '''class GoogleApi:
    """A valid Google docstring.

    Note: this conversion materializes the frame.

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
''',
        encoding="utf-8",
    )

    result = audit_public_docstrings(tmp_path)

    assert result.errors == ()
    assert result.checked_docstrings == 2


def test_audit_rejects_unclassified_docstring_dunder_drift(tmp_path: Path) -> None:
    source = tmp_path / "unclassified_dunder.py"
    source.write_text(
        '''class PublicApi:
    """A public class."""

    def __mystery__(self):
        """An unclassified user-defined protocol candidate."""

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
    assert "unclassified docstring-bearing dunder __mystery__" in result.errors[0]


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
    assert "mixes Google and NumPy structured evidence" in result.errors[0]
    assert "Google: parameters" in result.errors[0]
    assert "NumPy: examples" in result.errors[0]


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
    assert "mixes Google and NumPy structured evidence" in result.errors[0]
    assert "Google: attributes" in result.errors[0]
    assert "NumPy: examples" in result.errors[0]


@pytest.mark.parametrize(
    ("value", "parser", "identity"),
    [
        ("Args:\n    value: A value.", Parser.google, ("parameters", None, None)),
        ("Returns: Result value\n    int: A value.", Parser.google, ("returns", None, "result value")),
        ("Examples:\n    >>> PublicApi()", Parser.google, ("examples", None, None)),
        ("Note: Performance\n    Fast path.", Parser.google, ("admonition", "note", "performance")),
        ("Deprecated:\n    Use the replacement.", Parser.google, ("admonition", "deprecated", "deprecated")),
        ("Parameters\n----------\nvalue : int\n    A value.", Parser.numpy, ("parameters", None, None)),
        ("Parameters\n  ----------\nvalue : int\n    A value.", Parser.numpy, ("parameters", None, None)),
        ("Examples\n--------\n>>> PublicApi()", Parser.numpy, ("examples", None, None)),
        (
            "Implementation Details\n----------------------\nCustom.",
            Parser.numpy,
            ("admonition", "implementation-details", "implementation details"),
        ),
        ("Deprecated\n----------\n0.2.0\n    Use the replacement.", Parser.numpy, ("deprecated", None, None)),
    ],
)
def test_explicit_parser_matrix_defines_style_specific_evidence(
    value: str,
    parser: Parser,
    identity: tuple[str, str | None, str | None],
) -> None:
    identities = _parse_identities(_mask_fenced_code(f"Summary.\n\n{value}"), parser)

    assert identities == (identity,)


@pytest.mark.parametrize(
    "value",
    [
        "Summary.\n\nNote: this conversion materializes the frame.",
        "Summary.\n\nArgs:\nvalue: The body is not indented.",
        "Summary.\n\nParameters\nnot an underline\nvalue : int",
    ],
)
def test_plain_text_and_malformed_sections_are_not_style_evidence(value: str) -> None:
    masked = _mask_fenced_code(value)

    assert _parse_identities(masked, Parser.google) == ()
    assert _parse_identities(masked, Parser.numpy) == ()


@pytest.mark.parametrize(
    "literal",
    [
        "Args:\n    value: A value.",
        "Returns\n-------\nint\n    A value.",
        ":param value: A value.",
    ],
)
def test_parser_evidence_and_sphinx_scan_ignore_fenced_literal_syntax(literal: str) -> None:
    value = f"Summary.\n\n```text\n{literal}\n```"
    masked = _mask_fenced_code(value)

    assert masked.count("\n") == value.count("\n")
    assert _parse_identities(masked, Parser.google) == ()
    assert _parse_identities(masked, Parser.numpy) == ()
    assert _sphinx_fields(value) == ()


@pytest.mark.parametrize("fence_char", ["`", "~"])
def test_fence_mask_tracks_indented_outer_delimiter_and_info_string(fence_char: str) -> None:
    outer = fence_char * 4
    inner = fence_char * 3
    value = (
        "Summary.\n\n"
        f"   {outer}markdown\n"
        f"   {inner}python\n"
        "   Args:\n"
        "       value: Literal nested syntax.\n"
        f"   {inner}\n"
        f"   {outer}\n"
        "Returns:\n"
        "    int: A real section."
    )

    masked = _mask_fenced_code(value)

    assert masked.count("\n") == value.count("\n")
    assert "Args:" not in masked
    assert "Returns:" in masked


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


def test_audit_preserves_nested_markdown_fence_boundaries(tmp_path: Path) -> None:
    source = tmp_path / "nested_fences.py"
    source.write_text(
        '''class GoogleApi:
    """A valid Google docstring containing a nested fence example.

    Returns:
        int: A value.

    ````markdown
    ```python
    Args:
        value: Literal syntax inside the nested example.
    ```
    ````
    """

class NumpyApi:
    """Exercise the established NumPy parser.

    Parameters
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


def test_audit_models_auto_heuristic_on_unmasked_fenced_source(tmp_path: Path) -> None:
    source = tmp_path / "fenced_auto.py"
    source.write_text(
        '''class GoogleApi:
    """Exercise the established Google parser.

    Returns:
        int: A value.
    """

class NumpyApi:
    """A NumPy docstring whose fenced example misleads auto detection.

    Parameters
    ----------
    value : int
        A value.

    ```text
    Args:
        value: Google syntax shown as literal text.
    ```
    """
''',
        encoding="utf-8",
    )

    result = audit_public_docstrings(tmp_path)

    assert len(result.errors) == 1
    assert "Griffe auto structured identities (none)" in result.errors[0]
    assert "do not match explicit NumPy identities (parameters)" in result.errors[0]


def test_explicit_numpy_parser_matches_headings_case_insensitively() -> None:
    identities = _parse_identities("returns\n-------\nint\n    A value.", Parser.numpy)

    assert identities == (("returns", None, None),)


def test_explicit_google_parser_matches_headings_case_insensitively() -> None:
    identities = _parse_identities("Summary.\n\nreturns:\n    int: A value.", Parser.google)

    assert identities == (("returns", None, None),)


def test_audit_accepts_titled_google_admonition_identity(tmp_path: Path) -> None:
    source = tmp_path / "admonitions.py"
    source.write_text(
        '''class GoogleApi:
    """A titled Google admonition must retain its kind and title.

    Args:
        value: A value.

    Note: Performance
        This path avoids a copy.
    """

class NumpyApi:
    """Exercise the established NumPy parser.

    Parameters
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
    assert result.structured_sections == 3


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
    assert "mixes Google and NumPy structured evidence" in result.errors[0]
    assert "Google: parameters" in result.errors[0]
    assert "NumPy: returns" in result.errors[0]


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

def public_function(value, *args, **kwargs):
    """An unsupported Sphinx-style docstring.

    :param value: A value.
    :param *args: Positional values.
    :param **kwargs: Keyword values.
    :param list[str] values: Collection values.
    :param pathlib.Path path: A filesystem path.
    :returns: The value.
    """
''',
        encoding="utf-8",
    )

    result = audit_public_docstrings(tmp_path)

    assert len(result.errors) == 1
    assert "uses unsupported Sphinx field-list sections" in result.errors[0]
    assert "param at docstring line 3" in result.errors[0]
    assert "param at docstring line 4" in result.errors[0]
    assert "param at docstring line 5" in result.errors[0]
    assert "param at docstring line 6" in result.errors[0]
    assert "param at docstring line 7" in result.errors[0]
    assert "returns at docstring line 8" in result.errors[0]


@pytest.mark.parametrize(
    ("field", "name"),
    [
        (":param list[str] values: Collection values.", "param"),
        (":param pathlib.Path value: A qualified type.", "param"),
        (":param **kwargs: Keyword values.", "param"),
        (":type values: dict[str, pathlib.Path]", "type"),
        (":raises package.errors.ParseError: Invalid input.", "raises"),
        (":rtype: tuple[str, pathlib.Path]", "rtype"),
        (":ivar dict[str, int] counts: Stored counts.", "ivar"),
    ],
)
def test_sphinx_scan_rejects_complete_field_tails(field: str, name: str) -> None:
    assert _sphinx_fields(f"Summary.\n\n{field}") == ((name, 3),)


def test_audit_rejects_google_mixed_with_custom_numpy_heading(tmp_path: Path) -> None:
    source = tmp_path / "custom_numpy.py"
    source.write_text(
        '''class PublicApi:
    """An invalid mixed-style docstring.

    Args:
        value: A value.

    Implementation Details
    ----------------------
    This is a custom NumPy section.
    """
''',
        encoding="utf-8",
    )

    result = audit_public_docstrings(tmp_path)

    assert len(result.errors) == 1
    assert "mixes Google and NumPy structured evidence" in result.errors[0]
    assert "Google: parameters" in result.errors[0]
    assert "NumPy: admonition (implementation-details/implementation details)" in result.errors[0]


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
    assert "mixes Google and NumPy structured evidence" in result.errors[0]


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
    assert "Griffe auto structured identities (none)" in result.errors[0]
    assert "do not match explicit NumPy identities (examples)" in result.errors[0]


def test_documentation_governance_records_translation_and_compatibility_scope() -> None:
    contributing = (REPO_ROOT / "docs/src/contributing.md").read_text(encoding="utf-8")
    stability = (REPO_ROOT / "docs/src/explanation/public-api-stability.md").read_text(encoding="utf-8")
    previous_release_notes = (REPO_ROOT / "docs/src/release-notes/v0.6.1.md").read_text(encoding="utf-8")
    release_notes = (REPO_ROOT / "docs/src/release-notes/v0.6.2.md").read_text(encoding="utf-8")
    release_template = (REPO_ROOT / "docs/src/release-notes/template.md").read_text(encoding="utf-8")
    stability_flat = " ".join(stability.split()).casefold()
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
    compatibility_matrix = (
        "stable and supported serialized contracts may use `none` only",
        "experimental removals may use `none` without an exception",
        "internal-only",
    )
    assert all(rule in stability_flat for rule in compatibility_matrix)
    assert all(rule in release_template_flat.casefold() for rule in compatibility_matrix)

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
