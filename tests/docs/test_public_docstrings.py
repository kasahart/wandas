from pathlib import Path

from griffe import Parser

from scripts.check_public_docstrings import (
    MKDOCS_CONFIG,
    audit_public_docstrings,
    configured_docstring_style,
    public_docstrings,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_mkdocstrings_auto_parser_covers_both_established_styles() -> None:
    assert configured_docstring_style(MKDOCS_CONFIG) == Parser.auto.value

    result = audit_public_docstrings()

    assert result.errors == ()
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


def test_documentation_governance_records_translation_and_compatibility_scope() -> None:
    contributing = (REPO_ROOT / "docs/src/contributing.md").read_text(encoding="utf-8")
    stability = (REPO_ROOT / "docs/src/explanation/public-api-stability.md").read_text(encoding="utf-8")
    previous_release_notes = (REPO_ROOT / "docs/src/release-notes/v0.6.1.md").read_text(encoding="utf-8")
    release_notes = (REPO_ROOT / "docs/src/release-notes/v0.6.2.md").read_text(encoding="utf-8")
    release_template = (REPO_ROOT / "docs/src/release-notes/template.md").read_text(encoding="utf-8")
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
