"""Tests for the lightweight public export documentation check."""

from scripts.check_public_docstrings import check_public_exports


def test_public_exports_are_importable_and_documented() -> None:
    """Every declared public export has a usable runtime docstring."""
    assert check_public_exports() == []
