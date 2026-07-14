"""Documentation contract for the public Recipe surfaces."""

from __future__ import annotations

import inspect

import pytest

import wandas.pipeline as pipeline
from wandas.pipeline import sklearn as sklearn_adapters


@pytest.mark.parametrize("name", pipeline.__all__)
def test_pipeline_export_has_docstring(name: str) -> None:
    """Keep every documented top-level Recipe export self-describing."""
    assert inspect.getdoc(getattr(pipeline, name)), f"wandas.pipeline.{name} needs a docstring"


@pytest.mark.parametrize("name", sklearn_adapters.__all__)
def test_sklearn_adapter_export_has_docstring(name: str) -> None:
    """Keep every public sklearn adapter self-describing."""
    assert inspect.getdoc(getattr(sklearn_adapters, name)), f"wandas.pipeline.sklearn.{name} needs a docstring"
