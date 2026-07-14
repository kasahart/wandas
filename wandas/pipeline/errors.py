from __future__ import annotations


class RecipeExtractionError(ValueError):
    """Raised when a frame lineage cannot be represented by a recipe."""


class RecipeSerializationError(ValueError):
    """Raised when a Recipe payload violates the canonical schema."""


class RecipeValidationError(ValueError):
    """Raised when a Recipe graph or registered operation contract is invalid."""


class RecipeExecutionError(RuntimeError):
    """Raised when a validated Recipe cannot be applied to runtime inputs."""
