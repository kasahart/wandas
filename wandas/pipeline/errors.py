from __future__ import annotations


class RecipeExtractionError(ValueError):
    """Raised when a frame lineage cannot be represented by a recipe."""


class RecipeSerializationError(ValueError):
    """Raised when a Recipe payload violates the canonical schema."""
