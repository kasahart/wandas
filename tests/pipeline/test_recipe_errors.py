from wandas.pipeline import RecipeExtractionError as PublicRecipeExtractionError
from wandas.pipeline.errors import RecipeExtractionError


def test_recipe_extraction_error_has_public_and_structured_import_paths() -> None:
    assert RecipeExtractionError is PublicRecipeExtractionError
    assert issubclass(RecipeExtractionError, ValueError)
