from wandas.pipeline import RecipeExtractionError as PublicRecipeExtractionError
from wandas.pipeline.errors import RecipeExtractionError


def test_recipe_extraction_error_has_public_and_structured_import_paths() -> None:
    assert RecipeExtractionError is PublicRecipeExtractionError
    assert issubclass(RecipeExtractionError, ValueError)


def test_recipe_components_have_focused_import_paths() -> None:
    from wandas.pipeline.extraction import steps_from_graph
    from wandas.pipeline.params import restore_history_value
    from wandas.pipeline.specs import GraphRecipeSpec, NodeGraphRecipeSpec, RecipeSpec
    from wandas.pipeline.steps import OperationSpec, TerminalStep

    assert RecipeSpec is __import__("wandas.pipeline", fromlist=["RecipeSpec"]).RecipeSpec
    assert GraphRecipeSpec is __import__("wandas.pipeline", fromlist=["GraphRecipeSpec"]).GraphRecipeSpec
    assert NodeGraphRecipeSpec is __import__("wandas.pipeline", fromlist=["NodeGraphRecipeSpec"]).NodeGraphRecipeSpec
    assert OperationSpec is __import__("wandas.pipeline", fromlist=["OperationSpec"]).OperationSpec
    assert TerminalStep is __import__("wandas.pipeline", fromlist=["TerminalStep"]).TerminalStep
    assert callable(steps_from_graph)
    assert callable(restore_history_value)
