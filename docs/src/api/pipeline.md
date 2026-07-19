# Pipeline Recipes API

`wandas.pipeline` provides the portable Recipe surface. Most users need only
`RecipePlan`; registry and decorator APIs are for extensions with stable public Frame
methods.

Start with the [Recipe tutorial](../tutorial/pipeline-recipes.md) for a complete
workflow or the [Recipe how-to](../how-to/pipeline-recipes.md) for concise task recipes.

## RecipePlan

Create plans with `RecipePlan.from_frame()`, `RecipePlan.from_dict()`, or
`RecipePlan.load()`. Persist a standalone strict JSON artifact with
`RecipePlan.save()`. The direct
graph constructor is internal and is not a supported compatibility surface.

::: wandas.pipeline.RecipePlan

## Errors

::: wandas.pipeline.RecipeExtractionError

::: wandas.pipeline.RecipeSerializationError

::: wandas.pipeline.RecipeValidationError

::: wandas.pipeline.RecipeExecutionError

## Extension declarations

::: wandas.pipeline.recipe_operation

::: wandas.pipeline.recipe_definition

::: wandas.pipeline.OperationCapture

::: wandas.pipeline.RecipeOperation

::: wandas.pipeline.RecipeRegistry

::: wandas.pipeline.default_recipe_registry

## Scikit-learn adapters

These stateless estimators call declared public Frame methods and preserve sklearn's
clone and pipeline conventions.

::: wandas.pipeline.sklearn.WandasOperationTransformer

::: wandas.pipeline.sklearn.HighPassFilter

::: wandas.pipeline.sklearn.LowPassFilter

::: wandas.pipeline.sklearn.BandPassFilter

::: wandas.pipeline.sklearn.Normalize

::: wandas.pipeline.sklearn.RemoveDC
