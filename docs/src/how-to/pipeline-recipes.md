# Work with RecipePlan

Use a `RecipePlan` when a public Frame workflow must be replayed with different runtime
inputs. For a guided introduction, start with the
[Recipe tutorial](../tutorial/pipeline-recipes.md).

## Extract and apply a plan

Process a Frame normally, then compile its semantic lineage:

```python
from wandas.pipeline import RecipePlan

processed = source.remove_dc().normalize()
plan = RecipePlan.from_frame(processed, input_names=("signal",))
replayed = plan.apply({"signal": another_frame})
```

The number of `input_names` must match the runtime inputs discovered in lineage. Apply
requires exactly those names: missing and extra names are errors.

## Supply Frame and array inputs

Frame arithmetic, indexing, channel operations, typed transitions, and multi-input
calls use the same plan. External NumPy and Dask arrays are named `array` inputs; their
values, chunks, and container backends are never embedded in the Recipe.

```python
processed = source + external_array
plan = RecipePlan.from_frame(processed, input_names=("signal", "offset"))
replayed = plan.apply({"signal": another_frame, "offset": external_array})
```

Supported operation shapes are:

| Workflow shape | Recipe inputs |
| --- | --- |
| unary or typed Frame operation | one `frame` |
| scalar arithmetic | one `frame`; scalar stored as a parameter |
| Frame arithmetic | ordered `frame` inputs |
| `mix()` | `base` plus a `frame` or `array` input |
| NumPy/Dask arithmetic | ordered `frame` and `array` inputs |
| indexing | one `frame`; selector stored as a parameter |
| `add_channel()` | `base` plus a `frame` or `array` input |

## Save and load a standalone artifact

Persist reusable intent independently from Frame data:

```python
path = plan.save("analysis")  # analysis.recipe.json
restored = RecipePlan.load(path)
```

Existing mapping workflows remain available through `to_dict()` and `from_dict()`.
Saving refuses to overwrite by default; pass `overwrite=True` deliberately.

The loader accepts only `wandas.recipe` schema 2. It rejects unknown operations,
versions, fields, binding kinds, malformed values, dead nodes, and unused inputs.
Extraction, serialization, loading, and lazy graph construction do not compute Dask
arrays. Built-in operation IDs resolve through the default registry; extension plans
must use the same immutable registry for extraction, loading, and application.

## Understand the boundaries

- A plan returns a Frame; scalar terminal results are outside the Recipe contract.
- `Frame.apply(callable)` is runtime-only and fails Recipe extraction.
- Regex and callable channel queries are not portable.
- WDF stores one typed result plus display history, not an executable `RecipePlan`.
- Keep reusable evidence as an explicit pair such as `analysis.wdf` and
  `analysis.recipe.json`; their schemas evolve independently.
- `mix()` uses array-index alignment and preserves the base Frame's metadata, length,
  labels, and source-time offsets.
- Unsupported operations fail the whole extraction instead of silently cutting the
  graph into another input.

Arbitrary callables passed to `Frame.apply(...)` are runtime-only. To make an extension
portable, declare its public Frame method with `@recipe_operation`, derive an immutable
registry containing that declaration, and supply the same registry to extraction,
loading, and application. See [Extending Recipe v2](../explanation/pipeline-recipe-developer-guide.md).
