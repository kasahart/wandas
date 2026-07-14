# Compile and replay a RecipePlan

Recipe v2 has one public graph model. Process a frame normally, then compile its
semantic lineage:

```python
from wandas.pipeline import RecipePlan

processed = source.remove_dc().normalize()
plan = RecipePlan.from_frame(processed, input_names=("signal",))
replayed = plan.apply({"signal": another_frame})
```

Frame, external-array, binary, indexing, add-channel, typed transition, and multi-input
calls all use the same plan. External NumPy and Dask arrays are named inputs; their
values and container backends are never embedded in the Recipe.

```python
processed = source + external_array
plan = RecipePlan.from_frame(processed, input_names=("signal", "offset"))
replayed = plan.apply({"signal": another_frame, "offset": external_array})
```

Persist only the versioned canonical schema:

```python
payload = plan.to_dict()
restored = RecipePlan.from_dict(payload)
```

The loader accepts only `wandas.recipe` schema 2 and rejects unknown operations,
versions, fields, binding kinds, and malformed canonical values. Recipe extraction,
serialization, loading, and lazy graph construction do not compute Dask arrays.

Arbitrary callables passed to `Frame.apply(...)` are runtime-only. To make an extension
portable, declare its public Frame method with `@recipe_operation`, derive an immutable
registry containing that declaration, and supply the same registry to extraction,
loading, and application. See [Extending Recipe v2](../explanation/pipeline-recipe-developer-guide.md).
