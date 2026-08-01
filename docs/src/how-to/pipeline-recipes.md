# Work with RecipePlan

The [Recipe tutorial](../tutorial/pipeline-recipes.md) demonstrates the first
replay. Use these procedures when a concrete task needs more than that short
path.

## Extract and apply a plan

Process a Frame normally, then name the runtime inputs while compiling its
semantic lineage:

```python
from wandas.pipeline import RecipePlan

processed = source.remove_dc().normalize()
plan = RecipePlan.from_frame(processed, input_names=("signal",))
replayed = plan.apply({"signal": another_frame})
```

The names supplied during extraction must match the mapping passed to `apply()`.
The result is a Frame; materialize it only at the boundary required by the task.

## Save and load a standalone artifact

Save reusable operation intent separately from Frame data:

```python
path = plan.save("analysis")  # analysis.recipe.json
restored = RecipePlan.load(path)
replayed = restored.apply({"signal": another_frame})
```

Use `to_dict()` and `from_dict()` when the artifact must travel through another
JSON or storage layer. Saving does not overwrite an existing artifact unless
that is requested explicitly. The format-specific schema and API errors are in
the [Pipeline API Reference](../api/pipeline.md).

## Supply multiple inputs

Give every runtime Frame or external array a stable name. Input order and
alignment are defined by the public operation that created the plan.

```python
processed = source + external_array
plan = RecipePlan.from_frame(processed, input_names=("signal", "offset"))
replayed = plan.apply({"signal": another_frame, "offset": external_array})
```

External NumPy and Dask arrays remain named inputs; they are not embedded in the
artifact or wrapped in temporary Frames. Use the same container and compatible
shape at replay.

## Handle runtime-only operations

Some calls intentionally cannot be portable. Arbitrary callables passed to
`Frame.apply()`, callable or regex channel predicates, and opaque Python objects
must remain runtime-only. Recipe extraction fails at the unsupported operation
instead of silently dropping part of the workflow.

For a new portable operation, follow the Recipe-capable section of the
[Frame and Operation extension guide](../contributing/frame-operation-extensions.md).
The guide covers stable IDs, bindings, immutable registries, handler boundaries,
and the focused end-to-end test.
