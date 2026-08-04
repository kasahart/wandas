# Work with RecipePlan

For a guided, executable introduction, start with the
<a href="../../learning-path/06_reusable_pipeline_recipes.html">Reusable Pipeline Recipes Learning Path</a>.
Use this how-to when you already know the task you need to complete.

RecipePlanを初めて段階的に学ぶ場合は、実行可能な
<a href="../../learning-path/06_reusable_pipeline_recipes.html">Reusable Pipeline Recipes Learning Path</a>
から始めてください。具体的な作業が決まっている場合に、このHow-toを使います。

## Choose WDF or Recipe

WDF saves one concrete, typed Frame result for later inspection or loading.
Recipe saves reusable workflow intent so the same public operations can run on
another input. A Recipe does not contain the original sample data; choose WDF
when the result is the artifact and Recipe when the process is the artifact.

## What a Recipe stores

- `previous` is a process-local reference to the immediately preceding Frame; it
  is not portable Recipe input.
- `lineage` is the complete provenance of Frame inputs.
- Runtime-only callables, regex predicates, and opaque Python objects cannot be
  serialized into a Recipe; extraction fails explicitly at that operation.

Named Frame and external-array inputs are supplied again at replay. The
serialized artifact stores operation intent and bindings, not the input samples.

## Set up the examples

The snippets below form one small workflow. Prepare a source, a replacement
Frame, and an external array before choosing the task you need:

```python
import numpy as np
import wandas as wd

source = wd.from_numpy(
    np.array([[1.0, 2.0, 4.0, 7.0]]),
    sampling_rate=8_000,
    ch_labels=["sensor"],
)
another_frame = wd.from_numpy(
    np.array([[2.0, 5.0, 8.0, 14.0]]),
    sampling_rate=8_000,
    ch_labels=["sensor"],
)
external_array = np.ones((1, 4))
processed = source.remove_dc().normalize()
```

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
artifact or wrapped in temporary Frames. At replay, supply a NumPy or Dask array
with a shape and dtype accepted by the operation. The original container type and
Dask chunking are not stored in the Recipe.

## Handle runtime-only operations

Some calls intentionally cannot be portable. Arbitrary callables passed to
`Frame.apply()`, callable or regex channel predicates, and opaque Python objects
must remain runtime-only. Recipe extraction fails at the unsupported operation
instead of silently dropping part of the workflow.

For a new portable operation, follow the Recipe-capable section of the
[Frame and Operation extension guide](../contributing/frame-operation-extensions.md).
The guide covers the contributor-only extension procedure and focused end-to-end
tests.
