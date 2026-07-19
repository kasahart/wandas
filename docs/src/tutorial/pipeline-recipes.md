# Reuse a processing workflow with RecipePlan

When the same preprocessing steps must run on another recording, copying a method
chain also copies its input assumptions into application code. A `RecipePlan` records
the public Frame operations and their parameters separately from the data, so the same
workflow can be inspected, stored, and applied to a new input.

This tutorial builds a small preprocessing workflow, turns it into a Recipe, and proves
that replay produces the same result as calling the Frame methods directly.

## Build the workflow once

Create a representative input and process it with ordinary public Frame methods. No
special builder API is required.

```python exec="on" session="recipe_tutorial"
import json

import numpy as np
import wandas as wd

from wandas.pipeline import RecipePlan

template = wd.from_numpy(
    np.array([[1.0, 2.0, 4.0, 7.0]]),
    sampling_rate=8_000,
    ch_labels=["sensor"],
)
template_result = template.remove_dc().normalize()

plan = RecipePlan.from_frame(template_result, input_names=("signal",))
payload = plan.to_dict()

print("Recipe input:", payload["inputs"][0]["name"])
print("Operations:", [node["operation"] for node in payload["nodes"]])
```

`RecipePlan.from_frame()` reads semantic lineage already attached by the public calls.
One public call becomes one node, and the original sample values are not stored in the
plan.

## Store and load the portable schema

`to_dict()` returns a strict JSON-compatible schema. A JSON roundtrip demonstrates
that the payload does not retain live Python operation objects. Loading resolves its
stable operation IDs through the built-in registry. A plan does not retain a registry;
extensions must pass the same immutable registry to `from_frame()`, `from_dict()`, and
`apply()`.

```python exec="on" session="recipe_tutorial"
recipe_json = json.dumps(payload)
loaded_plan = RecipePlan.from_dict(json.loads(recipe_json))

print("Schema:", payload["schema"], payload["version"])
print("Serialized bytes:", len(recipe_json.encode("utf-8")))
```

## Apply it to a new Frame

Runtime inputs are supplied by the names chosen during extraction. Applying a plan
builds a lazy Frame workflow; numerical data is materialized only when this example
reads `frame.data` to verify the result.

```python exec="on" session="recipe_tutorial"
new_signal = wd.from_numpy(
    np.array([[2.0, 5.0, 8.0, 14.0]]),
    sampling_rate=8_000,
    metadata={"recording": "next"},
    ch_labels=["sensor"],
)

replayed = loaded_plan.apply({"signal": new_signal})
direct = new_signal.remove_dc().normalize()

np.testing.assert_allclose(replayed.data, direct.data)
assert replayed.metadata == {"recording": "next"}

print("Replay matches direct calls: yes")
print("Runtime metadata:", replayed.metadata)
print("History entries:", len(replayed.operation_history))
```

The plan supplies operation intent; the runtime Frame supplies samples, metadata,
labels, sampling rate, and source-time information. The input Frame remains unchanged.

## Multiple inputs remain explicit

Frame arithmetic, `mix()`, and external NumPy or Dask operands become additional named
inputs. Their values and container implementation are not embedded in the Recipe.

```python
mixed = base.mix(other)
mix_plan = RecipePlan.from_frame(mixed, input_names=("base", "other"))
result = mix_plan.apply({"base": next_base, "other": next_other})
```

`mix()` combines samples by array index even when source-time offsets describe
different source periods. Align recordings explicitly first when source-time matching
is required.

## Where to go next

- Run the executable
  <a href="../../learning-path/06_reusable_pipeline_recipes.html">Reusable Pipeline Recipes learning path</a>.
- Use the [RecipePlan how-to](../how-to/pipeline-recipes.md) for multi-input and error
  boundaries.
- Consult the [Pipeline API reference](../api/pipeline.md) for signatures and exceptions.
- Read [Recipe design](../explanation/pipeline-recipe-design.md) only when you need the
  persistence or extension model.
