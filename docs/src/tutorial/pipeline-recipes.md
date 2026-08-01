# Reuse a processing workflow with RecipePlan

When the same preprocessing steps should run on another recording, a
`RecipePlan` separates the public Frame operations from the input data. This
short tutorial builds one plan, serializes it in memory, replays it, and checks
the result against direct method calls.

## First replay

Start with ordinary Frame methods. Extract the plan from the processed template,
round-trip its JSON-compatible mapping, and apply it to a different Frame:

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

loaded = RecipePlan.from_dict(json.loads(json.dumps(plan.to_dict())))
new_signal = wd.from_numpy(
    np.array([[2.0, 5.0, 8.0, 14.0]]),
    sampling_rate=8_000,
    metadata={"recording": "next"},
    ch_labels=["sensor"],
)

replayed = loaded.apply({"signal": new_signal})
direct = new_signal.remove_dc().normalize()
np.testing.assert_allclose(replayed.data, direct.data)
assert replayed.metadata == {"recording": "next"}
print("Replay matches direct calls: yes")
```

The assertion is the important part: replay and direct calls have the same
numerical result, while runtime metadata comes from the new input. A plan stores
reusable operation intent, not the template's samples.

## Continue from here

- Run the <a href="../../learning-path/06_reusable_pipeline_recipes.html">Reusable Pipeline Recipes Learning Path</a>
  for a complete create, save, load, apply, and result-verification workflow.
- Use the [RecipePlan How-to](../how-to/pipeline-recipes.md) for file persistence,
  multiple inputs, external arrays, and runtime-only errors.
- Consult the [Pipeline API Reference](../api/pipeline.md) for generated
  signatures and exceptions.
- Read [Recipe Design](../explanation/pipeline-recipe-design.md) for the
  separation between workflow intent, data, and provenance.
