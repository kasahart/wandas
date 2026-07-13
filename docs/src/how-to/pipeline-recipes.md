# Compile and replay a RecipePlan

Recipe v2 has one public graph model. Process a frame normally, then compile its
semantic lineage:

```python
from wandas.pipeline import RecipePlan

processed = source.remove_dc().normalize()
plan = RecipePlan.from_frame(processed, input_names=("signal",))
replayed = plan.apply({"signal": another_frame})
```

Frame, external-array, binary, indexing, add-channel, typed transition, custom, and
multi-input calls all use the same plan. External NumPy and Dask arrays are named
inputs; their values are never embedded in the Recipe.

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

The loader rejects unknown schema, call, and operation versions. Custom callables must
be importable module-level functions. Recipe compilation and lazy graph construction do
not compute Dask arrays.
