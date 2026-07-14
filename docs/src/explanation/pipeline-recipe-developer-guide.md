# Extending Recipe v2

A Recipe-capable operation has one `@recipe_operation(...)` declaration. The
declaration owns its stable ID/version, accepted ordered bindings, parameter
validation, and Frame-returning handler. Parameters use the shared canonical value
grammar; an operation does not define a family-specific codec. The same declaration
captures public semantic lineage and supplies the immutable registry entry, so those
contracts cannot drift.

Extensions never mutate a process-wide registry. Start with an immutable registry and
derive another value:

```python
registry = default_recipe_registry().with_operation(my_operation_definition)
plan = RecipePlan.from_frame(result, registry=registry)
loaded = RecipePlan.from_dict(plan.to_dict(), registry=registry)
replayed = loaded.apply({"input_0": source}, registry=registry)
```

Only operations explicitly present in the supplied registry are portable. A runtime
operation may remain undeclared, but extraction then fails at that node rather than
cutting the graph or serializing a Python callable path.

## Complete extension probe

Test the public path, not a descriptor or registry entry in isolation:

```text
public Frame operation
  -> semantic lineage
  -> RecipePlan.from_frame
  -> RecipePlan.to_dict
  -> RecipePlan.from_dict
  -> RecipePlan.apply
```

Use that probe for a unary operation, a typed Frame transition, and a true multi-Frame
operation. Adding any of them must not modify the central model, compiler, validator,
executor, or serializer. If it does, improve the registration contract instead of
adding a family branch.

## Handler boundary

A handler receives ordered runtime inputs and decoded immutable parameters only. It
does not receive a compiler, executor, registry, import path, or mutable context.
Parameter validators are pure and run once during complete-plan validation. Handlers
validate operation-specific runtime shape, sampling rate, class, and metadata at apply
time. The common executor validates named inputs, graph kinds, and the authoritative
semantic lineage returned by Frame operations.

Use `frame` bindings for Frame operands and `array` for external NumPy/Dask operands.
Do not embed arrays, wrap them in temporary Frames, serialize container kinds, or
compute Dask values. Scalars and small JSON-like configuration belong in parameters.
Persist invocation intent: omitted arguments stay omitted and input-dependent defaults
are resolved by the handler when the plan is applied.

Add tests for metadata, source-time offsets, operand order, mutation isolation, Dask
laziness, deterministic schema roundtrip, and unknown operation/version rejection.
