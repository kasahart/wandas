# Extending Recipe v2

Choose the semantic family first. Add or opt in the runtime operation, then prove the
extension through the public path:

```text
public Frame operation
  -> semantic lineage
  -> RecipePlan.from_frame(...)
  -> RecipePlan.to_dict()
  -> RecipePlan.from_dict(...)
  -> RecipePlan.apply(...)
```

A descriptor-to-codec unit probe is useful local evidence, but it is not an extension
probe: it cannot prove that public semantic capture, compilation, persistence, loading,
and execution agree. The complete path is required for same-frame unary operations,
typed Frame transitions, and true multi-frame operations.

Adding a new operation within one of those families must not require changes to the
central graph model, compiler, validator, executor, or serializer. If it does, the
family contract or registration boundary is incomplete and should be fixed there
rather than adding another central dispatch branch.

New codecs are registered before a registry is frozen. A codec returns an edge-free
call plus ordered typed bindings; the registry verifies those bindings against the
descriptor contract. Add characterization tests for metadata, source time, operand
order, Dask laziness, schema roundtrip, and unknown-version rejection.

Do not recover semantics from display history, add graph references to calls, embed
external arrays in params, or introduce a second execution path.
