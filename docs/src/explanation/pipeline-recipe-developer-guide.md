# Extending Recipe v2

Choose the semantic family first. Add or opt in the runtime operation and test its
descriptor-to-call behavior. Existing families require no changes to `RecipePlan`, the
generic compiler, validator, executor, or serializer dispatch.

New codecs are registered before a registry is frozen. A codec returns an edge-free
call plus ordered typed bindings; the registry verifies those bindings against the
descriptor contract. Add characterization tests for metadata, source time, operand
order, Dask laziness, schema roundtrip, and unknown-version rejection.

Do not recover semantics from display history, add graph references to calls, embed
external arrays in params, or introduce a second execution path.
