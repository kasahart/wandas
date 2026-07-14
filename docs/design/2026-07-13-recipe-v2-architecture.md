# ADR: Recipe v2 canonical semantic plan

- **Status:** Accepted
- **Date:** 2026-07-14
- **Compatibility:** Recipe contracts introduced in v0.4.0 may be replaced

## Decision

Recipe is a portable record of public Frame invocation intent. It is not a copy of
runtime operation objects, a Dask graph, resolved defaults, or display history. The
only replay path is:

```text
public Frame operation
  -> immutable SemanticOperation on LineageNode
  -> RecipePlan.from_frame(..., registry=...)
  -> RecipePlan.to_dict() / RecipePlan.from_dict(..., registry=...)
  -> RecipePlan.apply(..., registry=...)
```

`LineageNode` is the sole provenance source. Every Frame, including a newly loaded or
constructed Frame, has a source node. There is no optional lineage state, lazy source
fallback, Dask marker inspection, operation list, operation graph, or separate summary
lineage. `previous` remains a navigation/data-comparison convenience and never affects
history or Recipe extraction.

## Canonical semantic operation

One immutable descriptor contains all replay meaning for one public invocation:

- stable namespaced operation ID, such as `wandas.audio.normalize`;
- positive operation version;
- ordered, role-named bindings whose kind is `frame` or `array`;
- canonical immutable parameters;
- Frame output (scalar terminal results are outside the v2 contract).

Lineage edges correspond to the ordered bindings. A Frame binding owns a parent
lineage. An external array binding has no lineage and becomes a distinct named Recipe
input for each semantic binding occurrence. NumPy and Dask are the same persisted
`array` kind; their implementation type, chunks, values, and identity do not enter
history or the schema. Scalars and small configuration values are parameters, not
graph inputs.

One public invocation creates exactly one semantic node. Nested helpers reuse the
already selected node and may not reconstruct its operation ID, parameters, or edges.
Structural Frame validation applies to external `BaseFrame` subclasses as well as
built-in classes: a public operation returning a Frame must return the authoritative
semantic node selected at entry.

Runtime `AudioOperation` instances contain numerical execution state only. They do
not provide replay descriptors and are not retained by semantic lineage. Binary array
operations call the data-level lazy kernel directly; external arrays are not wrapped
in temporary Frames. Recipe extraction and lazy graph construction must never call
Dask `compute()`.

## Registration and execution

`RecipeRegistry` is an immutable value. The built-in registry is the default;
extensions use `registry.with_operation(...)` and pass the resulting value explicitly
to extraction, loading, validation, and execution. There is no mutable global
extension registry.

Each Recipe-capable public operation has one `@recipe_operation(...)` declaration.
The declaration creates the semantic capture contract and the matching registry
entry. A registry entry owns the operation ID/version, accepted ordered bindings,
parameter validation/decoding, handler, and output contract. Handlers receive only
ordered runtime inputs and decoded immutable parameters. Serialized data never
contains Python module, class, method, or function paths.

Adding a unary operation, typed Frame transition, or true multi-Frame operation must
not add branches to the graph model, compiler, validator, executor, or serializer.
Unregistered or semantically impure/nondeterministic operations may execute normally
but fail whole-plan extraction with the node and operation ID. There is no implicit
cut that turns an unsupported intermediate result into a new input.

## Canonical values and schema

Recipe schema version 2 is a deliberate breaking schema. Version 1 and unknown fields
are rejected; no compatibility loader or migration layer is provided.

The canonical value grammar contains null, booleans, integers, strings, lossless
numeric literals, lists, and explicit map entries. Ordinary parameter maps require
string keys. An operation may explicitly opt into ordered key/value entries when
non-string keys are part of its public contract. Tuple/list inputs normalize to a
canonical list unless an operation decoder restores a tuple. NaN, infinities, complex
numbers, and NumPy scalar dtype/value pairs use collision-proof tagged literals.
Arrays, callables, arbitrary objects, and values that cannot round-trip losslessly are
rejected.

The persisted node contains `operation`, `version`, `inputs`, and `params`; execution
behavior comes solely from the supplied registry entry. Serialization is deterministic
and fail-closed for schema fields, value tags, graph references, registered versions,
binding kinds, Frame outputs, dead nodes, and unused inputs.

## Public APIs and persistence

The user-facing Recipe surface is limited to:

- `RecipePlan.from_frame(frame, ..., registry=None)`;
- `plan.apply(inputs, registry=None)`;
- `plan.to_dict()`;
- `RecipePlan.from_dict(payload, registry=None)`.

Call-family classes, codecs, builders, and callable-path loaders are not public APIs.
Every operation returns a Frame carrying the authoritative semantic lineage.

`operation_history` is the single public display history. It is a defensive JSON-safe
projection with records containing `operation`, `version`, and `params`. Traversal is
ordered depth-first/topologically, visits a shared node once by identity, and emits the
current operation last. WDF stores this display history, not an executable Recipe.
On load it becomes an immutable history prefix on the new source node; subsequent live
operations append normally. WDF never embeds handlers or executable plans.

## Frame behavior fixed with this redesign

- Frame operations are immutable; `inplace` arguments and Recipe patches for them are
  removed.
- `+` remains strict element-wise arithmetic. Frame/Frame operands require equal
  sampling rate, shape, and semantic axes. Frame/array operands retain NumPy/Dask
  broadcasting and laziness.
- Ambiguous `ChannelFrame.add()` is replaced by `mix(other, *, align="strict",
  snr_db=None)`. Scalars are invalid. Mixing ignores `source_time_offset` when matching
  samples: different source periods may mix by array index. Sampling rates must match.
  The left/base frame owns output length, metadata, labels, and offsets. `strict`
  requires equal length; `pad` accepts only a shorter other input and zero-pads it;
  `truncate` accepts only a longer other input and truncates it. Channel counts must
  match, except a mono other input broadcasts across base channels.
- `add_channel` accepts a raw 1-D array or `(1, samples)` array only. Multiple added
  channels require a `ChannelFrame`; 2-D multi-channel input is never flattened.
- Recipe indexing supports an integer or label, channel slice, integer/label list,
  one-dimensional integer array, one-dimensional boolean mask, and a channel selector
  followed by slices for remaining axes. Stepped or reversed time-axis slices are
  rejected at the public Frame layer. Full NumPy advanced indexing is out of scope.
  Label and JSON-literal query intent is resolved when applying the plan; regex and
  callable queries are not portable.

## Explicitly removed or superseded

- replay descriptor families and descriptor-to-codec dispatch;
- public call-family classes and `RecipePlanBuilder`;
- Dask operation marker tasks and graph extraction;
- `frame.operations`, `frame.operation_graph`, and `frame.operation_summaries`;
- `_source_lineage`, `_lineage_or_source()`, and optional lineage branches;
- duplicate WDF operation-summary snapshot state;
- serialized callable/import paths and mutable global extension registration;
- Recipe v1 schema and compatibility behavior.

DI context replacement, broad non-Recipe registry redesign, mechanical splitting of
`calls.py`, and a general-purpose arbitrary replay-value system are not part of this
change.
