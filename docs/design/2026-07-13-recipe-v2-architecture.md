# ADR: Recipe v2 canonical replay architecture

- **Status:** Implemented
- **Date:** 2026-07-13
- **Base:** `origin/develop@b808c8e`

## Decision

Recipe v2 is a destructive replacement for the three v1 spec families. Its public
model is one immutable `RecipePlan` graph. Legacy specs, edge-bearing steps, the old
dictionary schema, and compatibility projections are removed rather than deprecated.

The runtime-to-persistence boundary is fixed when lineage is created:

```text
public Frame operation
  -> runtime result + immutable family ReplayDescriptor
  -> LineageNode(inputs own runtime edges)
  -> one LineageRecipeCompiler traversal
  -> RecipePlan(nodes own executable edges)
  -> one validator / executor / serializer / loader
```

`ReplayDescriptor` is the only replay input consumed by codecs. Codecs never inspect
the live runtime operation. This prevents later mutation of operation parameters,
bindings, versions, callables, or operand order from changing history or Recipe output.

Runtime operations do not retain user-owned mutable values. Binary NumPy and Dask
operands are captured as immutable descriptors when the operation is created, and
add-channel parameters are snapshotted at the same boundary. Display history records
semantic values, not container implementation details such as NumPy versus Dask.

## Semantic lineage

One public Frame operation produces one semantic lineage node and one history record.
Internal helper calls are not lineage. Multidimensional indexing is one typed indexing
intent. Shared `LineageNode` identity is preserved and projections visit shared nodes
once by object identity; equal but separately invoked operations remain distinct.

The public semantic entry point is the sole owner of lineage creation. In particular,
`@semantic_index` creates the one indexing node used by every indexing branch;
branches may select data and metadata, but must not rebuild lineage or its parameters.
Semantic result validation uses the structural Frame contract rather than a defining
module name so that external `BaseFrame` subclasses receive the same atomicity check.

Source identity is in-memory only. Compiler memoization may use runtime object identity,
but persisted node IDs are deterministic graph references and never Python object IDs.

## Operation families

The common runtime envelope is deliberately small: operation ID, positive version,
purity, ordered typed input bindings, and immutable parameters. Replay remains split
into semantic families: unary audio, typed Frame method, binary/scalar/external array,
indexing, add-channel, custom function, terminal, and true multi-input.

Only unary same-frame audio operations may use generic replay, and only by explicit
opt-in. Calls and descriptors never contain graph references.

## Canonical graph

`RecipeInput` identifies named frame or external-array inputs. `RecipeNode.inputs` is
the sole executable edge owner. Validation enforces topological availability, call
arity and input kinds, frame/terminal output, terminal-at-output, and the absence of
dead nodes or unused inputs. Execution evaluates that one validated graph only.

At the compiler boundary, an external array is represented only by
`BoundInput.lineage is None`; there is no duplicate external flag. Frame inputs always
carry a source or operation lineage, so a missing parent cannot silently create a new
external Frame input. For add-channel replay, `AddChannelOperation.input_kind` is the
sole runtime owner of the second input kind. Replay descriptors derive that kind from
their contract rather than storing another synchronized field. Descriptors likewise
derive semantic operation identity from the contract and scalar execution values from
their immutable parameter tree; neither value has a second synchronized field.

## Persistence

The canonical schema starts at `{"schema": "wandas.recipe", "version": 1}`. Loader
validation is fail-closed for unknown fields, schema versions, call families, operation
versions, callable paths, duplicate mapping keys, and unsupported values. Callable
targets must be importable stable paths; method targets must be directly owned class
members with matching replay contracts. External NumPy and Dask arrays are named
inputs and are never embedded.

Runtime operations own defensive parameter snapshots, while persistence uses one
immutable, collision-proof value tree. User mappings are encoded as mapping nodes
rather than inferred from reserved JSON key shapes. Arbitrary objects and callables
are not serialized as parameters.

## Explicitly removed

- `RecipeSpec`, `GraphRecipeSpec`, `NodeGraphRecipeSpec`, `GraphNodeSpec`
- edge-bearing step DTOs and step/call conversion
- `steps_from_graph` and dictionary graph reconstruction helpers
- v1 `to_dict()` schema and compatibility serializer
- Recipe extraction from `operation_graph`

`operation_graph` remains a debug projection. WDF executable Recipe persistence and
Dask graph restoration remain separate work.

## Acceptance gates

The implementation must meet the supplied structural, size, behavior, laziness, and
performance KPIs. In addition to normal tests, adversarial gates cover snapshot
mutation, marker collisions, duplicate keys, callable ownership, dead graph elements,
shared DAG identity, reflected operand order, and compute bombs during extraction and
lazy graph construction.

## Public API

`RecipePlan.from_frame(frame, input_names=...)` compiles runtime semantic lineage.
`plan.apply({"name": frame_or_array})` executes it, while `plan.to_dict()` and
`RecipePlan.from_dict(payload)` are the only persistence entry points. For explicit
graphs, `RecipePlanBuilder` creates the same model and therefore cannot bypass the
validator or executor.

The public call families are edge-free values. Adding an operation within an existing
family changes the operation implementation and its tests; it does not change the
graph model, compiler traversal, validator, executor, or serializer dispatch.

## Deferred work

WDF may retain display history, but executable Recipe persistence in WDF is deferred.
The schema records graph identity and ordered roles, not a content-derived global
identity. Cross-file node identity and Dask graph restoration require a separate
persistence contract.
