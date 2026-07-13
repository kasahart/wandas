# ADR: Recipe v2 canonical replay architecture

- **Status:** Proposed for implementation
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

## Semantic lineage

One public Frame operation produces one semantic lineage node and one history record.
Internal helper calls are not lineage. Multidimensional indexing is one typed indexing
intent. Shared `LineageNode` identity is preserved and projections visit shared nodes
once by object identity; equal but separately invoked operations remain distinct.

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

## Persistence

The canonical schema starts at `{"schema": "wandas.recipe", "version": 1}`. Loader
validation is fail-closed for unknown fields, schema versions, call families, operation
versions, callable paths, duplicate mapping keys, and unsupported values. Callable
targets must be importable stable paths; method targets must be directly owned class
members with matching replay contracts. External NumPy and Dask arrays are named
inputs and are never embedded.

Runtime parameter snapshots and persistence use one immutable, collision-proof value
tree. User mappings are encoded as mapping nodes rather than inferred from reserved
JSON key shapes. Arbitrary objects and callables are not serialized as parameters.

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
