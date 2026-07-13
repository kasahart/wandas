# Recipe v2 design

```text
public Frame call
  -> immutable typed ReplayDescriptor in semantic lineage
  -> LineageRecipeCompiler + frozen ReplayCodecRegistry
  -> RecipePlan
  -> one validator / executor / serializer / loader
```

`RecipeNode.inputs` is the only executable edge owner. Calls describe behavior only.
Compiler memoization preserves shared runtime lineage, while persisted node IDs remain
document-local references rather than Python object or global semantic identities.

Operation families remain distinct: unary audio, typed transition, binary/external,
indexing, add-channel, custom, terminal, and true multi-input. A family codec converts
an immutable descriptor to an edge-free call. The generic traversal never branches on
operation names and never reads `operation_graph`.

The schema is `wandas.recipe` version 1. The loader validates the complete graph and
fails closed for unknown fields, families, versions, callable paths, and value-tree
shapes. WDF executable persistence and Dask graph restoration are deferred.
