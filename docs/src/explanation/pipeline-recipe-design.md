# Recipe v2 design

```text
public Frame call
  -> @recipe_operation captures one immutable SemanticOperation
  -> LineageNode (the sole provenance authority)
  -> LineageRecipeCompiler + immutable RecipeRegistry
  -> RecipePlan
  -> one validator / executor / serializer / loader
```

Each semantic operation contains a stable ID and version, ordered input bindings, and
canonical immutable parameters. `RecipeNode.inputs` is the only persisted edge owner.
Compiler memoization preserves shared runtime lineage, while persisted node IDs are
document-local references rather than Python object identities.

Every operation shape uses the same model. Unary operations, typed Frame transitions,
binary and external-array operations, indexing, channel addition, and true multi-Frame
operations differ only in their registry declaration and ordered bindings. Adding one
does not add a branch to the compiler, validator, executor, or serializer.

The schema is `wandas.recipe` version 2. It stores stable operation IDs, versions,
ordered edge references, and a tagged canonical value grammar. The loader validates
the complete graph and fails closed for unknown fields, operations, versions, ambiguous
binding kinds, and malformed values. Plans currently return Frames; scalar terminal
results are deliberately outside the Recipe contract.

WDF stores `operation_history` as a display-only source prefix. Loading a WDF starts a
new executable lineage source; it does not restore Python callables or a Dask graph.
