# How RecipePlan separates workflow intent from data

A Recipe is a portable description of public Frame calls. It is not a saved Frame,
Dask graph, runtime operation object, or copy of `operation_history`.

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

## Why external arrays do not become temporary Frames

NumPy and Dask operands do not carry a sampling rate, channel metadata, or source-time
meaning. Wrapping them in temporary Frames would invent those values and add special
cases for broadcasting and laziness. Recipe therefore models them as one persisted
`array` input kind while Frame methods continue to use the array-level lazy kernel.

## Supported and excluded intent

Unary operations, typed Frame transitions, scalar and Frame arithmetic, external-array
arithmetic, indexing, channel addition, and signal mixing all use the same node model.
An extension uses the same model when its public method has an explicit
`@recipe_operation` declaration.

Scalar terminal values, arbitrary callables, compiled regular expressions, and opaque
Python objects are deliberately excluded. Failing closed keeps a loaded plan
deterministic and prevents hidden executable imports.

The durable low-level contract is recorded in the repository ADR at
`docs/design/2026-07-13-recipe-v2-architecture.md`.
