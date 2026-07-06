# Recipe Dispatch Contract Design

## Goal

Close issue #265 by making the `RecipeSpec.from_frame(...)` graph boundary explicit for users and stable for type checkers.

Wandas now has three Recipe extractors:

- `RecipeSpec.from_frame(...)` for single-input linear recipes.
- `GraphRecipeSpec.from_frame(...)` for two frame inputs with one binary merge and an optional linear tail.
- `NodeGraphRecipeSpec.from_frame(...)` for tree-shaped graph recipes with multiple external inputs or external operands.

The open question is whether `RecipeSpec.from_frame(...)` should automatically return or delegate to the graph recipe types when it sees multi-input lineage.

## Decision

Do not add automatic graph dispatch to `RecipeSpec.from_frame(...)` in this issue.

Keep the current explicit API:

- `RecipeSpec.from_frame(...)` returns only `RecipeSpec`.
- Graph extraction uses `GraphRecipeSpec.from_frame(...)` or `NodeGraphRecipeSpec.from_frame(...)`.
- Multi-input lineage passed to `RecipeSpec.from_frame(...)` fails with `RecipeExtractionError`.
- The error message should explain that graph lineage needs a graph recipe extractor and should name the likely alternatives.

This keeps the public return type stable today while leaving room for a future higher-level factory.

## Why Not Dispatch Now

Automatic dispatch would make the first example feel easier:

```python
recipe = RecipeSpec.from_frame(signal + noise)
```

But it would also make the return type harder to reason about:

```python
RecipeSpec | GraphRecipeSpec | NodeGraphRecipeSpec
```

Those types have different `apply(...)` signatures:

- `RecipeSpec.apply(frame)`
- `GraphRecipeSpec.apply({"input_0": frame_a, "input_1": frame_b})`
- `NodeGraphRecipeSpec.apply({...})`

Returning graph recipes from `RecipeSpec.from_frame(...)` would make simple-looking code fail later when callers pass a single frame to `apply(...)`. It also makes static typing and beginner documentation harder because the same entry point no longer has one shape.

## Future Improvement Path

Automatic graph extraction remains a valid future UX improvement, but it should not change `RecipeSpec.from_frame(...)` directly.

If Wandas adds a convenience entry point later, prefer a new higher-level factory with an explicit union return type, for example:

```python
Recipe.from_frame(...)
# or
AnyRecipeSpec.from_frame(...)
```

That future API can document a discriminated recipe family and guide users to the correct `apply(...)` shape. It can also preserve `RecipeSpec.from_frame(...)` as the predictable linear extractor.

## UX Contract

When users call `RecipeSpec.from_frame(...)` on graph lineage, the failure should be actionable:

- state that the frame has graph or multi-input lineage;
- state that `RecipeSpec.from_frame(...)` only supports single-input linear recipes;
- suggest `GraphRecipeSpec.from_frame(...)` for one binary frame merge;
- suggest `NodeGraphRecipeSpec.from_frame(...)` for multiple merges, external operands, or `add_channel` graph inputs.

Example message shape:

```text
RecipeSpec.from_frame(...) cannot extract graph lineage as a linear recipe
  Use GraphRecipeSpec.from_frame(...) for one binary frame merge.
  Use NodeGraphRecipeSpec.from_frame(...) for tree-shaped graph recipes.
```

The exact text can follow the existing multiline `RecipeExtractionError` style.

## Implementation Scope

The implementation should stay small:

1. Keep `RecipeSpec.from_frame(...)` returning only `RecipeSpec`.
2. Improve graph-boundary error messages where current text is too terse or does not identify the right graph extractor.
3. Add or adjust tests that assert the type contract and actionable error text.
4. Update docs to state that automatic dispatch is intentionally deferred and graph recipes use explicit entry points.

No new recipe class, wrapper type, serialization format, or dispatch factory is added in this issue.

## Compatibility

This is primarily a clarification. Existing successful linear recipes keep working. Existing explicit graph recipes keep working.

The only intended behavior change is clearer error text for callers who pass graph lineage to `RecipeSpec.from_frame(...)`.

## Testing

Tests should cover:

- `RecipeSpec.from_frame(linear_frame)` still returns `RecipeSpec`.
- `RecipeSpec.from_frame(frame_a + frame_b)` raises `RecipeExtractionError` with graph extractor guidance.
- At least one graph case that needs `NodeGraphRecipeSpec.from_frame(...)` raises with node-graph guidance.
- `GraphRecipeSpec.from_frame(...)` and `NodeGraphRecipeSpec.from_frame(...)` remain supported for the same lineage examples.

The tests should assert stable message fragments, not the entire formatted error string.

## Documentation

Update the recipe how-to and extraction-boundary docs:

- explain that `RecipeSpec.from_frame(...)` is intentionally linear-only;
- keep the table that routes graph cases to `GraphRecipeSpec` or `NodeGraphRecipeSpec`;
- add a short future note that a higher-level automatic factory can be considered later, without changing the `RecipeSpec.from_frame(...)` return type.

## Issue Closure

The PR should use `Closes #265`.

It should not close:

- #264, true DAG identity;
- #257, WDF persistence;
- #258, interoperability/export;
- #227, source-time policy;
- #246, legacy lineage/history cleanup.
