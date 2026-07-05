# Recipe Input Name Defaults Design

## Goal

Close issue #263 by making graph Recipe input naming easier to explain and safer to replay.

The current implementation has two default naming styles:

- `GraphRecipeSpec.from_frame(processed)` uses `left` and `right`.
- `NodeGraphRecipeSpec.from_frame(processed)` uses `input_0`, `input_1`, and so on.

This is confusing because `left` and `right` can also mean audio channel labels, while Recipe graph inputs are external runtime inputs. The default names should be mechanical and consistent across graph recipe types.

## Decision

Do not infer semantic input names from Python variable names, frame labels, channel labels, or source metadata in this issue.

Instead, unify omitted `input_names` defaults:

- `GraphRecipeSpec.from_frame(..., input_names=None)` uses `input_0` and `input_1`.
- `NodeGraphRecipeSpec.from_frame(..., input_names=None)` continues to use `input_0`, `input_1`, ...
- The order is the operation-graph source order. For a binary expression, `input_0` is the left operand and `input_1` is the right operand.
- Users who want readable names should pass explicit `input_names=("signal", "noise")`.

## Why Not Infer Names Yet

Runtime lineage does not preserve Python variable names. Frame labels and channel labels are not the same as graph input names:

- a frame can contain multiple channels with labels such as `left` or `right`;
- multiple different frames can share the same frame label;
- array operands and raw `add_channel` data do not have frame labels;
- generated names must be stable enough to serialize and replay later.

Inferring names from unstable or ambiguous metadata would make a Recipe look more user-friendly while weakening the replay contract.

## UX Contract

The default names are intentionally plain:

```python
recipe = GraphRecipeSpec.from_frame(signal + noise)
replayed = recipe.apply({"input_0": new_signal, "input_1": new_noise})
```

Readable names remain explicit:

```python
recipe = GraphRecipeSpec.from_frame(
    signal + noise,
    input_names=("signal", "noise"),
)
replayed = recipe.apply({"signal": new_signal, "noise": new_noise})
```

This keeps the default API usable without pretending Wandas knows semantic names that were never recorded.

## Implementation Scope

Change only the default naming contract:

1. Update `GraphRecipeSpec.from_frame(...)` so omitted `input_names` resolves to `("input_0", "input_1")`.
2. Keep explicit `input_names` behavior unchanged.
3. Keep `NodeGraphRecipeSpec.from_frame(...)` default behavior unchanged.
4. Update tests that currently expect `left` / `right` defaults.
5. Add or update docs to state that no semantic name inference happens yet.

## Compatibility

This is a behavior change for callers who relied on omitted `GraphRecipeSpec` names and then applied recipes with `{"left": ..., "right": ...}`.

The change is acceptable because:

- `GraphRecipeSpec.from_frame(..., input_names=None)` is recent prototype API;
- explicit `input_names=("left", "right")` still supports the previous spelling;
- the new default aligns with existing `NodeGraphRecipeSpec` behavior and avoids channel-name ambiguity.

## Error Handling

No new error type is needed.

Existing validation remains:

- explicit `input_names` must have one name per parent;
- `GraphRecipeSpec` explicit names must be distinct;
- `NodeGraphRecipeSpec` external input names must be unique after duplicate source paths are collapsed by order-preserving uniqueness;
- empty or non-string names continue to fail.

## Testing

Focused tests should prove:

- `GraphRecipeSpec.from_frame(processed)` defaults to `input_0` / `input_1`;
- binary operand order is preserved for non-commutative operations such as subtraction;
- frame labels such as `"signal"` and `"noise"` are not inferred as Recipe input names;
- explicit `input_names=("signal", "noise")` still works;
- `NodeGraphRecipeSpec.from_frame(processed)` still defaults to `input_0`, `input_1`, ...

## Documentation

Update the how-to and extraction-boundaries docs:

- Replace `left` / `right` default examples with `input_0` / `input_1`.
- Explain that `input_0` is the left operand and `input_1` is the right operand for binary graph recipes.
- Recommend explicit `input_names` for user-facing or persisted recipes.
- State that Python variable-name and frame-label inference is outside the current contract.

## Issue Closure

The PR should use `Closes #263`.

It should not close:

- #264, true DAG identity;
- #265, automatic graph recipe dispatch from `RecipeSpec.from_frame(...)`;
- #257, WDF persistence;
- #258, interoperability/export.
