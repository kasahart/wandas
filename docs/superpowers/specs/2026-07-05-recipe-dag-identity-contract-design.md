# Recipe DAG Identity Contract Design

## Goal

Close issue #264 by making the current Recipe graph identity contract explicit.

The important user-visible question is whether a Recipe extracted from a graph with a shared branch preserves true shared node identity, or whether it represents the same work as duplicated parent paths.

## Decision

Do not add true DAG identity to Recipe extraction in this issue.

Keep the current explicit contract:

- `NodeGraphRecipeSpec.from_frame(...)` extracts a tree-shaped graph recipe.
- Shared runtime branches are represented as duplicated parent paths when the current runtime lineage exposes them that way.
- Extraction does not deduplicate parents by Python object identity, source object identity, frame label, channel label, or equivalent-looking operation arguments.
- Replay may recompute duplicated branch paths.
- Repeated external leaves may intentionally use the same runtime input name when the caller passes repeated explicit names such as `input_names=("base", "base")`.

This preserves replay meaning without claiming identity information that the current runtime lineage does not reliably store.

## Why Not Preserve True DAG Identity Now

True DAG identity is a larger runtime-lineage contract, not just a Recipe serialization tweak.

Wandas would first need to decide:

- whether each operation-graph node has a stable identity separate from its current parent path;
- how shared references are represented in serialized Recipe data;
- whether replay must reuse one computed intermediate or only preserve equivalent user-visible results;
- how stable node identity interacts with Dask laziness and delayed computation;
- how the same identity contract will work for WDF persistence and external export.

Adding partial deduplication inside Recipe extraction would be fragile because two paths can look equivalent without being the same user operation, and one shared Python object can be transformed through contexts where stable replay identity is not recorded.

## UX Contract

Users should read `NodeGraphRecipeSpec` as a tree-shaped replay recipe, not as a full DAG identity graph.

For example, if a processed frame combines the same base branch twice, the extracted recipe may have two source leaves:

```python
recipe = NodeGraphRecipeSpec.from_frame(
    processed,
    input_names=("base", "base"),
)
replayed = recipe.apply({"base": new_base})
```

The repeated name is explicit. It says both leaves should be supplied by the same external replay input. It does not mean the extracted graph stores one shared internal node.

Default names remain mechanical:

```python
recipe = NodeGraphRecipeSpec.from_frame(processed)
replayed = recipe.apply({"input_0": left_base, "input_1": right_base})
```

Callers who need semantic names for saved or user-facing recipes should pass explicit `input_names`.

## Implementation Scope

This issue should stay small and document the existing behavior rather than changing runtime lineage.

Implementation should:

1. Document that `NodeGraphRecipeSpec` stores tree-shaped graph recipes and may duplicate shared branch paths.
2. Add focused extraction and replay tests for a shared-branch shape.
3. Add or adjust tests that demonstrate repeated explicit input names such as `input_names=("base", "base")`.
4. Keep existing tree-shaped graph recipes working.

Implementation should not:

- add runtime operation node IDs;
- add a new Recipe graph schema;
- infer source identity;
- deduplicate equivalent-looking parent paths;
- change Dask execution behavior;
- implement WDF persistence or export support.

## Compatibility

This is primarily a clarification. Existing `NodeGraphRecipeSpec` recipes continue to work.

The intended compatibility rule is:

- existing tree-shaped graph recipe extraction remains valid;
- existing replay behavior remains valid;
- any new tests should pin the documented duplicated-path behavior, not require true DAG identity.

## Testing

Focused tests should cover:

- a graph shape where one processed branch is reused in more than one parent path;
- extraction produces a valid `NodeGraphRecipeSpec`;
- replay with repeated explicit input names uses the same supplied external frame for both leaves;
- replay preserves the same user-visible output as the original graph;
- existing tree-shaped multi-input extraction tests still pass.

Tests should not assert internal node identity because that identity is outside the current contract.

## Documentation

Update the Recipe docs where graph extraction is explained:

- support matrix: state that `NodeGraphRecipeSpec` supports tree-shaped graph recipes and duplicated shared branches;
- extraction boundaries: explain that true shared node identity is not preserved yet;
- how-to or developer guide: show repeated explicit input names for the same replay input when that is the intended user contract.

The beginner-facing wording should avoid runtime-only terms where possible. Use "the same branch may appear twice in the saved recipe" before introducing `operation_graph`.

## Follow-Up Issue

Track true runtime DAG identity separately in #270, "Define runtime node identity for true Recipe DAG sharing".

That issue covers:

- define stable runtime operation node identity;
- define Recipe references to shared nodes;
- decide whether replay reuses shared computed intermediates or only preserves equivalent output;
- coordinate with WDF Recipe persistence (#257) and Recipe export boundaries (#258).

That follow-up should not block #264 because #264 can be closed by documenting and testing the current tree-shaped contract.

## Issue Closure

The PR should use `Closes #264` only if it documents shared branch behavior and adds focused extraction/replay tests for shared and duplicated branches.

It should use `Related` for:

- #270, runtime node identity for true DAG sharing;
- #257, WDF persistence;
- #258, interoperability/export;
- #246, legacy lineage/history cleanup.

It should not touch #227 because source-time policy is being handled separately.
