# Recipe Interoperability And Export Boundaries Design

## Goal

Close issue #258 by making Recipe interoperability and export boundaries explicit without expanding the current runtime contract.

The immediate decision is not "make every Recipe exportable". The decision is which conversions are supported today, which are intentionally unsupported, and where future WDF persistence should stop for now.

## Decision

Keep Wandas-native Recipe specs as the canonical representation.

The current `wandas.pipeline.sklearn` module remains a thin optional adapter for sklearn-style `Pipeline.transform(frame)` usage. It is not a general Recipe interchange layer.

Supported today:

- `WandasOperationTransformer(operation, **params)` wraps one Wandas `apply_operation(...)` call.
- Named sklearn transformers such as `HighPassFilter`, `LowPassFilter`, `BandPassFilter`, `Normalize`, and `RemoveDC` wrap one known Wandas operation.
- `transform(frame)` delegates to existing frame operations.
- `to_spec()` returns one `OperationSpec`.
- sklearn remains an optional dependency loaded only through `wandas.pipeline.sklearn`.

Unsupported today:

- automatic `RecipeSpec -> sklearn.pipeline.Pipeline` conversion;
- automatic `sklearn.pipeline.Pipeline -> RecipeSpec` conversion;
- `GraphRecipeSpec` or `NodeGraphRecipeSpec` sklearn conversion;
- `TerminalStep`, `CustomFunctionStep`, `MethodStep`, `TypedMethodStep`, `IndexingStep`, `ScalarOperationStep`, `BinaryFrameStep`, `BinaryOperandStep`, `AddChannelStep`, and `AddChannelDataStep` conversion to sklearn transformers;
- official joblib or skops export support for Wandas Recipe specs;
- Dask-ML integration beyond preserving normal frame laziness when existing frame methods run.

This keeps the current adapter honest: it is a convenience wrapper around frame operations, not a serialization or interchange format.

## WDF Boundary

WDF should not store executable Recipe specs as part of this issue.

For now, WDF only needs to preserve operation history / operation summaries for inspection. That means loaded WDF frames can show what happened, but users should not expect `load(...)` to restore an executable `RecipeSpec`.

The WDF contract is an inspection-only snapshot boundary:

- WDF save does not write an executable Recipe payload.
- WDF load does not restore pre-save runtime lineage or `operation_graph`.
- A frame loaded from WDF keeps the pre-save operation summaries as an inspection-only snapshot.
- If users process that loaded frame, display summaries should look like `snapshot at load time + new post-load operation summaries`.
- Repeating `save -> load -> process -> save -> load` should preserve summaries without dropping, duplicating, or reordering them.
- `RecipeSpec.from_frame(...)` on a loaded frame should not be expected to recover pre-save operations, because those operations are snapshot metadata, not runtime lineage.

Executable Recipe persistence remains #257. That later issue should define schema versioning, load behavior, portable custom functions, graph recipe handling, and unsupported future schema behavior.

This #258 work should not add `recipe_json`, a Recipe group, or any executable Recipe field to WDF.

## Why Not Broaden Interop Now

General Recipe interoperability looks tempting but would blur contracts that are still separate:

- Wandas Recipe replay preserves frame metadata, history, and Dask laziness by delegating to frame APIs.
- sklearn transformers expect a narrower fit/transform shape and do not naturally represent multi-input graph recipes, terminal metrics, indexing, add-channel data inputs, or custom frame transitions.
- joblib/skops can serialize Python objects, but that is not the same as a stable Wandas Recipe schema.
- Dask-ML integration needs a separate user story. Current Recipe replay can keep Dask arrays lazy because frame operations do; it does not need Dask-ML wrappers to satisfy today's contract.

Treating sklearn/joblib/skops as official Recipe export targets before these boundaries are needed would create compatibility obligations with little current UX benefit.

## UX Contract

Users should see three different concepts:

1. **Wandas Recipe**: the canonical replay contract for Wandas frame work.
2. **sklearn adapter**: a convenience layer for users who want sklearn `Pipeline([...]).transform(frame)` around simple Wandas operations.
3. **WDF history**: persisted inspection metadata, not executable Recipe persistence.

Docs should say:

- use `RecipeSpec`, `GraphRecipeSpec`, and `NodeGraphRecipeSpec` when replay is the goal;
- use `wandas.pipeline.sklearn` when sklearn-style transform ergonomics are useful;
- do not use sklearn, joblib, skops, or WDF as the canonical Recipe format today.

## Implementation Scope

Implementation should be documentation-first with focused tests only where the current adapter behavior needs pinning.

Do:

1. Add a Recipe interoperability/export support matrix.
2. Update existing Recipe docs so #258 boundaries are visible from the support matrix and design pages.
3. Add focused sklearn adapter tests for the supported one-operation `to_spec()` contract if coverage is missing.
4. Add explicit documentation that joblib/skops export and Recipe WDF persistence are unsupported/deferred.

Do not:

- add new public conversion APIs;
- add joblib/skops code paths;
- add Dask-ML wrappers;
- add WDF Recipe persistence fields;
- broaden sklearn adapter behavior to graph, terminal, custom, indexing, method-aware, or add-channel steps.

## Testing

Tests should remain small:

- existing sklearn adapter tests should continue proving `fit`, `transform`, `get_params`, `set_params`, and `to_spec()`;
- add tests only if a supported or rejected behavior is not already covered;
- no tests should require joblib, skops, Dask-ML, or WDF Recipe persistence.

Documentation checks are enough for unsupported export targets unless an explicit public API is added later. This PR should not add that API.

## Issue Closure

The PR should use `Closes #258` if it documents:

- supported Wandas Recipe / sklearn adapter behavior;
- unsupported Recipe step conversions;
- joblib/skops export boundaries;
- Dask / Dask-ML boundaries;
- WDF non-goal: history/summaries only for now, executable Recipe persistence deferred to #257.

It should use `Related` for:

- #257, executable Recipe WDF persistence;
- #270, true DAG identity for Recipe graph sharing;
- #246, legacy lineage/history cleanup.

It should not touch #227.
