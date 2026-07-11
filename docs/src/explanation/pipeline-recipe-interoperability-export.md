# Pipeline Recipe Interoperability And Export Boundaries

## Canonical Representation

Wandas-native Recipe specs are the canonical replay representation. Use `RecipeSpec`, `GraphRecipeSpec`, and `NodeGraphRecipeSpec` when the goal is replaying Wandas frame work.

## Supported Today

| Boundary | Status | Contract |
| --- | --- | --- |
| `wandas.pipeline.sklearn.WandasOperationTransformer` | Supported | Wraps one Wandas `apply_operation(...)` call and exposes sklearn `fit`, `transform`, `get_params`, `set_params`, and `to_spec()`. |
| Named sklearn transformers | Supported | `HighPassFilter`, `LowPassFilter`, `BandPassFilter`, `Normalize`, and `RemoveDC` wrap one known operation each. |
| `to_spec()` | Supported | Returns one `OperationSpec`; it does not build a full `RecipeSpec`. |
| sklearn optional dependency | Supported | `wandas.pipeline.sklearn` owns the sklearn import boundary. Core `wandas` import does not require sklearn. |

## Unsupported Today

| Boundary | Status | Reason |
| --- | --- | --- |
| `RecipeSpec -> sklearn.pipeline.Pipeline` | Unsupported | Recipe steps include frame-specific replay semantics that are not all sklearn transformers. |
| `sklearn.pipeline.Pipeline -> RecipeSpec` | Unsupported | Arbitrary sklearn transformers are not Wandas operations. |
| Graph recipes to sklearn | Unsupported | `GraphRecipeSpec` and `NodeGraphRecipeSpec` need named multi-input replay, while sklearn `Pipeline` is a single flowing transform chain. |
| Terminal/custom/indexing/method-aware/add-channel steps to sklearn | Unsupported | These steps need frame APIs, importable user code, indexing intent, typed frame transitions, external inputs, or terminal outputs outside the simple transformer contract. |
| joblib/skops Recipe export | Unsupported | Python object serialization is not a stable Wandas Recipe schema. |
| Dask-ML integration | Deferred | Current replay preserves Dask laziness by delegating to frame operations; Dask-ML wrappers need a separate use case. |

## WDF Inspection-Only Snapshot

WDF does not store executable Recipe specs today. It stores operation summaries for inspection.

Runtime lineage and `operation_graph` are the source of truth for executable replay. Operation summaries snapshots are inspection-only copies that carry display history across save/load. WDF load does not rebuild lineage from the snapshot. New operations after load are recorded in runtime lineage; display summaries are composed as `snapshot + post-load lineage delta`; save stores the composed summaries as the next inspection snapshot.

Because `RecipeSpec.from_frame(...)` reads runtime lineage / `operation_graph`, it should not recover pre-load snapshot operations as Recipe steps.

Executable Recipe WDF persistence remains #257.

## Choosing The Right Tool

| Goal | Use |
| --- | --- |
| Replay Wandas frame work | Wandas Recipe specs |
| Use sklearn `Pipeline.transform(frame)` ergonomics for simple operations | `wandas.pipeline.sklearn` |
| Inspect processing history after WDF load | WDF operation summaries snapshot |
| Persist executable Recipes in WDF | Not supported yet; tracked by #257 |
| Export Recipes through joblib/skops | Not supported today |
