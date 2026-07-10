# ADR: Recipe Replay, Graph Inputs, and Interoperability Boundaries

- **Status**: Accepted
- **Date**: 2026-07-06
- **Context**: Consolidates the important Recipe decisions from temporary superpowers specs into durable design documentation.

## Context

Pipeline Recipe work introduced several prototype specs for replay intent, graph extraction, input-name defaults, DAG identity, dispatch boundaries, and sklearn/WDF export. Those specs were useful while the API was unsettled, but the durable design now lives in the user-facing Recipe explanation pages and this ADR.

The core design goal is replay with the same user-visible meaning, not serialization of every runtime detail. When Wandas cannot represent that meaning in a stable public step, extraction fails with `RecipeExtractionError` instead of silently returning a misleading recipe.

## Decision

Wandas-native Recipe specs are the canonical replay representation:

- `RecipeSpec` is single-input and linear.
- `GraphRecipeSpec` covers a readable two-input graph with one binary merge and an optional linear tail.
- `NodeGraphRecipeSpec` covers tree-shaped graph replay, external array operands, and `add_channel(...)` graph inputs.
- `RecipeSpec.from_frame(...)` intentionally does not auto-upgrade into a graph recipe. Multi-input work must be extracted explicitly through `GraphRecipeSpec.from_frame(...)` or `NodeGraphRecipeSpec.from_frame(...)`.

Recipe extraction reads `operation_graph`, because it preserves parent structure and source leaves. `operation_history` remains a display-oriented compatibility view.

## Replay Intent Contract

Recipe extraction stores public replay intent:

- Supported frame operations replay through `apply_operation(...)`, frame methods, frame indexing, frame operators, or importable custom functions.
- `fix_length(duration=...)` stores the resolved sample `length`, because the current replay contract promises output length rather than reinterpreting duration on a different sampling rate.
- Channel selection stores stable public forms such as names, indices, ranges, index lists, boolean masks, and literal dict queries.
- Callable/regex channel queries, non-literal query values, unsupported multidimensional indexing, and non-contiguous time point selections are rejected.
- `add_channel(ChannelFrame, ...)` stores the added frame as a named external input plus public options.
- `add_channel(ndarray|dask.array, ...)` stores the raw data as a named external input plus public options, including `source_time_offset`; it does not store array values.
- `add_with_snr(...)` stores `snr` and the two frame inputs; it does not surface helper length-alignment steps as separate Recipe steps.

Recipe replay delegates to existing frame APIs. Recipe code must not duplicate numerical processing, metadata propagation, source-time updates, or Dask graph construction.

## Graph Input Names and Identity

When explicit `input_names` are omitted, graph extraction uses mechanical source-leaf names: `input_0`, `input_1`, and so on. It does not infer names from Python variables, frame labels, channel labels, or metadata.

For binary graph recipes, `input_0` is the left operand and `input_1` is the right operand. This preserves non-commutative operations such as subtraction and division.

Current runtime graph extraction is tree-shaped. It can replay duplicated parent paths, including repeated explicit input names such as `("base", "base")`, but it does not preserve true shared-branch DAG identity as a first-class graph feature. True DAG identity remains future work.

## Interoperability Boundaries

The sklearn adapter is optional and thin:

- `WandasOperationTransformer(operation, **params)` wraps one `frame.apply_operation(...)` call.
- Named transformers such as `HighPassFilter`, `LowPassFilter`, `BandPassFilter`, `Normalize`, and `RemoveDC` wrap one known operation.
- `to_spec()` returns one `OperationSpec`, not a full `RecipeSpec`.
- The adapter is an ergonomics layer for sklearn-style `Pipeline.transform(frame)`, not a general Recipe interchange format.

Unsupported today:

- automatic `RecipeSpec -> sklearn.pipeline.Pipeline` conversion;
- automatic `sklearn.pipeline.Pipeline -> RecipeSpec` conversion;
- sklearn conversion for `GraphRecipeSpec`, `NodeGraphRecipeSpec`, terminal steps, custom steps, indexing steps, method-aware steps, scalar-operation steps, binary steps, or add-channel steps;
- official joblib or skops Recipe export; and
- executable Recipe persistence in WDF.

WDF stores operation summaries for inspection only. Executable Recipe persistence remains separate future work.

## Documentation Contract

Long-lived Recipe documentation is split by audience:

- `docs/src/explanation/pipeline-recipe-requirements.md` describes requirements and non-goals.
- `docs/src/explanation/pipeline-recipe-design.md` describes the architecture.
- `docs/src/explanation/pipeline-recipe-support-matrix.md` is the quick user-facing support table.
- `docs/src/explanation/pipeline-recipe-extraction-boundaries.md` is the detailed extraction and failure-boundary reference.
- `docs/src/explanation/pipeline-recipe-interoperability-export.md` describes sklearn, WDF, joblib/skops, and Dask-ML boundaries.

Temporary superpowers specs should not duplicate these pages after the implementation contract has landed.

## Validation Evidence

The current implementation aligns with this contract:

- `GraphRecipeSpec.from_frame(...)` defaults omitted names to `input_0`, `input_1`, validates two distinct inputs, and preserves left/right order.
- `NodeGraphRecipeSpec.from_frame(...)` assigns source-leaf names in order, accepts explicit duplicate names for repeated external inputs, and validates unused or missing names.
- Add-channel graph steps model frame and raw-data inputs as named external inputs.
- `WandasOperationTransformer.to_spec()` returns one `OperationSpec`.
- Recipe explanation docs already publish the support matrix, extraction boundaries, and interoperability/export boundaries.
