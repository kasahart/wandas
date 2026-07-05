# Pipeline Recipe Developer Guide / Recipe 開発者ガイド

This guide is the onboarding entry point for contributors who want to understand or extend Wandas Recipe support. It summarizes the current implementation model and points to the deeper design notes when details matter.

See also:

- [Pipeline Recipe Design](pipeline-recipe-design.md)
- [Pipeline Recipe Extraction Boundaries](pipeline-recipe-extraction-boundaries.md)
- [Pipeline Recipe Requirements](pipeline-recipe-requirements.md)

## Purpose

Recipe support makes exploratory frame calculations replayable. A user can process a `ChannelFrame` in a notebook, extract a recipe from the resulting frame, and replay the same calculation against another compatible frame.

The important design constraints are:

- Wandas Recipe is the source of truth. sklearn `Pipeline` is only an optional adapter.
- Recipe objects describe calls; they do not own signal-processing logic.
- Replay delegates to existing frame APIs such as `apply_operation()`, frame methods, frame operators, `frame[key]`, and `frame.apply(...)`.
- Frame immutability, metadata/history updates, and Dask laziness remain the responsibility of the existing frame implementation.
- Unsupported calculations must fail loudly with `RecipeExtractionError` rather than returning a partial or misleading recipe.

## Core Terms

| Term | Meaning |
| --- | --- |
| `lineage` | Runtime tree stored on a frame. It records the operation object and parent lineage nodes used to create the frame. |
| `operation_history` | Backward-compatible linear list derived from `lineage`. Good for user inspection, but not enough for graph extraction because it loses parent structure. |
| `operation_graph` | JSON-like tree derived from `lineage`. Recipe extraction reads this structure because it preserves operation params, parent edges, source leaves, and selected custom metadata. |
| `OperationSpec` | A replayable `frame.apply_operation(operation, **params)` call for registered single-input operations. |
| `RecipeSpec` | A single-input ordered sequence of replayable steps. It can be extracted from one linear parent chain with `RecipeSpec.from_frame(frame)`. |
| `MethodStep` | A replayable public frame method such as `get_channel()`, `sum()`, `fix_length()`, `remove_channel()`, or `rename_channels()`. |
| `TypedMethodStep` | A replayable frame method that may change the frame class, such as `fft()`, `stft()`, `ifft()`, `welch()`, or `roughness_dw_spec()`. |
| `CustomFunctionStep` | A replayable `frame.apply(...)` call for importable module-level functions, optionally with an importable `output_frame_class`. |
| `ScalarOperationStep` | A replayable scalar operator such as `frame + 1.0` or `2.0 - frame`. |
| `IndexingStep` | A replayable `frame[key]` selection for supported channel and slice-only multidimensional indexing. |
| `TerminalStep` | An explicit terminal property or method call that returns a non-frame value, such as `rms`, `magnitude`, `dB`, `loudness_zwst()`, or `sharpness_din_st()`. |
| `GraphRecipeSpec` | A readable two-input graph recipe for one binary frame merge plus an optional linear tail. |
| `NodeGraphRecipeSpec` | A general tree-shaped graph recipe for multiple unary/binary nodes, external array operands, and `add_channel` graph inputs. |
| `RecipeExtractionError` | A boundary signal. It means the frame calculation is real, but the current recipe model cannot replay it safely. |

## Data Flow

Frame operations create lineage as they run:

```text
source frame
  -> frame.remove_dc()
    -> LineageNode(operation="remove_dc", inputs=[source])
  -> .normalize()
    -> LineageNode(operation="normalize", inputs=[remove_dc node])
  -> processed frame
```

The frame exposes two derived views:

```text
lineage
  -> operation_history
       [{"operation": "remove_dc"}, {"operation": "normalize", ...}]
  -> operation_graph
       {"operation": "normalize", "inputs": [{"operation": "remove_dc", ...}], ...}
```

Recipe extraction reads `operation_graph`, not `operation_history`:

```text
processed.operation_graph
  -> RecipeSpec.from_frame(processed)
  -> RecipeSpec([OperationSpec("remove_dc"), OperationSpec("normalize", ...)])
  -> recipe.apply(other_frame)
  -> other_frame.apply_operation("remove_dc").apply_operation("normalize", ...)
```

For graph calculations, the same graph is read with a graph-aware entry point:

```text
(signal.remove_dc() + noise.normalize()).stft()
  -> operation_graph with two parents at the binary merge
  -> GraphRecipeSpec.from_frame(..., input_names=("signal", "noise"))
  -> graph_recipe.apply({"signal": new_signal, "noise": new_noise})
```

## Choosing an Extraction Entry Point

| Calculation shape | Use | Why |
| --- | --- | --- |
| One input, one linear chain, returns a frame | `RecipeSpec.from_frame(processed)` | Produces the simplest replayable recipe. |
| One input, explicit terminal result wanted | `RecipeSpec([... , TerminalStep(...)])` | Terminal arrays do not carry lineage, so extraction from the array is not possible. |
| Two frame inputs, one binary merge, optional linear tail | `GraphRecipeSpec.from_frame(processed, input_names=(...))` | Keeps the common graph case readable. |
| Multiple merges, external ndarray/dask operands, or `add_channel` graph inputs | `NodeGraphRecipeSpec.from_frame(processed, input_names=(...))` | Represents a tree of named inputs and nodes. |
| sklearn-style `fit` / `transform` integration | `wandas.pipeline.sklearn` | Optional wrapper for sklearn users; Wandas Recipe remains canonical. |

`RecipeSpec.from_frame(frame_a + frame_b)` intentionally does not return a graph recipe. `RecipeSpec` stays single-input and linear; use `GraphRecipeSpec` or `NodeGraphRecipeSpec` for multi-input calculations.

## Why sklearn Pipeline Is Not Canonical

sklearn `Pipeline` is a useful interface when a user already has ML tooling, but it is not a good source of truth for Wandas signal processing:

- sklearn does not know Wandas frame metadata, channel labels, source time offsets, or Dask graph semantics.
- Wandas already has operation registry keys and frame methods that define behavior.
- Some replayable calculations are not sklearn-style transformers, such as indexing, binary frame graphs, `add_channel`, and terminal properties.
- Optional sklearn support must not make normal `import wandas` import scikit-learn.

The sklearn adapter therefore stays thin: transformer classes call Wandas operations and can return an `OperationSpec` with `to_spec()`.

## Delegation Rule

Recipe steps must not duplicate frame calculation logic.

Good:

```python
def apply(self, frame):
    return frame.normalize(**self.params)
```

Bad:

```python
def apply(self, frame):
    # Reimplements normalization, channel metadata, label changes, and Dask details.
    ...
```

When a new operation needs recipe support, prefer adding a step that calls the same public API the user called. If the public API cannot express the runtime graph, define a narrow explicit boundary and raise `RecipeExtractionError`.

## Implemented Scope

The current prototype supports these broad categories:

- Linear single-input frame operations through `RecipeSpec`.
- Public frame methods that own metadata behavior through method-aware steps.
- Selected typed frame transitions through typed method steps.
- Numeric scalar operators through scalar operation steps.
- Supported channel and slice indexing through indexing steps.
- Two-input and tree-shaped graph recipes through `GraphRecipeSpec` and `NodeGraphRecipeSpec`.
- Named external frame or array inputs for graph-shaped calculations.
- Importable custom functions, including selected custom frame domain transitions.
- Explicit terminal properties and terminal methods through `TerminalStep`.

The exact operation/method matrix lives in [Pipeline Recipe Extraction Boundaries](pipeline-recipe-extraction-boundaries.md). Keep that page as the canonical support list so developer and user guides do not drift from implementation.

## Current Boundaries

The current recipe model intentionally rejects ambiguous or unreplayable categories:

- Non-importable custom code or custom frame classes.
- Runtime values that cannot be represented as recipe literals or named graph inputs.
- Selection/indexing forms whose intent would be lost during extraction.
- Multi-input calculations requested through the single-input `RecipeSpec` extractor.
- Python variable-name inference and true DAG identity.
- Automatic extraction from terminal arrays that no longer carry lineage.
- Display/report/evaluation actions as transform recipe steps.

These boundaries are not final product decisions. They keep the prototype replayable and understandable while the representation expands.

## Adding Recipe Support For A New Frame Calculation

Use this checklist before editing code:

- Identify the public API users call: `apply_operation`, a frame method, an operator, indexing, `frame.apply`, or graph input.
- Confirm the operation leaves enough information in `operation_graph` to replay the same intent.
- Decide whether the calculation is linear (`RecipeSpec`) or graph-shaped (`GraphRecipeSpec` / `NodeGraphRecipeSpec`).
- Prefer an existing step class. Add a new step only when no current step can express the public API call.
- Keep params recipe-literal unless a dedicated typed representation already exists.
- Make `apply()` delegate to the existing frame API.
- Make extraction reject ambiguous cases with `RecipeExtractionError`.
- Preserve existing behavior for `operation_history`.
- Add tests for replayed data, operation history, labels/metadata when relevant, source frame immutability, and Dask laziness when the operation is lazy.
- Update [Pipeline Recipe Extraction Boundaries](pipeline-recipe-extraction-boundaries.md) and user-facing docs.

Typical code locations:

- Public Recipe exports: `wandas/pipeline/__init__.py`
- Replayable support allowlists: `wandas/pipeline/registry.py`
- Recipe-literal param snapshot/restore helpers: `wandas/pipeline/params.py`
- Step classes and replay calls: `wandas/pipeline/steps.py`
- Lineage graph to step conversion helpers: `wandas/pipeline/extraction.py`
- `RecipeSpec`, `GraphRecipeSpec`, and `NodeGraphRecipeSpec`: `wandas/pipeline/specs.py`
- Recipe extraction error type and future boundary helpers: `wandas/pipeline/errors.py`
- sklearn adapter: `wandas/pipeline/sklearn.py`
- Runtime lineage and operation graph generation: `wandas/core/base_frame.py`
- Custom operation metadata: `wandas/processing/custom.py`
- Operation implementation and registration: `wandas/processing/`
- Frame methods and typed transitions: `wandas/frames/` and `wandas/frames/mixins/`
- Recipe behavior tests: `tests/pipeline/test_recipe.py`
- Recipe module boundary tests: `tests/pipeline/test_recipe_errors.py`
- Frame/processing behavior tests: add focused tests near the frame or operation contract being changed.

## Reading RecipeExtractionError

`RecipeExtractionError` is not a generic failure. It is a precise statement that the calculation is outside the currently replayable boundary.

Common messages:

| Error theme | Meaning | Usual next step |
| --- | --- | --- |
| "RecipeSpec.from_frame(...) cannot extract graph lineage as a linear recipe" | A linear `RecipeSpec` saw graph or multi-input lineage. | Use `GraphRecipeSpec.from_frame(...)` or `NodeGraphRecipeSpec.from_frame(...)`. |
| "Operation is outside the Stage 1 recipe allowlist" | The operation is registered, but not yet declared replayable. | Confirm replay semantics, then add allowlist/tests/docs. |
| "Typed operation requires frame method lineage" | A typed transform was called through a generic operation path. | Use the public method, or define a new typed extraction rule. |
| "Custom operation recipe extraction requires importable..." | The callable or output shape function cannot be imported by module path. | Move the function into an importable module-level function. |
| "Channel selection recipe extraction only supports..." | Selection intent would be lost if flattened to indices. | Add a typed query representation, or keep the operation outside recipe extraction. |
| "Scalar operation requires a stable numeric scalar operand" | Operand is not a supported scalar or has unstable equality such as NaN. | Use a supported scalar or graph recipe with an external operand. |

When adding a new error, include:

- The operation or method name.
- The unsupported value or shape.
- A short explanation of the currently supported path.

## Test Checklist

For each new recipe-supported calculation, add focused tests that answer:

- Does extraction produce the expected step object?
- Does `recipe.apply(source)` match the original processed frame?
- Does replay preserve `operation_history` order and params?
- Does replay preserve labels, metadata, channel metadata, source time offsets, and frame type when relevant?
- Does the source frame remain unchanged?
- Does a Dask-backed frame remain lazy until `.compute()` when the underlying operation is lazy?
- Does an unsupported nearby case raise `RecipeExtractionError` with a useful message?
- If a graph recipe is involved, are missing runtime inputs reported clearly?
- If custom code is involved, are non-importable callables rejected before producing an unreplayable recipe?

## Maintenance Rule

Do not expand an allowlist because a test is easy to write. Expand it only when the replay representation preserves user intent and delegates to the existing frame API. A smaller explicit boundary is better than a recipe that silently changes meaning.
