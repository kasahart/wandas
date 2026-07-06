# Pipeline Recipes / Recipe の使い方

Wandas Recipe lets you make frame calculations replayable. You can process one frame, extract a recipe from the result, and apply the same calculation to another frame.

Recipe is useful when you want to:

- turn exploratory notebook work into a repeatable preprocessing pipeline,
- apply the same signal-processing chain to many files,
- inspect exactly which Wandas operations were applied,
- keep Wandas frame metadata, lineage-derived history views, and Dask behavior intact.

## Quick Start: Linear Recipe

Start with normal frame methods:

```python
processed = (
    frame
    .remove_dc()
    .trim(start=0.1, end=0.5)
    .normalize()
)
```

Extract a recipe from the processed frame:

```python
from wandas.pipeline import RecipeSpec

recipe = RecipeSpec.from_frame(processed)
replayed = recipe.apply(other_frame)
```

`RecipeSpec.from_frame(...)` reads `processed.operation_graph`. It does not inspect Python source code or variable names.

The replayed result uses existing frame APIs, so normal Wandas behavior still applies:

- the input frame is not modified,
- `operation_history` is preserved on the output,
- Dask-backed data stays lazy when the underlying operation is lazy,
- metadata and channel labels are updated by existing frame methods.

## Optional: Build A Recipe Explicitly

Frame extraction is the recommended user entry point. Direct construction is available for tests,
generated configs, and explicit specifications:

```python
from wandas.pipeline import OperationSpec, RecipeSpec

recipe = RecipeSpec(
    [
        OperationSpec("remove_dc"),
        OperationSpec("highpass_filter", {"cutoff": 100.0, "order": 2}),
        OperationSpec("normalize"),
    ]
)

processed = recipe.apply(frame)
```

Operation names are Wandas operation registry keys such as `"remove_dc"` and `"highpass_filter"`. They are not necessarily the same as convenience method names such as `frame.high_pass_filter(...)`.

## Method And Indexing Recipes

Some frame calculations are replayed through frame methods rather than `apply_operation()` because the method owns metadata behavior:

```python
processed = (
    frame
    .get_channel(0)
    .fix_length(length=8000)
    .rename_channels({0: "front"})
)

recipe = RecipeSpec.from_frame(processed)
```

Supported indexing can also be extracted:

```python
processed = frame[0:2].trim(start=0.0, end=0.25)
recipe = RecipeSpec.from_frame(processed)
```

Supported indexing includes channel slices, integer lists, 1-D boolean masks, label lists, and slice-only multidimensional indexing.

## Scalar Operators

Numeric scalar operators are replayable:

```python
processed = (frame.remove_dc() * 2.0) + 0.1
recipe = RecipeSpec.from_frame(processed)
```

Scalar-left operators are also supported when the left operand is a stable numeric scalar:

```python
processed = 2.0 - frame.normalize()
recipe = RecipeSpec.from_frame(processed)
```

Array operands are graph inputs, not scalar values. Use `NodeGraphRecipeSpec` for `frame + ndarray` or `frame * dask_array`.

## Typed Frame Transitions

Recipes can replay selected methods that return another frame type:

```python
processed = frame.stft(n_fft=512, hop_length=128).get_frame_at(time_idx=0)
recipe = RecipeSpec.from_frame(processed)
spectral_frame = recipe.apply(other_frame)
```

Supported typed methods are listed in [Pipeline Recipe Extraction Boundaries](../explanation/pipeline-recipe-extraction-boundaries.md). If a typed transform is not listed there, keep it outside recipe extraction until replay semantics are added and tested.

## Graph Recipes

Use `GraphRecipeSpec` when the result has two frame parents and one merge point:

```python
from wandas.pipeline import GraphRecipeSpec

processed = (signal.remove_dc() + noise.normalize()).trim(start=0.0, end=0.25)

graph_recipe = GraphRecipeSpec.from_frame(
    processed,
    input_names=("signal", "noise"),
)

replayed = graph_recipe.apply(
    {
        "signal": new_signal,
        "noise": new_noise,
    }
)
```

`input_names` are explicit because runtime lineage does not know Python variable names, and graph recipes do not infer names from frame labels, channel labels, or metadata. If omitted, graph recipes use source leaf order to assign mechanical names such as `input_0` and `input_1`; for binary frame recipes, `input_0` is the left operand and `input_1` is the right operand.

Use `NodeGraphRecipeSpec` for more general tree-shaped graphs:

```python
from wandas.pipeline import NodeGraphRecipeSpec

processed = (left.remove_dc() + right.normalize()) + reference

node_recipe = NodeGraphRecipeSpec.from_frame(
    processed,
    input_names=("left", "right", "reference"),
)

replayed = node_recipe.apply(
    {
        "left": new_left,
        "right": new_right,
        "reference": new_reference,
    }
)
```

If the same processed branch is used twice, the current recipe stores it as two parent paths. Pass the same explicit input name for both source leaves when replay should use one runtime input:

```python
shared = base.normalize()
processed = shared.low_pass_filter(cutoff=3000.0).add(
    shared.high_pass_filter(cutoff=300.0),
    snr=3.0,
)

recipe = NodeGraphRecipeSpec.from_frame(
    processed,
    input_names=("base", "base"),
)
replayed = recipe.apply({"base": new_base})
```

This is not true DAG identity. True shared node identity is tracked separately in #270.

## add_channel Recipes

`RecipeSpec` is single-input, so `add_channel(...)` needs a graph recipe.

Frame input:

```python
from wandas.pipeline import NodeGraphRecipeSpec

processed = base.add_channel(added, label="reference")

recipe = NodeGraphRecipeSpec.from_frame(
    processed,
    input_names=("base", "added"),
)

replayed = recipe.apply({"base": new_base, "added": new_added})
```

Raw data input:

```python
processed = base.add_channel(raw_array, label="raw", source_time_offset=0.0)

recipe = NodeGraphRecipeSpec.from_frame(
    processed,
    input_names=("base", "raw"),
)

replayed = recipe.apply({"base": new_base, "raw": new_raw_array})
```

The recipe stores the input name and public add-channel options. It does not store raw array values.

## Custom Function Recipes

Importable module-level functions can be replayed:

```python
from my_project.audio_ops import scale
from my_project.shapes import same_shape
from wandas.pipeline import RecipeSpec

processed = frame.apply(scale, output_shape_func=same_shape, gain=2.0)

recipe = RecipeSpec.from_frame(processed)
replayed = recipe.apply(other_frame)
```

The function path is stored in the recipe. Replay imports and executes that function, so only use custom recipes from trusted code.

Custom domain transitions are supported when the output frame class is importable and constructor kwargs are recipe literals:

```python
from wandas.frames.spectral import SpectralFrame

processed = frame.apply(
    custom_rfft,
    output_shape_func=rfft_shape,
    output_frame_class=SpectralFrame,
    output_frame_kwargs={"n_fft": frame.n_samples, "window": "hann"},
)

recipe = RecipeSpec.from_frame(processed)
```

Not supported: lambdas, nested functions, closures, callable objects, bound methods, `functools.partial`, and functions defined only in `__main__` or a notebook cell.

## Terminal Steps

Terminal values such as `rms`, `magnitude`, and `dB` return arrays, not frames. Arrays do not carry Wandas lineage, so terminal steps are explicit:

```python
from wandas.pipeline import OperationSpec, RecipeSpec, TerminalStep

rms_recipe = RecipeSpec(
    [
        OperationSpec("remove_dc"),
        TerminalStep("rms"),
    ]
)

rms_values = rms_recipe.apply(frame)
```

For spectral properties:

```python
from wandas.pipeline import RecipeSpec, TerminalStep, TypedMethodStep

magnitude_recipe = RecipeSpec(
    [
        TypedMethodStep("fft", {"n_fft": 1024, "window": "hann"}),
        TerminalStep("magnitude"),
    ]
)

magnitude = magnitude_recipe.apply(frame)
```

Terminal extraction from `frame.rms` is not possible because the result is a NumPy array without `operation_graph`.

## sklearn Adapter

Install the optional dependency when you want sklearn-style objects:

```bash
pip install "wandas[sklearn]"
```

Example:

```python
from sklearn.pipeline import Pipeline

from wandas.pipeline.sklearn import HighPassFilter, Normalize, RemoveDC

pipeline = Pipeline(
    [
        ("dc", RemoveDC()),
        ("hp", HighPassFilter(cutoff=100.0, order=2)),
        ("norm", Normalize()),
    ]
)

processed = pipeline.transform(frame)
```

The adapter is thin. It calls Wandas operations and can convert back to `OperationSpec`:

```python
spec = HighPassFilter(cutoff=100.0, order=2).to_spec()
```

Wandas Recipe remains the canonical representation. sklearn `Pipeline` is for integration and familiar UX.

WDF currently preserves operation summaries for inspection only. Loading a WDF does not rebuild runtime lineage or restore an executable `RecipeSpec`; executable Recipe persistence is tracked separately in #257.

## Advanced Reference: Choosing A Recipe Class

Most users should start with `RecipeSpec.from_frame(processed)`. Use this table only when the frame
result crosses the single-input linear boundary or an integration layer requires another API.

`RecipeSpec.from_frame(...)` is intentionally linear-only and does not automatically return `GraphRecipeSpec` or `NodeGraphRecipeSpec`. When a result has graph lineage, choose the graph recipe class explicitly. A future higher-level factory may add automatic dispatch without changing the `RecipeSpec.from_frame(...)` return type.

| Your calculation | Recommended class | Example |
| --- | --- | --- |
| One input, linear frame result | `RecipeSpec` | `frame.remove_dc().normalize()` |
| One input, explicit terminal array/scalar | `RecipeSpec([... , TerminalStep(...)])` | `RecipeSpec([OperationSpec("remove_dc"), TerminalStep("rms")])` |
| Two frame inputs, one binary merge, optional linear tail | `GraphRecipeSpec` | `(signal.remove_dc() + noise.normalize()).trim(...)` |
| Multiple merges, external arrays, or `add_channel` graph inputs | `NodeGraphRecipeSpec` | `base.add_channel(raw_array)` or `(a + b) + c` |
| sklearn-style transform interface | `wandas.pipeline.sklearn` | `Pipeline([("dc", RemoveDC()), ...])` |

## What Is Not Supported Yet

Start with the short [Pipeline Recipe Support Matrix](../explanation/pipeline-recipe-support-matrix.md) when you only need to know whether a calculation is supported. The detailed boundary notes are maintained in [Pipeline Recipe Extraction Boundaries](../explanation/pipeline-recipe-extraction-boundaries.md).

These are the user-facing categories to remember:

| Boundary category | Alternative |
| --- | --- |
| Multi-input work through `RecipeSpec.from_frame(...)` | `RecipeSpec.from_frame(...)` stays linear-only; use `GraphRecipeSpec.from_frame(...)` or `NodeGraphRecipeSpec.from_frame(...)` explicitly. |
| Semantic runtime names or values | Pass explicit `input_names=(...)` and runtime inputs; omitted names are source-order mechanical `input_0`, `input_1`, ... labels, not inferred from frame/channel metadata. |
| Selection/indexing forms that cannot be replayed by intent | Use supported literal selections, or keep the operation outside recipe extraction. |
| Non-importable custom code | Move the function to an importable module-level function. |
| Terminal NumPy arrays without lineage | Build a recipe with an explicit `TerminalStep`. |
| Reporting/evaluation actions | Keep `plot()`, `describe()`, `info()`, and `compute()` outside transform recipes. |

`RecipeExtractionError` means you hit one of these boundaries. It is expected behavior: Wandas refuses to make a recipe that cannot be replayed safely.

## Try The Notebook

An executable notebook is included at:

```text
docs/src/tutorial/pipeline-recipes.ipynb
```

It creates small synthetic frames and demonstrates extraction, replay, graph recipes, custom functions, terminal steps, and failure cases without downloading data.
