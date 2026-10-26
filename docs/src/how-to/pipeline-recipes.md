# Work with RecipePlan

For a guided, executable introduction, start with the
<a href="../../learning-path/06_reusable_pipeline_recipes.html">Reusable Pipeline Recipes Learning Path</a>.
Use this how-to when you already know the task you need to complete.

RecipePlanを初めて段階的に学ぶ場合は、実行可能な
<a href="../../learning-path/06_reusable_pipeline_recipes.html">Reusable Pipeline Recipes Learning Path</a>
から始めてください。具体的な作業が決まっている場合に、このHow-toを使います。

## Choose WDF or Recipe

WDF saves one concrete, typed Frame result for later inspection or loading.
Recipe saves reusable workflow intent so the same public operations can run on
another input. A Recipe does not contain the original sample data; choose WDF
when the result is the artifact and Recipe when the process is the artifact.

Saving or loading WDF needs the `io` extra; a Recipe file does not.

```bash
pip install "wandas[io]"
```

WDFの保存と読み込みには`io` extraが必要です。Recipeファイルの保存には要りません。

## What a Recipe stores

- `previous` is a process-local reference to the immediately preceding Frame; it
  is not portable Recipe input.
- `lineage` is the complete provenance of Frame inputs.
- Runtime-only callables, regex predicates, and opaque Python objects cannot be
  serialized into a Recipe; extraction fails explicitly at that operation.

Named Frame and external-array inputs are supplied again at replay. The
serialized artifact stores operation intent and bindings, not the input samples.

## Set up the examples

The snippets below form one small workflow. Prepare a source, a replacement
Frame, and an external array before choosing the task you need:

```python
import numpy as np
import wandas as wd

source = wd.from_numpy(
    np.array([[1.0, 2.0, 4.0, 7.0]]),
    sampling_rate=8_000,
    ch_labels=["sensor"],
)
another_frame = wd.from_numpy(
    np.array([[2.0, 5.0, 8.0, 14.0]]),
    sampling_rate=8_000,
    ch_labels=["sensor"],
)
external_array = np.ones((1, 4))
processed = source.remove_dc().normalize()
```

## Extract and apply a plan

Process a Frame normally, then name the runtime inputs while compiling its
semantic lineage:

```python
from wandas.pipeline import RecipePlan

processed = source.remove_dc().normalize()
plan = RecipePlan.from_frame(processed, input_names=("signal",))
replayed = plan.apply({"signal": another_frame})
```

The names supplied during extraction must match the mapping passed to `apply()`.
The result is a Frame; materialize it only at the boundary required by the task.

## Save and load a standalone artifact

Save reusable operation intent separately from Frame data:

```python
path = plan.save("analysis")  # analysis.recipe.json
restored = RecipePlan.load(path)
replayed = restored.apply({"signal": another_frame})
```

Use `to_dict()` and `from_dict()` when the artifact must travel through another
JSON or storage layer. Saving does not overwrite an existing artifact unless
that is requested explicitly. The format-specific schema and API errors are in
the [Pipeline API Reference](../api/pipeline.md).

## Supply multiple inputs

Give every runtime Frame or external array a stable name. Input order and
alignment are defined by the public operation that created the plan.

```python
processed = source + external_array
plan = RecipePlan.from_frame(processed, input_names=("signal", "offset"))
replayed = plan.apply({"signal": another_frame, "offset": external_array})
```

External NumPy and Dask arrays remain named inputs; they are not embedded in the
artifact or wrapped in temporary Frames. At replay, supply a NumPy or Dask array
with a shape and dtype accepted by the operation. The original container type and
Dask chunking are not stored in the Recipe.

## List operations and their parameters

`RecipeRegistry.describe()` returns one description per registered operation:
its ID, version, method name, input bindings, and the method parameters
with their defaults. Parameters annotated with `Literal[...]` also report their
accepted values as `choices`. Use it to build parameter forms, documentation, or
tool schemas from the same registry that validates and replays Recipes.

```python
from wandas.pipeline import default_recipe_registry

catalog = default_recipe_registry().describe()
fft = next(item for item in catalog if item.method == "fft")
window = next(parameter for parameter in fft.parameters if parameter.name == "window")
print(window.default)  # hann
```

Contracts kept only to replay older Recipes are listed too. Their `method` is
private (one leading underscore, such as `_fft_recipe_v2`); public calls record
the operations whose `method` is public, including operator methods such as
`__add__`.

`RecipeRegistry.describe()`は登録済み操作ごとに、ID、version、メソッド名、入力の
binding、メソッドの引数と既定値を返します。`Literal[...]`で注釈した引数は受け付ける値を
`choices`として返します。パラメータ入力欄、ドキュメント、ツール定義を、Recipeの検証と再実行に
使うのと同じregistryから作れます。古いRecipeの再実行用に残した操作も含まれ、その`method`は
`_fft_recipe_v2`のように先頭が`_`1つの非公開名です。公開呼出しが記録するのは`method`が公開名の
操作で、`__add__`のような演算子メソッドも含みます。

## Annotate each runtime input

Labels, `metadata` and channel `extra` describe one recording, so a plan does
not record `with_label()`, `with_metadata()` or `with_channel_extra()`; replay
uses the annotations of the frame you pass in. A plan that selects channels by
`extra`, such as `get_channel(query={"position": "front"})`, therefore needs
the same annotation on every runtime input, for example from that recording's
catalog row. This example uses two-channel recordings of its own:

```python
recording = wd.from_numpy(
    np.random.default_rng(0).normal(size=(2, 800)),
    sampling_rate=8_000,
    ch_labels=["mic1", "mic2"],
)
annotated = recording.with_channel_extra("mic2", {"position": "front"})
front = annotated.get_channel(query={"position": "front"}).high_pass_filter(cutoff=100)
plan = RecipePlan.from_frame(front, input_names=("signal",))

next_recording = wd.from_numpy(
    np.random.default_rng(1).normal(size=(2, 800)),
    sampling_rate=8_000,
    ch_labels=["mic1", "mic2"],
)
other = next_recording.with_channel_extra("mic2", {"position": "front"})
replayed = plan.apply({"signal": other})  # selects mic2
```

Replaying on a frame where no channel has the `position` key raises
`RecipeExecutionError` naming the unknown metadata key. If the key exists but
no channel has the value `"front"`, the error says that no channel matches the
query. If a different channel carries `"front"`, replay selects that channel,
so annotate each input from its own catalog row.

ラベル・`metadata`・チャネルの`extra`は録音ごとの情報なので、Recipeには記録されず、
再生時は渡したフレームの注釈が使われます。`get_channel(query={"position": "front"})`のように
`extra`でチャネルを選ぶRecipeは、再生する各入力にも同じ注釈（たとえばその録音の諸元表の値）を
付けてから渡してください。どのチャネルにもそのキーがない入力では、未知のメタデータキーを示す
`RecipeExecutionError`になります。キーはあっても値が一致するチャネルがなければ、クエリに
一致するチャネルがないというエラーになります。別のチャネルに同じ値を付けると、そのチャネルが
選ばれるので、注釈は各録音の諸元表から付けてください。

## Find the step that failed

When an operation fails during `apply`, the `RecipeExecutionError` carries the
failing node as attributes, so a tool can mark that step without parsing the
message. `node_id` and `operation` are `None` for errors that do not belong to
one node, such as mismatched input names.

```python
from wandas.pipeline import RecipeExecutionError

try:
    plan.apply({"signal": source})
except RecipeExecutionError as error:
    print(error.node_id, error.operation, repr(error.cause))
```

`apply`で処理が失敗すると、`RecipeExecutionError`の`node_id`・`operation`・`cause`
（元の例外、`__cause__`と同じ）で失敗したノードが分かります。エラー文を読まずに、
失敗した手順へ印を付けられます。入力名の不一致のようにノードに関係しないエラーでは、
`node_id`と`operation`は`None`です。

## Handle runtime-only operations

Some calls intentionally cannot be portable. Arbitrary callables passed to
`Frame.apply()`, callable or regex channel predicates, and opaque Python objects
must remain runtime-only. Recipe extraction fails at the unsupported operation
instead of silently dropping part of the workflow.

To keep a portable operation in your own project, follow
[Add your own processing](custom-processing.md), including the registry passed
to extraction, loading, and replay.
自分のプロジェクト内でRecipe対応の独自処理を作る場合は、
[独自処理のHow-to](custom-processing.md)を参照してください。

For built-in contributions and advanced Frame contracts, use the
[Frame and Operation extension guide](../contributing/frame-operation-extensions.md).
