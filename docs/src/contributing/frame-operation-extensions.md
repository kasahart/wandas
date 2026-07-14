# Extending Frames and Operations / Frame・Operation 拡張ガイド

Use this guide when adding a new signal-processing operation, a public Frame
method, or a new Frame family. It defines the implementation path, the ownership
boundaries, and the minimum tests expected in one change.
新しい信号処理 Operation、公開 Frame メソッド、または新しい Frame family を追加する際は、
このガイドを使用してください。1つの変更で必要となる実装経路、責務境界、最低限のテストを定義します。

Repository agents must start from
[`AGENTS.md`](https://github.com/kasahart/wandas/blob/develop/AGENTS.md) and then use
this guide together with the path-specific instructions under
`.github/instructions/`. This document explains the workflow; those instruction
files remain the automatically loaded guardrails.
リポジトリ上の Agent は、まず
[`AGENTS.md`](https://github.com/kasahart/wandas/blob/develop/AGENTS.md)を読み、その後このガイドと
`.github/instructions/` 配下の対象別指示を併用してください。この文書は作業手順を説明し、
各 instruction ファイルは自動適用されるガードレールとして維持されます。

## Choose the smallest extension / 最小の拡張単位を選ぶ

Do not start by creating a class. First decide which public contract is missing.
最初から class を作らず、欠けている公開契約を先に決めます。

| Need / 必要なもの | Add / 追加するもの | Do not add / 追加しないもの |
| --- | --- | --- |
| Reusable numerical behavior only / 再利用可能な数値処理だけ | An `AudioOperation` in `wandas/processing/` and focused processing tests / `wandas/processing/` の `AudioOperation` と処理テスト | Frame metadata or public API / Frame metadata や公開API |
| A chainable user operation / chain可能な利用者向け処理 | The `AudioOperation`, a thin public Frame method, and processing + Frame tests / `AudioOperation`、薄い公開Frameメソッド、processing + Frameテスト | Numerical logic in the Frame method / Frameメソッド内の数値ロジック |
| A domain transition / 領域変換 | The operation plus construction of the existing destination Frame / Operationと既存の出力先Frame生成 | A new Frame when an existing domain model fits / 既存domain modelで表せる場合の新Frame |
| A genuinely new data domain / 本当に新しいデータ領域 | A new `BaseFrame` subclass with explicit axes and domain state / axisとdomain stateを明示した`BaseFrame` subclass | A Frame that only renames an existing result / 既存結果を名前だけ変えるFrame |
| Portable replay / Recipeで再利用可能な処理 | `@recipe_operation` and an end-to-end Recipe probe / `@recipe_operation`とRecipe完全経路テスト | Operation-specific branches in the Recipe model/compiler/serializer / Recipe中央層のoperation別分岐 |

A new Frame is justified only when data shape, axes, required constructor state,
or domain-specific behavior cannot be represented clearly by an existing Frame.
Different labels, units, or one extra convenience method are not enough.
新しい Frame は、data shape、axis、必須constructor state、またはdomain固有の振る舞いを
既存 Frame で明確に表現できない場合にだけ追加します。label、unit、便利メソッドが1つ違うだけでは
新しい Frame を作る理由になりません。

## Ownership map / 責務の配置

Keep one owner for each kind of state or behavior.
状態や振る舞いの種類ごとに、正本を1か所に限定します。

| Concern / 関心事 | Owner / 正本 |
| --- | --- |
| Numerical algorithm and parameter validation / 数値アルゴリズムとparameter検証 | `wandas/processing/` |
| Public method, input alignment, output Frame choice / 公開メソッド、入力整合、出力Frame選択 | `wandas/frames/` |
| User/recording/domain metadata and axes / 利用者・収録・domain metadataとaxis | Frame constructor/helper |
| Runtime provenance / runtime provenance | Immutable `lineage`; `operation_history` is derived / immutableな`lineage`。`operation_history`は派生view |
| Portable invocation intent / portableな呼出し意図 | `@recipe_operation` declaration |
| Numerical correctness / 数値的正しさ | `tests/processing/` |
| Public Frame contract / 公開Frame契約 | `tests/frames/` |
| Recipe extraction and replay / Recipe抽出と再実行 | `tests/pipeline/` |

Frame methods must not duplicate processing parameters in metadata, reconstruct
lineage manually, or call `.compute()` while building a result. Processing code
must not construct Frames or mutate Frame metadata.
Frameメソッドは、処理parameterをmetadataへ重複保存したり、lineageを手作業で再構築したり、
結果構築中に`.compute()`を呼んではいけません。processingコードはFrameを生成したり、
Frame metadataを変更してはいけません。

## Add an AudioOperation / AudioOperation を追加する

1. Choose the nearest module such as `filters.py`, `spectral.py`, `temporal.py`,
   `stats.py`, or `effects.py`.
   `filters.py`、`spectral.py`、`temporal.py`、`stats.py`、`effects.py`など、
   最も近いmoduleを選びます。
2. Subclass `AudioOperation`, give it a stable registry `name`, and pass all
   constructor configuration to `super().__init__`. The base class snapshots
   caller-owned configuration.
   `AudioOperation`を継承し、安定したregistry `name`を付け、constructor設定をすべて
   `super().__init__`へ渡します。base classが呼出し側所有の設定値をsnapshotします。
3. Validate operation parameters in `validate_params()` and implement the eager
   array kernel in `_process()`. `process()` is the shared lazy Dask boundary;
   do not replace it merely to implement the algorithm.
   `validate_params()`でparameterを検証し、`_process()`にeager array kernelを実装します。
   `process()`は共通のlazy Dask境界なので、アルゴリズム実装のためだけに上書きしません。
4. Override `calculate_output_shape()` and `calculate_output_dtype()` whenever
   shape or dtype differs from the input metadata. Dask needs both before compute.
   shapeまたはdtypeが入力metadataと異なる場合は、`calculate_output_shape()`と
   `calculate_output_dtype()`を上書きします。Daskはcompute前に両方を必要とします。
5. Use `get_metadata_updates()` only for real domain state changes such as a new
   sampling rate. It is not a second parameter store.
   `get_metadata_updates()`はsampling rate変更など実際のdomain state変更だけに使い、
   parameterの第2保存先にはしません。
6. Register the class with `register_operation()`. If the public class should be
   importable from `wandas.processing`, update its eager import or lazy-operation
   mapping and `__all__`.
   classを`register_operation()`へ登録します。`wandas.processing`から公開importする場合は、
   eager importまたはlazy-operation mappingと`__all__`も更新します。

The following sketch shows the required boundaries. Use the exact validation and
dtype appropriate for the real operation.
次の例は必要な責務境界を示します。実際のOperationに適した検証とdtypeを使用してください。

```python
class Gain(AudioOperation[NDArrayReal, NDArrayReal]):
    name = "gain"
    _display = "gain"

    def __init__(self, sampling_rate: float, factor: float) -> None:
        super().__init__(sampling_rate, factor=factor)

    def validate_params(self) -> None:
        factor = self._config_value("factor")
        if not isinstance(factor, int | float):
            raise TypeError(
                "Invalid gain factor\n"
                f"  Got: {type(factor).__name__}\n"
                "  Expected: int or float\n"
                "Pass a numeric amplitude multiplier."
            )

    def _process(self, data: NDArrayReal) -> NDArrayReal:
        return data * float(self._config_value("factor"))

    def calculate_output_dtype(
        self,
        input_dtype: np.dtype[Any],
        *input_dtypes: np.dtype[Any],
    ) -> np.dtype[Any]:
        return np.result_type(input_dtype, np.float64)


register_operation(Gain)
```

Never retain a user-supplied mutable list, mapping, or NumPy array on a separate
public attribute that must stay synchronized with the base configuration. Read it
through `_config_value()` or `_config_snapshot()`.
利用者が渡したmutableなlist、mapping、NumPy arrayを、base設定と同期が必要な別のpublic属性へ
保持しないでください。`_config_value()`または`_config_snapshot()`から読み出します。

## Add a public Frame method / 公開 Frame メソッドを追加する

The method is an orchestration boundary. It validates Frame-level compatibility,
creates or selects the operation, and returns a new Frame. For a same-domain unary
operation, prefer the existing application helper:
メソッドはorchestration境界です。Frame-levelの互換性を検証し、Operationを生成・選択して、
新しいFrameを返します。同一domainのunary operationでは既存の適用helperを優先します。

```python
@recipe_operation("wandas.audio.gain")
def gain(self, factor: float) -> ChannelFrame:
    return self._apply_operation_impl("gain", factor=factor)
```

Keep the public method name readable (`low_pass_filter`) even when the numerical
registry key differs (`lowpass_filter`). Treat these names as separate stable
contracts and test the mapping.
公開メソッド名は読みやすく保ち（`low_pass_filter`）、数値registry key
（`lowpass_filter`）と異なっていても構いません。両者を別々の安定契約として扱い、mappingをテストします。

For a domain transition, instantiate the destination Frame explicitly or use
`_apply_operation_instance(..., output_frame_class=..., output_frame_kwargs=...)`
when that helper expresses the contract without hidden state. Preserve channel IDs,
channel metadata, user metadata, sampling rate or its documented replacement,
`source_time_offset`, and the semantic lineage supplied by
`_required_semantic_lineage()`.
domain transitionでは、出力先Frameを明示的に生成するか、隠れた状態を増やさず契約を表現できる場合に
`_apply_operation_instance(..., output_frame_class=..., output_frame_kwargs=...)`を使います。
channel ID、channel metadata、利用者metadata、sampling rateまたは文書化した変更値、
`source_time_offset`、`_required_semantic_lineage()`が供給するsemantic lineageを維持します。

For multiple Frame or external array inputs, define input order and alignment in the
public method. Do not invent metadata for raw arrays. Preserve NumPy/Dask laziness and
snapshot mutable inputs at the operation boundary where the contract requires value
stability.
複数Frame入力やexternal array入力では、入力順序とalignmentを公開メソッドで定義します。
raw arrayにmetadataを捏造しません。NumPy/Daskのlazinessを維持し、値の安定性が契約上必要な場合は
Operation境界でmutable inputをsnapshotします。

## Add a new Frame family / 新しい Frame family を追加する

Before implementation, write down these invariants in the class docstring and tests:
実装前に、次のinvariantをclass docstringとtestへ記述します。

- array rank, channel axis, and domain axes;
  array rank、channel axis、domain axis。
- real or complex dtype expectations;
  real／complex dtypeの期待値。
- required constructor state and how it determines axes;
  必須constructor stateと、それがaxisを決める方法。
- sampling-rate and `source_time_offset` meaning;
  sampling rateと`source_time_offset`の意味。
- which operations preserve the family and which transition to another Frame.
  familyを維持するOperationと、別Frameへ遷移するOperation。

Then implement the smallest `BaseFrame` subclass that satisfies them:
その後、契約を満たす最小の`BaseFrame` subclassを実装します。

1. Validate and normalize shape in `__init__`, while keeping the internal array lazy.
   `__init__`でshapeを検証・正規化し、内部arrayはlazyなまま維持します。
2. Set `_xarray_dim_suffix` to the authoritative dimension names.
   `_xarray_dim_suffix`へ正本となるdimension名を設定します。
3. Implement required domain properties and `plot()`.
   必要なdomain propertyと`plot()`を実装します。
4. Override `_get_additional_init_kwargs()` for constructor state that must survive
   `_create_new_instance()`.
   `_create_new_instance()`後も維持すべきconstructor stateは
   `_get_additional_init_kwargs()`で返します。
5. Override `_get_dataframe_index()` when DataFrame export has a domain axis.
   DataFrame exportにdomain axisがある場合は`_get_dataframe_index()`を上書きします。
6. Export the class from `wandas.frames`; add a top-level `wandas` export only when
   that is the intended public UX. Add it to the Frame API reference.
   classを`wandas.frames`からexportします。top-level `wandas` exportは意図した公開UXの場合だけ
   追加し、Frame API referenceにも追加します。

`previous` is a compatibility/debug pointer, not provenance. Never derive history or
Recipe structure from it. `lineage` remains the only provenance state.
`previous`は互換性・debug用pointerであり、provenanceではありません。historyやRecipe構造を
`previous`から生成せず、`lineage`だけをprovenance stateにします。

## Make the operation Recipe-capable / Operation を Recipe 対応にする

Use `@recipe_operation` on the public Frame method when the call should be portable.
The stable operation ID and version describe the serialized behavior, not the Python
class path. Unary Frame operations use the default capture and handler. Multi-Frame,
Frame-or-array, positional-only, or variadic contracts need explicit bindings,
capture, and handler.
呼出しをportableにする場合は、公開Frameメソッドへ`@recipe_operation`を付けます。安定した
operation IDとversionはPython class pathではなくserializeされる振る舞いを表します。
unary Frame operationはdefault capture／handlerを使えます。multi-Frame、Frame-or-array、
positional-only、variadic contractでは明示的なbinding、capture、handlerが必要です。

Built-in methods declared directly on an owner already collected by
`wandas.pipeline.builtins` enter the default immutable registry. A new built-in Frame
owner must be added to that declaration collection. External extensions derive a new
registry with `default_recipe_registry().with_operation(recipe_definition(method))`;
they never mutate the default registry.
`wandas.pipeline.builtins`が既に収集するowner上のbuilt-inメソッドはdefault immutable registryへ
入ります。新しいbuilt-in Frame ownerは宣言収集対象へ追加します。外部拡張は
`default_recipe_registry().with_operation(recipe_definition(method))`で新しいregistryを派生し、
default registryを変更しません。

Prove portability through the complete public path:
portable性は次の公開完全経路で証明します。

```text
public Frame operation
  -> semantic lineage
  -> RecipePlan.from_frame
  -> RecipePlan.to_dict
  -> RecipePlan.from_dict
  -> RecipePlan.apply
```

Do not add operation-specific conditionals to the Recipe model, compiler, validator,
executor, or serializer. See the
[Recipe extension guide](../explanation/pipeline-recipe-developer-guide.md) for binding,
handler, immutable parameter, and external-input details.
Recipeのmodel、compiler、validator、executor、serializerへoperation固有の条件分岐を追加しません。
binding、handler、immutable parameter、external inputの詳細は
[Recipe extension guide](../explanation/pipeline-recipe-developer-guide.md)を参照してください。

## Add tests with the feature / 機能と同時にテストを追加する

Tests should describe the public contract, not reproduce private implementation.
Use deterministic known signals and independent expected values.
testはprivate実装を再現するのではなく、公開契約を記述します。決定論的な既知信号と、
実装から独立した期待値を使います。

### Processing tests / Processing テスト

Add focused tests under `tests/processing/` for:
`tests/processing/`へ次のfocused testを追加します。

- constructor validation and WHAT/WHY/HOW errors;
  constructor検証とWHAT/WHY/HOW形式のerror。
- operation registration and the public registry key;
  Operation登録と公開registry key。
- output shape and dtype metadata before compute;
  compute前の出力shape／dtype metadata。
- no eager `.compute()` while `process()` builds the graph;
  `process()`のgraph構築中にeagerな`.compute()`がないこと。
- numerical equivalence to SciPy/librosa/MoSQITo for wrappers, or an analytical
  result for custom algorithms;
  wrapperではSciPy／librosa／MoSQIToとの数値一致、独自algorithmでは解析解との一致。
- input and caller-owned configuration remain unchanged.
  入力と呼出し側所有の設定値が変更されないこと。

### Frame tests / Frame テスト

Add focused tests under `tests/frames/` for:
`tests/frames/`へ次のfocused testを追加します。

- correct return Frame type and chainability;
  正しい戻りFrame型とchainability。
- original data, metadata, labels, offsets, and lineage remain unchanged;
  元のdata、metadata、label、offset、lineageが変更されないこと。
- output metadata, axes, labels, and `source_time_offset` are correct;
  出力metadata、axis、label、`source_time_offset`が正しいこと。
- one public processing call creates exactly one lineage/history entry;
  公開処理1回がlineage／historyを正確に1件生成すること。
- the result remains Dask-backed and lazy;
  結果がDask-backedかつlazyであること。
- invalid shape, sampling rate, channel alignment, and boundary parameters fail
  explicitly;
  不正なshape、sampling rate、channel alignment、境界parameterが明示的に失敗すること。
- domain transitions produce theoretically correct axes and dimensions.
  domain transition後のaxisとdimensionが理論値に一致すること。

For a new Frame family, also test constructor normalization, `_create_new_instance()`
state preservation, xarray dimension names, DataFrame index behavior, public exports,
and at least one inbound and outbound domain transition.
新しいFrame familyでは、constructor正規化、`_create_new_instance()`によるstate維持、
xarray dimension名、DataFrame index、public export、少なくとも1つの入出domain transitionもテストします。

### Recipe and documentation tests / Recipe・文書テスト

If the operation is portable, add a complete extract/serialize/load/apply test under
`tests/pipeline/`. Cover mutation isolation, metadata and offset preservation, input
order for multiple inputs, lazy external Dask arrays, and unknown ID/version rejection
when relevant.
Operationがportableなら、`tests/pipeline/`へextract／serialize／load／apply完全経路テストを
追加します。該当する場合はmutation isolation、metadata／offset維持、複数入力の順序、
external Dask arrayのlaziness、未知ID／version拒否を含めます。

Update public docstrings, `docs/src/api/`, and a tutorial or how-to when users need a
new workflow explanation. Every public class and method must describe parameters,
return type, raised errors, laziness or compute behavior, metadata/axis effects, and a
minimal example.
新しいworkflow説明が必要なら、公開docstring、`docs/src/api/`、tutorialまたはhow-toも更新します。
すべての公開class／methodにparameter、戻り型、例外、lazinessまたはcompute動作、metadata／axisへの
影響、最小exampleを記載します。

## Definition of done / 完了条件

Run focused tests first, then the repository gates appropriate to the change.
最初にfocused testを実行し、その後変更に応じたrepository gateを実行します。

```bash
uv run pytest tests/processing/<module-test>.py -q
uv run pytest tests/frames/<frame-test>.py -q
uv run pytest tests/pipeline -q  # when Recipe behavior changes
uv run ruff check wandas tests scripts
uv run ty check wandas tests
uv run pytest -q
uv run mkdocs build --strict -f docs/mkdocs.yml
```

Before opening a PR, confirm:
PR作成前に次を確認します。

- no input Frame or caller-owned configuration is mutated;
  入力Frameや呼出し側所有の設定値が変更されていない。
- metadata, axes, labels, offsets, lineage, and output type change atomically;
  metadata、axis、label、offset、lineage、出力型がatomicに変更される。
- graph construction does not call Dask `.compute()`;
  graph構築中にDask `.compute()`を呼ばない。
- the operation has one numerical registry key and, when portable, one Recipe ID;
  Operationが1つの数値registry keyと、portableな場合は1つのRecipe IDを持つ。
- tests cover Unit, Domain, and Integration layers where applicable;
  該当するUnit、Domain、Integration層をtestがcoverする。
- no new compatibility shim, duplicate state, or operation-specific Recipe branch was
  introduced.
  新しいcompatibility shim、状態重複、operation固有Recipe分岐を追加していない。

## Agent route / Agent の参照順序

Agents should follow this route instead of relying on one example file:
Agentは1つのexample fileだけに依存せず、次の順序で参照します。

1. [`AGENTS.md`](https://github.com/kasahart/wandas/blob/develop/AGENTS.md) for
   repository-wide authority.
   repository全体の正本として
   [`AGENTS.md`](https://github.com/kasahart/wandas/blob/develop/AGENTS.md)。
2. This guide for the end-to-end extension workflow.
   拡張のend-to-end手順としてこのガイド。
3. `frames-design.instructions.md` and `processing-api.instructions.md` for the
   files being changed.
   変更対象に応じて`frames-design.instructions.md`と`processing-api.instructions.md`。
4. `test-grand-policy.instructions.md`, `test-frames-policy.instructions.md`, and
   `test-processing-policy.instructions.md` for test design.
   test設計では`test-grand-policy.instructions.md`、`test-frames-policy.instructions.md`、
   `test-processing-policy.instructions.md`。
5. The [Recipe extension guide](../explanation/pipeline-recipe-developer-guide.md)
   when the public operation must be portable.
   公開Operationをportableにする場合は
   [Recipe extension guide](../explanation/pipeline-recipe-developer-guide.md)。
