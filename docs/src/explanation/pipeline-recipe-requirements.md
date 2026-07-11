# Pipeline Recipe Requirements / パイプラインレシピ要件定義

## Purpose / 目的

Wandas の信号処理チェーンを、Notebook やバッチ処理で再利用できる小さな「レシピ」として表現する。scikit-learn の `Pipeline` を正本にせず、Wandas の frame API と `operation_graph` を replay の正本にすることで、既存のフレーム不変性、メタデータ伝播、Dask lazy execution を保つ。operation history は確認用の互換 view として扱う。

The long-term target is to make every calculation that can be expressed with a Wandas frame replayable as a Recipe. The current implementation is a staged subset: linear single-input recipes, selected method-aware and typed steps, explicit terminal steps, importable custom functions, and graph recipes for supported multi-input trees.

長期的な目標は、Wandas frame で表現できる計算をすべて Recipe として再生可能にすることである。現在の実装は段階的なサブセットであり、単一入力の直列 Recipe、対応済みの method-aware / typed step、明示 terminal step、importable custom function、対応済み複数入力 tree の graph recipe を対象とする。

## Target Users / 対象ユーザー

- 実験 Notebook で作った前処理手順を、後で同じ順序・同じパラメータで再実行したい信号処理ユーザー。
- Wandas の `Frame` 操作を使いながら、sklearn 風の `fit` / `transform` インターフェースで前処理を組みたい ML ユーザー。
- operation history view で「どの処理をいつ適用したか」を確認し、分析ノートやレビューで説明したい開発者。

## Functional Requirements / 機能要件

| ID | Requirement | Acceptance Criteria |
| --- | --- | --- |
| R1 | Replayable な単一 operation を表現できる | `OperationSpec(operation, params)` が operation registry key とパラメータを保持する |
| R2 | 複数 operation を直列適用できる | `RecipeSpec.apply(frame)` が `frame.apply_operation(...)` を順に呼び、最後の frame を返す |
| R3 | 元 frame を破壊しない | レシピ適用後も入力 frame の data、metadata、operation history view は変わらない |
| R4 | Dask laziness を維持する | Dask-backed frame に適用しても `.compute()` を呼ばず、lazy graph を返す |
| R5 | sklearn 互換 transformer を提供する | `fit(X, y=None) -> self`、`transform(X)`、`get_params()`、`set_params()`、`to_spec()` が動く |
| R6 | 代表的な signal operation をクラス化する | `HighPassFilter`、`LowPassFilter`、`BandPassFilter`、`Normalize`、`RemoveDC` を提供する |
| R7 | sklearn は optional dependency とする | 通常の `import wandas` では sklearn を import せず、`wandas.pipeline.sklearn` import 時だけ必要にする |
| R8 | 未導入時の案内を明確にする | sklearn 未導入時は `pip install "wandas[sklearn]"` を含む `ImportError` を出す |
| R9 | 探索的に処理した frame から直列 Recipe を抽出できる | `RecipeSpec.from_frame(processed)` が `processed.operation_graph` を読み、対応済みの単一入力 chain を replayable step 列に変換する |
| R10 | 複数入力 graph recipe を抽出できる | `GraphRecipeSpec.from_frame(...)` と `NodeGraphRecipeSpec.from_frame(...)` が対応済み frame-frame / external operand / `add_channel` graph を名前付き入力から replay できる |
| R11 | frame method / typed transition / indexing / scalar / custom / terminal を表現できる | `MethodStep`、`TypedMethodStep`、`IndexingStep`、`ScalarOperationStep`、`CustomFunctionStep`、`TerminalStep` が既存 frame API へ委譲して replay する |
| R12 | 抽出できない境界を明示できる | 対応外の callable、query、indexing、graph、terminal 自動抽出は黙って落とさず `RecipeExtractionError` を返す |

## Non-Functional Requirements / 非機能要件

- API surface は小さく保つ。v1 では recipe serialization schema や WDF 永続化を追加しない。
- operation 名は frame method 名ではなく registry key を使う。例: `high_pass_filter(...)` ではなく `"highpass_filter"`。
- `RecipeSpec` と `OperationSpec` は実行時に外部からパラメータを書き換えられない構造にする。
- sklearn adapter は薄い境界に留め、数値処理や frame 作成ロジックを持たない。
- Recipe 側で計算ロジック、metadata 更新、Dask graph 構築を複製しない。replay は既存 frame method、operator、`apply_operation(...)` へ委譲する。
- `RecipeExtractionError` は失敗ではなく、現在の replay contract が定義されていない境界を示すための明示的な signal とする。

## Out of Scope / スコープ外

- `RecipeSpec` と `sklearn.pipeline.Pipeline` の自動相互変換。
- WDF への `recipe_json` 保存。
- `FunctionTransformer`、lambda、closure、nested function、callable object、bound method、`functools.partial`、`__main__` 上の関数の replayable 判定。
- 対応表にない frame method、typed transition、indexing form、terminal metric の自動抽出。
- true DAG identity、shared branch identity、外部入力名の自動推定。
- joblib/skops export、Dask-ML 連携、モデル学習結果の保存。

## Success Metrics / 成功指標

- Executable requirement checks live in [Pipeline Recipe Requirements Check Notebook](../tutorial/pipeline-recipe-requirements-check.md).
  実行可能な要件確認は [Pipeline Recipe Requirements Check Notebook](../tutorial/pipeline-recipe-requirements-check.md) にあります。
- Notebook 上で `RecipeSpec` と sklearn `Pipeline` の両方を実行し、同じ operation history view を確認できる。
- Notebook 上で探索的に処理した frame から `RecipeSpec.from_frame(...)` を抽出し、別 frame へ replay できる。
- Notebook 上で `GraphRecipeSpec.from_frame(...)` または `NodeGraphRecipeSpec.from_frame(...)` を使い、対応済み graph recipe を replay できる。
- Notebook 上で対応外操作が `RecipeExtractionError` になることを確認できる。
- 既存の Wandas operation history view で、レシピ適用順とパラメータを確認できる。
- `uv run pytest tests/test_optional_dependencies.py tests/pipeline` が通る。
- `uv run ruff check wandas tests --config=pyproject.toml -v` と `uv run ty check wandas tests` が通る。
