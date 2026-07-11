# Pipeline Recipe Use Cases / パイプラインレシピのユースケース

## Installation / インストール

sklearn adapter を使う場合:

```bash
pip install "wandas[sklearn]"
```

Wandas-native の `RecipeSpec` だけを使う場合、sklearn は不要である。

## Use Case 1: Notebook の前処理を再実行可能にする

研究や測定の Notebook では、試行錯誤で次のような前処理が固まることがある。
Recipeの入口は、この通常のframe操作でよい。

```python
processed = (
    frame
    .remove_dc()
    .high_pass_filter(cutoff=100.0, order=2)
    .normalize()
)
```

同じ処理を別データへ適用したい場合、処理済みframeからRecipeを抽出する。

```python
from wandas.pipeline import RecipeSpec

recipe = RecipeSpec.from_frame(processed)
replayed = recipe.apply(other_frame)
```

ユーザーが最初から `OperationSpec` を手で並べたり、sklearn `Pipeline` を構成したりする必要はない。
必要なら、抽出されたstepを後から読める。

```python
for step in recipe.steps:
    print(step.operation, dict(step.params))
```

確認ポイント:

- 入力 frame は変更されない。
- replay結果 frame の `operation_history` に `remove_dc -> highpass_filter -> normalize` が残る。
- underlying operation が lazy な場合、Dask-backed frame でも recipe 適用時点では計算を開始しない。

## Use Case 2: 任意でRecipeを直接構成する

明示的なRecipe構築は、テスト、生成された設定、監査用の仕様などで有用である。
ただし初学者ユーザーの標準入口ではない。

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

## Use Case 3: sklearn Pipeline の形で前処理を組む

ML チームや既存コードが sklearn の `Pipeline` に慣れている場合、Wandas frame を sklearn 風に変換できる。
これは連携用のオプションであり、Wandas Recipeの正本ではない。

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

この `Pipeline` は Wandas operation を呼ぶ薄い wrapper であり、sklearn を処理履歴の正本にしない。結果確認は Wandas frame の `operation_history` で行う。

## Use Case 4: sklearn 風パラメータ編集から Wandas spec に戻す

sklearn adapter は `get_params()` / `set_params()` を持つため、Notebook UI や探索コードでパラメータを編集しやすい。

```python
hp = HighPassFilter(cutoff=80.0)
hp.set_params(cutoff=120.0, order=6)

operation_spec = hp.to_spec()
```

`to_spec()` により、sklearn 風 transformer から Wandas-native な `OperationSpec` に戻せる。

## Use Case 5: Frame-first UX をmarimoで確認する

まず触るべきRecipe UXは、frame method chainからの抽出とreplayである。

- <a href="../learning-path/06_pipeline_recipe_ux.html">Learning Path — 06_Frame-First Recipe UX</a>
- source: `learning-path/06_pipeline_recipe_ux.py`

この marimo アプリでは次を確認できる。

- 通常のframe method chainで探索する。
- `RecipeSpec.from_frame(...)` でRecipeを抽出する。
- 別frameに `recipe.apply(other_frame)` で再現する。
- 明示Recipe構築とsklearn adapterは任意セクションとして扱う。

## Use Case 6: Advanced design exploration をmarimoで確認する

Recipe のあるべき姿は、ユーザーが最初から pipeline 固有の class を選ぶことではない。
まず通常の frame 操作で探索し、処理済み frame から replayable な recipe を作り、別 frame で再現実験できることである。
このアプリは初学者向けの最初の導線ではなく、graph recipeや将来facade案を検討するための補助資料である。

frame 中心の UX を確認する marimo アプリ:

- <a href="../learning-path/07_frame_centric_recipe_ux.html">Advanced design app — 07_Frame-Centric Recipe UX</a>
- source: `learning-path/07_frame_centric_recipe_ux.py`

この marimo アプリでは次を確認できる。

- 通常の frame method chain から `RecipeSpec.from_frame(...)` で抽出する。
- notebook-local facade を使い、`RecipeSpec` / `GraphRecipeSpec` / `NodeGraphRecipeSpec` の選択をユーザーから隠す理想形を確認する。
- 別 frame への replay、graph recipe、外部データ、terminal step、抽出境界を小さなデータで実行する。
- 現在の public API と、将来の `processed.to_recipe(...)` のような入口候補を区別して確認する。

## Decision Guide / 使い分け

| Need | Recommended API |
| --- | --- |
| Wandas 内で処理順を再利用したい | `RecipeSpec` |
| sklearn 風の `fit` / `transform` に合わせたい | `wandas.pipeline.sklearn` |
| 保存形式や長期互換性を設計したい | 次フェーズの JSON/WDF 設計 |
| 任意 Python 関数を replay したい | v1 では対象外。明示的な Wandas operation 化を検討する |
