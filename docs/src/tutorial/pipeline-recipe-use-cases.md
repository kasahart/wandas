# Pipeline Recipe Use Cases / パイプラインレシピのユースケース

## Installation / インストール

sklearn adapter を使う場合:

```bash
pip install "wandas[sklearn]"
```

Wandas-native の `RecipeSpec` だけを使う場合、sklearn は不要である。

## Use Case 1: Notebook の前処理を再実行可能にする

研究や測定の Notebook では、試行錯誤で次のような前処理が固まることがある。

```python
processed = (
    frame
    .remove_dc()
    .high_pass_filter(cutoff=100.0, order=2)
    .normalize()
)
```

同じ処理を別データへ適用したい場合、`RecipeSpec` にすると順序とパラメータが明示される。

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

確認ポイント:

- 入力 frame は変更されない。
- 結果 frame の `operation_history` に `remove_dc -> highpass_filter -> normalize` が残る。
- Dask-backed frame でも recipe 適用時点では計算を開始しない。

## Use Case 2: sklearn Pipeline の形で前処理を組む

ML チームや既存コードが sklearn の `Pipeline` に慣れている場合、Wandas frame を sklearn 風に変換できる。

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

## Use Case 3: sklearn 風パラメータ編集から Wandas spec に戻す

sklearn adapter は `get_params()` / `set_params()` を持つため、Notebook UI や探索コードでパラメータを編集しやすい。

```python
hp = HighPassFilter(cutoff=80.0)
hp.set_params(cutoff=120.0, order=6)

operation_spec = hp.to_spec()
```

`to_spec()` により、sklearn 風 transformer から Wandas-native な `OperationSpec` に戻せる。

## Use Case 4: UX Notebook で処理履歴を確認する

実行可能な UX ノートブック:

```bash
learning-path/06_pipeline_recipe_ux.ipynb
```

この Notebook では次を確認できる。

- 合成信号の作成。
- `RecipeSpec` での前処理。
- sklearn `Pipeline` での同じ前処理。
- `operation_history` による replayability の確認。
- 処理前後の簡単な数値比較。

## Decision Guide / 使い分け

| Need | Recommended API |
| --- | --- |
| Wandas 内で処理順を再利用したい | `RecipeSpec` |
| sklearn 風の `fit` / `transform` に合わせたい | `wandas.pipeline.sklearn` |
| 保存形式や長期互換性を設計したい | 次フェーズの JSON/WDF 設計 |
| 任意 Python 関数を replay したい | v1 では対象外。明示的な Wandas operation 化を検討する |
