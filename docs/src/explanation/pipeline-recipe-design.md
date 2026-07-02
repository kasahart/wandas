# Pipeline Recipe Design / パイプラインレシピ設計書

## Design Summary / 設計概要

Wandas 側に小さな replayable spec を置き、sklearn は optional adapter として扱う。中心は `OperationSpec` と `RecipeSpec` であり、実行は既存の `Frame.apply_operation(...)` に委譲する。これにより、frame immutability、metadata/history、Dask lazy execution は既存 frame operation の契約に従う。

sklearn 互換の `WandasOperationTransformer` は、`BaseEstimator` / `TransformerMixin` と同じ呼び出し形を提供するだけの薄いラッパーである。信号処理の正本は Wandas operation registry key であり、sklearn `Pipeline` は UI/連携の選択肢に留める。

## Components / コンポーネント

### `wandas.pipeline.OperationSpec`

- `operation: str`
- `params: Mapping[str, Any]`

単一 Wandas operation の replayable な呼び出しを表す。`operation` は `"normalize"` や `"highpass_filter"` のような registry key を使う。`params` は作成時に snapshot されるため、呼び出し元の dict を後から変更しても spec は変わらない。

### `wandas.pipeline.RecipeSpec`

- `steps: tuple[OperationSpec, ...]`
- `apply(frame) -> frame`

`steps` を順番に処理し、各 step で `frame.apply_operation(step.operation, **step.params)` を呼ぶ。`RecipeSpec` 自体は frame を保持しないため、同じレシピを複数 frame に再適用できる。

### `wandas.pipeline.sklearn.WandasOperationTransformer`

- `fit(X, y=None) -> self`
- `transform(X) -> X.apply_operation(operation, **params)`
- `to_spec() -> OperationSpec`
- sklearn 標準の `get_params()` / `set_params()` を提供する

この transformer は stateful learning をしない。sklearn `Pipeline.transform(frame)` の UX を使うための stateless transformer として扱う。

### Named Transformers / 個別 transformer

| Class | Wandas operation | Parameters |
| --- | --- | --- |
| `HighPassFilter` | `highpass_filter` | `cutoff`, `order=4` |
| `LowPassFilter` | `lowpass_filter` | `cutoff`, `order=4` |
| `BandPassFilter` | `bandpass_filter` | `low_cutoff`, `high_cutoff`, `order=4` |
| `Normalize` | `normalize` | `norm=np.inf`, `axis=-1`, `threshold=None`, `fill=None` |
| `RemoveDC` | `remove_dc` | none |

## Data Flow / データフロー

```text
ChannelFrame
  -> RecipeSpec.apply(frame)
    -> OperationSpec("remove_dc")
      -> frame.apply_operation("remove_dc")
    -> OperationSpec("highpass_filter", {"cutoff": 100.0})
      -> frame.apply_operation("highpass_filter", cutoff=100.0)
    -> OperationSpec("normalize")
      -> frame.apply_operation("normalize")
  -> new ChannelFrame with operation_history
```

sklearn adapter の場合も、最終的には同じ `apply_operation` に到達する。

```text
sklearn.pipeline.Pipeline
  -> HighPassFilter.transform(frame)
    -> frame.apply_operation("highpass_filter", cutoff=...)
  -> Normalize.transform(frame)
    -> frame.apply_operation("normalize", ...)
```

## UX Principles / UX原則

- Wandas ユーザーには `RecipeSpec([...]).apply(frame)` を見せる。これは「この順序で同じ処理を再実行する」意図が明確である。
- sklearn ユーザーには `Pipeline([("hp", HighPassFilter(...)), ...]).transform(frame)` を見せる。これは既存の ML 前処理文脈に馴染む。
- operation history を最終確認画面として使う。ユーザーは結果 frame を見れば、処理順とパラメータを確認できる。
- `to_spec()` により、sklearn transformer から Wandas-native spec へ戻れる。

## Error Handling / エラー処理

- sklearn 未導入で `wandas.pipeline.sklearn` を import した場合は、optional dependency registry を通じて `pip install "wandas[sklearn]"` を含む `ImportError` を出す。
- operation 名やパラメータが不正な場合は、既存の `apply_operation` / operation class の validation に委譲する。
- v1 では複数入力 operation を recipe から呼ばない。必要な場合は operation-specific method を使う。

## Testing Strategy / テスト方針

- Optional dependency tests: `sklearn` extra と dependency registry の整合性、未導入時の `ImportError` を確認する。
- Recipe tests: operation 順序、入力 frame 不変性、Dask lazy graph の維持を確認する。
- sklearn adapter tests: `get_params()` / `set_params()`、sklearn `Pipeline.transform(frame)`、`to_spec()` を確認する。
- Notebook smoke: 合成信号で `RecipeSpec` と sklearn `Pipeline` の UX を実行し、operation history と簡単な数値指標を表示する。

## Future Extensions / 次フェーズ候補

- frame の探索的メソッドチェインから `RecipeSpec.from_frame(processed)` で Recipe を抽出する UX。
- `FrameMethodStep` を追加し、`sum()`、`mean()`、`fix_length()` のような method 固有 metadata 処理を replay する。
- `GraphRecipeSpec` を追加し、binary operation、複数入力 operation、shared branch を表現する。
- `RecipeSpec` の JSON schema と WDF metadata 保存。
- `RecipeSpec` と sklearn `Pipeline` の明示的な相互変換。
- domain transition を含む recipe の型制約。
- notebook から保存済み recipe を読み込み、バッチ処理に適用する workflow。

See also: [Pipeline Recipe Extraction Boundaries](pipeline-recipe-extraction-boundaries.md).
