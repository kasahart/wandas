# Pipeline Recipe Design / パイプラインレシピ設計書

## Design Summary / 設計概要

Wandas 側に小さな replayable spec を置き、sklearn は optional adapter として扱う。中心は `OperationSpec` と `RecipeSpec` であり、実行は既存の `Frame.apply_operation(...)` に委譲する。これにより、frame immutability、metadata/history、Dask lazy execution は既存 frame operation の契約に従う。

sklearn 互換の `WandasOperationTransformer` は、`BaseEstimator` / `TransformerMixin` と同じ呼び出し形を提供するだけの薄いラッパーである。信号処理の正本は Wandas operation registry key であり、sklearn `Pipeline` は UI/連携の選択肢に留める。

## Components / コンポーネント

### `wandas.pipeline.OperationSpec`

- `operation: str`
- `params: Mapping[str, Any]`

単一 Wandas operation の replayable な呼び出しを表す。`operation` は `"normalize"`、`"highpass_filter"`、`"fade"` のような registry key を使う。`params` は作成時に snapshot されるため、呼び出し元の dict を後から変更しても spec は変わらない。

Stage 1 では、`params` は recipe literal に限定する。対応する値は `None`、`bool`、`int`、`float`、`str`、およびそれらだけを要素に持つ浅い `list` / `tuple` である。sequence は作成時に snapshot され、`to_dict()` では JSON 的に扱いやすい `list` として返る。nested sequence、dict、callable、array-like object、NaN は、保存形式と equality policy を決めるまで受け付けない。

`rename_channels()` は例外的に `MethodStep` 側で dedicated mapping snapshot を持つ。Python dict をそのまま JSON object にすると int key が文字列化されるため、`to_dict()` では `{"mapping_items": [[0, "left"], [1, "right"]]}` の pair list として保存する。これは汎用 mapping params のサポートではなく、既存 frame method の replay に必要な最小の typed 表現である。

### `wandas.pipeline.RecipeSpec`

- `steps: tuple[OperationSpec | MethodStep | TypedMethodStep | IndexingStep | ScalarOperationStep, ...]`
- `apply(frame) -> Any`
- `from_frame(frame) -> RecipeSpec`

`steps` を順番に処理し、`OperationSpec` は `frame.apply_operation(step.operation, **step.params)`、`MethodStep` / `TypedMethodStep` は対応する frame method、`IndexingStep` は既存 `frame[key]`、`ScalarOperationStep` は既存の frame 演算子を呼ぶ。`RecipeSpec` 自体は frame を保持しないため、同じレシピを複数 frame に再適用できる。

`from_frame(frame)` は、探索的に処理した frame の `operation_graph` から Stage 1 で安全に replay できる直列 chain だけを抽出する。対応外の計算は黙って落とさず、`RecipeExtractionError` で現在の境界を返す。

### `wandas.pipeline.MethodStep`

- `method: str`
- `params: Mapping[str, Any]`

`sum()`、`mean()`、`fix_length()`、`channel_difference()`、`get_channel()`、`remove_channel()`、`rename_channels()` のように、metadata 変換を frame method 側に持つ処理を replay する。Recipe 側では metadata 再構成を複製せず、既存 method を呼ぶ。

### `wandas.pipeline.TypedMethodStep`

- `method: str`
- `params: Mapping[str, Any]`

`fft()`、`stft()`、`get_frame_at()`、`ifft()`、`istft()`、`welch()`、`noct_spectrum()`、`noct_synthesis()`、`coherence()`、`csd()`、`transfer_function()`、`roughness_dw_spec()` のように、出力 frame class が変わる method を replay する。戻り型は入力 frame と同じとは限らないため、`RecipeSpec.apply(frame)` の戻り値は実行した step chain に従う。

### `wandas.pipeline.ScalarOperationStep`

- `symbol: str`
- `operand: int | float`

`frame + 0.1`、`frame * 2` のような単一 numeric scalar operand を持つ frame 演算子を replay する。対応演算子は `+`、`-`、`*`、`/`、`**` に限定する。

この step も演算ロジックを Recipe 側に複製しない。`apply()` 時に `frame + operand` のように既存 frame operator を呼び、metadata/history/Dask laziness は frame 本体の実装に委譲する。operation graph に値として保存された Python / NumPy real scalar は replay できる。frame-frame 演算、ndarray/dask array operand、値を持たない scalar-like descriptor は、graph recipe と operand serialization policy が決まるまで対象外である。

### `wandas.pipeline.IndexingStep`

- `key: slice | list[int] | list[str] | tuple[int | slice | list[int] | list[str], ...]`

`frame[0:2]` のような channel slice、`frame[[1, 0]]` や `frame[np.array([1, 0])]` のような integer selection、`frame[np.array([False, True])]` のような 1-D boolean mask、`frame[["left", "right"]]` のような label list selection、`frame[:, 100:400]` のような slice-only tuple indexing を replay する。Recipe は selection logic や metadata 更新を複製せず、保存した key で既存 `frame[key]` を呼ぶ。Point/fancy time selection は、indexing intent を保存する schema が必要なため対象外である。

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
    -> MethodStep("fix_length", {"length": 8000})
      -> frame.fix_length(length=8000)
    -> MethodStep("get_channel", {"channel_idx": 0})
      -> frame.get_channel(channel_idx=0)
    -> OperationSpec("highpass_filter", {"cutoff": 100.0})
      -> frame.apply_operation("highpass_filter", cutoff=100.0)
    -> ScalarOperationStep("+", 0.25)
      -> frame + 0.25
    -> OperationSpec("fade", {"fade_ms": 10.0})
      -> frame.apply_operation("fade", fade_ms=10.0)
    -> OperationSpec("rms_trend", {"frame_length": 512, "hop_length": 128, "ref": [1.0]})
      -> frame.apply_operation("rms_trend", frame_length=512, hop_length=128, ref=[1.0])
    -> OperationSpec("roughness_dw", {"overlap": 0.5})
      -> frame.apply_operation("roughness_dw", overlap=0.5)
    -> OperationSpec("normalize")
      -> frame.apply_operation("normalize")
  -> new ChannelFrame with operation_history
```

型遷移を含む場合、戻り値の frame class は step chain に従う。

```text
ChannelFrame
  -> RecipeSpec.apply(frame)
    -> TypedMethodStep("fft", {"n_fft": 1024, "window": "hann"})
      -> frame.fft(n_fft=1024, window="hann")
  -> SpectralFrame with operation_history

ChannelFrame
  -> RecipeSpec.apply(frame)
    -> TypedMethodStep("stft", {"n_fft": 512, "hop_length": 128, "win_length": 512, "window": "hann"})
      -> frame.stft(n_fft=512, hop_length=128, win_length=512, window="hann")
    -> TypedMethodStep("istft")
      -> spectrogram.istft()
  -> ChannelFrame with operation_history

ChannelFrame
  -> RecipeSpec.apply(frame)
    -> TypedMethodStep("coherence", {"n_fft": 512, "hop_length": 128, "win_length": 512, "window": "hann", "detrend": "constant"})
      -> frame.coherence(n_fft=512, hop_length=128, win_length=512, window="hann", detrend="constant")
  -> SpectralFrame with pairwise channel metadata

SpectralFrame
  -> RecipeSpec.apply(frame)
    -> TypedMethodStep("noct_synthesis", {"fmin": 25, "fmax": 20000, "n": 3, "G": 10, "fr": 1000})
      -> spectral_frame.noct_synthesis(fmin=25, fmax=20000, n=3, G=10, fr=1000)
  -> NOctFrame

ChannelFrame
  -> RecipeSpec.apply(frame)
    -> TypedMethodStep("roughness_dw_spec", {"overlap": 0.5})
      -> frame.roughness_dw_spec(overlap=0.5)
  -> RoughnessFrame
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
- sequence params は浅い literal sequence だけを受け付ける。`rms_trend()` / `sound_level()` の `ref` や HPSS の `kernel_size` / `margin` は replay に必要な値として保存するが、nested list や array operand は graph recipe の範囲として拒否する。
- `loudness_zwtv()`、`roughness_dw()`、`sharpness_din()` のような frame を返す psychoacoustic operation は `OperationSpec` で replay する。`loudness_zwst()`、`sharpness_din_st()` のように配列を返す terminal metric は現在の frame transform Recipe には含めない。
- `get_channel()` は explicit index/list、exact label query、literal dict query だけを抽出する。callable、regex、regex 値を含む dict query は query intent を保存する schema が必要なため拒否する。
- `__getitem__` は channel slice、integer list/ndarray、1-D boolean mask ndarray、label list、slice-only tuple indexing だけを `IndexingStep` として抽出する。point/fancy time selection は indexing intent と source time offset contract の追加 schema が必要なため現在は拒否する。
- `add_channel()` は外部 data または別 frame 入力が必要なため、現行の単一入力 `RecipeSpec` では replay しない。非 `inplace` 実行では lineage を残し、抽出時に明示的に拒否して部分 Recipe を返さない。
- `add_channel()` は外部データ入力を含むため graph recipe の範囲として扱う。
- v1 では複数入力 operation を recipe から呼ばない。必要な場合は operation-specific method を使う。
- method-aware step は明示 allowlist に含まれる method だけを抽出する。現在は `get_channel`、`remove_channel`、`rename_channels`、`fix_length`、`sum`、`mean`、`channel_difference` に限定する。
- typed method step は明示 allowlist に含まれる method だけを抽出する。現在は `fft`、`stft`、`get_frame_at`、`ifft`、`istft`、`welch`、`noct_spectrum`、`noct_synthesis`、`coherence`、`csd`、`transfer_function`、`roughness_dw_spec` に限定する。
- scalar operation step は numeric scalar operand だけを抽出する。frame operand、array operand、reverse operator は graph recipe の範囲として扱う。

## Testing Strategy / テスト方針

- Optional dependency tests: `sklearn` extra と dependency registry の整合性、未導入時の `ImportError` を確認する。
- Recipe tests: operation 順序、入力 frame 不変性、Dask lazy graph の維持を確認する。
- sklearn adapter tests: `get_params()` / `set_params()`、sklearn `Pipeline.transform(frame)`、`to_spec()` を確認する。
- Notebook smoke: 合成信号で `RecipeSpec` と sklearn `Pipeline` の UX を実行し、operation history と簡単な数値指標を表示する。

## Future Extensions / 次フェーズ候補

- method-aware extraction を、metadata 変換を持つ他の frame method へ広げる。
- terminal metric / report recipe の設計。
- `GraphRecipeSpec` を追加し、binary operation、複数入力 operation、shared branch を表現する。
- `RecipeSpec.from_frame(processed)` を graph recipe まで拡張する。
- `RecipeSpec` の JSON schema と WDF metadata 保存。
- `RecipeSpec` と sklearn `Pipeline` の明示的な相互変換。
- domain transition を含む recipe の型制約。
- notebook から保存済み recipe を読み込み、バッチ処理に適用する workflow。

See also: [Pipeline Recipe Extraction Boundaries](pipeline-recipe-extraction-boundaries.md).
