# Pipeline Recipe Design / パイプラインレシピ設計書

## Design Summary / 設計概要

Wandas 側に小さな replayable spec を置き、sklearn は optional adapter として扱う。中心は `OperationSpec` と `RecipeSpec` であり、実行は既存の `Frame.apply_operation(...)`、frame method、frame operator に委譲する。これにより、frame immutability、metadata/history、Dask lazy execution は既存 frame operation の契約に従う。

Recipe の主要UXは二つある。ひとつは `RecipeSpec([...]).apply(frame)` のように明示的に組む入口であり、もうひとつは探索的に frame method chain を実行した後で `RecipeSpec.from_frame(processed)` / `GraphRecipeSpec.from_frame(processed)` / `NodeGraphRecipeSpec.from_frame(processed)` により `operation_graph` から抽出する入口である。長期的なあるべき姿は frame で計算できるものをすべて Recipe 化できることだが、現在は replay contract が定義済みの step だけを抽出し、未定義の境界は `RecipeExtractionError` として明示する。

sklearn 互換の `WandasOperationTransformer` は、`BaseEstimator` / `TransformerMixin` と同じ呼び出し形を提供するだけの薄いラッパーである。信号処理の正本は Wandas operation registry key であり、sklearn `Pipeline` は UI/連携の選択肢に留める。

## Components / コンポーネント

### `wandas.pipeline.OperationSpec`

- `operation: str`
- `params: Mapping[str, Any]`

単一 Wandas operation の replayable な呼び出しを表す。`operation` は `"normalize"`、`"highpass_filter"`、`"fade"` のような registry key を使う。`params` は作成時に snapshot されるため、呼び出し元の dict を後から変更しても spec は変わらない。

Stage 1 では、`params` は recipe literal に限定する。対応する値は `None`、`bool`、`int`、`float`、`str`、およびそれらだけを要素に持つ浅い `list` / `tuple` である。sequence は作成時に snapshot され、`to_dict()` では JSON 的に扱いやすい `list` として返る。nested sequence、dict、callable、array-like object、NaN は、保存形式と equality policy を決めるまで受け付けない。

`rename_channels()` は例外的に `MethodStep` 側で dedicated mapping snapshot を持つ。Python dict をそのまま JSON object にすると int key が文字列化されるため、`to_dict()` では `{"mapping_items": [[0, "left"], [1, "right"]]}` の pair list として保存する。これは汎用 mapping params のサポートではなく、既存 frame method の replay に必要な最小の typed 表現である。

### `wandas.pipeline.RecipeSpec`

- `steps: tuple[OperationSpec | MethodStep | TypedMethodStep | CustomFunctionStep | IndexingStep | ScalarOperationStep | TerminalStep, ...]`
- `apply(frame) -> Any`
- `from_frame(frame) -> RecipeSpec`

`steps` を順番に処理し、`OperationSpec` は `frame.apply_operation(step.operation, **step.params)`、`MethodStep` / `TypedMethodStep` は対応する frame method、`CustomFunctionStep` は import path から関数を読み込んで `frame.apply(...)`、`IndexingStep` は既存 `frame[key]`、`ScalarOperationStep` は既存の frame 演算子、`TerminalStep` は既存の terminal property を呼ぶ。`RecipeSpec` 自体は frame を保持しないため、同じレシピを複数 frame に再適用できる。

`from_frame(frame)` は、探索的に処理した frame の `operation_graph` から安全に replay できる直列 chain だけを抽出する。対応外の計算は黙って落とさず、`RecipeExtractionError` で現在の境界を返す。`operation_history` は人間が処理順を読むための記録であり、抽出の正本は親関係や source leaf を保持する `operation_graph` である。

### `wandas.pipeline.TerminalStep`

- `metric: str`

`rms`、`crest_factor`、`magnitude`、`phase`、`dB`、`time` のような terminal property、および `loudness_zwst()`、`sharpness_din_st()` のように frame ではなく NumPy 配列を返す terminal method を replay する。`TerminalStep` は Recipe の末尾で使うことを想定し、既存 frame property / method を呼ぶだけで評価ロジックを複製しない。Stage 1 では明示 allowlist に含まれる terminal metric / property に限定し、`RecipeSpec.from_frame(...)` による自動抽出は行わない。terminal 結果は frame lineage を保持しないため、探索後の自動抽出には別の report/metric recipe UX が必要である。

### `wandas.pipeline.GraphRecipeSpec`

- `input_recipes: Mapping[str, RecipeSpec]`
- `output: BinaryFrameStep`
- `tail_recipe: RecipeSpec | None = None`

複数入力 frame 計算の最小表現である。名前付き入力ごとに既存の直列 `RecipeSpec` を適用し、`BinaryFrameStep` で frame-frame `+` / `-` / `*` / `/` / `**` または `add_with_snr` を実行し、必要なら merge 後にもう一度直列 `RecipeSpec` を適用する。演算は `left + right` のような既存 operator または `left.add(right, snr=...)` に委譲し、shape、sampling rate、channel metadata、Dask laziness の契約は既存 frame 実装に従う。

`GraphRecipeSpec` は明示的に組む API である。`GraphRecipeSpec.from_frame(processed, input_names=(...))` は frame graph が 2 本の linear chain、1 つの binary merge、0 個以上の linear tail として表現できる場合だけ抽出する。入力名は lineage から推定せず、呼び出し側が渡す。

任意 DAG の自動抽出には、外部入力名の決定、未加工入力の表現、shared branch の扱い、array/dask operand の保存ポリシーが必要になるため、Stage 1 では扱わない。

### `wandas.pipeline.NodeGraphRecipeSpec`

- `inputs: tuple[str, ...]`
- `nodes: tuple[GraphNodeSpec, ...]`
- `output: str`

複数 merge を含む replayable tree graph の表現である。`GraphRecipeSpec` は「2 入力・1 merge・任意の linear tail」の読みやすい最小 API として残し、それを超える graph は `NodeGraphRecipeSpec` が扱う。

各 `GraphNodeSpec` は既存の `OperationSpec` / `MethodStep` / `TypedMethodStep` / `IndexingStep` / `ScalarOperationStep` / `BinaryFrameStep` / `BinaryOperandStep` をそのまま持つ。replay は外部入力と node 結果を dict に入れて topological order で進めるだけで、演算や metadata 更新は既存 frame method/operator に委譲する。

`CustomFunctionStep` も unary node として扱える。実行時は import path からユーザー関数を import して `frame.apply(...)` に渡すため、trusted user code として扱う。Recipe ファイルや notebook から外部由来の custom function path を実行する場合は、通常の Python import と同じ権限でコードが動く。

`add_channel(ChannelFrame, ...)` は `AddChannelStep` として扱う。これは二つの named frame input を受け取り、既存 `base.add_channel(added, align=..., label=..., suffix_on_dup=...)` に委譲するだけの graph 専用 step である。

`add_channel(ndarray|dask.array, ...)` は `AddChannelDataStep` として扱う。Recipe は raw data を保存せず、実行時に named external data input として受け取る。`source_time_offset` は raw data の時間意味を決める public option なので lineage params に保存し、replay 時に `base.add_channel(data, align=..., label=..., suffix_on_dup=..., source_time_offset=...)` へ渡す。

`frame + ndarray` や `frame * dask_array` は `BinaryOperandStep` として扱う。Recipe は array 値を保存せず、実行時に named external operand として受け取る。`operation_graph` の descriptor は operand が `ndarray` / `dask.array` だったことを判定するためだけに使い、shape/dtype/chunks から値を再構成しない。

現在の `operation_graph` は shared object identity を保持しない tree 表現である。そのため `NodeGraphRecipeSpec.from_frame(...)` は shared branch を同じ構造の duplicated parent path として抽出する。必要なら同じ外部入力名を複数 source leaf に渡して同じ入力 frame から再生できるが、真の DAG node identity の保存は lineage 側の拡張が必要なので別段階とする。

### `wandas.pipeline.MethodStep`

- `method: str`
- `params: Mapping[str, Any]`

`sum()`、`mean()`、`fix_length()`、`channel_difference()`、`get_channel()`、`remove_channel()`、`rename_channels()` のように、metadata 変換を frame method 側に持つ処理を replay する。Recipe 側では metadata 再構成を複製せず、既存 method を呼ぶ。

### `wandas.pipeline.TypedMethodStep`

- `method: str`
- `params: Mapping[str, Any]`

`fft()`、`stft()`、`get_frame_at()`、`ifft()`、`istft()`、`welch()`、`noct_spectrum()`、`noct_synthesis()`、`coherence()`、`csd()`、`transfer_function()`、`roughness_dw_spec()` のように、出力 frame class が変わる method を replay する。戻り型は入力 frame と同じとは限らないため、`RecipeSpec.apply(frame)` の戻り値は実行した step chain に従う。

### `wandas.pipeline.CustomFunctionStep`

- `function: str`
- `output_shape_function: str | None`
- `params: Mapping[str, Any]`
- `dask_pure: bool`

`frame.apply(custom_func, output_shape_func=..., **params)` のうち、importable custom function を replay する。対象は module-level function に限定し、lambda、closure、nested function、callable object、bound method、`functools.partial`、`__main__` 上の関数は抽出しない。`output_shape_func` も指定する場合は同じく importable module-level function に限定する。

`output_frame_class` を使う custom domain transition は、frame class が importable な `BaseFrame` subclass で、`output_frame_kwargs` が recipe literal だけを含む場合に限って replay する。実行時は Recipe 側で constructor logic を複製せず、import した class と kwargs を既存 `frame.apply(...)` に渡す。

`operation_history` には従来どおり `{"operation": "custom", "params": ...}` だけを残し、callable identity は `operation_graph` の custom metadata と `CustomFunctionStep` にだけ保持する。Recipe 適用時は import path の Python コードを実行するため、信頼できる環境・信頼できる recipe だけを対象にする。

### `wandas.pipeline.ScalarOperationStep`

- `symbol: str`
- `operand: int | float`

`frame + 0.1`、`frame * 2`、`2 - frame` のような単一 numeric scalar operand を持つ frame 演算子を replay する。対応演算子は `+`、`-`、`*`、`/`、`**` に限定する。scalar が左辺にある場合は `ScalarOperationStep(reverse=True)` として保存する。

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

Full Recipe/Pipeline conversion, joblib/skops export, and Dask-ML integration are outside the current contract; see [Pipeline Recipe Interoperability And Export Boundaries](pipeline-recipe-interoperability-export.md).

### Named Transformers / 個別 transformer

| Class | Wandas operation | Parameters |
| --- | --- | --- |
| `HighPassFilter` | `highpass_filter` | `cutoff`, `order=4` |
| `LowPassFilter` | `lowpass_filter` | `cutoff`, `order=4` |
| `BandPassFilter` | `bandpass_filter` | `low_cutoff`, `high_cutoff`, `order=4` |
| `Normalize` | `normalize` | `norm=np.inf`, `axis=-1`, `threshold=None`, `fill=None` |
| `RemoveDC` | `remove_dc` | none |

## Data Flow / データフロー

探索的解析から Recipe を作る場合の data flow は次の通りである。

```text
frame method chain
  -> new frame with operation_history and operation_graph
  -> RecipeSpec.from_frame(processed)
    -> validate graph shape and step support
    -> OperationSpec / MethodStep / TypedMethodStep / IndexingStep / ScalarOperationStep / CustomFunctionStep
  -> recipe.apply(other_frame)
    -> existing frame API calls
  -> replayed frame with new operation_history
```

複数入力の場合は、`GraphRecipeSpec.from_frame(...)` または `NodeGraphRecipeSpec.from_frame(...)` が source leaf を名前付き入力へ対応づける。

```text
frame_a, frame_b, external data or external operand
  -> frame graph calculation
  -> result frame with operation_graph tree
  -> GraphRecipeSpec.from_frame(...) or NodeGraphRecipeSpec.from_frame(...)
    -> named external inputs + graph nodes
  -> graph_recipe.apply({...})
    -> existing frame API calls in graph order
```

明示的に Recipe を組む場合の replay flow は次の通りである。

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

- Wandas ユーザーには、探索的解析では通常の frame method chain を使わせ、その結果から `RecipeSpec.from_frame(processed)` を抽出する入口を見せる。これは「まず計算し、後で再利用可能なRecipeへ落とす」UXである。
- 明示的な定義が必要な場面では `RecipeSpec([...]).apply(frame)` を見せる。これは「この順序で同じ処理を再実行する」意図が明確である。
- 複数入力の再利用には `GraphRecipeSpec.from_frame(...)` または `NodeGraphRecipeSpec.from_frame(...)` を使い、外部入力名は呼び出し側が明示する。Python変数名やframe labelから入力名を推定しない。
- sklearn ユーザーには `Pipeline([("hp", HighPassFilter(...)), ...]).transform(frame)` を見せる。これは既存の ML 前処理文脈に馴染む。
- operation history を最終確認画面として使う。ユーザーは結果 frame を見れば、処理順とパラメータを確認できる。
- `to_spec()` により、sklearn transformer から Wandas-native spec へ戻れる。
- `RecipeExtractionError` はUX上の失敗ではなく、まだRecipe化できない境界を説明するためのメッセージとして扱う。

## Error Handling / エラー処理

- sklearn 未導入で `wandas.pipeline.sklearn` を import した場合は、optional dependency registry を通じて `pip install "wandas[sklearn]"` を含む `ImportError` を出す。
- operation 名やパラメータが不正な場合は、既存の `apply_operation` / operation class の validation に委譲する。
- sequence params は浅い literal sequence だけを受け付ける。`rms_trend()` / `sound_level()` の `ref` や HPSS の `kernel_size` / `margin` は replay に必要な値として保存するが、nested list や array operand は graph recipe の範囲として拒否する。
- `loudness_zwtv()`、`roughness_dw()`、`sharpness_din()` のような frame を返す psychoacoustic operation は `OperationSpec` で replay する。`loudness_zwst()`、`sharpness_din_st()` や `magnitude` のように配列を返す terminal metric / property は、明示 `TerminalStep` としてだけ扱う。
- `get_channel()` は explicit index、integer sequence/ndarray、1-D boolean mask ndarray、exact label query、literal dict query だけを抽出する。callable、regex、regex 値を含む dict query は query intent を保存する schema が必要なため拒否する。
- `__getitem__` は channel slice、integer list/ndarray、1-D boolean mask ndarray、label list、slice-only tuple indexing だけを `IndexingStep` として抽出する。point/fancy time selection は indexing intent と source time offset contract の追加 schema が必要なため現在は拒否する。
- `add_channel()` は外部 data または別 frame 入力が必要なため、現行の単一入力 `RecipeSpec` では replay しない。非 `inplace` 実行では lineage を残し、抽出時に明示的に拒否して部分 Recipe を返さない。
- `add_channel(ChannelFrame, ...)` は graph recipe の範囲として扱い、`NodeGraphRecipeSpec` が `AddChannelStep` に抽出する。
- `add_channel(ndarray|dask.array, ...)` は graph recipe の範囲として扱い、`NodeGraphRecipeSpec` が `AddChannelDataStep` に抽出する。raw data 自体は保存せず、replay 時の named external input とする。
- `frame.apply(...)` の custom callable は、function と `output_shape_func` が importable module-level function として検証できる場合だけ `CustomFunctionStep` に抽出する。custom domain transition は importable `BaseFrame` subclass と literal `output_frame_kwargs` に限定する。lambda、closure、nested function、callable object、bound method、`functools.partial`、`__main__` 上の関数は拒否する。
- `RecipeSpec.from_frame(...)` は単一入力の直列 chain だけを返す。複数入力 graph は `GraphRecipeSpec.from_frame(...)` または `NodeGraphRecipeSpec.from_frame(...)` で扱い、`RecipeSpec.from_frame(...)` から暗黙に graph recipe へ昇格しない。
- method-aware step は明示 allowlist に含まれる method だけを抽出する。現在は `get_channel`、`remove_channel`、`rename_channels`、`fix_length`、`sum`、`mean`、`channel_difference` に限定する。
- typed method step は明示 allowlist に含まれる method だけを抽出する。現在は `fft`、`stft`、`get_frame_at`、`ifft`、`istft`、`welch`、`noct_spectrum`、`noct_synthesis`、`coherence`、`csd`、`transfer_function`、`roughness_dw_spec` に限定する。
- scalar operation step は numeric scalar operand だけを抽出する。frame operand と array/dask operand は graph recipe の範囲として扱う。array が左辺にある reverse operator は、array library 側の dispatch と外部入力表現を追加で決める必要があるため現在は対象外である。

## Testing Strategy / テスト方針

- Optional dependency tests: `sklearn` extra と dependency registry の整合性、未導入時の `ImportError` を確認する。
- Recipe tests: operation 順序、入力 frame 不変性、Dask lazy graph の維持を確認する。
- Extraction tests: `RecipeSpec.from_frame(...)` が対応済み linear chain を抽出し、対応外 boundary を `RecipeExtractionError` として返すことを確認する。
- Graph extraction tests: `GraphRecipeSpec.from_frame(...)` / `NodeGraphRecipeSpec.from_frame(...)` が対応済み複数入力 graph を named input から replay できることを確認する。
- sklearn adapter tests: `get_params()` / `set_params()`、sklearn `Pipeline.transform(frame)`、`to_spec()` を確認する。
- Notebook smoke: 合成信号で明示 Recipe、`from_frame` 抽出、graph recipe、sklearn `Pipeline` の UX を実行し、operation history と簡単な数値指標を表示する。

## Future Extensions / 次フェーズ候補

- method-aware extraction を、metadata 変換を持つ他の frame method へ広げる。
- terminal metric / report recipe の設計。
- true DAG identity を保持する graph lineage。
- `RecipeSpec` の JSON schema と WDF metadata 保存。
- `RecipeSpec` と sklearn `Pipeline` の明示的な相互変換は現在非対応であり、必要な use case が固まってから再設計する。
- import path 以外で custom callable / custom frame class を参照する registry UX。
- notebook から保存済み recipe を読み込み、バッチ処理に適用する workflow。

See also: [Pipeline Recipe Extraction Boundaries](pipeline-recipe-extraction-boundaries.md).
