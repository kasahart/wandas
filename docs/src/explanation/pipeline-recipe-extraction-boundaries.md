# Pipeline Recipe Extraction Boundaries / Recipe 化境界検証

## Goal / 目標

あるべき姿は、frame で計算できる処理をすべて Recipe 化できることである。ただし、現在の `RecipeSpec` は「単一入力・直列・`apply_operation` replay」の最小表現なので、すべての frame 計算を表すには段階的な拡張が必要である。

このページは、試作時点で実際に frame lineage / operation graph を確認した境界をまとめる。

## Current Extraction Entry Point / 現在の抽出入口

```python
processed = (
    frame
    .remove_dc()
    .trim(start=0.1, end=0.5)
    .resampling(target_sr=8000)
    .normalize(norm=2.0)
)

recipe = RecipeSpec.from_frame(processed)
replayed = recipe.apply(frame)
```

`RecipeSpec.from_frame(processed)` は `processed.operation_graph` を読み、1 本の親 chain として表現できる operation だけを `OperationSpec` / `MethodStep` / `TypedMethodStep` / `IndexingStep` / `ScalarOperationStep` に変換する。

## Stage 1: Linear `apply_operation` Replay / 直列 apply_operation 再生

Status: implemented in prototype.

対象:

- `remove_dc`
- `highpass_filter`
- `lowpass_filter`
- `bandpass_filter`
- `normalize`
- `trim`
- `resampling`
- `abs`
- `power`
- `a_weighting`
- `fade`
- `rms_trend`
- `sound_level`
- `hpss_harmonic`
- `hpss_percussive`
- `loudness_zwtv`
- `roughness_dw`
- `sharpness_din`

条件:

- operation graph が 1 本の chain である。
- 各 node が registered Wandas operation である。
- operation が generic `frame.apply_operation(operation, **params)` で同じ意味に replay できる。
- パラメータが Recipe 内で値として保持できる。対応値は `None`、`bool`、`int`、`float`、`str`、およびそれらだけを要素に持つ浅い `list` / `tuple` である。

`rms_trend()` と `sound_level()` は channel metadata から解決した `ref` を operation graph に保存する。Recipe ではこの `ref` を落とさず `OperationSpec` に残す。そうしないと、別 frame に replay した時に `ref=1.0` へ戻り、特に `dB=True` の結果が変わる。

HPSS の `kernel_size` / `margin` は public API が scalar と 2要素 sequence を受け付けるため、浅い sequence literal として replay する。ただし callable window、array-like、nested dict/list は現在の直列 Recipe では保存形式を定義しない。

`loudness_zwtv()`、`roughness_dw()`、`sharpness_din()` は optional `mosqito` backend を使うが、Recipe は依存関係を横取りしない。抽出・再生は既存 `apply_operation` に委譲し、backend が無い場合のエラーも既存 optional dependency 契約に従う。

この段階では、Wandas の既存 frame operation が持つ不変性、metadata/history、Dask laziness に依存する。

## Stage 2: Method-Aware Linear Steps / frame method aware な直列 step

Status: partially implemented for `fix_length`, `sum`, `mean`, `channel_difference`, simple `get_channel` selection, literal dict channel queries, channel-only `__getitem__` selection including `list[int]`, integer `ndarray`, and 1-D boolean mask `ndarray`, slice-only multidimensional `__getitem__`, `remove_channel`, and `rename_channels`.

検証で見えた例:

| Frame calculation | Recipe step | Why method replay is used |
| --- | --- | --- |
| `frame.fix_length(length=8000)` | `MethodStep("fix_length", {"length": 8000})` | operation history は `target_length` を持つため、frame method の public argument に戻す |
| `frame.fix_length(duration=0.25)` | `MethodStep("fix_length", {"length": int(0.25 * sampling_rate)})` | duration intent ではなく、実行時に解決済みの target length を replay して同じ出力長を再現する |
| `frame.sum()` | `MethodStep("sum")` | channel metadata を 2ch から 1ch に再構成する frame method 固有処理を再利用する |
| `frame.mean()` | `MethodStep("mean")` | `sum` と同じく channel metadata 再構成を frame method に委譲する |
| `frame.channel_difference(other_channel=0)` | `MethodStep("channel_difference", {"other_channel": 0})` | channel label / metadata 更新を frame method に委譲する |
| `frame.get_channel(1)` | `MethodStep("get_channel", {"channel_idx": 1})` | channel metadata、channel ids、source time offset の selection を frame method に委譲する |
| `frame.get_channel(np.array([False, True]))` | `MethodStep("get_channel", {"channel_mask": [False, True]})` | mask intent を integer index へ潰さず、既存 `get_channel(mask)` に委譲する |
| `frame.get_channel(query="right")` | `MethodStep("get_channel", {"query": "right", "validate_query_keys": True})` | exact label query は値として保存できる |
| `frame.get_channel(query={"unit": "Pa"})` | `MethodStep("get_channel", {"query": {"unit": "Pa"}, "validate_query_keys": True})` | literal metadata query は query intent を保存できる |
| `frame[0:2]` | `IndexingStep(slice(0, 2))` | slice intent を index list へ潰さず、既存 `frame[key]` に委譲する |
| `frame[[1, 0]]` | `IndexingStep([1, 0])` | explicit integer list intent を保存し、既存 `frame[key]` に委譲する |
| `frame[np.array([1, 0])]` | `IndexingStep([1, 0])` | explicit integer selection として保存し、container 型ではなく選択値を replay する |
| `frame[np.array([False, True])]` | `IndexingStep(np.array([False, True]))` | mask intent を integer index へ潰さず、boolean mask として replay する |
| `frame[["right", "rear"]]` | `IndexingStep(["right", "rear"])` | label list intent を index list へ潰さず、既存 `frame[key]` に委譲する |
| `frame[:, 100:400]` | `IndexingStep((slice(None), slice(100, 400)))` | channel selection と time slice を分離せず、元の tuple indexing を既存 `frame[key]` に委譲する |
| `frame.remove_channel("right")` | `MethodStep("remove_channel", {"key": "right"})` | channel metadata、channel ids、source time offset の removal を frame method に委譲する |
| `frame.rename_channels({0: "left"})` | `MethodStep("rename_channels", {"mapping": {0: "left"}})` | channel metadata の label 更新を frame method に委譲する。保存表現では int key を保つため `mapping_items` を使う |

現在の表現:

```text
MethodStep(method="fix_length", kwargs={"length": 8000})
MethodStep(method="sum", kwargs={})
MethodStep(method="mean", kwargs={})
MethodStep(method="channel_difference", kwargs={"other_channel": 0})
MethodStep(method="get_channel", kwargs={"channel_idx": [0, 2]})
IndexingStep(key=slice(0, 2))
IndexingStep(key=[1, 0])
IndexingStep(key=["right", "rear"])
IndexingStep(key=(slice(None), slice(100, 400)))
MethodStep(method="remove_channel", kwargs={"key": "right"})
MethodStep(method="rename_channels", kwargs={"mapping": {0: "left", 1: "right"}})
```

`rename_channels()` の `to_dict()` 表現では JSON object key の文字列化で int key intent が失われないよう、`{"mapping_items": [[0, "left"], [1, "right"]]}` の pair list として保存する。

この段階では metadata 変換ロジックを Recipe 側に複製しない。既存 frame method を呼ぶことで、frame immutability、metadata/history、Dask laziness は既存契約に従う。

`get_channel()` の callable / regex query と regex 値を含む dict query は、選択時点の index へ潰すと query intent が失われるため現在は抽出時に拒否する。literal dict query は `None`、`bool`、`int`、`float`、`str` だけを値に持つ場合に限って保存する。

`IndexingStep` の保存表現は `{"getitem": {"type": "channel_slice", "start": 0, "stop": 2, "step": null}}`、`{"getitem": {"type": "integer_list", "indices": [1, 0]}}`、`{"getitem": {"type": "boolean_mask", "mask": [false, true]}}`、`{"getitem": {"type": "label_list", "labels": ["right", "rear"]}}`、または `{"getitem": {"type": "multidimensional_slice", "channel": {"type": "slice", "start": null, "stop": null, "step": null}, "axis_slices": [{"start": 100, "stop": 400, "step": null}]}}` とする。Recipe 側では indexing の metadata 処理を複製せず、`frame[key]` を呼ぶ。

`frame["right", 10:20]` のような channel + time の tuple indexing は、channel selector が単一 index、channel slice、integer list、1-D boolean mask、または label list で、残りの軸が `slice` だけの場合に限って抽出する。source time offset の更新は Recipe に複製せず、既存 `frame[key]` に委譲する。Point/fancy time selection は、selection intent と source time offset contract を追加で定義するまで拒否する。

`add_channel()` は新しい signal data または別 frame を Recipe step 内で参照する必要があるため、現在の単一入力 linear Recipe では扱わない。非 `inplace` の `add_channel()` は明示的な lineage を残すが、`RecipeSpec.from_frame()` は部分 Recipe を返さず `RecipeExtractionError` で拒否する。

## Stage 3: Typed Domain Transitions / 型遷移を含む Recipe

Status: partially implemented for `fft`, `stft`, `get_frame_at`, `ifft`, `istft`, `welch`, `noct_spectrum`, `noct_synthesis`, `coherence`, `csd`, `transfer_function`, and `roughness_dw_spec`.

対象例:

Implemented:

- `frame.fft()` -> `SpectralFrame`
- `frame.stft()` -> `SpectrogramFrame`
- `spectrogram_frame.get_frame_at()` -> `SpectralFrame`
- `spectral_frame.ifft()` -> `ChannelFrame`
- `spectrogram_frame.istft()` -> `ChannelFrame`
- `frame.welch()` -> `SpectralFrame`
- `frame.noct_spectrum()` -> `NOctFrame`
- `spectral_frame.noct_synthesis()` -> `NOctFrame`
- `frame.coherence()` -> `SpectralFrame`
- `frame.csd()` -> `SpectralFrame`
- `frame.transfer_function()` -> `SpectralFrame`
- `frame.roughness_dw_spec()` -> `RoughnessFrame`

Not implemented yet:

- terminal metric arrays such as `loudness_zwst()` and `sharpness_din_st()`

これらは operation graph としては線形に見えるが、出力 frame class と constructor metadata が変わる。実装済み範囲では `TypedMethodStep` が既存 frame method を呼び、型遷移と metadata 構築を既存実装に委譲する。

Typed domain transition は public frame method 経由の lineage だけを抽出する。`frame.apply_operation("fft", ...)` のような generic operation call は、同じ operation 名でも戻り frame type と metadata 構築の契約が public method と異なるため、`TypedMethodStep` には変換しない。

現在の表現:

```text
TypedMethodStep(method="fft", kwargs={"n_fft": 16000, "window": "hann"})
TypedMethodStep(method="stft", kwargs={"n_fft": 2048, "hop_length": 512, "win_length": 2048, "window": "hann"})
TypedMethodStep(method="get_frame_at", kwargs={"time_idx": 2})
TypedMethodStep(method="welch", kwargs={"n_fft": 2048, "hop_length": 512, "win_length": 2048, "window": "hann", "average": "mean"})
TypedMethodStep(method="noct_spectrum", kwargs={"fmin": 25, "fmax": 20000, "n": 3, "G": 10, "fr": 1000})
TypedMethodStep(method="ifft", kwargs={})
TypedMethodStep(method="istft", kwargs={})
TypedMethodStep(method="coherence", kwargs={"n_fft": 2048, "hop_length": 512, "win_length": 2048, "window": "hann", "detrend": "constant"})
TypedMethodStep(method="csd", kwargs={"n_fft": 2048, "hop_length": 512, "win_length": 2048, "window": "hann", "detrend": "constant", "scaling": "spectrum", "average": "mean"})
TypedMethodStep(method="transfer_function", kwargs={"n_fft": 2048, "hop_length": 512, "win_length": 2048, "window": "hann", "detrend": "constant", "scaling": "spectrum", "average": "mean"})
TypedMethodStep(method="noct_synthesis", kwargs={"fmin": 25, "fmax": 20000, "n": 3, "G": 10, "fr": 1000})
TypedMethodStep(method="roughness_dw_spec", kwargs={"overlap": 0.5})
```

`ifft()` と `istft()` は、入力 `SpectralFrame` / `SpectrogramFrame` の metadata から必要な逆変換パラメータを取得する既存 method なので、Recipe には追加パラメータを持たせない。

`welch()` は public `ChannelFrame.welch()` で replay できる範囲だけを抽出する。operation graph に `detrend` が残っていても、public method が表現できる default `detrend="constant"` の場合だけ受け付ける。`frame.apply_operation("welch", detrend="linear")` のような graph は、数値結果が変わるため境界外として拒否する。

`noct_spectrum()` と `noct_synthesis()` は optional `mosqito` backend を使うが、Recipe は依存関係を横取りしない。抽出・再生は既存 method に委譲し、backend が無い場合のエラーも既存 optional dependency 契約に従う。

`roughness_dw_spec()` も optional `mosqito` backend を使う。Recipe は `RoughnessFrame` の Bark 軸や metadata 構築を複製せず、既存 frame method に委譲する。

`persist()` は計算結果のキャッシュ/materialization state を変える実行制御であり、信号処理の意味を変える frame calculation ではないため Recipe には含めない。

## Stage 4: Graph / Multi-Input / Binary Calculations

Status: partially implemented for numeric scalar operands, explicit two-input graph recipes, and root binary graph extraction with caller-provided input names.

Implemented:

- `frame + 0.1`
- `frame - 0.1`
- `frame * 2`
- `frame / 2`
- `frame ** 2`
- `GraphRecipeSpec(..., BinaryFrameStep("+", ...))`
- `GraphRecipeSpec(..., BinaryFrameStep("add_with_snr", ..., params={"snr": 6.0}))`
- `GraphRecipeSpec.from_frame(processed, input_names=("signal", "noise"))` for root `+` / `add_with_snr` graphs with two linear parents

現在の表現:

```text
ScalarOperationStep(symbol="+", operand=0.1)
ScalarOperationStep(symbol="*", operand=2)
```

`ScalarOperationStep` は既存 frame operator を呼ぶだけで、二項演算の metadata/history/Dask laziness は frame 本体に委譲する。対応 operand は operation graph に値として保存された Python / NumPy real scalar に限定する。NaN は recipe equality が安定しないため拒否する。

`GraphRecipeSpec` は名前付き入力ごとに linear `RecipeSpec` を適用し、最後に `BinaryFrameStep` で既存 frame-frame 演算を呼ぶ。`from_frame(..., input_names=...)` は root binary graph だけを対象にし、入力名は呼び出し側が与える。

Not implemented yet:

- 入力名を推定する `RecipeSpec.from_frame(frame_a + frame_b)` の自動 graph 抽出
- `frame + np.ones(frame.shape)`
- `frame + dask_array`
- 入力名を推定する `RecipeSpec.from_frame(signal.add(noise, snr=6.0))` の自動 graph 抽出
- shared branch を持つ graph: `base.normalize()` から signal/noise branch を作って合成する処理

これらは直列 Recipe では表現できない。特に `operation_history` だけを見ると `normalize -> lowpass_filter -> add_with_snr` のように直列に見えることがあるが、`operation_graph` では複数 parent や外部 operand が必要である。array operand も shape、chunking、保存形式を Recipe 側で決める必要があるため、scalar operand と同じ扱いにはしない。

必要な拡張:

```text
GraphRecipeSpec(
    inputs=["signal", "noise"],
    nodes=[
        Node(id="signal_norm", input="signal", operation="normalize"),
        Node(id="noise_lp", input="noise", operation="lowpass_filter", params={"cutoff": 1000}),
        Node(id="mix", operation="add_with_snr", inputs=["signal_norm", "noise_lp"], params={"snr": 6.0}),
    ],
    output="mix",
)
```

この段階では、外部入力の名前付け、shape/sampling_rate/channel metadata の整合性、operand serialization policy が必要になる。

## Stage 5: Custom Callable Calculations

Status: not implemented; required for full target only with constraints.

対象例:

```python
frame.apply(lambda data, gain: data * gain, output_shape_func=lambda shape: shape, gain=2.0)
```

現在の history には `operation="custom"` と parameter は残るが、callable 本体と `output_shape_func` は replayable な形では残らない。任意 lambda の完全保存は Python 実行環境に依存するため、無条件に Recipe 化するべきではない。

必要な拡張:

- 登録済み callable 名を使う。
- import path で解決できる関数だけを許可する。
- lambda や closure は strict mode で拒否する。

## Stage 6: Terminal / Display Operations

Status: partially implemented for explicit terminal metrics.

Implemented:

- `TerminalStep("rms")`
- `TerminalStep("crest_factor")`
- `TerminalStep("loudness_zwst", {"field_type": "free"})`
- `TerminalStep("sharpness_din_st", {"weighting": "din", "field_type": "free"})`

これらは `RecipeSpec.from_frame(...)` で自動抽出しない。terminal metric は NumPy 配列を返し、結果そのものは frame lineage を保持しないためである。明示的に `RecipeSpec([OperationSpec("remove_dc"), TerminalStep("rms")])` のように組む場合だけ、変換 chain の末尾で既存 property / method を呼ぶ。

未対応の terminal metric / report 自動抽出には、frame lineage を持たない結果からどの frame と terminal call を結び付けるかを決める UX が必要である。

`compute()`, `plot()`, `describe()`, `info()` は frame を変換する Recipe step ではなく、評価・表示・レポート生成に近い。全体の UX としては、変換 Recipe と report Recipe を分けるのが自然である。

## Interim Rule / 当面のルール

`RecipeSpec.from_frame(frame)` は、現在の `RecipeSpec` で安全に replay できるものだけを返す。安全に replay できない operation は黙って落とさず、`RecipeExtractionError` で次に必要な Recipe 表現を示す。

これは最終スコープを狭めるためではなく、すべての frame 計算を Recipe 化するために必要な表現力を段階的に検証するための制約である。
