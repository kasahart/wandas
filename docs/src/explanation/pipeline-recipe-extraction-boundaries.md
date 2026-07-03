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

`RecipeSpec.from_frame(processed)` は `processed.operation_graph` を読み、1 本の親 chain として表現できる operation だけを `OperationSpec` / `MethodStep` / `TypedMethodStep` / `ScalarOperationStep` に変換する。

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

条件:

- operation graph が 1 本の chain である。
- 各 node が registered Wandas operation である。
- operation が generic `frame.apply_operation(operation, **params)` で同じ意味に replay できる。
- パラメータが Recipe 内で値として保持できる。

この段階では、Wandas の既存 frame operation が持つ不変性、metadata/history、Dask laziness に依存する。

## Stage 2: Method-Aware Linear Steps / frame method aware な直列 step

Status: partially implemented for `fix_length`, `sum`, and `mean`.

検証で見えた例:

| Frame calculation | Recipe step | Why method replay is used |
| --- | --- | --- |
| `frame.fix_length(length=8000)` | `MethodStep("fix_length", {"length": 8000})` | operation history は `target_length` を持つため、frame method の public argument に戻す |
| `frame.sum()` | `MethodStep("sum")` | channel metadata を 2ch から 1ch に再構成する frame method 固有処理を再利用する |
| `frame.mean()` | `MethodStep("mean")` | `sum` と同じく channel metadata 再構成を frame method に委譲する |

現在の表現:

```text
FrameMethodStep(method="fix_length", kwargs={"length": 8000})
FrameMethodStep(method="sum", kwargs={})
FrameMethodStep(method="mean", kwargs={})
```

この段階では metadata 変換ロジックを Recipe 側に複製しない。既存 frame method を呼ぶことで、frame immutability、metadata/history、Dask laziness は既存契約に従う。

## Stage 3: Typed Domain Transitions / 型遷移を含む Recipe

Status: partially implemented for `fft`, `stft`, `ifft`, and `istft`.

対象例:

Implemented:

- `frame.fft()` -> `SpectralFrame`
- `frame.stft()` -> `SpectrogramFrame`
- `spectral_frame.ifft()` -> `ChannelFrame`
- `spectrogram_frame.istft()` -> `ChannelFrame`

Not implemented yet:

- `frame.welch()` -> `SpectralFrame`
- `frame.noct_spectrum()` -> `NOctFrame`
- cross-channel transforms such as `coherence()`, `csd()`, `transfer_function()`

これらは operation graph としては線形に見えるが、出力 frame class と constructor metadata が変わる。実装済み範囲では `TypedMethodStep` が既存 frame method を呼び、型遷移と metadata 構築を既存実装に委譲する。

現在の表現:

```text
TypedMethodStep(method="fft", kwargs={"n_fft": 16000, "window": "hann"})
TypedMethodStep(method="stft", kwargs={"n_fft": 2048, "hop_length": 512, "win_length": 2048, "window": "hann"})
TypedMethodStep(method="ifft", kwargs={})
TypedMethodStep(method="istft", kwargs={})
```

`ifft()` と `istft()` は、入力 `SpectralFrame` / `SpectrogramFrame` の metadata から必要な逆変換パラメータを取得する既存 method なので、Recipe には追加パラメータを持たせない。

## Stage 4: Graph / Multi-Input / Binary Calculations

Status: partially implemented for numeric scalar operands; full graph recipe is still required.

Implemented:

- `frame + 0.1`
- `frame - 0.1`
- `frame * 2`
- `frame / 2`
- `frame ** 2`

現在の表現:

```text
ScalarOperationStep(symbol="+", operand=0.1)
ScalarOperationStep(symbol="*", operand=2)
```

`ScalarOperationStep` は既存 frame operator を呼ぶだけで、二項演算の metadata/history/Dask laziness は frame 本体に委譲する。対応 operand は operation graph に値として保存された Python / NumPy real scalar に限定する。NaN は recipe equality が安定しないため拒否する。

Not implemented yet:

- `frame_a + frame_b`
- `frame + np.ones(frame.shape)`
- `frame + dask_array`
- `signal.add(noise, snr=6.0)`
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

Status: separate concern.

`compute()`, `plot()`, `describe()`, `info()` は frame を変換する Recipe step ではなく、評価・表示・レポート生成に近い。全体の UX としては、変換 Recipe と report Recipe を分けるのが自然である。

## Interim Rule / 当面のルール

`RecipeSpec.from_frame(frame)` は、現在の `RecipeSpec` で安全に replay できるものだけを返す。安全に replay できない operation は黙って落とさず、`RecipeExtractionError` で次に必要な Recipe 表現を示す。

これは最終スコープを狭めるためではなく、すべての frame 計算を Recipe 化するために必要な表現力を段階的に検証するための制約である。
