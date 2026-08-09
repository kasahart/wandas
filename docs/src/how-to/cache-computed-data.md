# Reuse Computed Frame Data

Use `Frame.cache()` when a bounded Frame fits in memory and you will materialize
or process its result repeatedly. Evaluation happens synchronously at `cache()`.

メモリに収まるbounded Frameを繰り返し実体化または処理する場合は、`Frame.cache()`を
使用します。評価は`cache()`の呼び出し時に同期的に行われます。

```python
import wandas as wd

audio = wd.read("motor.wav")
spectrogram = audio.stft(n_fft=2048, hop_length=512)

cached = spectrogram.cache()
levels = cached.dB         # reuses the computed STFT
magnitude = cached.abs()   # subsequent operations reuse it too
```

Unlike `.data`, which returns calibrated NumPy values for one access, `cache()`
returns a chainable Frame backed by computed raw data. The original Frame is unchanged.

`.data`は1回のアクセス用にcalibration適用済みNumPy値を返しますが、`cache()`は計算済みraw
dataを持つchain可能なFrameを返します。元のFrameは変更されません。

## Reduce the resident cache dtype explicitly / cache常駐dtypeを明示的に縮小する

Place `astype()` before `cache()` when the precision tradeoff is acceptable and the
resident raw tensor should use less memory:

精度とのtrade-offを許容でき、常駐するraw tensorのメモリを削減したい場合は、`cache()`の前に
`astype()`を置きます。

```python
audio32 = audio.astype("float32").cache()
spectrogram64 = spectrogram.astype("complex64").cache()
```

For the same shape, a cached `float32` raw tensor uses half the bytes of `float64`;
`complex64` likewise uses half the bytes of `complex128`. Reduced precision changes
numerical meaning, so `astype()` is lazy, immutable, and recorded as one
`wandas.frame.astype` lineage and Recipe node. `cache()` remains a no-argument
execution boundary and adds no lineage or Recipe node.

同じshapeなら、cacheされた`float32` raw tensorは`float64`の半分、`complex64`は
`complex128`の半分のbyte数になります。精度縮小は数値的意味を変えるため、`astype()`は
lazyかつimmutableな`wandas.frame.astype` lineage／Recipe nodeとして1件記録されます。
`cache()`は引数なしの実行境界のままで、lineage／Recipe nodeを追加しません。

This is a resident-cache guarantee, not an all-stage peak-memory guarantee. An
upstream operation may still create a temporary `float64` or `complex128` result
during materialization before `astype()` produces the smaller resident tensor.
Frames also retain their immediate receiver through `previous`. If the source is
already backed by a resident NumPy array or another cache, that wider source remains
reachable from the compact result; only the new cached raw tensor is guaranteed to
shrink. Total process-memory savings therefore depend on the upstream graph and
retained Frame chain.
Channel calibration is preserved separately, so applying a non-unit factor through
`.data` or another numerical API can also promote a `float32` raw tensor to
`float64`. Check accuracy on representative signals before reducing precision.

これはcache作成後の常駐量に対する保証であり、全計算段階のpeak memory保証ではありません。
上流Operationが`float64`／`complex128`を生成する場合、materialization中には`astype()`が
小さい常駐tensorを作る前の一時配列が残ります。Frameは`previous`経由で直前の
receiverも保持するため、入力が常駐NumPy配列や別のcacheである場合、その広いdtypeのsourceも
compact結果からreachableなままです。保証されるは新しいcached raw tensorの縮小であり、
プロセス全体の削減量は上流graphと保持されるFrame chainに依存します。またchannel calibrationは別に保持されるため、
1以外のfactorを`.data`や数値APIで適用すると、`float32` raw tensorが`float64`へ昇格する
場合があります。精度を縮小する前に代表signalでaccuracyを確認してください。

The [`BaseFrame.cache()` API Reference](../api/core.md#wandas.core.base_frame.BaseFrame.cache)
is the authoritative source for memory, ownership, lineage, and exception behavior.
The implementation and representative review evidence are recorded in
[PR #439](https://github.com/kasahart/wandas/pull/439) for `cache()` and
[PR #441](https://github.com/kasahart/wandas/pull/441) for dtype reduction.

memory、ownership、lineage、例外の契約は
[`BaseFrame.cache()` API Reference](../api/core.md#wandas.core.base_frame.BaseFrame.cache)を正本とします。
`cache()`の実装と代表review evidenceは
[PR #439](https://github.com/kasahart/wandas/pull/439)、dtype縮小については
[PR #441](https://github.com/kasahart/wandas/pull/441)に記録しています。
