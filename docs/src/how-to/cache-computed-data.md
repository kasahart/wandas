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

The [`BaseFrame.cache()` API Reference](../api/core.md#wandas.core.base_frame.BaseFrame.cache)
is the authoritative source for memory, ownership, lineage, and exception behavior.
Representative timing evidence is archived with the
[Issue #326 benchmark assets](../assets/benchmarks/issue-326/README.md).

memory、ownership、lineage、例外の契約は
[`BaseFrame.cache()` API Reference](../api/core.md#wandas.core.base_frame.BaseFrame.cache)を正本とします。
代表timing evidenceは
[Issue #326 benchmark assets](../assets/benchmarks/issue-326/README.md)に保存しています。
