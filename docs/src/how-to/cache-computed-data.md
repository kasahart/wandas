# Reuse Computed Frame Data

Use `Frame.cache()` when a bounded Frame fits in memory and you will materialize
it or build more operations from it several times. `cache()` synchronously evaluates
the current raw Dask graph once and returns a new Frame of the same concrete type,
backed by the resulting in-memory NumPy array.

メモリに収まるbounded Frameを繰り返し実体化したり、その結果から後続処理を構築したりする場合は、
`Frame.cache()`を使います。`cache()`は現在のraw Dask graphを同期的に一度評価し、得られた
in-memory NumPy arrayを内部に持つ、同じ具象型の新しいFrameを返します。

## Cache a reusable result / 再利用する結果をcacheする

```python
import wandas as wd

audio = wd.read("motor.wav")
spectrogram = audio.stft(n_fft=2048, hop_length=512)

cached = spectrogram.cache()  # computation happens synchronously here
levels = cached.dB             # reuses the computed STFT
magnitude = cached.abs()       # its graph starts from the computed STFT
second_levels = cached.dB      # does not rerun the original STFT graph
```

The source `spectrogram` is unchanged. The returned Frame preserves the exact Frame
type, metadata, channel calibration, axes, source-time offsets, Frame-specific state,
and lineage. Cache creation is an execution detail: it adds no operation-history or
Recipe node, and it does not change WDF or Recipe schemas.

元の`spectrogram`は変更されません。返されるFrameは、具象Frame型、metadata、channel
calibration、axes、source-time offset、Frame固有state、lineageを維持します。cache作成は
実行上の詳細であり、operation history／Recipe nodeを追加せず、WDF／Recipe schemaも変更しません。

## Choose between `.data` and `.cache()` / `.data`との使い分け

`frame.data` computes calibrated values and returns a NumPy array for that access.
It is appropriate when you need values once outside Wandas. `frame.cache()` computes
the raw internal tensor and returns a chainable Frame; calibration remains attached
to the Frame and is still applied exactly once by numerical public APIs.

`frame.data`はcalibration適用後の値を計算し、そのアクセス用のNumPy arrayを返します。
Wandas外で値を一度だけ使う場合に適しています。`frame.cache()`はraw内部tensorを計算して
chain可能なFrameを返します。calibrationはFrameに保持され、数値public APIで一度だけ適用されます。

`cache()` has no arguments, cache-state query, explicit release, capacity limit,
scheduler selection, automatic eviction, or `persist()` alias. The complete raw
tensor must fit in local process memory. Computation errors, including `MemoryError`,
are propagated unchanged and leave the source Frame untouched. Cached samples become
eligible for garbage collection after the returned Frame and all Frames derived from
it are no longer referenced.

`cache()`には引数、cache状態照会、明示的release、容量制限、scheduler指定、自動eviction、
`persist()` aliasはありません。raw tensor全体がlocal process memoryに収まる必要があります。
`MemoryError`を含む計算例外はそのまま送出され、元Frameは変化しません。返されたFrameと
その派生Frameへの参照がなくなると、cached sampleはgarbage collectionの対象になります。

## Representative timing evidence / 代表timing evidence

Issue [#326](https://github.com/kasahart/wandas/issues/326) used a 10-second,
48 kHz mono sine wave and an STFT with `n_fft=2048`, `hop_length=512`, and a Hann
window. Each recorded trial measured three `.dB` accesses. The synchronous scheduler,
a warm-up pass, and a fixed signal make the three phases directly comparable; these
timings are evidence from one environment, not a performance guarantee.

Issue [#326](https://github.com/kasahart/wandas/issues/326)では、10秒・48 kHz・monoの
正弦波と、`n_fft=2048`、`hop_length=512`、Hann windowのSTFTを使用しました。各trialは
`.dB`を3回評価しています。同期scheduler、warm-up、固定signalにより3段階を直接比較できますが、
このtimingは1環境でのevidenceであり、性能保証ではありません。

Environment: Linux 7.0.0-28-generic x86-64, Intel Core i7-13700KF, Python 3.10.20;
base commit `9041e21b4f697b9f32026ce280108131ecba5900`; uncommitted candidate working
tree based on that commit (so no candidate commit SHA exists), with the
`wandas/core/base_frame.py` diff SHA-256
`9784c8242d77f0121d7c926422ea55f19b4a907e7d8d6a5eaa5dde63f18ca1dc`;
`uv.lock` SHA-256
`4ef530af6b76bfa0ad997392615109f16cbe677b374f5ce9c87aacff4f242db4`.
The base revision had no cache API, so no before-cache result exists; repeated
uncached `.dB` on the candidate is the control.

| Run | Three uncached `.dB` accesses (ms) | `cache()` creation (ms) | Three cached `.dB` accesses (ms) |
| --- | --- | --- | --- |
| A trials | 109.34, 90.65, 88.75, 90.15, 82.82 | 36.96, 30.12, 30.43, 30.81, 30.04 | 43.32, 31.82, 23.51, 22.86, 26.25 |
| A median | 90.15 | 30.43 | 26.25 |
| B trials | 85.75, 90.63, 85.15, 91.36, 83.65 | 31.88, 30.88, 29.82, 31.34, 29.65 | 25.58, 23.29, 25.15, 22.60, 24.87 |
| B median | 85.75 | 30.88 | 24.87 |

Reproduce the measurement from the repository root:

```bash
uv run --locked python - <<'PY'
import json
import platform
import time

import dask
import numpy as np
import wandas as wd

sampling_rate = 48_000
samples = 10 * sampling_rate
accesses = 3

def measure(callable_):
    start = time.perf_counter()
    callable_()
    return time.perf_counter() - start

def make_spectrogram():
    t = np.arange(samples, dtype=np.float64) / sampling_rate
    signal = np.sin(2 * np.pi * 1_000 * t)
    return wd.ChannelFrame.from_numpy(signal, sampling_rate=sampling_rate).stft(
        n_fft=2048, hop_length=512, win_length=2048, window="hann"
    )

with dask.config.set(scheduler="synchronous"):
    warmup = make_spectrogram()
    _ = warmup.dB
    _ = warmup.cache().dB
    trials = []
    for trial in range(5):
        spectrogram = make_spectrogram()
        uncached = measure(lambda: [spectrogram.dB for _ in range(accesses)])
        holder = []
        creation = measure(lambda: holder.append(spectrogram.cache()))
        cached = holder[0]
        reused = measure(lambda: [cached.dB for _ in range(accesses)])
        trials.append([trial + 1, uncached, creation, reused])

print(json.dumps({"python": platform.python_version(), "trials": trials}, indent=2))
PY
```
