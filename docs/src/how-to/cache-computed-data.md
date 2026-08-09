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

Environment: Linux 7.0.0-28-generic x86-64, Intel Core i7-13700KF, Python
3.10.20, NumPy 2.2.6, and Dask 2025.11.0. The clean worktrees used base commit
`7f52b534a09559d35b459dfebe05db3dc2e429ac` and candidate commit
`926f1ed961a64c69132995a66374a094c82bb6a4`. Both used `uv.lock` SHA-256
`4ef530af6b76bfa0ad997392615109f16cbe677b374f5ce9c87aacff4f242db4`.
The candidate contains every runtime and test change on the ordinary
`np.ndarray` path measured here. Later commit `289db826` adds a separate
`np.ma.MaskedArray` ownership/copy branch and its behavioral test after this timing
was recorded. The benchmark constructs an ordinary NumPy array, so it does not
exercise that branch and provides no timing evidence for masked-array caching; the
ordinary-array branch measured by `926f1ed9` is unchanged in the final candidate.

The exact two-run raw results are preserved as
[base JSON](../assets/benchmarks/issue-326/base-7f52b534.json) and
[candidate JSON](../assets/benchmarks/issue-326/candidate-926f1ed9.json).
The base has no `cache()` API, so its cache fields are intentionally absent and
non-comparable; its repeated uncached `.dB` timing is the before control.

| Revision and run | Three uncached `.dB` accesses, median (ms) | `cache()` creation, median (ms) | Three cached `.dB` accesses, median (ms) |
| --- | --- | --- | --- |
| Base run 1 | 93.03 | N/A | N/A |
| Base run 2 | 91.99 | N/A | N/A |
| Candidate run 1 | 88.63 | 35.28 | 29.27 |
| Candidate run 2 | 95.97 | 37.02 | 32.22 |

The unchanged uncached control stayed within ordinary timing variation. On this
bounded recording, three cached `.dB` accesses took less time than three uncached
accesses in both candidate runs, even though every cached materialization returns an
isolated array to preserve Frame immutability. This is representative evidence, not
a repository-wide performance budget.

Reproduce the measurement from the repository root with the committed one-off
[bridge script](../assets/benchmarks/issue-326/benchmark.py), whose SHA-256 is
`807e853d70618c805137570795bdc5a35cb1d5aad27e88002ed60340d92393d9`:

```bash
repo_root=$PWD
benchmark_script=$repo_root/docs/src/assets/benchmarks/issue-326/benchmark.py
base_output=$repo_root/docs/src/assets/benchmarks/issue-326/base-7f52b534.json
candidate_output=$repo_root/docs/src/assets/benchmarks/issue-326/candidate-926f1ed9.json

git worktree add --detach /tmp/wandas-issue326-base 7f52b534a09559d35b459dfebe05db3dc2e429ac
cd /tmp/wandas-issue326-base
uv run --locked python "$benchmark_script" > "$base_output"

cd "$repo_root"
git worktree add --detach /tmp/wandas-issue326-candidate 926f1ed961a64c69132995a66374a094c82bb6a4
cd /tmp/wandas-issue326-candidate
uv run --locked python "$benchmark_script" > "$candidate_output"
```
