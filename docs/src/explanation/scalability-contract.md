# Scalability contract / スケーラビリティ契約

Wandas scales primarily across collections of bounded recordings while preserving
the continuous-time assumptions of signal processing. Stored and lazy Frame data
retain a channel axis. Operations that explicitly declare the channel-independent
contract can execute one complete channel per lazy kernel task; other delayed
`AudioOperation` transforms keep the conservative whole-Frame boundary. Wandas
therefore does not promise arbitrary channel-count or time-axis distribution for one
enormous Frame.

Wandas は主に、サイズを制御した多数の収録ファイルを扱う方向へ拡張します。
Frame の保存・遅延データはチャンネル軸を保持します。channel-independent 契約を
明示した operation は、完全な 1 チャンネルごとに遅延 kernel task を実行できます。
その他の遅延 `AudioOperation` transform は、保守的な whole-Frame boundary を維持します。
したがって Wandas は、単一の巨大な Frame をチャンネル数または時間方向へ自由に分散できるとは約束しません。

## What scales well / 得意な処理

- Discover many files as a lazy `ChannelFrameDataset`.
- Select files from path/CSV metadata before reading waveform samples.
- Load only selected files and keep each loaded multi-channel Frame bounded.
- Build and apply a `RecipePlan` without computing Frame samples.
- Preserve the channel axis in stored and lazy Frame data while each continuous
  signal axis normally remains one chunk.

## Current limits / 現在の制約

- Filters, FFT, STFT, and other continuity-sensitive operations normally require a
  single time chunk per channel.
- Most delayed `AudioOperation` transforms wrap the complete channel-first Dask array
  in one call. Explicitly adopted operations such as `RemoveDC`, Butterworth filters,
  resampling, and N-octave spectrum analysis instead build independent channel tasks,
  while every task still materializes one complete continuous time series. Eligible
  `Normalize` configurations use the same private graph mechanics.
- Whole-frame operations can therefore exceed memory as either channel count or
  per-channel signal size grows. Independent channel-task execution reduces an
  adopted operation's kernel boundary across channels, but per-channel signal size
  remains bounded by available memory. N-octave spectrum analysis still receives the
  complete time axis within each channel task.
- WDF 0.4 passes internal source chunks to the writer without first computing the
  complete tensor. This bounds the writer's upstream data access by source chunking,
  although backend and compression buffers still contribute to RSS.
- A WDF-loaded Frame owns access to its source internally. Keep the source path
  unchanged while that Frame or Frames derived from it are in use; obtain NumPy
  values through `frame.data` without managing the storage backend.
- Tensor conversion and most external ML framework hand-offs materialize data.

The revision-addressed N-octave spectrum comparison used 240,000 float64 samples per
channel. At eight channels, the public `Frame.data` path changed from 36 to 56 tasks,
while median materialization changed from 0.2636 s to 0.1270 s and median worker peak
RSS changed from 270.9 MiB to 257.2 MiB. Base and candidate outputs were exactly equal.
These are same-environment observations, not portable thresholds; see the
[execution rationale and formal raw evidence](audio-operation-execution.md#adopted-operation-n-octave-spectrum).

## Recommended dataset workflow / 推奨 workflow

```python
import wandas as wd

dataset = wd.from_folder(
    "recordings/",
    recursive=True,
    path_metadata=True,
)
selected = dataset.select(machine="fan", split="train")
processed = selected.trim(0, 5).resample(16_000).normalize()
```

Select first, then load/process. Prefer several bounded recordings over concatenating
an entire corpus into one Frame. Chunk topology remains an internal implementation and
benchmark concern, not part of the normal Frame workflow.

## Reproducible benchmark / 再現可能 benchmark

Run the repository benchmark with the I/O extra:

```bash
uv run --no-dev --extra io python scripts/scalability_benchmark.py
```

Defaults cover 10-second and 100-second stereo Frames at 48 kHz with 1-second and
10-second source chunks. Every `channels × samples × chunk-samples × execution-path`
combination runs in an isolated worker process. The schema-version-2 JSON reports the
effective time chunk size,
chunks per channel, lazy graph construction time/peak Python allocation, and the
concrete task-key count from the benchmark's internal Dask collection graph (not the
number of HighLevelGraph layers). Operation metrics compare the same `remove_dc`
kernel through forced `whole-frame` and prototype `channel-wise` paths, including
compute time and absolute peak RSS observed immediately after operation execution.
Recipe extraction runs after those operation measurements, and contributes only the
separate `recipe_nodes` structural metric. WDF save time and file size use the
unprocessed chunked source Frame, so neither Recipe nor writer behavior is conflated
with the operation boundary. A benchmark-only internal fixture installs the
synthetic source chunks directly in xarray storage and verifies their actual topology
immediately before save; it does not change the public Frame workflow. Absolute peak
RSS covers the complete worker lifetime and is comparable only between workers using
the same platform, environment, and dependency lock. Use smaller values for a smoke
run:

```bash
uv run --no-dev --extra io python scripts/scalability_benchmark.py --samples 8000 --chunk-samples 1000 4000
```

These measurements characterize bounded upstream writer access, not a fixed RSS ceiling
across platforms or HDF5 configurations. WDF preserves typed Frame state, axes,
metadata, and deterministic failure behavior without precomputing the complete tensor.
Independent channel-task execution applies only to operations explicitly classified
under the channel-independent contract. See
[AudioOperation execution dependencies](audio-operation-execution.md) for the
internal contract and the classification of operations that remain whole-frame.
