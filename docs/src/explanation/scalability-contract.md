# Scalability contract / スケーラビリティ契約

Wandas scales primarily across collections of bounded recordings while preserving
the continuous-time assumptions of signal processing. Stored and lazy Frame data
retain a channel axis. `RemoveDC` is the first prototype operation that executes one
complete channel per lazy kernel task; other delayed `AudioOperation` transforms keep
the conservative whole-Frame boundary. Wandas therefore does not promise arbitrary
channel-count or time-axis distribution for one enormous Frame.

Wandas は主に、サイズを制御した多数の収録ファイルを扱う方向へ拡張します。Frame の保存・遅延データはチャンネル軸を保持しますが、一般的な `AudioOperation` は現在、1 つの Frame の全チャンネルをまとめて実体化します。信号処理の時間連続性を守るため、単一の巨大な Frame をチャンネル数または時間方向へ自由に分散できるとは約束しません。

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
  in one call. `RemoveDC` instead builds independent channel tasks, while every task
  still materializes one complete continuous time series.
- Whole-frame operations can therefore exceed memory as either channel count or
  per-channel signal size grows. The `RemoveDC` prototype reduces its kernel boundary
  across channels, but per-channel signal size remains bounded by available memory.
- WDF 0.4 passes internal source chunks to the writer without first computing the
  complete tensor. This bounds the writer's upstream data access by source chunking,
  although backend and compression buffers still contribute to RSS.
- A WDF-loaded Frame owns access to its source internally. Keep the source path
  unchanged while that Frame or Frames derived from it are in use; obtain NumPy
  values through `frame.data` without managing the storage backend.
- Tensor conversion and most external ML framework hand-offs materialize data.

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
Independent channel-task execution is currently promised only for `RemoveDC`. See
[AudioOperation execution dependencies](audio-operation-execution.md) for the internal
prototype contract and the classification of operations that remain whole-frame.
