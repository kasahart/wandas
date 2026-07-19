# Scalability contract / スケーラビリティ契約

Wandas scales primarily across collections of bounded recordings while preserving
the continuous-time assumptions of signal processing. Stored and lazy Frame data
retain a channel axis, but common `AudioOperation` transforms currently materialize
all channels in one Frame together. Wandas therefore does not promise arbitrary
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
- `AudioOperation.process()` currently wraps the complete channel-first Dask array in
  one delayed call. Common transforms therefore materialize all channels in a Frame
  before invoking the operation, rather than processing channel chunks independently.
- One Frame can therefore exceed memory as either channel count or per-channel signal
  size grows, even though graph construction is lazy.
- WDF 0.4 passes the source Dask chunks through xarray to the HDF5 writer without
  first computing the complete tensor. This bounds the writer's upstream data access
  by source chunking, although backend and compression buffers still contribute to RSS.
- WDF loading returns a backend-backed Dask array and keeps the source file open until
  the lazy data is computed, persisted, or released.
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
an entire corpus into one Frame. Keep time rechunking explicit and validate an
operation before changing the default `(channel=1, time=all)` policy.

## Reproducible benchmark / 再現可能 benchmark

Run the repository benchmark with the I/O extra:

```bash
uv run --no-dev --extra io python scripts/scalability_benchmark.py
```

Defaults cover 10-second and 100-second stereo Frames at 48 kHz with 1-second and
10-second source chunks. Every `samples × chunk-samples` pair runs in an isolated
worker process. The schema-version-2 JSON reports the effective time chunk size,
chunks per channel, lazy graph construction time/peak Python allocation, and the
concrete task-key count returned from the public
`Frame.xr.data` Dask collection graph protocol (not the number of HighLevelGraph
layers). Operation-graph metrics use a processed Frame; WDF save time and file size use
the unprocessed chunked source Frame, so writer behavior is not conflated with the
`AudioOperation` whole-Frame boundary. A benchmark-only internal fixture installs the
synthetic source chunks directly in xarray storage and verifies their actual topology
immediately before save; it does not change the public Frame constructor or the normal
`(channel=1, time=all)` chunk policy. Absolute peak RSS covers the complete worker
lifetime and is comparable only between workers using the same platform, environment,
and dependency lock. Use smaller
values for a smoke run:

```bash
uv run --no-dev --extra io python scripts/scalability_benchmark.py --samples 8000 --chunk-samples 1000 4000
```

These measurements characterize bounded upstream writer access, not a fixed RSS ceiling
across platforms or HDF5 configurations. WDF preserves typed Frame state, axes,
metadata, and deterministic failure behavior without precomputing the complete tensor.
Independent channel-chunk execution remains a possible future scalability target, not
behavior promised by the current `AudioOperation.process()` implementation.
