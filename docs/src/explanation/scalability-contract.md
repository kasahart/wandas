# Scalability contract / スケーラビリティ契約

Wandas scales primarily across recordings and channels while preserving the
continuous-time assumptions of signal processing. It does not promise arbitrary
time-axis distribution for one enormous waveform.

Wandas は主に「多数の収録ファイル」と「チャンネル」の方向へ拡張します。信号処理の時間連続性を守るため、単一の巨大波形を時間方向へ自由に分散できるとは約束しません。

## What scales well / 得意な処理

- Discover many files as a lazy `ChannelFrameDataset`.
- Select files from path/CSV metadata before reading waveform samples.
- Load only selected files and process channels independently.
- Build and apply a `RecipePlan` without computing Frame samples.
- Keep the channel axis chunked while each continuous signal axis normally remains one chunk.

## Current limits / 現在の制約

- Filters, FFT, STFT, and other continuity-sensitive operations normally require a
  single time chunk per channel.
- One very large Frame can therefore exceed memory during filter/STFT computation even
  though graph construction is lazy.
- WDF 0.3 `save()` calls `frame.compute()` before writing its rank-preserving tensor.
- WDF loading currently reads the stored tensor before wrapping it in a Dask array;
  it is not a streaming or memory-mapped reader.
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
uv run --extra io python scripts/scalability_benchmark.py
```

Defaults cover 10-second and 100-second stereo-equivalent Frames at 48 kHz. Each case
runs in an isolated worker process. The JSON result reports lazy graph construction
time/peak Python allocation, the concrete task-key count returned from the public
`Frame.xr.data` Dask collection graph protocol (not the number of HighLevelGraph
layers), WDF save time and file size, and the worker's absolute process peak RSS. The
RSS field covers the complete worker lifetime, not only the WDF save phase. Use smaller
values for a smoke run:

```bash
uv run --extra io python scripts/scalability_benchmark.py --samples 8000
```

These measurements characterize the current contract; they are not a promise that WDF
is streaming. A future chunked writer must avoid whole-Frame materialization while
preserving typed Frame state, axes, metadata, and deterministic failure behavior.
