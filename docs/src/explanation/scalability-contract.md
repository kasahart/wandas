# Scalability contract / スケーラビリティ契約

Wandas scales primarily across collections of bounded recordings while
preserving the continuous-time assumptions of signal processing. It does not
promise arbitrary channel-count or time-axis distribution for one enormous
Frame.

Wandas は主に、サイズを制御した多数の収録ファイルを扱う方向へ拡張します。
単一の巨大な Frame をチャンネル数または時間方向へ自由に分散できるとは約束しません。

## Supported flow / 推奨する流れ

1. Discover a `ChannelFrameDataset` from paths and metadata.
2. Select recordings before reading waveform samples.
3. Load only the selected, bounded recordings.
4. Build a lazy graph and materialize only at the operation's execution boundary.

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

Prefer several bounded recordings over concatenating an entire corpus into one
Frame. Selection metadata is resolved before waveform samples are read, so the
selection step is the natural place to control the amount of data entering the
workflow.

## Materialization boundaries / 実体化境界

- Method chains construct lazy graphs; graph construction itself does not promise
  that samples have been computed.
- A numerical kernel may require one complete, continuous time axis for each
  channel. Channel-independent execution can reduce the number of channels at a
  kernel boundary without distributing that time axis.
- Conservative whole-Frame operations may materialize all channels at once, so
  channel count and recording length remain memory constraints.
- `frame.data`, NumPy conversion, and tensor or external ML-framework hand-offs
  materialize the requested result.
- `frame.cache()` synchronously materializes one complete raw Frame tensor into
  local process memory and returns an equivalent Dask-backed Frame that can reuse
  those samples. It is intended only for bounded recordings that fit in memory;
  there is no automatic eviction, distributed-worker placement, or scheduler API.
  A raw `np.ma.MaskedArray` compute result is rejected because mask representation
  is not consistent across supported xarray versions.

These are execution boundaries, not scheduler or Dask-topology guarantees. The
class hierarchy, operation implementation, and tests define current eligibility;
private chunking and scheduler choices may change.

Caching is not a processing or persistence operation. It preserves runtime lineage
and contributes no Recipe node, WDF state, cache-status flag, or release protocol.
The computed tensor becomes eligible for garbage collection when the cached Frame
and all Frames derived from it are no longer referenced.

For operation authors, see the [Frame and Operation extension guide](../contributing/frame-operation-extensions.md).
For WDF implementation invariants, start at the [Contributing Overview](../contributing.md), which links to the [I/O contract guide](../contributing/io-contracts.md).
