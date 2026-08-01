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

These are execution boundaries, not scheduler or Dask-topology guarantees. The
class hierarchy, operation implementation, and tests define current eligibility;
private chunking and scheduler choices may change.

For operation authors, see [AudioOperation execution dependencies](audio-operation-execution.md).
For WDF persistence details, see the [I/O contract guide](../contributing/io-contracts.md).
