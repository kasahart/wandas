# Inspect audio before analysis / 音声の事前検査と実フレーム時刻

Use `inspect` when recording limits should be checked before even constructing
a lazy sample graph. It returns a normal dictionary; no new record type is needed.

```python
import wandas as wd

info = wd.inspect("recording.wav")
if not 1 <= info["channels"] <= 8 or info["duration"] > 180:
    raise ValueError("Choose a shorter recording with fewer channels")
recording = wd.read("recording.wav")
```

`inspect` reads audio headers synchronously and creates neither PCM arrays nor
Dask graphs. It supports the built-in SoundFile formats, local paths, bytes-like
values and seekable binary streams. Bytes use the same format inference as `read`.
Borrowed streams are inspected from zero, restored to their original position on
success or failure, and never closed. It rejects URLs, CSV, WDF, custom readers,
text streams and non-seekable streams. Inspection is a snapshot, not a guarantee
that a mutable source will still contain the same recording when read later.
`read` independently inspects the source again. A browser must still provide the
selected recording's bytes; this API grants no filesystem access.

`inspect` は音声ヘッダーだけを同期検査し、PCM配列やDask graphを作りません。
入力streamの位置を戻し、借りたstreamを閉じません。CSVの全表解析やURL downloadは
事前検査の保証と異なるため対象外です。後続の `read` は改めてヘッダーを検査します。

## Physical STFT centers / 実際のSTFTフレーム中心

```python
spectrum = recording.trim(0.25, 1.25).stft(n_fft=2048, hop_length=512)
centers = spectrum.frame_center_times[0]
```

`frame_center_times` is an array shaped `(n_channels, n_frames)` in seconds on
each channel's source timeline. It includes padding centers before or after the
recording. At 16 kHz with a 2048-sample Hann window and hop 512, the first center
is -0.032 seconds relative to the analyzed segment's start. Trimming, resampling,
channel selection, contiguous time slicing, magnitude/dtype transforms, caching,
cepstrogram/envelope conversion and Recipe replay preserve placement without
computing samples to retrieve the axis. Binary arithmetic uses the left operand's
placement and performs no time alignment, matching the existing offset contract.
The existing zero-based `times`, `source_times`, plotting and ISTFT behavior are
unchanged. In particular, `get_frame_at` retains its existing local-time offset
semantics; use this new axis when physical frame centers are needed.

The read-only `frame_time_origin` is the first center relative to the analyzed
input's start. STFT supplies it from the same numerical operation that generates
the coefficients. Manual Spectrogram/Cepstrogram construction may supply a known
origin in seconds with `frame_time_origin=...`; the default is `None` (unknown).
Requesting `frame_center_times` when the origin is unknown raises `ValueError`
instead of guessing from window parameters or display history.

実時刻はchannelごとの原音timelineで返し、padding分を含みます。既存のゼロ始点の軸や
plot/ISTFTは変更しません。手動構築や旧保存形式で情報がない場合は推測せず明示的に失敗します。

## Saved results / 保存互換性

Frames with a known physical time origin use the additive WDF 0.5 constructor
extension. Other Frames still save as WDF 0.4. The new reader accepts both strict
schemas. A WDF 0.4 time-frequency result loads with `frame_time_origin=None`;
its existing data and axes remain usable. Recompute STFT from its source when
physical centers are required. History is not used to invent missing state.
An older reader rejects WDF 0.5 explicitly rather than silently losing placement.
Recipe parameters and versions are unchanged: replay constructs the new derived
axis state from the numerical STFT operation.

実時刻が既知の結果だけWDF 0.5で保存します。旧WDF 0.4は従来のdata/axisを保って読み込めます。
旧readerは0.5を拒否するため、実時刻を保持した保存結果を旧版で開く用途には注意してください。
