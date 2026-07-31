# IO Module / 入出力モジュール

The `wandas.io` module provides reading and writing capabilities for various file formats.
`wandas.io` モジュールは、様々なファイル形式の読み書き機能を提供します。

## Recommended Entry Points / 推奨入口

Use `wd.read(...)` for external source data such as WAV, CSV, supported audio files, URLs, bytes, and file-like objects.
WAV、CSV、対応音声ファイル、URL、bytes、file-like object などの外部ソースデータには `wd.read(...)` を使います。

Use `wd.load(...)` for Wandas native WDF files.
Wandas native WDF ファイルには `wd.load(...)` を使います。

`read_wav()` and `read_csv()` remain available for compatibility, but new documentation and examples prefer `read()`.
互換性のため `read_wav()` と `read_csv()` は残りますが、新しいドキュメントと例では `read()` を優先します。

## `wd.read()` API

::: wandas.read

### Source and format selection / source と format の選択

Local paths select a registered reader from the path suffix. HTTP/HTTPS URLs use
the URL path suffix, or an explicit `file_type` when supplied. For `bytes`,
`bytearray`, `memoryview`, and file-like objects, `wd.read()` uses the first
available format hint in this order:

1. explicit `file_type`;
2. a file-like object's `.name` suffix;
3. the `source_name` suffix (for HTTP/HTTPS URLs, the URL path suffix before
   any query or fragment);
4. `.wav` for an otherwise anonymous in-memory source.

local path は path suffix、HTTP/HTTPS URL は URL path suffix（または明示した
`file_type`）から登録済みreaderを選びます。`bytes`、`bytearray`、`memoryview`、
file-like object では、`file_type`、file-like `.name` の suffix、`source_name`
の suffix（HTTP/HTTPS URL は query／fragment より前の URL path suffix）、匿名入力の
`.wav` の順で最初に利用できる hint を使います。

Only the URL path participates in inference. A filename hint found solely in a
query or fragment, such as `?filename=data.csv`, is intentionally ignored; pass
`file_type=".csv"` explicitly. URL 推論では URL path だけを使います。
`?filename=data.csv` のように query／fragment だけにある filename hint は意図的に
無視するため、`file_type=".csv"` を明示してください。

The anonymous-WAV fallback is retained for compatibility with existing
`wd.read(wav_bytes)` calls. It does not inspect or sniff the content. Anonymous
CSV or other formats must provide `file_type`; alternatively, give
`source_name` a registered suffix. `file_type` is case-insensitive and accepts
both `"csv"` and `".csv"`.

匿名入力を WAV とする fallback は既存の `wd.read(wav_bytes)` との互換性のため
維持されます。content sniffing は行いません。匿名 CSV などは `file_type` を渡すか、
登録済み suffix を持つ `source_name` を指定してください。`file_type` は大文字小文字を
区別せず、`"csv"` と `".csv"` の両方を受け付けます。

`source_name` has two roles for in-memory input: its suffix can select the
reader, and its value supplies the Frame label and `_source_file` provenance
metadata. It never opens or downloads that name. A file-like `.name` takes
format precedence over an explicit `source_name`, while the explicit
`source_name` remains the recorded provenance.

in-memory input の `source_name` には、suffix による reader 選択と、Frame label／
`_source_file` 由来metadataという2つの役割があります。その名前を別途open／download
することはありません。file-like `.name` は明示した `source_name` より format 推論で
優先されますが、記録される由来情報には明示した `source_name` を使います。

```python
import io
import wandas as wd

wav_frame = wd.read(wav_bytes)  # anonymous in-memory input defaults to WAV
csv_frame = wd.read(csv_bytes, source_name="sensor.csv")

stream = io.BytesIO(csv_bytes)
stream.name = "upload.csv"
csv_frame = wd.read(stream)
```

An unavailable suffix raises `ValueError` instead of trying another reader.
Missing local paths raise `FileNotFoundError`; URL transport failures raise
`OSError`. WDF is a separate persistence boundary: `wd.read("result.wdf")`
raises with guidance to use `wd.load("result.wdf")`.

未登録の suffix は別 reader を試さず `ValueError`、存在しない local path は
`FileNotFoundError`、URL transport の失敗は `OSError` になります。WDF は別の
永続化境界であり、`wd.read("result.wdf")` は `wd.load("result.wdf")` を使うよう
案内して失敗します。

## Canonical numeric and loading contract / 正規化数値・読込契約

Built-in readers always return Dask-backed, channel-first `float64` data. Audio
sample decoding is deferred until the Dask data is computed. CSV is different:
its complete table is parsed synchronously once to determine exact shape and
sampling rate before the Frame is returned, then parsed again when the Dask
sample data is computed. Equal file content has equal values whether it comes
from a local path, URL, bytes, `bytearray`, `memoryview`, or a file-like object.

built-in readerは常にDask-backedのchannel-first `float64`を返します。audio sampleの
decodeはDask dataをcomputeするまで遅延されます。CSVは異なり、正確なshapeと
sampling rateを決めるため返却前に全tableを同期的に1回parseし、Dask sample dataの
compute時に再度parseします。同じファイル内容なら、local path、URL、bytes、
`bytearray`、`memoryview`、file-like objectのどれから読んでも値は同じです。

| Input / 入力 | `wd.read()` numeric rule / 数値規則 |
| --- | --- |
| WAV (`PCM_U8`, `PCM_16`, `PCM_24`, `PCM_32`) | libsndfile full-scale conversion; unsigned 8-bit PCM is zero-centered / libsndfileのfull-scale変換、符号なし8-bit PCMもゼロ中心 |
| WAV (`FLOAT`, `DOUBLE`) | Values are preserved as `float64`; no clipping, so values may exceed ±1 / 値を`float64`で保持し、クリップしないため±1を超える場合がある |
| FLAC, OGG, AIFF/AIF, SND | libsndfile full-scale `float64` audio / libsndfileのfull-scale `float64`音声 |
| CSV | Non-time numeric values are preserved and cast to `float64`; non-numeric channels are rejected / 時間列以外の数値を維持して`float64`化し、非数値chは拒否 |

This is decode normalization, not peak normalization: Wandas never divides by
the maximum value of an individual waveform. `frame.normalize()` and playback's
`normalize` option remain separate processing and presentation features.

これは波形ごとの最大値で割るpeak normalizationではなく、decode時の正規化です。
`frame.normalize()`と再生時の`normalize`は別の処理・表示機能です。

### Migration / 移行

Local integer WAV files previously defaulted to raw PCM counts cast to
`float32`. They now use the same full-scale `float64` decoding as every other
transport. Calibration factors derived for raw counts must be derived again
from a reference recording read under the new contract. `wd.load()` preserves
the dtype stored in WDF, and `wd.from_numpy()` preserves the user-selected
array dtype; neither contract changes here.

従来local integer WAVはraw PCM countを`float32`へcastしていましたが、今後は他の
transportと同じfull-scale `float64`です。raw count向けの既知係数は、新契約で読んだ
参照収録から再導出してください。`wd.load()`はWDF保存dtype、`wd.from_numpy()`は
利用者指定dtypeを維持し、これらの契約は変更しません。

## File Readers / ファイルリーダー

Provides functionality to read data from various file formats.
様々なファイル形式からデータを読み込む機能を提供します。

::: wandas.io.readers

## WAV File IO / WAVファイル入出力

Provides functions for reading and writing WAV files.
WAVファイルの読み書き機能を提供します。

::: wandas.io.wav_io

## WDF File IO / WDFファイル入出力

WDF stores the concrete typed Frame state described in the
[WDF 0.4 contract](wdf_io.md). It does not preserve every runtime object or an
executable Recipe.
WDF は [WDF 0.4 契約](wdf_io.md)に記載された具体的な型付き Frame state を保存します。
すべての runtime object や実行可能 Recipe を保存するものではありません。

::: wandas.io.wdf_io
