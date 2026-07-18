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

## Canonical numeric contract / 正規化数値契約

Built-in readers always produce lazy, channel-first `float64` data. Equal file
content has equal values whether it comes from a local path, URL, bytes,
`bytearray`, `memoryview`, or a file-like object.

built-in readerは常に遅延実行のchannel-first `float64`を返します。同じファイル内容なら、
local path、URL、bytes、`bytearray`、`memoryview`、file-like objectのどれから読んでも値は同じです。

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

Provides functions for reading and writing WDF (Wandas Data File) format, which enables complete preservation including metadata.
WDF（Wandas Data File）形式の読み書き機能を提供します。このフォーマットはメタデータを含む完全な保存が可能です。

::: wandas.io.wdf_io
