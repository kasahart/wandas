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
