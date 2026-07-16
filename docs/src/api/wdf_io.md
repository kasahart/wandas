# WDF File I/O / WDFファイル入出力

The `wandas.io.wdf_io` module saves and loads built-in typed Frames in the WDF (Wandas Data File) format.
`wandas.io.wdf_io` モジュールは built-in typed Frame を WDF (Wandas Data File) 形式で保存・読み込みします。

The WDF format is based on HDF5 and preserves not only the data but also all metadata such as sampling rate, units, and channel labels.
WDFフォーマットは HDF5 をベースとし、データだけでなくサンプリングレート、単位、チャンネルラベルなどのメタデータも完全に保存します。

## WDF Format Overview / WDFフォーマット概要

The WDF format has the following features:
WDFフォーマットは以下の特徴を持ちます:

- HDF5-based hierarchical data structure.
  HDF5ベースの階層的なデータ構造。
- Typed round-trip for Channel, spectral, spectrogram, cepstral, cepstrogram,
  N-octave, and roughness Frames.
  Channel、spectrum、spectrogram、cepstrum、cepstrogram、N-octave、roughness Frame の型付き往復。
- Size optimization through data compression and chunking.
  データ圧縮とチャンク化によるサイズ最適化。
- Version management for future extensions.
  将来の拡張に対応するバージョン管理。

File structure / ファイル構造:

```
/data           : Complete rank-preserving Frame tensor / Frame tensor 全体
/channels/{i}   : Channel metadata / channel metadata
/coordinates    : Persisted represented-axis coordinates / 表現済み axis coordinate
/meta           : Frame-level metadata (JSON) / Frame metadata (JSON)
/attrs          : WDF, Frame-state, and display-history schemas
```

WDF 0.3 restores the exact built-in Frame type and its analysis parameters. WDF 0.1
and 0.2 remain readable as `ChannelFrame`. Unsupported future versions fail explicitly.
Runtime lineage and Dask graphs are not restored; `operation_history` is display-only.

## Saving WDF Files / WDFファイル保存

::: wandas.io.wdf_io.save

## Loading WDF Files / WDFファイル読み込み

::: wandas.io.wdf_io.load

## Usage Examples / 利用例

```python
import wandas as wd

# Any built-in typed Frame can save itself
frame = wd.read("audio.wav").stft(n_fft=2048)
frame.save("analysis.wdf")

# Specifying options when saving
# 保存時のオプション指定
frame.save(
    "high_quality.wdf",
    compress="gzip",  # Compression method / 圧縮方式
    dtype="complex64",  # Keep complex analysis data complex / 複素解析データを保持
    overwrite=True    # Allow overwriting / 上書き許可
)

# Restore the concrete stored type (SpectrogramFrame here)
restored = wd.load("analysis.wdf")
```

WDF save currently materializes the complete Frame before writing, and load reads the
stored tensor before wrapping it in Dask.
Converting a complex Frame to a real `dtype` is rejected because it would discard the
imaginary component of the analysis result.
