# WDF File I/O / WDFファイル入出力

The `wandas.io.wdf_io` module provides functionality for saving and loading `ChannelFrame` objects in the WDF (Wandas Data File) format.
`wandas.io.wdf_io` モジュールは、`ChannelFrame` オブジェクトを WDF (Wandas Data File) 形式で保存・読み込みするための機能を提供します。

The WDF format is based on HDF5 and preserves not only the data but also all metadata such as sampling rate, units, and channel labels.
WDFフォーマットは HDF5 をベースとし、データだけでなくサンプリングレート、単位、チャンネルラベルなどのメタデータも完全に保存します。

## WDF Format Overview / WDFフォーマット概要

The WDF format has the following features:
WDFフォーマットは以下の特徴を持ちます:

- HDF5-based hierarchical data structure.
  HDF5ベースの階層的なデータ構造。
- Complete preservation of channel data and metadata.
  チャンネルデータとメタデータの完全な保持。
- Size optimization through data compression and chunking.
  データ圧縮とチャンク化によるサイズ最適化。
- Version management for future extensions.
  将来の拡張に対応するバージョン管理。

File structure / ファイル構造:

```
/meta           : Frame-level metadata (JSON format) / Frame 全体のメタデータ (JSON形式)
/channels/{i}   : Individual channel data and metadata / 個々のチャンネルデータとメタデータ
    ├─ data           : Waveform data (numpy array) / 波形データ (numpy array)
    └─ attrs          : Channel attributes (labels, units, etc.) / チャンネル属性 (ラベル、単位など)
```

## Saving WDF Files / WDFファイル保存

::: wandas.io.wdf_io.save

## Loading WDF Files / WDFファイル読み込み

::: wandas.io.wdf_io.load

## Usage Examples / 利用例

```python
# Save a ChannelFrame in WDF format
# ChannelFrame を WDF形式で保存
cf = wd.read_wav("audio.wav")
cf.save("audio_data.wdf")

# Specifying options when saving
# 保存時のオプション指定
cf.save(
    "high_quality.wdf",
    compress="gzip",  # Compression method / 圧縮方式
    dtype="float64",  # Data type / データ型
    overwrite=True    # Allow overwriting / 上書き許可
)

# Load a ChannelFrame from a WDF file
# WDFファイルから ChannelFrame を読み込み
cf2 = wd.ChannelFrame.load("audio_data.wdf")
```
