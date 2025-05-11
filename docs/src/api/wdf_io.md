# WDFファイル入出力

`wandas.io.wdf_io` モジュールは、`ChannelFrame` オブジェクトを WDF (Wandas Data File) 形式で保存・読み込みするための機能を提供します。
WDFフォーマットは HDF5 をベースとし、データだけでなくサンプリングレート、単位、チャンネルラベルなどのメタデータも完全に保存します。

## WDFフォーマット概要

WDFフォーマットは以下の特徴を持ちます:

- HDF5ベースの階層的なデータ構造
- チャンネルデータとメタデータの完全な保持
- データ圧縮とチャンク化によるサイズ最適化
- 将来の拡張に対応するバージョン管理

ファイル構造:

```
/meta           : Frame 全体のメタデータ (JSON形式)
/channels/{i}   : 個々のチャンネルデータとメタデータ
    ├─ data           : 波形データ (numpy array)
    └─ attrs          : チャンネル属性 (ラベル、単位など)
```

## WDFファイル保存

::: wandas.io.wdf_io.save

## WDFファイル読み込み

::: wandas.io.wdf_io.load

## 利用例

```python
# ChannelFrame を WDF形式で保存
cf = wd.read_wav("audio.wav")
cf.save("audio_data.wdf")

# 保存時のオプション指定
cf.save(
    "high_quality.wdf",
    compress="gzip",  # 圧縮方式
    dtype="float64",  # データ型
    overwrite=True    # 上書き許可
)

# WDFファイルから ChannelFrame を読み込み
cf2 = wd.ChannelFrame.load("audio_data.wdf")
```

詳細な使用例は [ファイル入出力ガイド](/how_to/file_io) を参照してください。
