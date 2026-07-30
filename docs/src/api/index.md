# API Reference / APIリファレンス

API reference for the main components and functions of the Wandas library.
Wandasライブラリの主要コンポーネントと関数のAPIリファレンスです。

## Top-Level API / トップレベルAPI

The primary top-level API is intentionally small:
主要なトップレベル API は意図的に小さくしています。

- `wd.read(...)` - Read external source data into a `ChannelFrame` / 外部ソースデータを `ChannelFrame` として読み込む
- `wd.load(...)` - Load Wandas native WDF files / Wandas native WDF ファイルを読み込む
- `wd.from_numpy(...)` - Create a `ChannelFrame` from a NumPy array / NumPy 配列から `ChannelFrame` を作る
- `wd.from_folder(...)` - Create a `ChannelFrameDataset` from a folder / フォルダから `ChannelFrameDataset` を作る
- `wd.supported_formats()` - List registered reader suffixes / 登録済み reader suffix を一覧表示
- `wd.ChannelFrame`, `wd.SpectralFrame`, `wd.CepstralFrame`, `wd.SpectrogramFrame`, `wd.CepstrogramFrame`, `wd.NOctFrame`, `wd.ChannelFrameDataset`, `wd.ChannelCalibration` - Public frame and calibration classes / 公開フレーム・校正クラス

`read_wav()` and `read_csv()` are stable compatibility conveniences outside
`wandas.__all__`; new code normally uses `read()`. `from_ndarray()` is deprecated
compatibility outside `__all__`: use `from_numpy()` instead. It has been deprecated
since 0.2.0, remains supported through 0.6.x, and will not be removed before 0.7.0.
`read_wav()` と `read_csv()` は `wandas.__all__` 外の stable な互換 convenience で、
新規コードでは通常 `read()` を使います。`from_ndarray()` は `__all__` 外の非推奨互換
API です。代わりに `from_numpy()` を使用してください。0.2.0 から非推奨で、0.6.x の間は
維持され、0.7.0 より前には削除されません。

`generate_sin()` and `setup_wandas_logging()` are experimental conveniences outside
`wandas.__all__`. `generate_sin()` is for self-contained learning examples;
`setup_wandas_logging()` configures the `wandas` logger but applications may instead
use the standard `logging` module. Their contracts may change in a feature release.
`generate_sin()` と `setup_wandas_logging()` は `wandas.__all__` 外の実験的 convenience
です。`generate_sin()` は自己完結した学習例向け、`setup_wandas_logging()` は `wandas`
logger の設定向けです。feature release で契約が変わる可能性があります。

The machine-readable source for these classifications and every package export is
`wandas._public_api.PUBLIC_API_INVENTORY`; package `__all__` lists are derived from
it. The stability categories are defined in the
[public API stability guide](../explanation/public-api-stability.md).
これらの分類と package export の機械可読な正本は
`wandas._public_api.PUBLIC_API_INVENTORY` で、各 `__all__` はそこから導出されます。
分類の意味は [public API stability guide](../explanation/public-api-stability.md) を
参照してください。

## Modules / モジュール

Browse the detailed API documentation for each module:
各モジュールの詳細なAPIドキュメントを参照してください：

### [Core Module / コアモジュール](core.md)

The core module provides the basic functionality of Wandas, including base classes and metadata management.
コアモジュールはWandasの基本機能（基底クラスやメタデータ管理など）を提供します。

- `BaseFrame` - Base class for all frames / すべてのフレームの基底クラス
- `ChannelMetadata` - Channel metadata management / チャンネルメタデータ管理

### [Frames Module / フレームモジュール](frames.md)

The frames module defines different types of data frames for time-domain, frequency-domain, and time-frequency-domain data.
フレームモジュールは、時間領域、周波数領域、時間-周波数領域データのための様々なデータフレームを定義します。

- `ChannelFrame` - Time-domain waveform data / 時間領域波形データ
- `SpectralFrame` - Frequency-domain data / 周波数領域データ
- `CepstralFrame` - Quefrency-domain data / ケフレンシー領域データ
- `SpectrogramFrame` - Time-frequency domain data / 時間-周波数領域データ
- `CepstrogramFrame` - Time-quefrency domain data / 時間-ケフレンシー領域データ
- `NOctFrame` - N-octave band analysis / Nオクターブバンド解析
- `RoughnessFrame` - Psychoacoustic roughness analysis results / 心理音響ラフネス解析結果

### [Processing Module / 処理モジュール](processing.md)

The processing module provides various processing functions for audio data, including filters, effects, and analysis.
処理モジュールは、フィルタ、エフェクト、分析など、オーディオデータに対する様々な処理機能を提供します。

- Filters / フィルター - Digital filters for signal processing / 信号処理用デジタルフィルター
- Effects / エフェクト - Audio effects processing / オーディオエフェクト処理
- Spectral / スペクトル - Spectral analysis functions / スペクトル解析機能
- Temporal / 時間領域 - Time-domain processing / 時間領域処理
- Stats / 統計 - Statistical analysis / 統計分析

### [IO Module / 入出力モジュール](io.md)

The IO module provides file reading and writing functions for various formats.
入出力モジュールは、様々なフォーマットのファイル読み書き機能を提供します。

- WAV file I/O / WAVファイル入出力
- WDF file I/O / WDFファイル入出力 - See also: [WDF Format Details](wdf_io.md) / 詳細: [WDFフォーマット詳細](wdf_io.md)
- File readers / ファイルリーダー

### [Visualization Module / 可視化モジュール](visualization.md)

The visualization module provides data visualization functions using Matplotlib.
可視化モジュールは、Matplotlibを使用したデータ視覚化機能を提供します。

- Plotting functions / プロッティング関数
- Plot strategies for different frame types / 異なるフレームタイプ用のプロット戦略

### [Utilities Module / ユーティリティモジュール](utils.md)

The utilities module provides auxiliary functions including dataset management and sample generation.
ユーティリティモジュールは、データセット管理やサンプル生成などの補助機能を提供します。

- Frame datasets / フレームデータセット - Batch processing of audio files / 音声ファイルのバッチ処理
- Sample generation / サンプル生成 - Generate test signals / テスト信号生成
- Type definitions / 型定義

### [Datasets Module / データセットモジュール](datasets.md)

The datasets namespace currently has no public exports or packaged sample assets.
Use experimental `wd.generate_sin()` for a known signal, or stable `wd.read()` and
`wd.from_folder()` for application-owned recordings.
datasets namespaceには現在public exportやpackage同梱sample assetがありません。
既知信号には実験的な`wd.generate_sin()`、application所有のrecordingにはstableな
`wd.read()`または`wd.from_folder()`を使用します。

- No public `wandas.datasets` symbols / publicな`wandas.datasets` symbolなし

### [Pipeline Recipes API](pipeline.md)

The pipeline module records a public Frame workflow as a portable, validated
`RecipePlan` and applies it to named runtime inputs.
pipelineモジュールは公開Frame処理をportableな`RecipePlan`として記録し、名前付きの
runtime入力へ適用します。
