# Theory Background and Architecture / 理論背景とアーキテクチャ

This section explains the design philosophy, internal architecture, and theoretical background used in the Wandas library.
このセクションでは、Wandasライブラリの設計思想、内部アーキテクチャ、およびライブラリで使用されている理論的背景について説明します。

## Design Philosophy / 設計思想

Wandas is developed based on the following design principles:
Wandasは以下の設計原則に基づいて開発されています：

1. **Intuitive API Design** - Consistent interface that users can easily use.
   **直感的なAPI設計** - ユーザーが簡単に使える一貫性のあるインターフェース。
2. **Efficient Memory Usage** - Memory-efficient implementation suitable for processing large-scale data.
   **効率的なメモリ使用** - 大規模データの処理に適したメモリ効率の良い実装。
3. **Extensibility** - Expandable architecture that makes it easy to add new features and algorithms.
   **拡張性** - 新しい機能やアルゴリズムを追加しやすい拡張可能なアーキテクチャ。

## Core Architecture / コアアーキテクチャ

### Data Model / データモデル

The central data model of the Wandas library is structured around immutable frames:
Wandasライブラリの中心となるデータモデルは、不変（Immutable）なフレームを中心に構成されています：

```
frames/
 ├── ChannelFrame (Time-domain signals / 時間領域信号)
 ├── SpectralFrame (Frequency-domain data / 周波数領域データ)
 └── SpectrogramFrame (Time-Frequency domain data / 時間-周波数領域データ)
```

Responsibilities of each class:
各クラスの責任：

- **ChannelFrame**: Handles multi-channel time-domain waveform data. Manages axes, metadata, and operation history.
  **ChannelFrame**: マルチチャンネルの時間領域波形データを扱います。軸、メタデータ、操作履歴を管理します。
- **SpectralFrame**: Handles frequency-domain data (e.g., FFT results).
  **SpectralFrame**: 周波数領域データ（FFT結果など）を扱います。
- **SpectrogramFrame**: Handles time-frequency domain data (e.g., STFT results).
  **SpectrogramFrame**: 時間-周波数領域データ（STFT結果など）を扱います。

### Separation of Concerns / 関心の分離

- **frames/**: User-facing data structures. Responsible for orchestration and metadata management.
  **frames/**: ユーザー向けのデータ構造。オーケストレーションとメタデータ管理を担当します。
- **processing/**: Pure numerical logic (filters, spectral analysis, etc.). Frame methods delegate to these functions.
  **processing/**: 純粋な数値ロジック（フィルタ、スペクトル分析など）。フレームのメソッドはこれらの関数に処理を委譲します。
- **io/**: I/O helpers for WAV, WDF, CSV, etc.
  **io/**: WAV, WDF, CSVなどのI/Oヘルパー。

### Data Processing Flow / データ処理フロー

1. **Input Stage**: Generate `ChannelFrame` objects from files using `io` helpers.
   **入力段階**: `io` ヘルパーを使用してファイルから `ChannelFrame` オブジェクトを生成します。
2. **Processing Stage**: Apply processing such as filtering and resampling. Operations return new frame objects (immutability).
   **処理段階**: フィルタリング、リサンプリングなどの処理を適用します。操作は新しいフレームオブジェクトを返します（不変性）。
3. **Analysis Stage**: Analyze signal characteristics (spectrum, level, etc.).
   **分析段階**: 信号の特性（スペクトル、レベル等）を分析します。
4. **Output Stage**: Save processing results to files or visualize as graphs.
   **出力段階**: 処理結果をファイルに保存またはグラフとして可視化します。

## Implementation Details / 実装詳細

### Memory Efficiency / メモリ効率

Wandas ensures memory efficiency for handling large audio data through the following methods:
Wandasは大規模なオーディオデータを扱うために、以下の方法でメモリ効率を確保しています：

- **Lazy Evaluation**: A mechanism that delays calculations until needed (using Dask).
  **遅延評価**: 必要になるまで計算を遅延させる仕組み（Daskを使用）。
- **Memory Mapping**: Access to large files without loading them entirely into memory.
  **メモリマッピング**: 大きなファイルでもメモリに全て読み込まずにアクセス。

### Signal Processing Algorithms / 信号処理アルゴリズム

Wandas implements signal processing algorithms such as:
Wandasは以下のような信号処理アルゴリズムを実装しています：

- **Digital Filters**: IIR/FIR filters such as Butterworth filters.
  **デジタルフィルタ**: バターワースフィルタなどのIIR/FIRフィルタ。
- **Spectral Analysis**: Frequency analysis based on Fast Fourier Transform (FFT).
  **スペクトル分析**: 高速フーリエ変換（FFT）に基づく周波数分析。
- **Time-Frequency Analysis**: Short-Time Fourier Transform (STFT), spectrograms.
  **時間-周波数分析**: 短時間フーリエ変換（STFT）、スペクトログラム。
- **Statistical Analysis**: Calculation of signal characteristics such as RMS, peak values, crest factor.
  **統計的分析**: RMS、ピーク値、クレストファクターなどの信号特性の計算。

## Psychoacoustic Metrics / 心理音響メトリクス

Wandas provides psychoacoustic metrics for analyzing audio signals based on human perception. These metrics are calculated using standardized methods and the MoSQITo library.:
Wandasは、人間の知覚に基づく音響信号を分析するための心理音響メトリクスを提供します。これらのメトリクスは、標準化された手法とMoSQIToライブラリを使用して計算されます。：

- **Loudness Calculation / ラウドネス計算**: Time-varying loudness calculation using Zwicker method according to ISO 532-1:2017.
  ISO 532-1:2017に準拠したZwicker法による時間変化するラウドネス計算。
- **Sharpness Calculation / シャープネス計算**: Sharpness calculation based on Aures method according to DIN 45692.
  DIN 45692に準拠したAures法によるシャープネス計算。
- **Roughness Calculation / ラフネス計算**: Roughness calculation using Daniel and Weber method.
  Daniel and Weber法によるラフネス計算。
