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
 ├── CepstralFrame (Quefrency-domain data / ケフレンシー領域データ)
 ├── SpectrogramFrame (Time-Frequency domain data / 時間-周波数領域データ)
 └── CepstrogramFrame (Time-Quefrency domain data / 時間-ケフレンシー領域データ)
```

Responsibilities of each class:
各クラスの責任：

- **ChannelFrame**: Handles multi-channel time-domain waveform data. Manages axes, metadata, and lineage-derived operation history views.
  **ChannelFrame**: マルチチャンネルの時間領域波形データを扱います。軸、メタデータ、操作履歴を管理します。
- **SpectralFrame**: Handles frequency-domain data (e.g., FFT results).
  **SpectralFrame**: 周波数領域データ（FFT結果など）を扱います。
- **SpectrogramFrame**: Handles time-frequency domain data (e.g., STFT results).
  **SpectrogramFrame**: 時間-周波数領域データ（STFT結果など）を扱います。
- **CepstralFrame**: Handles one real cepstrum on a quefrency axis.
  **CepstralFrame**: ケフレンシー軸上の1つの実ケプストラムを扱います。
- **CepstrogramFrame**: Handles a real cepstrum evolving over STFT time frames.
  **CepstrogramFrame**: STFT時間フレームごとに変化する実ケプストラムを扱います。

### Separation of Concerns / 関心の分離

- **frames/**: User-facing data structures. Responsible for orchestration and metadata management.
  **frames/**: ユーザー向けのデータ構造。オーケストレーションとメタデータ管理を担当します。
- **processing/**: Pure numerical logic (filters, spectral analysis, etc.). Frame methods delegate to these functions.
  **processing/**: 純粋な数値ロジック（フィルタ、スペクトル分析など）。フレームのメソッドはこれらの関数に処理を委譲します。
- **io/**: I/O helpers for WAV, WDF, CSV, etc.
  **io/**: WAV, WDF, CSVなどのI/Oヘルパー。

### xarray-backed architecture / xarrayベースのアーキテクチャ

Wandas uses xarray internally as the labelled storage and frame-state layer while keeping Wandas as the waveform analysis API. In the current migration stage, xarray owns data, named dimensions, selected coordinates, and frame-level attrs. Wandas still owns validation, channel metadata objects, lineage-derived operation history view semantics, and operation execution.

Wandasは内部でxarrayをラベル付きストレージとフレーム状態の層として使い、波形解析APIとしての責務はWandasに残します。現在の移行段階では、xarrayがデータ、名前付き次元、選択された座標、フレーム単位の属性を担当し、Wandasが検証、チャンネルメタデータオブジェクト、操作履歴の意味づけ、操作の実行を担当します。

The storage record is in `docs/design/2026-06-11-xarray-migration-consolidation.md`;
the current state-update contract is in
`docs/design/2026-07-21-immutable-frame-state-updates.md`.

storage の設計記録は `docs/design/2026-06-11-xarray-migration-consolidation.md`、
現在の state 更新契約は `docs/design/2026-07-21-immutable-frame-state-updates.md` にあります。

Compatibility notes for this migration:

- Use `with_label()`, `with_metadata()`, `with_channel_extra()`, or atomic `with_annotations()` to return an annotation-updated Frame. Direct and nested mutation remains effective in v0.7 with `DeprecationWarning` and becomes read-only in v0.8.
- `frame.channels` is a sequence-like xarray-backed metadata view, not a `list`. Use `rename_channels()` for labels and `with_calibration()` with `ChannelCalibration.with_factor()`, `with_unit()`, or `with_ref()` for physical-domain state.
- `frame.channels.to_list()` returns a list snapshot of `ChannelMetadata` value objects when list semantics are needed.

この移行に関する互換性メモ:

- annotation 更新には `with_label()`、`with_metadata()`、`with_channel_extra()`、または atomic な `with_annotations()` を使います。直接・nested mutation は v0.7 では `DeprecationWarning` 付きで反映され、v0.8 で read-only になります。
- `frame.channels` は `list` ではなく xarray backed の sequence-like metadata view です。label は `rename_channels()`、physical-domain state は `with_calibration()` と `ChannelCalibration.with_factor()`、`with_unit()`、`with_ref()` で更新します。
- list semantics が必要な場合は、`frame.channels.to_list()` で `ChannelMetadata` value object の list snapshot を取得できます。

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

### Pipeline Recipe Documentation / Pipeline Recipe ドキュメント

- [Pipeline Recipe Design](pipeline-recipe-design.md): representation and replay design.
- [Pipeline Recipe Developer Guide](pipeline-recipe-developer-guide.md): contributor-oriented overview, terms, extension checklist, and testing guidance.
- [Scalability Contract](scalability-contract.md): exact lazy/channel-wise scaling promises and limits.
- [Public API and Schema Stability](public-api-stability.md): stable, experimental, optional, and serialization surfaces.

### Memory Efficiency / メモリ効率

Wandas ensures memory efficiency for handling large audio data through the following methods:
Wandasは大規模なオーディオデータを扱うために、以下の方法でメモリ効率を確保しています：

- **Lazy Evaluation**: A mechanism that delays calculations until needed (using Dask).
  **遅延評価**: 必要になるまで計算を遅延させる仕組み（Daskを使用）。
- **Dataset-first selection**: Discover and filter many recordings before loading waveform samples.
  **Dataset-first selection**: 多数の収録を探索し、波形 sample の読み込み前に絞り込み。

WDF 0.4 writes internal chunks without precomputing the complete tensor. Users keep the
normal Frame workflow and obtain NumPy values through `frame.data`; xarray, Dask, and
storage chunk topology remain implementation details. See the
[scalability contract](scalability-contract.md) for exact limits and the contributor
benchmark command.

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
