# Utilities Module / ユーティリティモジュール

The `wandas.utils` module provides various utility functions used in the Wandas library.
`wandas.utils` モジュールは、Wandasライブラリで使用される様々なユーティリティ機能を提供します。

## Frame Dataset / フレームデータセット

Provides dataset utilities for managing multiple data frames.
複数のデータフレームを管理するためのデータセットユーティリティを提供します。

### Overview / 概要

The `FrameDataset` classes enable efficient batch processing of audio files in a folder. Key features include:
`FrameDataset` クラスは、フォルダ内の音声ファイルの効率的なバッチ処理を可能にします。主な機能：

- **Lazy Loading**: Load files only when accessed, reducing memory usage.
  **遅延読み込み**: アクセス時のみファイルを読み込み、メモリ使用量を削減。
- **Transformation Chaining**: Apply multiple processing operations efficiently.
  **変換のチェーン**: 複数の処理操作を効率的に適用。
- **Sampling**: Extract random subsets for testing or analysis.
  **サンプリング**: テストや分析のためにランダムなサブセットを抽出。
- **Metadata Tracking**: Keep track of dataset properties and processing history.
  **メタデータ追跡**: データセットのプロパティと処理履歴を記録。

### Main Classes / 主なクラス

- **`ChannelFrameDataset`**: For time-domain audio data (WAV, MP3, FLAC, CSV files).
  **`ChannelFrameDataset`**: 時間領域の音声データ用（WAV、MP3、FLAC、CSVファイル）。
- **`SpectrogramFrameDataset`**: For time-frequency domain data (typically created from STFT).
  **`SpectrogramFrameDataset`**: 時間周波数領域データ用（通常はSTFTから作成）。

### Basic Usage / 基本的な使用方法

```python
from wandas.utils.frame_dataset import ChannelFrameDataset

# Create a dataset from a folder
# フォルダからデータセットを作成
dataset = ChannelFrameDataset.from_folder(
    folder_path="path/to/audio/files",
    sampling_rate=16000,  # Optional: resample all files to this rate / オプション: すべてのファイルをこのレートにリサンプリング
    file_extensions=[".wav", ".mp3"],  # File types to include / 含めるファイルタイプ
    recursive=True,  # Search subdirectories / サブディレクトリを検索
    lazy_loading=True  # Load files on demand (recommended) / オンデマンドでファイルを読み込む（推奨）
)

# Access individual files
# 個別のファイルにアクセス
first_file = dataset[0]
print(f"File: {first_file.label}")
print(f"Duration: {first_file.duration}s")

# Get dataset information
# データセット情報を取得
metadata = dataset.get_metadata()
print(f"Total files: {metadata['file_count']}")
print(f"Loaded files: {metadata['loaded_count']}")
```

### Sampling / サンプリング

Extract random subsets of the dataset for testing or analysis:
テストや分析のためにデータセットのランダムなサブセットを抽出：

```python
# Sample by number of files
# ファイル数でサンプリング
sampled = dataset.sample(n=10, seed=42)

# Sample by ratio
# 比率でサンプリング
sampled = dataset.sample(ratio=0.1, seed=42)

# Default: 10% or minimum 1 file
# デフォルト: 10% または最低1ファイル
sampled = dataset.sample(seed=42)
```

### Transformations / 変換

Apply processing operations to all files in the dataset:
データセット内のすべてのファイルに処理操作を適用：

```python
# Built-in transformations
# 組み込みの変換
resampled = dataset.resample(target_sr=8000)
trimmed = dataset.trim(start=0.5, end=2.0)

# Chain multiple transformations
# 複数の変換をチェーン
processed = (
    dataset
    .resample(target_sr=8000)
    .trim(start=0.5, end=2.0)
)

# Custom transformation
# カスタム変換
def custom_filter(frame):
    return frame.low_pass_filter(cutoff=1000)

filtered = dataset.apply(custom_filter)
```

### STFT - Spectrogram Generation / STFT - スペクトログラム生成

Convert time-domain data to spectrograms:
時間領域データをスペクトログラムに変換：

```python
# Create spectrogram dataset
# スペクトログラムデータセットを作成
spec_dataset = dataset.stft(
    n_fft=2048,
    hop_length=512,
    window="hann"
)

# Access a spectrogram
# スペクトログラムにアクセス
spec_frame = spec_dataset[0]
spec_frame.plot()
```

### Iteration / 反復処理

Process all files in the dataset:
データセット内のすべてのファイルを処理：

```python
for i in range(len(dataset)):
    frame = dataset[i]
    if frame is not None:
        # Process the frame
        # フレームを処理
        print(f"Processing {frame.label}...")
```

### Key Parameters / 主なパラメータ

**folder_path** (str): Path to the folder containing audio files.
音声ファイルを含むフォルダへのパス。

**sampling_rate** (Optional[int]): Target sampling rate. Files will be resampled if different from this rate.
ターゲットサンプリングレート。このレートと異なる場合、ファイルはリサンプリングされます。

**file_extensions** (Optional[list[str]]): List of file extensions to include. Default: `[".wav", ".mp3", ".flac", ".csv"]`.
含めるファイル拡張子のリスト。デフォルト: `[".wav", ".mp3", ".flac", ".csv"]`。

**lazy_loading** (bool): If True, files are loaded only when accessed. Default: True.
Trueの場合、ファイルはアクセス時にのみ読み込まれます。デフォルト: True。

**recursive** (bool): If True, search subdirectories recursively. Default: False.
Trueの場合、サブディレクトリを再帰的に検索します。デフォルト: False。

### Examples / 使用例

For detailed examples, see the [FrameDataset Usage Guide](../../examples/03_frame_dataset_usage.ipynb) notebook.
詳細な例については、[FrameDataset Usage Guide](../../examples/03_frame_dataset_usage.ipynb) ノートブックを参照してください。

### API Reference / APIリファレンス

::: wandas.utils.frame_dataset

## Sample Generation / サンプル生成

Provides functions for generating sample data for testing.
テスト用のサンプルデータを生成する機能を提供します。

::: wandas.utils.generate_sample

## Type Definitions / 型定義

Provides type definitions used in Wandas.
Wandasで使用される型定義を提供します。

::: wandas.utils.types

## General Utilities / 一般ユーティリティ

Provides other general utility functions.
その他の一般的なユーティリティ機能を提供します。

::: wandas.utils.util
