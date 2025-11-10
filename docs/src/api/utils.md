# ユーティリティモジュール

`wandas.utils` モジュールは、Wandasライブラリで使用される様々なユーティリティ機能を提供します。

## フレームデータセット

複数のデータフレームを管理するためのデータセットユーティリティを提供します。

### 概要

`FrameDataset` クラスは、フォルダ内の音声ファイルの効率的なバッチ処理を可能にします。主な機能：

- **遅延読み込み**: アクセス時のみファイルを読み込み、メモリ使用量を削減
- **変換のチェーン**: 複数の処理操作を効率的に適用
- **サンプリング**: テストや分析のためにランダムなサブセットを抽出
- **メタデータ追跡**: データセットのプロパティと処理履歴を記録

### 主なクラス

- **`ChannelFrameDataset`**: 時間領域の音声データ用（WAV、MP3、FLAC、CSVファイル）
- **`SpectrogramFrameDataset`**: 時間周波数領域データ用（通常はSTFTから作成）

### 基本的な使用方法

```python
from wandas.utils.frame_dataset import ChannelFrameDataset

# フォルダからデータセットを作成
dataset = ChannelFrameDataset.from_folder(
    folder_path="path/to/audio/files",
    sampling_rate=16000,  # オプション: すべてのファイルをこのレートにリサンプリング
    file_extensions=[".wav", ".mp3"],  # 含めるファイルタイプ
    recursive=True,  # サブディレクトリを検索
    lazy_loading=True  # オンデマンドでファイルを読み込む（推奨）
)

# 個別のファイルにアクセス
first_file = dataset[0]
print(f"ファイル: {first_file.label}")
print(f"長さ: {first_file.duration}秒")

# データセット情報を取得
metadata = dataset.get_metadata()
print(f"総ファイル数: {metadata['file_count']}")
print(f"読み込み済みファイル数: {metadata['loaded_count']}")
```

### サンプリング

テストや分析のためにデータセットのランダムなサブセットを抽出：

```python
# ファイル数でサンプリング
sampled = dataset.sample(n=10, seed=42)

# 比率でサンプリング
sampled = dataset.sample(ratio=0.1, seed=42)

# デフォルト: 10%または最小1ファイル
sampled = dataset.sample(seed=42)
```

### 変換

データセット内のすべてのファイルに処理操作を適用：

```python
# 組み込み変換
resampled = dataset.resample(target_sr=8000)
trimmed = dataset.trim(start=0.5, end=2.0)

# 複数の変換をチェーン
processed = (
    dataset
    .resample(target_sr=8000)
    .trim(start=0.5, end=2.0)
)

# カスタム変換
def custom_filter(frame):
    return frame.low_pass_filter(cutoff=1000)

filtered = dataset.apply(custom_filter)
```

### STFT - スペクトログラム生成

時間領域データをスペクトログラムに変換：

```python
# スペクトログラムデータセットを作成
spec_dataset = dataset.stft(
    n_fft=2048,
    hop_length=512,
    window="hann"
)

# スペクトログラムにアクセス
spec_frame = spec_dataset[0]
spec_frame.plot()
```

### 反復処理

データセット内のすべてのファイルを処理：

```python
for i in range(len(dataset)):
    frame = dataset[i]
    if frame is not None:
        # フレームを処理
        print(f"{frame.label} を処理中...")
```

### 主要なパラメータ

**folder_path** (str): 音声ファイルを含むフォルダへのパス

**sampling_rate** (Optional[int]): 目標サンプリングレート。このレートと異なる場合、ファイルはリサンプリングされます

**file_extensions** (Optional[list[str]]): 含めるファイル拡張子のリスト。デフォルト: `[".wav", ".mp3", ".flac", ".csv"]`

**lazy_loading** (bool): True の場合、アクセス時のみファイルを読み込む。デフォルト: True

**recursive** (bool): True の場合、サブディレクトリを再帰的に検索。デフォルト: False

### 使用例

詳細な例については、[FrameDataset 使用ガイド](../../examples/03_frame_dataset_usage.ipynb) ノートブックを参照してください。

### APIリファレンス

::: wandas.utils.frame_dataset

## サンプル生成

テスト用のサンプルデータを生成する機能を提供します。

::: wandas.utils.generate_sample

## 型定義

Wandasで使用される型定義を提供します。

::: wandas.utils.types

## 一般的なユーティリティ

その他の一般的なユーティリティ関数を提供します。

::: wandas.utils.util
