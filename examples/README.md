# Wandas Examples / Wandas サンプル集

This directory contains example notebooks and scripts demonstrating various features of the Wandas library.

このディレクトリには、Wandasライブラリのさまざまな機能を示すサンプルノートブックとスクリプトが含まれています。

## Available Examples / 利用可能なサンプル

### Basic Usage / 基本的な使い方

- **[00_file_read.ipynb](./00_file_read.ipynb)** - File reading and basic operations / ファイルの読み込みと基本操作
- **[01_basic_usage.ipynb](./01_basic_usage.ipynb)** - Basic usage examples / 基本的な使用例
- **[02_signal_processing.ipynb](./02_signal_processing.ipynb)** - Signal processing examples / 信号処理の例

### Dataset Processing / データセット処理

- **[03_frame_dataset_usage.ipynb](./03_frame_dataset_usage.ipynb)** - **FrameDataset usage guide** / **FrameDataset 使用ガイド**
  - Comprehensive guide for batch processing multiple audio files
  - 複数の音声ファイルのバッチ処理に関する包括的なガイド
  - Topics covered / カバーされるトピック:
    - Basic initialization / 基本的な初期化
    - File access and iteration / ファイルへのアクセスと反復処理
    - Sampling / サンプリング
    - Transformations and chaining / 変換とチェーン
    - STFT and spectrogram generation / STFT とスペクトログラム生成
    - Metadata retrieval / メタデータの取得
    - Best practices / ベストプラクティス

- **[FolderFrame.ipynb](./FolderFrame.ipynb)** - Legacy example for folder-based processing / フォルダベース処理の従来の例

### Advanced Topics / 高度なトピック

- **[calculation.ipynb](./calculation.ipynb)** - Calculation examples / 計算の例
- **[matrix_usage.ipynb](./matrix_usage.ipynb)** - Matrix operations / 行列操作
- **[compare_signal_usage.ipynb](./compare_signal_usage.ipynb)** - Signal comparison / 信号の比較

## Getting Started / はじめに

### Prerequisites / 前提条件

Make sure you have Wandas installed:

Wandas がインストールされていることを確認してください：

```bash
pip install git+https://github.com/endolith/waveform-analysis.git@master
pip install wandas
```

### Running the Examples / サンプルの実行

1. Clone the repository / リポジトリをクローン:
   ```bash
   git clone https://github.com/kasahart/wandas.git
   cd wandas/examples
   ```

2. Start Jupyter Notebook / Jupyter Notebook を起動:
   ```bash
   jupyter notebook
   ```

3. Open any example notebook and run the cells / 任意のサンプルノートブックを開いてセルを実行

## Quick Reference: FrameDataset / クイックリファレンス: FrameDataset

For users looking to process multiple audio files efficiently, here's a quick start with `FrameDataset`:

複数の音声ファイルを効率的に処理したいユーザー向けに、`FrameDataset` のクイックスタート：

```python
from wandas.utils.frame_dataset import ChannelFrameDataset

# Create a dataset from a folder
# フォルダからデータセットを作成
dataset = ChannelFrameDataset.from_folder(
    folder_path="path/to/audio/files",
    recursive=True,  # Search subdirectories / サブディレクトリを検索
    lazy_loading=True  # Load files on demand / オンデマンドでファイルを読み込む
)

# Access individual files
# 個別のファイルにアクセス
first_file = dataset[0]
print(f"File: {first_file.label}, Duration: {first_file.duration}s")

# Sample a subset
# サブセットをサンプリング
sampled = dataset.sample(n=10, seed=42)

# Apply transformations
# 変換を適用
processed = (
    dataset
    .resample(target_sr=8000)
    .normalize()
    .trim(start=0.5, end=2.0)
)

# Create spectrograms
# スペクトログラムを作成
spectrograms = dataset.stft(n_fft=2048, hop_length=512)
```

For detailed examples and explanations, see **[03_frame_dataset_usage.ipynb](./03_frame_dataset_usage.ipynb)**.

詳細な例と説明については、**[03_frame_dataset_usage.ipynb](./03_frame_dataset_usage.ipynb)** を参照してください。

## Documentation / ドキュメント

For comprehensive documentation, visit:

包括的なドキュメントについては、以下をご覧ください：

- **API Reference / APIリファレンス**: [https://kasahart.github.io/wandas/api/utils/](https://kasahart.github.io/wandas/api/utils/)
- **Tutorial / チュートリアル**: [https://kasahart.github.io/wandas/tutorial/](https://kasahart.github.io/wandas/tutorial/)
- **GitHub Repository / GitHub リポジトリ**: [https://github.com/kasahart/wandas](https://github.com/kasahart/wandas)

## Contributing / 貢献

If you have suggestions for additional examples or improvements, please:

追加のサンプルや改善の提案がある場合は、以下をご利用ください：

- Open an issue / Issue を開く: [https://github.com/kasahart/wandas/issues](https://github.com/kasahart/wandas/issues)
- Submit a pull request / プルリクエストを送信
- Join the discussion / ディスカッションに参加: [https://github.com/kasahart/wandas/discussions](https://github.com/kasahart/wandas/discussions)

## License / ライセンス

These examples are part of the Wandas project and are released under the [MIT License](https://opensource.org/licenses/MIT).

これらのサンプルは Wandas プロジェクトの一部であり、[MIT ライセンス](https://opensource.org/licenses/MIT)の下で公開されています。
