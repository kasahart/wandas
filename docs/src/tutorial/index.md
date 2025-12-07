# Tutorial / チュートリアル

This tutorial will teach you the basics of the Wandas library in 5 minutes.
このチュートリアルでは、Wandasライブラリの基本的な使い方を5分で学べます。

## Installation / インストール

```bash
pip install git+https://github.com/endolith/waveform-analysis.git@master
pip install wandas
```

## Basic Usage / 基本的な使い方

### 1. Import the Library / ライブラリのインポート

```python exec="on" session="wd_demo"
from io import StringIO
import matplotlib.pyplot as plt
```

```python exec="on" source="above" session="wd_demo"
import wandas as wd
```

### 2. Load Audio Files / 音声ファイルの読み込み

```python
# Load a WAV file / WAVファイルを読み込む
url = "https://github.com/kasahart/wandas/raw/main/examples/data/summer_streets1.wav"

audio = wd.read_wav(url)
print(f"Sampling rate / サンプリングレート: {audio.sampling_rate} Hz")
print(f"Number of channels / チャンネル数: {audio.n_channels}")
print(f"Duration / 長さ: {audio.duration} s")
```

```python exec="on" session="wd_demo"
# Load a WAV file / WAVファイルを読み込む
url = "https://github.com/kasahart/wandas/raw/main/examples/data/summer_streets1.wav"

audio = wd.read_wav(url)
print(f"Sampling rate / サンプリングレート: {audio.sampling_rate} Hz  ")
print(f"Number of channels / チャンネル数: {audio.n_channels}  ")
print(f"Duration / 長さ: {audio.duration} s  ")
```

### 3. Visualize Signals / 信号の可視化

```python
# Display waveform / 波形を表示
audio.describe()
```

```python exec="on" html="true" session="wd_demo"
audio.describe(is_close=False)
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())
```

<audio controls src="https://github.com/kasahart/wandas/raw/main/examples/data/summer_streets1.wav"></audio>

### 4. Basic Signal Processing / 基本的な信号処理

```python
# Apply a low-pass filter (passing frequencies below 1kHz)
# ローパスフィルタを適用（1kHz以下の周波数を通過）
filtered = audio.low_pass_filter(cutoff=1000)

# Visualize and compare results
# 結果を可視化して比較
filtered.previous.plot(title="Original")
filtered.plot(title="filtered")
```

```python exec="on" html="true" session="wd_demo"
filtered = audio.low_pass_filter(cutoff=1000)
filtered.previous.plot(title="Original")
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())

filtered.plot(title="filtered")
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())
```

## Next Steps / 次のステップ

- [API Reference / APIリファレンス](../api/index.md)
  - Detailed API specifications.
  - 詳細な機能やAPI仕様を調べる。
- [Theory Background / 理論背景](../explanation/index.md)
  - Design philosophy and algorithm explanations.
  - ライブラリの設計思想やアルゴリズムを理解する。

## Recipes by Use Case / ユースケース別レシピ

This section provides links to tutorial notebooks that demonstrate more detailed features and application examples of the Wandas library.
このセクションでは、Wandasライブラリのより詳細な機能や応用例を、以下のチュートリアルノートブックを通じて学ぶことができます。

- [00_setup.ipynb: Setup and basic configuration / セットアップと基本的な設定](/tutorial/00_setup.ipynb)
- [01_io_basics.ipynb: File reading/writing and basic operations / ファイルの読み書きと基本的な操作](/tutorial/01_io_basics.ipynb)
- [02_signal_processing_basics.ipynb: Basic signal processing / 基本的な信号処理](/tutorial/02_signal_processing_basics.ipynb)
