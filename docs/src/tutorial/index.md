# チュートリアル

このチュートリアルでは、Wandasライブラリの基本的な使い方を5分で学べます。

## インストール

```bash
pip install git+https://github.com/endolith/waveform-analysis.git@master
pip install wandas
```

## 基本的な使い方

### 1. ライブラリのインポート

```python exec="on" session="wd_demo"
from io import StringIO
import matplotlib.pyplot as plt
```

```python exec="on" source="above" session="wd_demo"
import wandas as wd

```

### 2. 音声ファイルの読み込み

```python
# URLからデータを取得
url = "https://github.com/kasahart/wandas/raw/main/examples/data/summer_streets1.wav"

audio = wd.read_wav(url)
print(f"サンプリングレート: {audio.sampling_rate} Hz")
print(f"チャンネル数: {audio.n_channels}")
print(f"長さ: {audio.duration} s")

```

```python exec="on" session="wd_demo"
# URLからデータを取得
url = "https://github.com/kasahart/wandas/raw/main/examples/data/summer_streets1.wav"

audio = wd.read_wav(url)
print(f"サンプリングレート: {audio.sampling_rate} Hz  ")
print(f"チャンネル数: {audio.n_channels}  ")
print(f"長さ: {audio.duration} s  ")

```

### 3. 信号の可視化

```python
# 波形を表示
audio.describe()
```

```python exec="on" html="true" session="wd_demo"
audio.describe(is_close=False)
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())
```

<audio controls src="https://github.com/kasahart/wandas/raw/main/examples/data/summer_streets1.wav"></audio>

### 4. 基本的な信号処理

```python
# ローパスフィルタを適用（1kHz以下の周波数を通過）
filtered = audio.low_pass_filter(cutoff=1000)

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

## 次のステップ

- [APIリファレンス](../api/index.md) で詳細な機能を調べる
- [理論背景](../explanation/index.md) でライブラリの設計思想を理解する

## ユースケース別レシピ

このセクションでは、Wandasライブラリのより詳細な機能や応用例を、以下のチュートリアルノートブックを通じて学ぶことができます。

- [00_setup.ipynb: セットアップと基本的な設定](/tutorial/00_setup.ipynb)
- [01_io_basics.ipynb: ファイルの読み書きと基本的な操作](/tutorial/01_io_basics.ipynb)
- [02_signal_processing_basics.ipynb:基本的な信号処理](/tutorial/02_signal_processing_basics.ipynb)
- [03_visualization.ipynb: データの可視化](/tutorial/03_visualization.ipynb)
- [04_time_frequency.ipynb: 時間周波数分析](/tutorial/04_time_frequency.ipynb)
- [05_lazy_and_dask.ipynb: 遅延評価とDaskによる大規模データ処理](/tutorial/05_lazy_and_dask.ipynb)
- [06_metadata_history.ipynb: メタデータと処理履歴の活用](/tutorial/06_metadata_history.ipynb)
- [07_batch_processing.ipynb: 複数ファイルへの一括処理](/tutorial/07_batch_processing.ipynb)
- [08_frame_dataset_usage.ipynb: FrameDataset 使用ガイド - フォルダ内の複数音声ファイルの効率的な処理](../../examples/03_frame_dataset_usage.ipynb)
- [09_extending_api.ipynb: カスタム関数の追加とAPIの拡張](/tutorial/08_extending_api.ipynb)
- [10_interoperability.ipynb: 他のライブラリとの連携](/tutorial/08_interoperability.ipynb)
- [11_case_studies.ipynb: 実践的なユースケーススタディ](/tutorial/09_case_studies.ipynb)

!!! tip "ヒント"
    各ノートブックは特定のトピックに焦点を当てています。興味のあるものから順に、または必要に応じて参照してください。Wandasの基本的な使い方については、このチュートリアルの冒頭部分も合わせてご覧ください。
