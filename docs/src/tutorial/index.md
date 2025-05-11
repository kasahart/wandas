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

- [クックブック](../how_to/index.md) で様々な応用例を確認する
- [APIリファレンス](../api/index.md) で詳細な機能を調べる
- [理論背景](../explanation/index.md) でライブラリの設計思想を理解する
