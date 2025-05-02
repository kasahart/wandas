# チュートリアル

このチュートリアルでは、Wandasライブラリの基本的な使い方を5分で学べます。

## インストール

```bash
pip install wandas
```

## 基本的な使い方

### 1. ライブラリのインポート

```python
import wandas as wd
import matplotlib.pyplot as plt
```

### 2. 音声ファイルの読み込み

```python
# WAVファイルを読み込む
audio = wd.io.read_wav("path/to/audio.wav")
print(f"サンプリングレート: {audio.sampling_rate} Hz")
print(f"チャンネル数: {len(audio)}")
```

### 3. 信号の可視化

```python
# 波形を表示
audio.plot()
plt.show()
```

### 4. 基本的な信号処理

```python
# ローパスフィルタを適用（1kHz以下の周波数を通過）
filtered = audio.filter(lowpass=1000)

# 結果を可視化して比較
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
audio.plot(ax=axes[0], title="オリジナル信号")
filtered.plot(ax=axes[1], title="フィルタ後の信号")
plt.tight_layout()
plt.show()
```

### 5. 処理結果の保存

```python
# 処理した信号をWAVファイルとして保存
wd.io.write_wav(filtered, "filtered_audio.wav")
```

## 次のステップ

- [クックブック](../how_to/index.md) で様々な応用例を確認する
- [APIリファレンス](../api/index.md) で詳細な機能を調べる
- [理論背景](../explanation/index.md) でライブラリの設計思想を理解する