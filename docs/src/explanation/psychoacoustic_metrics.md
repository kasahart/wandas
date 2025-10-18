# 心理音響メトリクス

Wandasは、人間の知覚に基づく音響信号を分析するための心理音響メトリクスを提供します。これらのメトリクスは、標準化された手法とMoSQIToライブラリを使用して計算されます。

## ラウドネス（非定常信号）

### 概要

`loudness_zwtv()` メソッドは、ISO 532-1:2017に従ったZwicker法を使用して、非定常信号の時間変化するラウドネスを計算します。この方法は、人間の知覚とよく相関する知覚ラウドネスの尺度を提供します。

### ラウドネスとは？

ラウドネスは **ソーン（sone）** で測定される知覚単位です：

- **1 sone** は40 phon（約40 dB SPLの1 kHz純音のラウドネス）のラウドネスレベルに相当
- **ソーンが2倍** になると、知覚ラウドネスも2倍になります
- 関係式：音Aのソーン値が音Bの2倍であれば、2倍の大きさに聞こえます

### 典型的なラウドネス値

| 環境/音源 | おおよそのラウドネス |
|---------|------------------|
| 静かな図書館 | ~0.5-1 sone |
| 静かな会話 | ~2-4 sones |
| 通常の会話 | ~4-8 sones |
| 賑やかなオフィス | ~8-16 sones |
| 大音量の音楽 | ~32+ sones |
| 非常に大きな騒音 | ~100+ sones |

### 使用方法

#### 基本的な使い方

```python
import wandas as wd

# 音声ファイルを読み込む
signal = wd.read_wav("audio.wav")

# ラウドネスを計算（自由音場）
loudness = signal.loudness_zwtv()

# 時間変化するラウドネスをプロット
loudness.plot(title="時間変化するラウドネス")
```

#### 音場タイプの選択

このメソッドは2種類の音場をサポートしています：

- **自由音場** (`field_type="free"`): 特定の方向から到来する音（例：リスナーの前方にあるスピーカー）
- **拡散音場** (`field_type="diffuse"`): 全方向から均一に到来する音（例：残響室）

```python
# 自由音場（デフォルト）
loudness_free = signal.loudness_zwtv(field_type="free")

# 拡散音場
loudness_diffuse = signal.loudness_zwtv(field_type="diffuse")
```

### メソッドシグネチャ

```python
def loudness_zwtv(self, field_type: str = "free") -> ChannelFrame:
    """
    Zwicker法を使用して時間変化するラウドネスを計算
    
    Parameters
    ----------
    field_type : str, default="free"
        音場のタイプ（'free' または 'diffuse'）
    
    Returns
    -------
    ChannelFrame
        ソーン単位の時間変化するラウドネス値
    """
```

### 出力

このメソッドは以下を含む `ChannelFrame` を返します：

- **時間変化するラウドネス値**（ソーン単位）
- **時間分解能**: 約2ms（0.002秒）
- **マルチチャンネル処理**: 各チャンネルが独立して処理されます

### 使用例

#### 例1: 基本的な使い方

```python
import wandas as wd
import numpy as np

# 音声ファイルを読み込む
signal = wd.read_wav("audio.wav")

# ラウドネスを計算（デフォルトは自由音場）
loudness = signal.loudness_zwtv()

# 時間変化するラウドネスをプロット
loudness.plot(title="時間変化するラウドネス（sone）")
```

#### 例2: テスト信号の生成

```python
import wandas as wd
import numpy as np

# 1 kHz 正弦波を中程度のレベルで生成
signal = wd.generate_sin(freqs=[1000], duration=2.0, sampling_rate=48000)

# 約70 dB SPLにスケーリング
signal = signal * 0.063

# ラウドネスを計算
loudness = signal.loudness_zwtv()

# 統計情報を表示
print(f"平均ラウドネス: {loudness.mean():.2f} sones")
print(f"最大ラウドネス: {loudness.max():.2f} sones")
print(f"最小ラウドネス: {loudness.min():.2f} sones")
```

#### 例3: 自由音場と拡散音場の比較

```python
import wandas as wd
import matplotlib.pyplot as plt

# 信号を読み込む
signal = wd.read_wav("audio.wav")

# 両方の音場タイプで計算
loudness_free = signal.loudness_zwtv(field_type="free")
loudness_diffuse = signal.loudness_zwtv(field_type="diffuse")

# 比較プロット
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
loudness_free.plot(ax=axes[0], title="自由音場のラウドネス")
loudness_diffuse.plot(ax=axes[1], title="拡散音場のラウドネス")
plt.tight_layout()
plt.show()
```

#### 例4: マルチチャンネル処理

```python
import wandas as wd

# ステレオ音声を読み込む
stereo_signal = wd.read_wav("stereo_audio.wav")

# ラウドネスを計算（各チャンネルが独立して処理される）
loudness = stereo_signal.loudness_zwtv()

# 個々のチャンネルにアクセス
left_loudness = loudness[0]
right_loudness = loudness[1]

# 両方のチャンネルをプロット
loudness.plot(overlay=True, title="ステレオラウドネス比較")
```

#### 例5: MoSQIToを直接使用

より詳細な出力（特定ラウドネス、バーク軸など）が必要な場合は、MoSQIToを直接使用できます：

```python
from mosqito.sq_metrics.loudness.loudness_zwtv import loudness_zwtv
import wandas as wd

signal = wd.read_wav("audio.wav")
data = signal.data[0]  # 最初のチャンネルを取得

# MoSQIToを直接呼び出す
N, N_spec, bark_axis, time_axis = loudness_zwtv(
    data, signal.sampling_rate, field_type="free"
)

print(f"ラウドネスの形状: {N.shape}")
print(f"特定ラウドネスの形状: {N_spec.shape}")
print(f"時間軸: {time_axis[:10]}...")  # 最初の10個の時間点
```

### 技術的詳細

#### アルゴリズム

この実装は、MoSQIToの `loudness_zwtv` 関数を使用しており、以下を実装しています：

1. **外耳伝達関数**: 外耳のフィルタリング効果をシミュレート
2. **中耳伝達関数**: 中耳の伝達をモデル化
3. **励起パターン**: 基底膜に沿った励起を計算
4. **特定ラウドネス**: 各臨界帯域でラウドネスを決定
5. **総ラウドネス**: すべての臨界帯域にわたって特定ラウドネスを積分

#### 時間分解能

ラウドネス計算は約2msの時間分解能で値を生成します。1秒の信号に対して、約500個のラウドネス値が期待できます。

#### 計算の複雑さ

- アルゴリズムは効率性のためブロック単位で信号を処理します
- 処理時間は信号の長さに対して線形にスケールします
- メモリ使用量は中程度（時間変化するラウドネス値を保存）

### 制限事項

1. **サンプリングレート**: 44.1 kHz以上のサンプリングレートで最良の結果が得られます
2. **信号レベル**: 可聴範囲内の信号（通常20-100 dB SPL）で正確です
3. **定常性の仮定**: 非定常信号用に設計されていますが、極端に急激な過渡現象は完全には捉えられない場合があります
4. **キャリブレーション**: 物理単位（Pa）への適切な信号キャリブレーションを前提としています

### 標準と参考文献

- **ISO 532-1:2017**: "Acoustics — Methods for calculating loudness — Part 1: Zwicker method"
- **Zwicker, E., & Fastl, H. (1999)**: Psychoacoustics: Facts and models (2nd ed.). Springer.
- **MoSQIToライブラリ**: https://mosqito.readthedocs.io/en/latest/

---

## ラウドネス（定常信号）

### 概要

`loudness_zwst()` メソッドは、ISO 532-1:2017に従ったZwicker法を使用して、定常信号のラウドネスを計算します。このメソッドは、ファンノイズ、定常的な機械音などの持続的な音の評価に適しています。

### 非定常ラウドネスとの違い

| 特性 | 時間変化（`loudness_zwtv`） | 定常（`loudness_zwst`） |
|------|---------------------------|----------------------|
| **対象信号** | 非定常（時間変化する音） | 定常（持続的な音） |
| **使用例** | 音声、音楽、過渡音 | ファンノイズ、定常機械音 |
| **出力** | 時系列のラウドネス値 | 単一のラウドネス値 |
| **出力形状** | (channels, time_samples) | (n_channels,) |
| **サンプリングレート** | ~500 Hzに更新 | 変更なし（単一値） |

### 使用方法

#### 基本的な使い方

```python
import wandas as wd

# 定常信号を読み込む（ファンノイズなど）
signal = wd.read_wav("fan_noise.wav")

# 定常ラウドネスを計算（自由音場）
loudness = signal.loudness_zwst()

# 結果を表示
print(f"定常ラウドネス: {loudness[0]:.2f} sones")
```

#### 音場タイプの選択

時間変化するラウドネスと同様に、2種類の音場をサポートしています：

```python
# 自由音場（デフォルト）
loudness_free = signal.loudness_zwst(field_type="free")

# 拡散音場
loudness_diffuse = signal.loudness_zwst(field_type="diffuse")

print(f"自由音場: {loudness_free[0]:.2f} sones")
print(f"拡散音場: {loudness_diffuse[0]:.2f} sones")
```

### メソッドシグネチャ

```python
def loudness_zwst(self, field_type: str = "free") -> NDArrayReal:
    """
    Zwicker法を使用して定常ラウドネスを計算
    
    Parameters
    ----------
    field_type : str, default="free"
        音場のタイプ（'free' または 'diffuse'）
    
    Returns
    -------
    NDArrayReal
        ソーン単位の定常ラウドネス値（各チャンネルごとに1つの値）
        形状: (n_channels,)
    """
```

### 出力

このメソッドは以下を含む `NDArrayReal` を返します：

- **単一のラウドネス値**（ソーン単位、各チャンネルごと）
- **出力形状**: (n_channels,) - 1D配列
- **マルチチャンネル処理**: 各チャンネルが独立して処理されます
- **NumPy互換**: 直接NumPy操作が可能（`loudness[0]`, `loudness.mean()`など）

### 使用例

#### 例1: ファンノイズの評価

```python
import wandas as wd

# ファンノイズを読み込む
fan_signal = wd.read_wav("fan_noise.wav")

# 定常ラウドネスを計算
loudness = fan_signal.loudness_zwst(field_type="free")

# 結果を表示
print(f"ファンノイズのラウドネス: {loudness[0]:.2f} sones")
```

#### 例2: 複数の定常音源の比較

```python
import wandas as wd

# 異なる定常音源を読み込む
fan1 = wd.read_wav("fan1.wav")
fan2 = wd.read_wav("fan2.wav")

# 定常ラウドネスを計算
loudness1 = fan1.loudness_zwst()
loudness2 = fan2.loudness_zwst()

# 比較
print(f"Fan 1: {loudness1[0]:.2f} sones")
print(f"Fan 2: {loudness2[0]:.2f} sones")

if loudness1[0] > loudness2[0]:
    print("Fan 1 is louder")
else:
    print("Fan 2 is louder")
```

#### 例3: ステレオ定常音源の処理

```python
import wandas as wd

# ステレオの定常音源を読み込む
stereo_signal = wd.read_wav("stereo_steady_noise.wav")

# 定常ラウドネスを計算（各チャンネル独立）
loudness = stereo_signal.loudness_zwst()

# 各チャンネルの結果を表示
print(f"左チャンネル: {loudness[0]:.2f} sones")
print(f"右チャンネル: {loudness[1]:.2f} sones")
```

#### 例4: 自由音場と拡散音場の比較

```python
import wandas as wd

# 定常信号を読み込む
signal = wd.read_wav("steady_noise.wav")

# 両方の音場タイプで計算
loudness_free = signal.loudness_zwst(field_type="free")
loudness_diffuse = signal.loudness_zwst(field_type="diffuse")

# 比較
print(f"自由音場: {loudness_free[0]:.2f} sones")
print(f"拡散音場: {loudness_diffuse[0]:.2f} sones")
print(f"差: {abs(loudness_free[0] - loudness_diffuse[0]):.2f} sones")
```

#### 例5: MoSQIToを直接使用

より詳細な出力が必要な場合は、MoSQIToを直接使用できます：

```python
from mosqito.sq_metrics.loudness.loudness_zwst import loudness_zwst
import wandas as wd

signal = wd.read_wav("steady_noise.wav")
data = signal.data[0]  # 最初のチャンネルを取得

# MoSQIToを直接呼び出す
N, N_spec, bark_axis = loudness_zwst(
    data, signal.sampling_rate, field_type="free"
)

print(f"ラウドネス: {N:.2f} sones")
print(f"特定ラウドネスの形状: {N_spec.shape}")
print(f"バーク軸: {bark_axis}")
```

### 技術的詳細

#### アルゴリズム

定常ラウドネスの計算は、時間変化するラウドネスと同じZwicker法に基づいていますが、単一の代表値を出力します：

1. **外耳伝達関数**: 外耳のフィルタリング効果をシミュレート
2. **中耳伝達関数**: 中耳の伝達をモデル化
3. **励起パターン**: 基底膜に沿った励起を計算
4. **特定ラウドネス**: 各臨界帯域でラウドネスを決定
5. **総ラウドネス**: すべての臨界帯域にわたって特定ラウドネスを積分

#### 計算の複雑さ

- 定常信号を想定しているため、時間変化するラウドネスよりも計算が簡略化されます
- 処理時間は信号の長さに依存しますが、単一の値のみを出力します
- メモリ使用量は小さい（単一のラウドネス値のみを保存）

### 制限事項

1. **サンプリングレート**: 44.1 kHz以上のサンプリングレートで最良の結果が得られます
2. **信号レベル**: 可聴範囲内の信号（通常20-100 dB SPL）で正確です
3. **定常性の仮定**: このメソッドは定常信号用に設計されています。時間変化する信号には `loudness_zwtv()` を使用してください
4. **キャリブレーション**: 物理単位（Pa）への適切な信号キャリブレーションを前提としています

### 標準と参考文献

- **ISO 532-1:2017**: "Acoustics — Methods for calculating loudness — Part 1: Zwicker method"
- **Zwicker, E., & Fastl, H. (1999)**: Psychoacoustics: Facts and models (2nd ed.). Springer.
- **MoSQIToライブラリ**: https://mosqito.readthedocs.io/en/latest/

---

### 関連する操作

- `loudness_zwtv()`: 時間変化するラウドネスを計算（非定常信号用）
- `loudness_zwst()`: 定常ラウドネスを計算（定常信号用）
- `a_weighting()`: A特性フィルタを適用（人間の聴覚を近似する周波数重み付け）
- `noct_spectrum()`: Nオクターブバンドスペクトルを計算
- `rms_trend()`: 時間に沿ったRMSトレンドを計算

### 参照

- [MoSQITo Documentation](https://mosqito.readthedocs.io/en/latest/)
- [ISO 532-1:2017 Standard](https://www.iso.org/standard/63077.html)
- [Psychoacoustics Fundamentals](https://en.wikipedia.org/wiki/Psychoacoustics)
