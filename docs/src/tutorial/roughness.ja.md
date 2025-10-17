# ラフネス計算

ラフネス（粗さ）は、音の知覚される粗さを定量化する心理音響メトリクスです。15-300 Hz範囲の急速な振幅変調に関連しています。

## 概要

wandasライブラリは、MoSQITo (Modular Sound Quality Indicators Tools) ライブラリを介して実装されたDaniel & Weber法を使用したラフネス計算を提供します。

**主な機能:**
- `roughness_dw_time`を使用した時間領域計算
- `roughness_dw_freq`を使用した周波数領域計算
- モノラルおよびマルチチャンネル信号のサポート
- 時間領域法用の設定可能なオーバーラップパラメータ

## 使用方法

### 基本的な使用方法

```python
import wandas as wd

# オーディオファイルを読み込む
signal = wd.read_wav("audio.wav")

# 時間領域法を使用してラフネスを計算（デフォルト）
roughness = signal.roughness(method='time')
print(f"ラフネス: {roughness:.3f} asper")
```

### メソッドの選択

2つの計算方法が利用可能です：

```python
# 時間領域計算（デフォルト）
roughness_time = signal.roughness(method='time')

# 周波数領域計算
roughness_freq = signal.roughness(method='freq')
```

### オーバーラップパラメータ

時間領域法では、オーバーラップ比率を指定できます：

```python
# オーバーラップなし（デフォルト）
roughness_no_overlap = signal.roughness(method='time', overlap=0.0)

# 50%オーバーラップで時間平均化を滑らかに
roughness_with_overlap = signal.roughness(method='time', overlap=0.5)
```

### マルチチャンネル信号

ステレオまたはマルチチャンネル信号の場合、ラフネスは各チャンネルで独立して計算されます：

```python
# ステレオ信号
stereo_signal = wd.read_wav("stereo_audio.wav")
roughness = stereo_signal.roughness(method='time')

# 結果はチャンネルごとに1つの値を持つ配列
print(f"左チャンネル:  {roughness[0]:.3f} asper")
print(f"右チャンネル: {roughness[1]:.3f} asper")
```

## ラフネス値の理解

### 単位: Asper（アスパー）

ラフネスは **asper** で測定されます。基準は以下の通りです：
- **1 asper** = 60 dB SPLの1 kHzトーン、70 Hzで100%振幅変調されたもののラフネス

### 典型的な値

- **純音**: < 0.5 asper（非常に滑らか）
- **変調音（70 Hz）**: ピークラフネス（通常1-3 asper）
- **変調音（100 Hz）**: 中程度のラフネス
- **複雑な信号**: 変動あり、通常0-10 asper

### 知覚特性

ラフネスの知覚は、約70 Hz付近の振幅変調で最も高く、それより低い周波数でも高い周波数でも減少します：

- **< 15 Hz**: 個別のパルスとして知覚され、ラフネスとしては認識されない
- **15-70 Hz**: ラフネスの増加
- **70 Hz**: ピークラフネス知覚
- **70-300 Hz**: ラフネスの減少
- **> 300 Hz**: 最小限のラフネス知覚

## 例: 異なる信号の比較

```python
import numpy as np
import wandas as wd

# パラメータ
sample_rate = 44100
duration = 1.0
t = np.linspace(0, duration, int(sample_rate * duration))

# 純音（低ラフネス）
pure_tone = np.sin(2 * np.pi * 1000 * t)
signal_pure = wd.ChannelFrame(data=pure_tone, sampling_rate=sample_rate)
roughness_pure = signal_pure.roughness(method='time')
print(f"純音のラフネス: {roughness_pure:.3f} asper")

# 70 Hzで変調された音（高ラフネス）
carrier = np.sin(2 * np.pi * 1000 * t)
modulator = 0.5 * (1 + np.sin(2 * np.pi * 70 * t))
modulated = carrier * modulator
signal_mod = wd.ChannelFrame(data=modulated, sampling_rate=sample_rate)
roughness_mod = signal_mod.roughness(method='time')
print(f"変調音のラフネス: {roughness_mod:.3f} asper")

# 期待される結果: roughness_mod >> roughness_pure
```

## 技術的詳細

### Daniel & Weber法

この実装はDaniel & Weber法を使用しており、以下に基づいています：
- 臨界帯域（Barkスケール）へのバンドパスフィルタリング
- 各帯域でのエンベロープ抽出
- 変調深度解析
- 周波数帯域にわたる統合

### 参考文献

1. Daniel, P., & Weber, R. (1997). Psychoacoustical roughness: implementation of an optimized model. *Acustica*, 83, 113-123.

2. ECMA-418-2:2022 - Psychoacoustic metrics for ITT equipment - Part 2: Models based on human perception

3. MoSQIToドキュメント: [https://mosqito.readthedocs.io](https://mosqito.readthedocs.io)

## APIリファレンス

詳細なパラメータの説明と戻り値の型については、[APIドキュメント](../api/processing.md#心理音響メトリクス)を参照してください。
