# Wandas: **W**aveform **An**alysis **Da**ta **S**tructures

[![PyPi](https://img.shields.io/pypi/v/wandas)](https://pypi.org/project/wandas/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/wandas)
[![CI](https://github.com/kasahart/wandas/actions/workflows/ci.yml/badge.svg)](https://github.com/kasahart/wandas/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/kasahart/wandas/graph/badge.svg?token=53NPNQQZZ8)](https://codecov.io/gh/kasahart/wandas)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/kasahart/wandas/blob/main/LICENSE)
[![Typing](https://img.shields.io/pypi/types/wandas)](https://pypi.org/project/wandas/)

**Wandas** は、Pythonによる効率的な信号解析のためのオープンソースライブラリです。Wandas は、信号処理のための包括的な機能を提供し、Matplotlibとのシームレスな統合を実現しています。

## 機能

- **包括的な信号処理機能**: フィルタリング、フーリエ変換、STFTなど、基本的な信号処理操作を簡単に実行可能
- **可視化ライブラリとの統合**: Matplotlibとシームレスに統合してデータを簡単に可視化可能。

## インストール

```bash
pip install wandas
```

## クイックスタート

```python
import wandas as wd

cf = wd.read_wav("data/summer_streets1.wav")
cf.describe()
```

![cf.describe](https://github.com/kasahart/wandas/blob/main/images/read_wav_describe.png?raw=true)

```python
cf.describe(
    axis_config={
        "time_plot": {"xlim": (0, 15), "ylim": (-30000, 30000)},
        "freq_plot": {"xlim": (60, 120), "ylim": (0, 16000)},
    },
    cbar_config={"vmin": 10, "vmax": 70},
)
```

![cf.describe](https://github.com/kasahart/wandas/blob/main/images/read_wav_describe_set_config.png?raw=true)

```python
cf = wd.read_csv("data/test_signals.csv", time_column="Time")
cf.plot(title="Plot of test_signals.csv using wandas", overlay=False)
```

![cf.plot](https://github.com/kasahart/wandas/blob/main/images/plot_csv_using_wandas.png?raw=true)

### 信号処理

```python
signal = wd.generate_sin(freqs=[5000, 1000], duration=1)
# ローパスフィルタを適用
signal.low_pass_filter(cutoff=1000).fft().plot()
```

![signal.low_pass_filter](https://github.com/kasahart/wandas/blob/main/images/low_pass_filter.png?raw=true)

```python
# フィルタ済み信号を WAV ファイルに保存
signal.low_pass_filter(cutoff=1000).to_wav('filtered_audio.wav')
# Audioコントロール表示
signal.to_audio()
```

## ドキュメント

詳細な使用方法は`/exsampls`を参照してください

## 対応データ形式

- **音声ファイル**: WAV
- **データファイル**: CSV

## バグ報告と機能リクエスト

- **バグ報告**: [Issue Tracker](https://github.com/kasahart/wandas/issues) に詳細を記載してください。
- **機能リクエスト**: 新機能や改善案があれば、気軽に Issue をオープンしてください。

## ライセンス

このプロジェクトは [MIT ライセンス](LICENSE) の下で公開されています。

---

Wandas を使って効率的な信号解析体験を！
