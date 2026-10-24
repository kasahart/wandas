# Wandas

[English](https://github.com/kasahart/wandas/blob/main/README.md) | 日本語

![Wandas logo](https://github.com/kasahart/wandas/blob/main/images/logo.png?raw=true)

[![PyPI](https://img.shields.io/pypi/v/wandas)](https://pypi.org/project/wandas/)
[![CI](https://github.com/kasahart/wandas/actions/workflows/ci.yml/badge.svg)](https://github.com/kasahart/wandas/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/kasahart/wandas/graph/badge.svg?token=53NPNQQZZ8)](https://codecov.io/gh/kasahart/wandas)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/kasahart/wandas/blob/main/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/wandas)](https://pypi.org/project/wandas/)

**信号解析を、データフレームを扱うように。**

Wandas は、音・振動・センサーの波形を、サンプリング周波数・チャンネル名・単位・処理履歴ごと 1 つの frame（`ChannelFrame`）として扱う Python ライブラリです。読み込みから前処理、周波数解析、可視化までをメソッドチェーンで書けます。

```python
import wandas as wd

x = wd.read("https://raw.githubusercontent.com/kasahart/wandas/main/learning-path/sample_audio.wav", end=15)  # first 15 s
x.describe()                                               # waveform, spectrogram and spectrum at a glance
x.remove_dc().high_pass_filter(cutoff=100).welch().plot()  # preprocess, then a Welch spectrum
```

![describe() の出力](https://raw.githubusercontent.com/kasahart/wandas/main/images/readme_sample_audio_describe_0.png)

## なぜ Wandas か

NumPy / SciPy だけで書くと、サンプリング周波数やチャンネル名、単位、「何をしたか」を自分で管理することになります。

```python
# NumPy / SciPy: you carry the array and the sampling rate yourself
import scipy.signal

a, fs = x.data, x.sampling_rate
sos = scipy.signal.butter(4, 100, "highpass", fs=fs, output="sos")
y_np = scipy.signal.sosfiltfilt(sos, a - a.mean(axis=-1, keepdims=True))

# Wandas: the same 4th-order zero-phase Butterworth
y = x.remove_dc().high_pass_filter(cutoff=100)
y.operation_history  # [{'operation': 'wandas.audio.remove_dc', ...}, ...]
```

- 各メソッドは元の frame を変更せず、新しい frame を返します。処理履歴は結果に残ります。
- 処理手順を保存して、別の録音に再適用できます（[`RecipePlan`](https://kasahart.github.io/wandas/how-to/pipeline-recipes/)）。

## できること

| 分類 | 主な機能 |
|------|----------|
| 読み込み | WAV / FLAC / OGG / MP3 / CSV、URL、NumPy 配列 |
| 前処理 | DC 除去、ハイ／ロー／バンドパス、A 特性、リサンプリング、トリミング |
| レベル | チャンネルごとの校正、RMS、dB SPL |
| 周波数解析 | FFT、Welch、STFT、1/N オクターブバンド、コヒーレンス、伝達関数 |
| 心理音響 | ラウドネス、ラフネス、シャープネス（MoSQITo を利用） |
| 多数ファイル | フォルダや CSV から録音一覧を作り、まとめて処理 |

## インストール

```bash
pip install wandas
pip install "wandas[marimo]"          # Jupyter / marimo で describe() の音声再生を使う場合
pip install "wandas[psychoacoustic]"  # 心理音響指標を使う場合
```

## 答えがわかっている信号で確かめる

750 Hz と 1500 Hz の正弦波に DC を加え、DC 除去と 1 kHz ローパスをかけて比べます。

```python
import numpy as np
import wandas as wd

sr = 48_000
t = np.arange(sr) / sr
samples = 0.25 + 0.20 * np.sin(2 * np.pi * 750 * t) + 0.05 * np.sin(2 * np.pi * 1500 * t)

signal = wd.from_numpy(samples, sampling_rate=sr, ch_labels=["Original"], ch_units="Pa")
processed = signal.remove_dc().low_pass_filter(cutoff=1_000).rename_channels({0: "After DC removal + 1 kHz low-pass"})

comparison = signal.concat_frame(processed)
comparison.plot(overlay=True, xlim=(0, 0.02))
comparison.fft().plot(overlay=True, xlim=(0, 4_000)).set_ylim(30, 90)  # amplitude level, dB re 20 µPa
```

![波形の重ね描き](https://raw.githubusercontent.com/kasahart/wandas/main/images/readme_known_signal_waveform.png)

![FFT の重ね描き](https://raw.githubusercontent.com/kasahart/wandas/main/images/readme_known_signal_spectrum.png)

750 Hz は残り、1500 Hz は減衰しています。

## 多数のファイルを扱う

```python
import wandas as wd

# recordings/
# ├── label=ok/  a.wav, b.wav, ...
# └── label=ng/  c.wav, d.wav, ...
dataset = wd.from_folder("recordings/", recursive=True, path_metadata=True)
ok = dataset.select(label="ok")
spectra = ok.trim(0, 5).resample(16_000).stft(n_fft=512)  # first 5 s
spectra[0].data  # NumPy array (channel, frequency, time); mono has no channel axis
```

`from_folder()` と `select()` の時点では音声ファイルを開かないので、ファイル数が多くても選択はすぐ終わります。読み込みと計算は、`spectra[0]` のように結果を取り出したときに 1 本ずつ行われます。

## もっと学ぶ

- [ドキュメント](https://kasahart.github.io/wandas/)
- [日本語 Learning Path](https://kasahart.github.io/wandas/learning-path/00_why_wandas.html)

## ライセンス

[MIT License](https://github.com/kasahart/wandas/blob/main/LICENSE)。バグ報告や提案は [Issue](https://github.com/kasahart/wandas/issues) へどうぞ。
