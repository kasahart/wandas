# Wandas

[English](README.md) | 日本語

![Wandas logo](https://github.com/kasahart/wandas/blob/main/images/logo.png?raw=true)

[![PyPI](https://img.shields.io/pypi/v/wandas)](https://pypi.org/project/wandas/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/wandas)](https://pypi.org/project/wandas/)
[![CI](https://github.com/kasahart/wandas/actions/workflows/ci.yml/badge.svg)](https://github.com/kasahart/wandas/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/kasahart/wandas/graph/badge.svg?token=53NPNQQZZ8)](https://codecov.io/gh/kasahart/wandas)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/kasahart/wandas/blob/main/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/wandas)](https://pypi.org/project/wandas/)

**信号解析を、データフレームを扱うように。**

Wandas は、波形・時系列データを `ChannelFrame` として扱う Python ライブラリです。サンプル列だけでなく、サンプリング周波数、チャンネル名、単位、メタデータ、処理履歴を一緒に持ったまま、確認、前処理、変換、可視化まで進められます。

`array`、`sampling_rate`、`channels`、処理メモを別々に持ち回る代わりに、読みやすい 1 本の解析チェーンとして書けます。

```python
import numpy as np
import wandas as wd

sr = 48_000
t = np.arange(sr) / sr
samples = np.vstack([
    np.sin(2 * np.pi * 440 * t),
    0.5 * np.sin(2 * np.pi * 880 * t),
]).astype(np.float32)

signal = wd.from_numpy(
    samples,
    sampling_rate=sr,
    label="demo tone",
    ch_labels=["440 Hz", "880 Hz"],
)

clean = signal.remove_dc().normalize()
clean.describe()

spectrum = clean.welch(n_fft=4096)
spectrogram = clean.stft(n_fft=1024)
```

`describe()` で、波形、スペクトル、スペクトログラムをまとめて素早く確認できます。

![Wandas describe output](https://github.com/kasahart/wandas/blob/main/images/read_wav_describe.png?raw=true)

## Wandas を試したくなるところ

- **フレーム指向の信号解析**: サンプリング周波数、長さ、チャンネル、ラベル、単位、メタデータを知っているオブジェクトとして扱えます。
- **生データから洞察までが短い**: 読み込み、トリミング、フィルタ、正規化、リサンプリング、要約、変換、プロットを一貫したメソッドでつなげられます。
- **時間・周波数・時間周波数を行き来できる**: `ChannelFrame` から `SpectralFrame`、`SpectrogramFrame`、`NOctFrame` へ、文脈を失わずに移れます。
- **実用的な音響解析も同じ流れで**: RMS トレンド、騒音レベル、A 特性、オクターブバンド、ラウドネス、粗さを必要に応じて扱えます。
- **実データで使いやすい**: WAV/FLAC/OGG/AIFF/SND/CSV、URL、bytes、file-like、NumPy 配列、録音フォルダ、`io` extra を使った Wandas WDF ファイルを扱えます。
- **探索に向いている**: `describe()`、Matplotlib と親和性の高いプロット、marimo 学習アプリで、まず見る・試すがすぐできます。

## インストール

最初に試すなら、インタラクティブ表示と学習アプリを含む extra 付きがおすすめです。

```bash
pip install "wandas[marimo]"
```

最小構成で入れる場合:

```bash
pip install wandas
```

core-only インストールでも、波形フレーム、CSV/WAV 読み込み、処理、プロット、`is_close=False` や `image_save` を使った `describe()` の図作成・保存ワークフローは利用できます。デフォルトのインタラクティブな `frame.describe()` 表示には `marimo` extra を使います。

必要な機能に応じて optional extras を組み合わせられます。

```bash
pip install "wandas[io]"              # WDF の保存・読み込み
pip install "wandas[effects]"         # librosa ベースのオーディオエフェクト
pip install "wandas[marimo]"          # marimo 学習アプリとインタラクティブ表示
pip install "wandas[psychoacoustic]"  # ラウドネス、粗さ、オクターブバンド補助機能
pip install "wandas[ml]"              # Torch/TensorFlow テンソル補助機能

pip install "wandas[marimo,io,effects,psychoacoustic]"
```

## 手元のデータで始める

### 録音を読み込んで確認する

```python
import wandas as wd

signal = wd.read("recording.wav", start=0, end=10, normalize=True)
signal.info()
signal.describe(fmin=20, fmax=8_000)
```

### 解析前に整える

```python
clean = (
    signal
    .remove_dc()
    .band_pass_filter(80, 8_000)
    .normalize()
)

clean.rms_plot(Aw=True)
```

### 周波数成分を見る

```python
spectrum = clean.welch(n_fft=2048)
spectrum.plot()

spectrogram = clean.stft(n_fft=2048, hop_length=512)
spectrogram.plot()

# オクターブバンド解析には psychoacoustic extra が必要です。
third_octave = clean.noct_spectrum(n=3)  # wandas[psychoacoustic] が必要
third_octave.plot()
```

### チャンネル比較や音響指標を見る

```python
# SPL として dB 表示する場合は、先に音圧校正を設定します。
for channel in signal.channels:
    channel.unit = "Pa"
    channel.ref = 20e-6

level = signal.sound_level(freq_weighting="A", time_weighting="Fast", dB=True)
level.plot(ylabel="LA Fast [dB re 20 uPa]")

# 心理音響指標には psychoacoustic extra が必要です。
loudness = signal.loudness_zwtv(field_type="free")
roughness = signal.roughness_dw(overlap=0.5)
```

## 小さな top-level API

```python
import numpy as np
import wandas as wd

signal = wd.read("audio.wav")          # WAV、CSV、対応音声、URL、bytes、file-like
saved = wd.load("analysis.wdf")        # Wandas ネイティブ WDF; wandas[io] が必要
data = np.zeros((2, 48_000), dtype=np.float32)
array_signal = wd.from_numpy(data, sampling_rate=48_000)
dataset = wd.from_folder("recordings/", recursive=True)
formats = wd.supported_formats()
```

既存コード向けに `read_wav()`、`read_csv()`、`from_ndarray()` は残っていますが、新しい例では `read()` と `from_numpy()` を使います。

## 主なオブジェクト

- `ChannelFrame`: チャンネルを持つ時間領域の波形・センサーデータ。
- `SpectralFrame`: FFT、Welch、コヒーレンス、CSD、伝達関数の結果。
- `SpectrogramFrame`: STFT などの時間周波数データ。
- `NOctFrame`: オクターブ、分数オクターブスペクトル。
- `ChannelFrameDataset`: フォルダ内の録音をまとめて扱うバッチ処理向けコレクション。

## 向いている用途

Wandas は、特に次のような場面で便利です。

- Notebook や marimo アプリで信号処理パイプラインを試作したい。
- フィルタや変換を試す間も、チャンネルメタデータを失いたくない。
- 音響録音をすばやく確認してから、詳細解析に進みたい。
- 複数の WAV/CSV ファイルを同じ API で比較したい。
- 信号処理の学習・説明用に読みやすいサンプルを作りたい。

## 次に読む

- [公式ドキュメント](https://kasahart.github.io/wandas/) - ガイド、API リファレンス、使用例。
- [学習パス](https://github.com/kasahart/wandas/tree/main/learning-path/) - marimo アプリベースのステップ別チュートリアル。
- [チュートリアル](https://kasahart.github.io/wandas/tutorial/) - 基本ワークフローを順に確認できます。
- [Issue Tracker](https://github.com/kasahart/wandas/issues) - バグ報告や機能提案。

## プロジェクトの状態

Wandas は現在も活発に改善中です。Python 3.10+ を対象にし、MIT License の下で公開されています。本番ワークフローで使う場合は、バージョンを固定し、アップグレード時にリリースノートを確認してください。

## 貢献

貢献を歓迎します。

開発環境セットアップ、品質チェック、ドキュメント規約、プルリクエスト手順は [docs/src/contributing.md](https://kasahart.github.io/wandas/contributing/) を参照してください。

## ライセンス

このプロジェクトは [MIT License](https://github.com/kasahart/wandas/blob/main/LICENSE) の下で公開されています。
