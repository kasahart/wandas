# Wandas

![Wandas logo](https://github.com/kasahart/wandas/blob/main/images/logo.png?raw=true)

[![PyPI](https://img.shields.io/pypi/v/wandas)](https://pypi.org/project/wandas/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/wandas)](https://pypi.org/project/wandas/)
[![CI](https://github.com/kasahart/wandas/actions/workflows/ci.yml/badge.svg)](https://github.com/kasahart/wandas/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/kasahart/wandas/graph/badge.svg?token=53NPNQQZZ8)](https://codecov.io/gh/kasahart/wandas)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/kasahart/wandas/blob/main/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/wandas)](https://pypi.org/project/wandas/)

Data structures for waveform analysis.
Python で波形・信号データを扱うためのデータ構造ライブラリです。

Wandas brings pandas-like workflows to time-domain, spectral, and spectrogram analysis.
WAV や CSV を読み込み、メタデータを保ちながら、可視化や周波数解析まで一貫して進められます。

## Hero / ヒーロー

Wandas is an open-source Python library for signal and waveform analysis with chainable, frame-based APIs.
Wandas は、メソッドチェーンしやすいフレーム指向 API で信号解析を進められる、オープンソースの Python ライブラリです。

It helps you move from raw data to inspection, filtering, spectral analysis, and plotting without losing context such as sampling rate, channel labels, and metadata.
サンプリング周波数、チャネル名、付随メタデータを保ちながら、読み込みから確認、フィルタリング、周波数解析、可視化までつなげられます。

## Why Wandas / なぜ Wandas か

- Work with waveform data using familiar, pandas-like objects instead of ad hoc NumPy arrays.
  NumPy 配列を都度組み合わせる代わりに、pandas ライクなオブジェクトで波形データを扱えます。
- Keep metadata, channel information, and operation history attached as analysis grows.
  解析が進んでも、メタデータ、チャネル情報、処理履歴を一緒に保てます。
- Move smoothly between time-domain, spectral, and spectrogram views with a consistent API.
  時間領域、周波数領域、スペクトログラムを一貫した API で行き来できます。
- Use built-in plotting and summary helpers to inspect signals quickly.
  組み込みの可視化と要約機能で、信号をすぐに確認できます。
- Scale to larger data with Dask-backed lazy execution where available.
  必要に応じて Dask ベースの遅延実行で大きなデータにも対応できます。

## Quick Start / クイックスタート

Install from PyPI:
PyPI からインストールします。

```bash
pip install wandas
```

Then load a WAV file and inspect it in one short path:
次に、WAV ファイルを読み込んで、そのまま確認できます。

```python
import wandas as wd

# Load a WAV file and inspect it.
# WAV ファイルを読み込んで確認する。
signal = wd.read_wav("audio.wav")
signal.describe()
```

`describe()` gives you a quick visual summary of the waveform, spectrum, and spectrogram.
`describe()` で、波形、スペクトル、スペクトログラムをまとめて素早く確認できます。

![cf.describe](https://github.com/kasahart/wandas/blob/main/images/read_wav_describe.png?raw=true)

## What You Can Do / できること

- Read waveform and sensor data from formats such as WAV, CSV, and WDF.
  WAV、CSV、WDF などから波形やセンサーデータを読み込めます。
- Filter, resample, normalize, and summarize signals with method chaining.
  フィルタ、リサンプリング、正規化、要約をメソッドチェーンで進められます。
- Run FFT, STFT, Welch, coherence, transfer-function, and octave-style analyses.
  FFT、STFT、Welch、コヒーレンス、伝達関数、オクターブ系解析を実行できます。
- Compute psychoacoustic metrics such as loudness and roughness.
  ラウドネスや粗さなどの心理音響指標を扱えます。
- Plot waveforms, spectra, and spectrograms directly with Matplotlib-friendly APIs.
  波形、スペクトル、スペクトログラムを Matplotlib と親和性の高い API で描画できます。

## Learn More / 次に読む

- [Documentation](https://kasahart.github.io/wandas/) - Guides, API reference, and examples.
  [公式ドキュメント](https://kasahart.github.io/wandas/) - ガイド、API リファレンス、使用例。
- [Learning Path](https://github.com/kasahart/wandas/tree/main/learning-path/) - Notebook-based walkthroughs.
  [Learning Path](https://github.com/kasahart/wandas/tree/main/learning-path/) - Notebook ベースのステップ別チュートリアル。
- [Examples](https://github.com/kasahart/wandas/tree/main/examples/) - Small runnable scripts and sample data.
  [examples](https://github.com/kasahart/wandas/tree/main/examples/) - 小さな実行例とサンプルデータ。

## Contributing / 貢献

Contributions are welcome.
貢献を歓迎します。

For setup, quality checks, documentation rules, and pull request workflow, see [docs/src/contributing.md](https://kasahart.github.io/wandas/contributing/).
開発環境セットアップ、品質チェック、ドキュメント規約、プルリクエスト手順は [docs/src/contributing.md](https://kasahart.github.io/wandas/contributing/) を参照してください。

If you want to report a bug or propose an idea, please use the [Issue Tracker](https://github.com/kasahart/wandas/issues).
バグ報告や機能提案は [Issue Tracker](https://github.com/kasahart/wandas/issues) を利用してください。

## License / ライセンス

Released under the [MIT License](https://github.com/kasahart/wandas/blob/main/LICENSE).
このプロジェクトは [MIT License](https://github.com/kasahart/wandas/blob/main/LICENSE) の下で公開されています。
