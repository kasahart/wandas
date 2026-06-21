# Wandas

[English](README.md) | 日本語

![Wandas logo](https://github.com/kasahart/wandas/blob/main/images/logo.png?raw=true)

[![PyPI](https://img.shields.io/pypi/v/wandas)](https://pypi.org/project/wandas/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/wandas)](https://pypi.org/project/wandas/)
[![CI](https://github.com/kasahart/wandas/actions/workflows/ci.yml/badge.svg)](https://github.com/kasahart/wandas/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/kasahart/wandas/graph/badge.svg?token=53NPNQQZZ8)](https://codecov.io/gh/kasahart/wandas)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/kasahart/wandas/blob/main/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/wandas)](https://pypi.org/project/wandas/)

Python で波形・信号データを扱うためのデータ構造ライブラリです。

WAV や CSV を読み込み、メタデータを保ちながら、可視化や周波数解析まで一貫して進められます。

## 概要

Wandas は、メソッドチェーンしやすいフレーム指向 API で信号解析を進められる、オープンソースの Python ライブラリです。

サンプリング周波数、チャネル名、付随メタデータを保ちながら、読み込みから確認、フィルタリング、周波数解析、可視化までつなげられます。

## なぜ Wandas か

- NumPy 配列を都度組み合わせる代わりに、pandas ライクなオブジェクトで波形データを扱えます。
- 解析が進んでも、メタデータ、チャネル情報、処理履歴を一緒に保てます。
- 時間領域、周波数領域、スペクトログラムを一貫した API で行き来できます。
- 組み込みの可視化と要約機能で、信号をすぐに確認できます。
- 必要に応じて Dask ベースの遅延実行で大きなデータにも対応できます。

## クイックスタート

推奨の marimo extra 付きで PyPI からインストールします。

```bash
pip install "wandas[marimo]"
```

最小の core-only インストールの場合:

```bash
pip install wandas
```

### インストールオプション

core-only インストールでは、optional extras なしで波形データ、CSV/WAV、処理、プロットを利用でき、`is_close=False` や `image_save` などの非表示オプションを使う `describe()` の図作成・保存ワークフローも利用できます。デフォルトのインタラクティブな `frame.describe()` 表示には `wandas[marimo]` が必要です。

追加のファイル形式や重めの解析機能が必要な場合は、optional extras を追加してインストールします。

```bash
pip install "wandas[io]"              # WDF の保存・読み込み
pip install "wandas[effects]"         # librosa ベースのオーディオエフェクト
pip install "wandas[marimo]"          # marimo 学習アプリとインタラクティブ表示
pip install "wandas[psychoacoustic]"  # ラウドネス、粗さ、オクターブバンド補助機能
pip install "wandas[ml]"              # Torch/TensorFlow テンソル補助機能
```

必要に応じて extras は組み合わせられます。

```bash
pip install "wandas[marimo,io,effects,psychoacoustic]"
```

次に、信号ファイルを読み込んで、そのまま確認できます。

```python
import wandas as wd

# 信号ファイルを読み込んで確認する。
signal = wd.read("audio.wav")
signal.describe()
```

`describe()` で、波形、スペクトル、スペクトログラムをまとめて素早く確認できます。

![cf.describe](https://github.com/kasahart/wandas/blob/main/images/read_wav_describe.png?raw=true)

## 公開 API

多くのワークフローでは、小さな top-level API から始めます。

```python
import numpy as np
import wandas as wd

signal = wd.read("audio.wav")      # WAV、CSV、対応音声、URL、bytes、file-like
saved = wd.load("analysis.wdf")    # Wandas ネイティブ WDF
data = np.zeros((1, 48000), dtype=np.float32)
array_signal = wd.from_numpy(data, sampling_rate=48000)
dataset = wd.from_folder("recordings/")
```

既存コード向けに `read_wav()`、`read_csv()`、`from_ndarray()` は残りますが、新しい例では `read()` と `from_numpy()` を使います。

## できること

- 登録済み reader 形式（WAV、FLAC、OGG、AIFF/AIF、SND、CSV）から波形やセンサーデータを読み込めます。WDF は別の save/load API で扱えます。
- フィルタ、リサンプリング、正規化、要約をメソッドチェーンで進められます。
- FFT、STFT、Welch、コヒーレンス、伝達関数、オクターブ系解析を実行できます。
- ラウドネスや粗さなどの心理音響指標を扱えます。
- 波形、スペクトル、スペクトログラムを Matplotlib と親和性の高い API で描画できます。

## 次に読む

- [公式ドキュメント](https://kasahart.github.io/wandas/) - ガイド、API リファレンス、使用例。
- [学習パス](https://github.com/kasahart/wandas/tree/main/learning-path/) - marimo アプリベースのステップ別チュートリアル。
- [使用例](https://github.com/kasahart/wandas/tree/main/examples/) - 小さな実行例とサンプルデータ。

## 貢献

貢献を歓迎します。

開発環境セットアップ、品質チェック、ドキュメント規約、プルリクエスト手順は [docs/src/contributing.md](https://kasahart.github.io/wandas/contributing/) を参照してください。

バグ報告や機能提案は [Issue Tracker](https://github.com/kasahart/wandas/issues) を利用してください。

## ライセンス

このプロジェクトは [MIT License](https://github.com/kasahart/wandas/blob/main/LICENSE) の下で公開されています。
