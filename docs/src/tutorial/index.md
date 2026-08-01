# Tutorial / チュートリアル

This is a five-minute introduction to the core Wandas workflow: create or read a
signal, apply one operation, and inspect the result.

このページでは、信号の作成または読込、処理、結果の確認というWandasの基本workflowを
5分で体験します。

## Installation / インストール

```bash
pip install wandas
```

The optional `io`, `marimo`, and analysis extras are documented with the
workflow that needs them. The learning apps include their own setup context.

必要なworkflowに応じた`io`、`marimo`、解析extraは各ドキュメントで説明しています。
learning appにも必要なセットアップ情報があります。

## Import / インポート

```python exec="on" session="wd_demo"
import wandas as wd
```

## Create or read a signal / 信号の作成または読込

The executable example uses a generated signal, so it runs without a file or
network connection.

実行例では生成信号を使うため、ファイルやネットワーク接続は不要です。

```python exec="on" session="wd_demo"
audio = wd.generate_sin(freqs=[440, 1_000], duration=1.0, sampling_rate=16_000)
print(f"Sampling rate / サンプリングレート: {audio.sampling_rate} Hz")
print(f"Channels / チャンネル数: {audio.n_channels}")
print(f"Duration / 長さ: {audio.duration} s")
```

For a local WAV file, the same workflow starts with one ordinary read call:

手元のWAVファイルを使う場合も、最初の読込は次の1行です:

```python
audio = wd.read("recording.wav")
```

## Basic processing / 基本的な処理

Apply a low-pass filter and keep the returned Frame for the next step. The
input Frame remains unchanged.

ローパスフィルタを適用し、返されたFrameを次の処理へ渡します。入力Frameは変更されません。

```python exec="on" session="wd_demo"
filtered = audio.low_pass_filter(cutoff=800)
```

## Visualization / 可視化

```python exec="on" session="wd_demo"
filtered.plot(title="Low-pass filtered signal / ローパスフィルタ後の信号")
```

For one representative channel selection, use a label query such as:

代表的なチャンネル選択では、ラベルqueryを次のように使えます:

```python
audio.rename_channels({0: "acc_x"}).get_channel(query="acc_x")
```

The generated API reference documents the other query forms and their
validation rules: [ChannelFrame.get_channel](../api/frames.md).

詳細なquery形式と検証規則は生成された[ChannelFrame.get_channel API reference](../api/frames.md)
を参照してください。

## Next steps / 次に読むもの

- <a href="../learning-path/01_getting_started.html">Getting Started learning app</a>: setup and the first interactive workflow.
- <a href="../learning-path/02_working_with_data.html">Working with Data learning app</a>: read and inspect local data files.
- <a href="../learning-path/03_signal_processing_basics.html">Signal Processing Basics learning app</a>: filtering and frequency analysis.
- [RecipePlan tutorial](pipeline-recipes.md): reuse a public Frame workflow on another input.
- [API Reference](../api/index.md): generated contracts for public symbols.
