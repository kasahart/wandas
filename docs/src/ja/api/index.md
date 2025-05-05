# APIリファレンス

このセクションでは、Wandasライブラリの全てのモジュール、クラス、メソッドの詳細なリファレンスを提供します。ライブラリのソースコードから自動的に生成されたドキュメントを含んでいます。

## コアコンポーネント

### BaseFrame

`BaseFrame`クラスはWandasの基本単位となるデータ構造です。

```python
from wandas.core.base_frame import BaseFrame
```

::: wandas.core.base_frame.BaseFrame
    handler: python
    selection:
      members:
        - __init__
        - data
        - metadata
        - shape
        - n_channels
        - channels
        - sampling_rate
        - compute
        - plot
        - persist

## フレームモジュール

### ChannelFrame

`ChannelFrame`は時間領域の信号データを扱うクラスです。

```python
from wandas.frames.channel import ChannelFrame
```

または、以下のように直接インポートすることもできます：

```python
import wandas
# 以下のように使用できます
frame = wandas.read_wav("audio.wav")
```

::: wandas.frames.channel.ChannelFrame
    handler: python
    selection:
      members:
        - __init__
        - plot
        - filter
        - resample

## 入出力 (I/O)

Wandasでは、以下のような方法で音声ファイルやCSVファイルを読み込むことができます。

### WAVファイルの読み込みと書き込み

```python
# WAVファイルの読み込み
from wandas.frames.channel import ChannelFrame
frame = ChannelFrame.read_wav("audio.wav")

# または直接インポートを使用
import wandas
frame = wandas.read_wav("audio.wav")

# WAVファイルの書き込み
frame.save("output.wav")
```

::: wandas.frames.channel.ChannelFrame.read_wav
    handler: python

::: wandas.frames.channel.ChannelFrame.save
    handler: python

### CSVファイルの読み込み

```python
# CSVファイルの読み込み
from wandas.frames.channel import ChannelFrame
frame = ChannelFrame.read_csv("data.csv")

# または直接インポートを使用
import wandas
frame = wandas.read_csv("data.csv")
```

::: wandas.frames.channel.ChannelFrame.read_csv
    handler: python

## 信号処理

Wandasは様々な信号処理機能を提供します。

::: wandas.processing

## ユーティリティ

::: wandas.utils
    handler: python
