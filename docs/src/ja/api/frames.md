# フレームモジュール（Frames）

このページではWandasのフレームモジュールについて説明します。フレームモジュールには、さまざまなタイプのデータフレームが含まれています。

## ChannelFrame

`ChannelFrame`は時間領域の信号データを扱うフレームです。

```python
from wandas.frames.channel import ChannelFrame

# または
import wandas
frame = wandas.read_wav("audio.wav")  # ChannelFrameインスタンスを返します
```

::: wandas.frames.channel.ChannelFrame
    handler: python
    selection:
      members:
        - __init__
        - __getitem__
        - __len__
        - plot
        - filter
        - resample
        - read_wav
        - write_wav
        - read_csv

## SpectralFrame

`SpectralFrame`は周波数領域のデータを扱うフレームです。

```python
from wandas.frames.spectral import SpectralFrame

# または ChannelFrameから変換
channel_frame = wandas.read_wav("audio.wav")
spectral_frame = channel_frame.to_spectral()
```

::: wandas.frames.spectral.SpectralFrame
    handler: python
    selection:
      members:
        - __init__
        - __getitem__
        - plot
        - to_channel_frame

## SpectrogramFrame

`SpectrogramFrame`は時間周波数表現（スペクトログラム）のデータを扱うフレームです。

```python
from wandas.frames.spectrogram import SpectrogramFrame

# または ChannelFrameから変換
channel_frame = wandas.read_wav("audio.wav")
spectrogram_frame = channel_frame.to_spectrogram()
```

::: wandas.frames.spectrogram.SpectrogramFrame
    handler: python
    selection:
      members:
        - __init__
        - plot
        - to_spectral_frame

## NOctFrame

`NOctFrame`はNオクターブ分析のデータを扱うフレームです。

```python
from wandas.frames.noct import NOctFrame

# または ChannelFrameから変換
channel_frame = wandas.read_wav("audio.wav")
noct_frame = channel_frame.to_noct()
```

::: wandas.frames.noct.NOctFrame
    handler: python
    selection:
      members:
        - __init__
        - plot
        - dB
        - dBA
        - freqs
