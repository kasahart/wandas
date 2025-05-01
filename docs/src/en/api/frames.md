# Frame Module

This page explains the frame modules in Wandas. The frame module includes various types of data frames.

## ChannelFrame

`ChannelFrame` is a frame that handles time-domain signal data.

```python
from wandas.frames.channel import ChannelFrame

# Or
import wandas
frame = wandas.read_wav("audio.wav")  # Returns a ChannelFrame instance
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

`SpectralFrame` is a frame that handles frequency-domain data.

```python
from wandas.frames.spectral import SpectralFrame

# Or convert from ChannelFrame
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

`SpectrogramFrame` is a frame that handles time-frequency representation (spectrogram) data.

```python
from wandas.frames.spectrogram import SpectrogramFrame

# Or convert from ChannelFrame
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

`NOctFrame` is a frame that handles N-octave analysis data.

```python
from wandas.frames.noct import NOctFrame

# Or convert from ChannelFrame
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
