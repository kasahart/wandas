# API Reference

This section provides detailed references for all modules, classes, and methods in the Wandas library. It includes documentation automatically generated from the library's source code.

## Core Components

### BaseFrame

The `BaseFrame` class is the basic data structure unit of Wandas.

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

## Frame Module

### ChannelFrame

`ChannelFrame` is a class that handles time-domain signal data.

```python
from wandas.frames.channel import ChannelFrame
```

You can also import it directly:

```python
import wandas
# Use it as follows
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

## Input/Output (I/O)

Wandas provides various methods to read audio files and CSV files.

### WAV File Operations

```python
# Reading WAV files
from wandas.frames.channel import ChannelFrame
frame = ChannelFrame.read_wav("audio.wav")

# Or using direct import
import wandas
frame = wandas.read_wav("audio.wav")

# Writing WAV files
frame.save("output.wav")
```

::: wandas.frames.channel.ChannelFrame.read_wav
    handler: python

::: wandas.frames.channel.ChannelFrame.save
    handler: python

### CSV File Operations

```python
# Reading CSV files
from wandas.frames.channel import ChannelFrame
frame = ChannelFrame.read_csv("data.csv")

# Or using direct import
import wandas
frame = wandas.read_csv("data.csv")
```

::: wandas.frames.channel.ChannelFrame.read_csv
    handler: python

## Signal Processing

Wandas provides various signal processing functions.

::: wandas.processing

## Utilities

::: wandas.utils
    handler: python
