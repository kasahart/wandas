---
name: wandas-getting-started
description: Use when starting with the wandas library, loading audio or sensor data from WAV or CSV files, creating signals from NumPy arrays, understanding ChannelFrame and other frame types, or needing a quick overview of wandas I/O and inspection functions.
---

# wandas: Getting Started

wandas provides pandas-like Frame objects for waveform analysis. All operations return new immutable frames; data is stored as lazy Dask arrays.

## Load Data

| Function | Returns | Notes |
|----------|---------|-------|
| `wd.read_wav(path)` | ChannelFrame | WAV file |
| `wd.read_csv(path, time_column="Time")` | ChannelFrame | CSV sensor data |
| `wd.from_numpy(data, sampling_rate=sr)` | ChannelFrame | 2D array (channels x samples) |
| `wd.generate_sin(freqs=[440], duration=1.0, sampling_rate=44100)` | ChannelFrame | Synthetic signal |
| `wd.from_folder(path)` | ChannelFrameDataset | Lazy batch load |

## Frame Types

| Frame | Domain | Created by |
|-------|--------|-----------|
| `ChannelFrame` | Time | I/O, `from_numpy` |
| `SpectralFrame` | Frequency | `.fft()`, `.welch()` |
| `SpectrogramFrame` | Time-frequency | `.stft()` |
| `NOctFrame` | Octave bands | `.noct_spectrum()` |
| `RoughnessFrame` | Psychoacoustics | `.roughness_dw_spec()` |

## Inspect

```python
signal.info()                 # print metadata
signal.describe()             # waveform + spectrum + spectrogram summary
signal.sampling_rate          # int: Hz
signal.n_channels             # int
signal.duration               # float: seconds
signal.labels                 # list[str]: channel names
```

## Quick Example

```python
import wandas as wd

signal = wd.read_wav("audio.wav")
signal.info()
signal.describe()
print(signal.sampling_rate, signal.n_channels, signal.duration)
```

## Common Mistakes

- **Lazy by default**: All operations build a Dask graph. Call `.compute()` to get a NumPy array.
- **Shape**: `from_numpy` expects shape `(channels, samples)`. A 1-D array is auto-reshaped to a single channel; 3-D+ raises an error.
- **CSV time column**: `read_csv` requires `time_column` to identify the time axis.