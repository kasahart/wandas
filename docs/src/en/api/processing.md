# Processing Module

This page explains the processing module of Wandas. The processing module provides various processing functions for time-series data.

## Time Series Processing

These are processing functions for time-series data. These functions are typically used through `ChannelFrame` methods.

```python
# Common usage
import wandas
frame = wandas.read_wav("audio.wav")
filtered_frame = frame.filter(cutoff=1000, filter_type="lowpass")
resampled_frame = frame.resample(target_rate=16000)
```

### Key Processing Classes

The filtering and other signal processing functions are internally implemented by the following classes:

::: wandas.processing.time_series.HighPassFilter
    handler: python

::: wandas.processing.time_series.LowPassFilter
    handler: python

::: wandas.processing.time_series.ReSampling
    handler: python

::: wandas.processing.time_series.AWeighting
    handler: python

## AudioOperation

The `AudioOperation` class enables abstraction and chaining of audio processing operations.

```python
from wandas.processing.time_series import AudioOperation

# Usage example
import wandas
frame = wandas.read_wav("audio.wav")

# Chain multiple processing steps
operation = (
    AudioOperation()
    .add_step("filter", cutoff=1000, filter_type="lowpass")
    .add_step("normalize")
)

# Apply the processing
processed_frame = operation.apply(frame)
```

::: wandas.processing.time_series.AudioOperation
    handler: python
    selection:
      members:
        - __init__
        - add_step
        - apply
