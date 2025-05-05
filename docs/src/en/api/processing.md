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

::: wandas.processing.HighPassFilter

::: wandas.processing.LowPassFilter

::: wandas.processing.ReSampling

::: wandas.processing.AWeighting

## AudioOperation

The `AudioOperation` class enables abstraction and chaining of audio processing operations.

```python
from wandas.processing import AudioOperation

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

## Creating Custom Operations

```python
from wandas.processing import AudioOperation

class MyCustomOperation(AudioOperation):
    def __init__(self, sampling_rate, **kwargs):
        super().__init__(sampling_rate)
        # Initialize parameters

    def process(self, data):
        # Process data
        return processed_data
```

You can then register and use your custom operation:

```python
from wandas.processing import register_operation

register_operation("my_custom_op", MyCustomOperation)
```

::: wandas.processing.AudioOperation
