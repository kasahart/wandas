# Example: Calculating Loudness for Non-Stationary Signals

This example demonstrates how to use the `loudness_zwtv` method to calculate time-varying loudness according to the Zwicker method (ISO 532-1:2017).

## Basic Usage

```python
import wandas as wd
import numpy as np

# Load an audio file
signal = wd.read_wav("audio.wav")

# Calculate loudness (free field by default)
loudness = signal.loudness_zwtv()

# Plot the loudness over time
loudness.plot(title="Time-varying Loudness (sones)")
```

## Comparing Free Field vs Diffuse Field

```python
import wandas as wd
import matplotlib.pyplot as plt

# Load signal
signal = wd.read_wav("audio.wav")

# Calculate for both field types
loudness_free = signal.loudness_zwtv(field_type="free")
loudness_diffuse = signal.loudness_zwtv(field_type="diffuse")

# Plot comparison
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
loudness_free.plot(ax=axes[0], title="Free Field Loudness")
loudness_diffuse.plot(ax=axes[1], title="Diffuse Field Loudness")
plt.tight_layout()
plt.show()
```

## Creating a Test Signal

```python
import wandas as wd
import numpy as np

# Generate a 1 kHz sine wave at moderate level
signal = wd.generate_sin(freqs=[1000], duration=2.0, sampling_rate=48000)

# Scale to approximately 70 dB SPL
signal = signal * 0.063

# Calculate loudness
loudness = signal.loudness_zwtv()

# Print statistics
print(f"Mean loudness: {loudness.mean():.2f} sones")
print(f"Max loudness: {loudness.max():.2f} sones")
print(f"Min loudness: {loudness.min():.2f} sones")
```

## Multi-channel Processing

```python
import wandas as wd

# Load stereo audio
stereo_signal = wd.read_wav("stereo_audio.wav")

# Calculate loudness (each channel processed independently)
loudness = stereo_signal.loudness_zwtv()

# Access individual channels
left_loudness = loudness[0]
right_loudness = loudness[1]

# Plot both channels
loudness.plot(overlay=True, title="Stereo Loudness Comparison")
```

## Understanding Loudness Values

The loudness is measured in **sones**, where:
- 1 sone corresponds to a loudness level of 40 phon
- Doubling the loudness in sones corresponds to doubling the perceived loudness
- Typical values:
  - Quiet room: ~1 sone
  - Normal conversation: ~4-8 sones
  - Loud music: ~32+ sones

## Method Parameters

### `loudness_zwtv(field_type="free")`

**Parameters:**
- `field_type` (str): Type of sound field
  - `"free"`: Sound arriving from a specific direction (default)
  - `"diffuse"`: Sound arriving uniformly from all directions

**Returns:**
- A new `ChannelFrame` containing time-varying loudness values in sones

**Notes:**
- The time resolution is approximately 2ms (determined by the algorithm)
- For multi-channel signals, each channel is processed independently
- The method follows ISO 532-1:2017 standard

## Advanced: Accessing MoSQITo Directly

If you need more detailed output (specific loudness, bark axis, etc.), you can use MoSQITo directly:

```python
from mosqito.sq_metrics.loudness.loudness_zwtv import loudness_zwtv
import wandas as wd

signal = wd.read_wav("audio.wav")
data = signal.data[0]  # Get first channel

# Call MoSQITo directly
N, N_spec, bark_axis, time_axis = loudness_zwtv(
    data, signal.sampling_rate, field_type="free"
)

print(f"Loudness shape: {N.shape}")
print(f"Specific loudness shape: {N_spec.shape}")
print(f"Time axis: {time_axis[:10]}...")  # First 10 time points
```

## References

1. ISO 532-1:2017, "Acoustics — Methods for calculating loudness — Part 1: Zwicker method"
2. MoSQITo documentation: https://mosqito.readthedocs.io/en/latest/
3. Zwicker, E., & Fastl, H. (1999). Psychoacoustics: Facts and models (2nd ed.). Springer.
