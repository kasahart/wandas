# Psychoacoustic Metrics

Wandas provides psychoacoustic metrics for analyzing audio signals based on human perception. These metrics are calculated using standardized methods and the MoSQITo library.

## Loudness (Non-Stationary Signals)

### Overview

The `loudness_zwtv()` method calculates time-varying loudness for non-stationary signals using the Zwicker method according to ISO 532-1:2017. This method provides a measure of perceived loudness that correlates well with human perception.

### What is Loudness?

Loudness is measured in **sones**, a perceptual unit where:

- **1 sone** corresponds to a loudness level of 40 phon (approximately the loudness of a 1 kHz tone at 40 dB SPL)
- **Doubling the sones** corresponds to doubling the perceived loudness
- The relationship is: if sound A has twice the loudness in sones as sound B, it will sound twice as loud

### Typical Loudness Values

| Environment/Sound | Approximate Loudness |
|-------------------|---------------------|
| Quiet library | ~0.5-1 sone |
| Quiet conversation | ~2-4 sones |
| Normal conversation | ~4-8 sones |
| Busy office | ~8-16 sones |
| Loud music | ~32+ sones |
| Very loud noise | ~100+ sones |

### Usage

#### Basic Usage

```python
import wandas as wd

# Load audio file
signal = wd.read_wav("audio.wav")

# Calculate loudness (free field)
loudness = signal.loudness_zwtv()

# Plot loudness over time
loudness.plot(title="Time-varying Loudness")
```

#### Field Type Selection

The method supports two types of sound fields:

- **Free field** (`field_type="free"`): Sound arriving from a specific direction (e.g., loudspeaker in front of listener)
- **Diffuse field** (`field_type="diffuse"`): Sound arriving uniformly from all directions (e.g., reverberant room)

```python
# Free field (default)
loudness_free = signal.loudness_zwtv(field_type="free")

# Diffuse field
loudness_diffuse = signal.loudness_zwtv(field_type="diffuse")
```

### Method Signature

```python
def loudness_zwtv(self, field_type: str = "free") -> ChannelFrame:
    """
    Calculate time-varying loudness using Zwicker method.
    
    Parameters
    ----------
    field_type : str, default="free"
        Type of sound field ('free' or 'diffuse')
    
    Returns
    -------
    ChannelFrame
        Time-varying loudness values in sones
    """
```

### Output

The method returns a `ChannelFrame` containing:

- **Time-varying loudness values** in sones
- **Time resolution**: Approximately 2ms (0.002 seconds)
- **Multi-channel handling**: Each channel is processed independently

### Examples

#### Example 1: Basic Usage

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

#### Example 2: Creating a Test Signal

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

#### Example 3: Comparing Free vs Diffuse Field

```python
import wandas as wd
import numpy as np

# Generate 1 kHz sine wave at moderate level
signal = wd.generate_sin(freqs=[1000], duration=2.0, sampling_rate=48000)
signal = signal * 0.063  # Scale to ~70 dB SPL

# Calculate loudness
loudness = signal.loudness_zwtv()

# Get statistics
print(f"Mean loudness: {loudness.mean():.2f} sones")
print(f"Max loudness: {loudness.max():.2f} sones")
```

#### Example 3: Comparing Free vs Diffuse Field

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

#### Example 4: Multi-Channel Processing

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

#### Example 5: Accessing MoSQITo Directly

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

### Technical Details

#### Algorithm

The implementation uses MoSQITo's `loudness_zwtv` function, which implements:

1. **Outer ear transfer function**: Simulates the filtering effect of the outer ear
2. **Middle ear transfer function**: Models middle ear transmission
3. **Excitation patterns**: Calculates excitation along the basilar membrane
4. **Specific loudness**: Determines loudness in each critical band
5. **Total loudness**: Integrates specific loudness over all critical bands

#### Time Resolution

The loudness calculation produces values with approximately 2ms time resolution. For a 1-second signal, you can expect around 500 loudness values.

#### Computational Complexity

- The algorithm processes signals in blocks for efficiency
- Processing time scales linearly with signal duration
- Memory usage is moderate (stores time-varying loudness values)

### Limitations

1. **Sampling rate**: Best results with sampling rates ≥ 44.1 kHz
2. **Signal level**: Accurate for signals in the range of audibility (typically 20-100 dB SPL)
3. **Stationary assumption**: While designed for non-stationary signals, extremely rapid transients may not be fully captured
4. **Calibration**: Assumes proper signal calibration to physical units (Pa)

### Standards and References

- **ISO 532-1:2017**: "Acoustics — Methods for calculating loudness — Part 1: Zwicker method"
- **Zwicker, E., & Fastl, H. (1999)**: Psychoacoustics: Facts and models (2nd ed.). Springer.
- **MoSQITo library**: https://mosqito.readthedocs.io/en/latest/

---

## Loudness (Steady Signals)

### Overview

The `loudness_zwst()` method calculates steady-state loudness for stationary (steady) signals using the Zwicker method according to ISO 532-1:2017. This method is suitable for evaluating continuous sounds such as fan noise, steady machinery sounds, and other stationary signals.

### Differences from Time-Varying Loudness

| Feature | Time-varying (`loudness_zwtv`) | Steady-state (`loudness_zwst`) |
|---------|-------------------------------|--------------------------------|
| **Signal type** | Non-stationary (time-varying) | Stationary (steady) |
| **Use cases** | Speech, music, transient sounds | Fan noise, constant machinery |
| **Output** | Time series of loudness values | Single loudness value |
| **Output shape** | (channels, time_samples) | (n_channels,) |
| **Sampling rate** | Updated to ~500 Hz | Not changed (single value) |

### Usage

#### Basic Usage

```python
import wandas as wd

# Load steady signal (e.g., fan noise)
signal = wd.read_wav("fan_noise.wav")

# Calculate steady-state loudness (free field)
loudness = signal.loudness_zwst()

# Display result
print(f"Steady-state loudness: {loudness[0]:.2f} sones")
```

#### Field Type Selection

Like time-varying loudness, this method supports two types of sound fields:

```python
# Free field (default)
loudness_free = signal.loudness_zwst(field_type="free")

# Diffuse field
loudness_diffuse = signal.loudness_zwst(field_type="diffuse")

print(f"Free field: {loudness_free[0]:.2f} sones")
print(f"Diffuse field: {loudness_diffuse[0]:.2f} sones")
```

### Method Signature

```python
def loudness_zwst(self, field_type: str = "free") -> NDArrayReal:
    """
    Calculate steady-state loudness using Zwicker method
    
    Parameters
    ----------
    field_type : str, default="free"
        Type of sound field ('free' or 'diffuse')
    
    Returns
    -------
    NDArrayReal
        Steady-state loudness values in sones (one value per channel)
        Shape: (n_channels,)
    """
```

### Output

The method returns `NDArrayReal` containing:

- **Single loudness value** in sones for each channel
- **Output shape**: (n_channels,) - 1D array
- **Multi-channel handling**: Each channel is processed independently
- **NumPy compatible**: Direct NumPy operations possible (`loudness[0]`, `loudness.mean()`, etc.)

### Examples

#### Example 1: Fan Noise Evaluation

```python
import wandas as wd

# Load fan noise
fan_signal = wd.read_wav("fan_noise.wav")

# Calculate steady-state loudness
loudness = fan_signal.loudness_zwst(field_type="free")

# Display result
print(f"Fan noise loudness: {loudness[0]:.2f} sones")
```

#### Example 2: Comparing Multiple Steady Sound Sources

```python
import wandas as wd

# Load different steady sound sources
fan1 = wd.read_wav("fan1.wav")
fan2 = wd.read_wav("fan2.wav")

# Calculate steady-state loudness
loudness1 = fan1.loudness_zwst()
loudness2 = fan2.loudness_zwst()

# Compare
print(f"Fan 1: {loudness1[0]:.2f} sones")
print(f"Fan 2: {loudness2[0]:.2f} sones")

if loudness1[0] > loudness2[0]:
    print("Fan 1 is louder")
else:
    print("Fan 2 is louder")
```

#### Example 3: Stereo Steady Sound Processing

```python
import wandas as wd

# Load stereo steady sound source
stereo_signal = wd.read_wav("stereo_steady_noise.wav")

# Calculate steady-state loudness (each channel independently)
loudness = stereo_signal.loudness_zwst()

# Display results for each channel
print(f"Left channel: {loudness[0]:.2f} sones")
print(f"Right channel: {loudness[1]:.2f} sones")
```

#### Example 4: Comparing Free Field and Diffuse Field

```python
import wandas as wd

# Load steady signal
signal = wd.read_wav("steady_noise.wav")

# Calculate for both field types
loudness_free = signal.loudness_zwst(field_type="free")
loudness_diffuse = signal.loudness_zwst(field_type="diffuse")

# Compare
print(f"Free field: {loudness_free[0]:.2f} sones")
print(f"Diffuse field: {loudness_diffuse[0]:.2f} sones")
print(f"Difference: {abs(loudness_free[0] - loudness_diffuse[0]):.2f} sones")
```

#### Example 5: Accessing MoSQITo Directly

If you need more detailed output (specific loudness, bark axis, etc.), you can use MoSQITo directly:

```python
from mosqito.sq_metrics.loudness.loudness_zwst import loudness_zwst
import wandas as wd

signal = wd.read_wav("steady_noise.wav")
data = signal.data[0]  # Get first channel

# Call MoSQITo directly
N, N_spec, bark_axis = loudness_zwst(
    data, signal.sampling_rate, field_type="free"
)

print(f"Loudness: {N:.2f} sones")
print(f"Specific loudness shape: {N_spec.shape}")
print(f"Bark axis: {bark_axis}")
```

### Technical Details

#### Algorithm

Steady-state loudness calculation is based on the same Zwicker method as time-varying loudness, but outputs a single representative value:

1. **Outer ear transfer function**: Simulates the filtering effect of the outer ear
2. **Middle ear transfer function**: Models middle ear transmission
3. **Excitation patterns**: Calculates excitation along the basilar membrane
4. **Specific loudness**: Determines loudness in each critical band
5. **Total loudness**: Integrates specific loudness over all critical bands

#### Computational Complexity

- Calculation is simplified compared to time-varying loudness since steady signals are assumed
- Processing time depends on signal length, but only a single value is output
- Memory usage is small (stores only a single loudness value)

### Limitations

1. **Sampling rate**: Best results with sampling rates ≥ 44.1 kHz
2. **Signal level**: Accurate for signals in the range of audibility (typically 20-100 dB SPL)
3. **Stationarity assumption**: This method is designed for stationary signals. For time-varying signals, use `loudness_zwtv()` instead
4. **Calibration**: Assumes proper signal calibration to physical units (Pa)

### Standards and References

- **ISO 532-1:2017**: "Acoustics — Methods for calculating loudness — Part 1: Zwicker method"
- **Zwicker, E., & Fastl, H. (1999)**: Psychoacoustics: Facts and models (2nd ed.). Springer.
- **MoSQITo library**: https://mosqito.readthedocs.io/en/latest/

---

### Related Operations

- `loudness_zwtv()`: Calculate time-varying loudness (for non-stationary signals)
- `loudness_zwst()`: Calculate steady-state loudness (for stationary signals)
- `a_weighting()`: Apply A-weighting filter (frequency weighting approximating human hearing)
- `noct_spectrum()`: Calculate N-octave band spectrum
- `rms_trend()`: Calculate RMS trend over time

### See Also

- [MoSQITo Documentation](https://mosqito.readthedocs.io/en/latest/)
- [ISO 532-1:2017 Standard](https://www.iso.org/standard/63077.html)
- [Psychoacoustics Fundamentals](https://en.wikipedia.org/wiki/Psychoacoustics)
