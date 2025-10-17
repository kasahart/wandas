# Roughness Calculation

Roughness is a psychoacoustic metric that quantifies the perceived roughness of a sound. It is related to rapid amplitude modulations in the 15-300 Hz range.

## Overview

The wandas library provides roughness calculation using the Daniel & Weber method, implemented via the MoSQITo (Modular Sound Quality Indicators Tools) library.

**Key Features:**
- Time-domain calculation using `roughness_dw_time`
- Frequency-domain calculation using `roughness_dw_freq`
- Support for mono and multi-channel signals
- Configurable overlap parameter for time-domain method

## Usage

### Basic Usage

```python
import wandas as wd

# Load an audio file
signal = wd.read_wav("audio.wav")

# Calculate roughness using time-domain method (default)
roughness = signal.roughness(method='time')
print(f"Roughness: {roughness:.3f} asper")
```

### Method Selection

Two calculation methods are available:

```python
# Time-domain calculation (default)
roughness_time = signal.roughness(method='time')

# Frequency-domain calculation
roughness_freq = signal.roughness(method='freq')
```

### Overlap Parameter

For the time-domain method, you can specify an overlap ratio:

```python
# No overlap (default)
roughness_no_overlap = signal.roughness(method='time', overlap=0.0)

# 50% overlap for smoother temporal averaging
roughness_with_overlap = signal.roughness(method='time', overlap=0.5)
```

### Multi-channel Signals

For stereo or multi-channel signals, roughness is calculated independently for each channel:

```python
# Stereo signal
stereo_signal = wd.read_wav("stereo_audio.wav")
roughness = stereo_signal.roughness(method='time')

# Result is an array with one value per channel
print(f"Left channel:  {roughness[0]:.3f} asper")
print(f"Right channel: {roughness[1]:.3f} asper")
```

## Understanding Roughness Values

### Unit: Asper

Roughness is measured in **asper**. The reference is:
- **1 asper** = roughness of a 1 kHz tone at 60 dB SPL, 100% amplitude modulated at 70 Hz

### Typical Values

- **Pure tone**: < 0.5 asper (very smooth)
- **Modulated tone (70 Hz)**: Peak roughness (typically 1-3 asper)
- **Modulated tone (100 Hz)**: Moderate roughness
- **Complex signals**: Variable, typically 0-10 asper

### Perceptual Characteristics

Roughness perception is highest for amplitude modulations around 70 Hz and decreases for both lower and higher modulation frequencies:

- **< 15 Hz**: Perceived as individual pulses, not roughness
- **15-70 Hz**: Increasing roughness
- **70 Hz**: Peak roughness perception
- **70-300 Hz**: Decreasing roughness
- **> 300 Hz**: Minimal roughness perception

## Example: Comparing Different Signals

```python
import numpy as np
import wandas as wd

# Parameters
sample_rate = 44100
duration = 1.0
t = np.linspace(0, duration, int(sample_rate * duration))

# Pure tone (low roughness)
pure_tone = np.sin(2 * np.pi * 1000 * t)
signal_pure = wd.ChannelFrame(data=pure_tone, sampling_rate=sample_rate)
roughness_pure = signal_pure.roughness(method='time')
print(f"Pure tone roughness: {roughness_pure:.3f} asper")

# Modulated tone at 70 Hz (high roughness)
carrier = np.sin(2 * np.pi * 1000 * t)
modulator = 0.5 * (1 + np.sin(2 * np.pi * 70 * t))
modulated = carrier * modulator
signal_mod = wd.ChannelFrame(data=modulated, sampling_rate=sample_rate)
roughness_mod = signal_mod.roughness(method='time')
print(f"Modulated tone roughness: {roughness_mod:.3f} asper")

# Expected: roughness_mod >> roughness_pure
```

## Technical Details

### Daniel & Weber Method

The implementation uses the Daniel & Weber method, which is based on:
- Bandpass filtering into critical bands (Bark scale)
- Envelope extraction in each band
- Modulation depth analysis
- Integration across frequency bands

### References

1. Daniel, P., & Weber, R. (1997). Psychoacoustical roughness: implementation of an optimized model. *Acustica*, 83, 113-123.

2. ECMA-418-2:2022 - Psychoacoustic metrics for ITT equipment - Part 2: Models based on human perception

3. MoSQITo Documentation: [https://mosqito.readthedocs.io](https://mosqito.readthedocs.io)

## API Reference

See the [API documentation](../api/processing.md#psychoacoustic-metrics) for detailed parameter descriptions and return types.
