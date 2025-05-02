# Tutorial

This tutorial will teach you the basics of the Wandas library in 5 minutes.

## Installation

```bash
pip install git+https://github.com/endolith/waveform-analysis.git@master
pip install wandas
```

## Basic Usage

### 1. Import the Library

```python
import wandas as wd
import matplotlib.pyplot as plt
```

### 2. Load Audio Files

```python
# Load a WAV file
audio = wd.io.read_wav("path/to/audio.wav")
print(f"Sampling rate: {audio.sampling_rate} Hz")
print(f"Number of channels: {len(audio)}")
```

### 3. Visualize Signals

```python
# Display waveform
audio.plot()
plt.show()
```

### 4. Basic Signal Processing

```python
# Apply a low-pass filter (passing frequencies below 1kHz)
filtered = audio.filter(lowpass=1000)

# Visualize and compare results
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
audio.plot(ax=axes[0], title="Original Signal")
filtered.plot(ax=axes[1], title="Filtered Signal")
plt.tight_layout()
plt.show()
```

### 5. Save Processing Results

```python
# Save the processed signal as a WAV file
wd.io.write_wav(filtered, "filtered_audio.wav")
```

## Next Steps

- Check out various applications in the [Cookbook](../how_to/index.md)
- Look up detailed functions in the [API Reference](../api/index.md)
- Understand the library's design philosophy in the [Theory Background](../explanation/index.md)
