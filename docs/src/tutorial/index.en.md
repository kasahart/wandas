# Tutorial

This tutorial will teach you the basics of the Wandas library in 5 minutes.

## Installation

```bash
pip install git+https://github.com/endolith/waveform-analysis.git@master
pip install wandas
```

## Basic Usage

### 1. Import the Library

```python exec="on" session="wd_demo"
from io import StringIO
import matplotlib.pyplot as plt
```

```python exec="on" source="above" session="wd_demo"
import wandas as wd

```

### 2. Load Audio Files

```python
# Load a WAV file
url = "https://github.com/kasahart/wandas/raw/main/examples/data/summer_streets1.wav"

audio = wd.read_wav(url)
print(f"Sampling rate: {audio.sampling_rate} Hz")
print(f"Number of channels: {len(audio)}")
print(f"Duration: {audio.duration} s")
```

```python exec="on" session="wd_demo"
url = "https://github.com/kasahart/wandas/raw/main/examples/data/summer_streets1.wav"

audio = wd.read_wav(url)
print(f"Sampling rate: {audio.sampling_rate} Hz  ")
print(f"Number of channels: {audio.n_channels}  ")
print(f"Duration: {audio.duration} s  ")

```

### 3. Visualize Signals

```python
# Display waveform
audio.describe()
```

```python exec="on" html="true" session="wd_demo"
audio.describe(is_close=False)
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())
```

<audio controls src="https://github.com/kasahart/wandas/raw/main/examples/data/summer_streets1.wav"></audio>

### 4. Basic Signal Processing

```python
# Apply a low-pass filter (passing frequencies below 1kHz)
filtered = audio.low_pass_filter(cutoff=1000)

# Visualize and compare results
filtered.previous.plot(title="Original")
filtered.plot(title="filtered")
```

```python exec="on" html="true" session="wd_demo"
filtered = audio.low_pass_filter(cutoff=1000)
filtered.previous.plot(title="Original")
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())

filtered.plot(title="filtered")
buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())
```

## Next Steps

- Check out various applications in the [Cookbook](../how_to/index.md)
- Look up detailed functions in the [API Reference](../api/index.md)
- Understand the library's design philosophy in the [Theory Background](../explanation/index.md)
