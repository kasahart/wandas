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

## Recipes by Use Case

This section provides links to tutorial notebooks that demonstrate more detailed features and application examples of the Wandas library.

- [00_setup.ipynb: Setup and basic configuration](/tutorial/00_setup.ipynb)
- [01_io_basics.ipynb: File reading/writing and basic operations](/tutorial/01_io_basics.ipynb)
- [02_signal_processing_basics.ipynb: Basic signal processing](/tutorial/02_signal_processing_basics.ipynb)
- [03_visualization.ipynb: Data visualization](/tutorial/03_visualization.ipynb)
- [04_time_frequency.ipynb: Time-frequency analysis](/tutorial/04_time_frequency.ipynb)
- [05_lazy_and_dask.ipynb: Lazy evaluation and large-scale data processing with Dask](/tutorial/05_lazy_and_dask.ipynb)
- [06_metadata_history.ipynb: Utilizing metadata and processing history](/tutorial/06_metadata_history.ipynb)
- [07_batch_processing.ipynb: Batch processing for multiple files](/tutorial/07_batch_processing.ipynb)
- [08_frame_dataset_usage.ipynb: FrameDataset Usage Guide - Efficient processing of multiple audio files in a folder](../../examples/03_frame_dataset_usage.ipynb)
- [09_extending_api.ipynb: Adding custom functions and extending the API](/tutorial/08_extending_api.ipynb)
- [10_interoperability.ipynb: Integration with other libraries](/tutorial/08_interoperability.ipynb)
- [11_case_studies.ipynb: Practical use case studies](/tutorial/09_case_studies.ipynb)

!!! tip "Hint"
    Each notebook focuses on a specific topic. Refer to them sequentially or as needed based on your interests. For basic usage of Wandas, please also see the "Basic Usage" section at the beginning of this document.
