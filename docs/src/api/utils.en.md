# Utilities Module

The `wandas.utils` module provides various utility functions used in the Wandas library.

## Frame Dataset

Provides dataset utilities for managing multiple data frames.

### Overview

The `FrameDataset` classes enable efficient batch processing of audio files in a folder. Key features include:

- **Lazy Loading**: Load files only when accessed, reducing memory usage
- **Transformation Chaining**: Apply multiple processing operations efficiently
- **Sampling**: Extract random subsets for testing or analysis
- **Metadata Tracking**: Keep track of dataset properties and processing history

### Main Classes

- **`ChannelFrameDataset`**: For time-domain audio data (WAV, MP3, FLAC, CSV files)
- **`SpectrogramFrameDataset`**: For time-frequency domain data (typically created from STFT)

### Basic Usage

```python
from wandas.utils.frame_dataset import ChannelFrameDataset

# Create a dataset from a folder
dataset = ChannelFrameDataset.from_folder(
    folder_path="path/to/audio/files",
    sampling_rate=16000,  # Optional: resample all files to this rate
    file_extensions=[".wav", ".mp3"],  # File types to include
    recursive=True,  # Search subdirectories
    lazy_loading=True  # Load files on demand (recommended)
)

# Access individual files
first_file = dataset[0]
print(f"File: {first_file.label}")
print(f"Duration: {first_file.duration}s")

# Get dataset information
metadata = dataset.get_metadata()
print(f"Total files: {metadata['file_count']}")
print(f"Loaded files: {metadata['loaded_count']}")
```

### Sampling

Extract random subsets of the dataset for testing or analysis:

```python
# Sample by number of files
sampled = dataset.sample(n=10, seed=42)

# Sample by ratio
sampled = dataset.sample(ratio=0.1, seed=42)

# Default: 10% or minimum 1 file
sampled = dataset.sample(seed=42)
```

### Transformations

Apply processing operations to all files in the dataset:

```python
# Built-in transformations
resampled = dataset.resample(target_sr=8000)
trimmed = dataset.trim(start=0.5, end=2.0)
normalized = dataset.normalize()

# Chain multiple transformations
processed = (
    dataset
    .resample(target_sr=8000)
    .trim(start=0.5, end=2.0)
    .normalize()
)

# Custom transformation
def custom_filter(frame):
    return frame.low_pass_filter(cutoff=1000)

filtered = dataset.apply(custom_filter)
```

### STFT - Spectrogram Generation

Convert time-domain data to spectrograms:

```python
# Create spectrogram dataset
spec_dataset = dataset.stft(
    n_fft=2048,
    hop_length=512,
    window="hann"
)

# Access a spectrogram
spec_frame = spec_dataset[0]
spec_frame.plot()
```

### Iteration

Process all files in the dataset:

```python
for i in range(len(dataset)):
    frame = dataset[i]
    if frame is not None:
        # Process the frame
        print(f"Processing {frame.label}...")
```

### Key Parameters

**folder_path** (str): Path to the folder containing audio files

**sampling_rate** (Optional[int]): Target sampling rate. Files will be resampled if different from this rate

**file_extensions** (Optional[list[str]]): List of file extensions to include. Default: `[".wav", ".mp3", ".flac", ".csv"]`

**lazy_loading** (bool): If True, files are loaded only when accessed. Default: True

**recursive** (bool): If True, search subdirectories recursively. Default: False

### Examples

For detailed examples, see the [FrameDataset Usage Guide](../../examples/03_frame_dataset_usage.ipynb) notebook.

### API Reference

::: wandas.utils.frame_dataset

## Sample Generation

Provides functions for generating sample data for testing.

::: wandas.utils.generate_sample

## Type Definitions

Provides type definitions used in Wandas.

::: wandas.utils.types

## General Utilities

Provides other general utility functions.

::: wandas.utils.util
