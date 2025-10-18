# Theory Background and Architecture

This section explains the design philosophy, internal architecture, and theoretical background used in the Wandas library.

## Design Philosophy

Wandas is developed based on the following design principles:

1. **Intuitive API Design** - Consistent interface that users can easily use
2. **Efficient Memory Usage** - Memory-efficient implementation suitable for processing large-scale data
3. **Extensibility** - Expandable architecture that makes it easy to add new features and algorithms
4. **Scientific Accuracy** - Accurate implementation based on acoustic signal processing theory

## Core Architecture

### Data Model

The central data model of the Wandas library is hierarchically structured:

```
BaseChannel (base class)
 ├── Channel (time-domain signal)
 │    └── FrequencyChannel (frequency-domain signal)
 │         └── TimeFrequencyChannel (time-frequency domain signal)
 └── ChannelFrame (container for multiple channels)
      ├── FileFrame (file-based multiple channels)
      └── FrequencyChannelFrame (multiple channels in frequency domain)
```

Responsibilities of each class:

- **BaseChannel**: Base class for all channels. Provides basic functionality for data access and metadata management
- **Channel**: Implements time-domain signal data and processing methods
- **FrequencyChannel**: Implements FFT-based frequency-domain data and processing
- **TimeFrequencyChannel**: Implements time-frequency domain representations such as Short-Time Fourier Transform (STFT)
- **ChannelFrame**: Manages multiple channels and enables batch processing

### Data Processing Flow

1. **Input Stage**: Generate `Channel` or `ChannelFrame` objects from files such as WAV and CSV
2. **Processing Stage**: Apply processing such as filtering and resampling
3. **Analysis Stage**: Analyze signal characteristics (spectrum, level, etc.)
4. **Output Stage**: Save processing results to files or visualize as graphs

## Implementation Details

### Memory Efficiency

Wandas ensures memory efficiency for handling large audio data through the following methods:

- **Lazy Evaluation**: A mechanism that delays calculations until needed
- **Memory Mapping**: Access to large files without loading them entirely into memory
- **Dask and H5PY**: Utilizing libraries suitable for large-scale data processing

### Signal Processing Algorithms

Wandas implements signal processing algorithms such as:

- **Digital Filters**: IIR/FIR filters such as Butterworth filters
- **Spectral Analysis**: Frequency analysis based on Fast Fourier Transform (FFT)
- **Time-Frequency Analysis**: Short-Time Fourier Transform (STFT), spectrograms
- **Statistical Analysis**: Calculation of signal characteristics such as RMS, peak values, crest factor

## Performance Considerations

Performance considerations when using Wandas:

- When processing large amounts of data, consider processing in chunks
- When building complex processing chains, improve performance by caching intermediate results
- Multi-channel processing efficiently utilizes multi-core processors

## Psychoacoustic Metrics

Wandas provides psychoacoustic metrics for analyzing audio signals based on human perception:

- **[Loudness Calculation](psychoacoustic_metrics.en.md)**: Time-varying loudness calculation using Zwicker method according to ISO 532-1:2017

## References

1. Smith, J. O. (2011). Spectral Audio Signal Processing. W3K Publishing.
2. Müller, M. (2015). Fundamentals of Music Processing: Audio, Analysis, Algorithms, Applications. Springer.
3. Zölzer, U. (2008). Digital Audio Signal Processing. Wiley.
