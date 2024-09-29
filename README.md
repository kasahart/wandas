# Wandas: **W**aveform **An**alysis **Da**ta **S**tructures

**Wandas** is an open-source library for signal processing in Python. It adopts the user-friendly data structures and APIs of pandas, enabling efficient analysis of waveform and time-series data.

## Features

- **Intuitive Data Structures**: Provides `WaveFrame` for handling waveform data and `SignalSeries` for one-dimensional signal data.
- **Comprehensive Signal Processing Functions**: Easily perform filtering, Fourier transforms, spectral analysis, and other essential signal processing operations.
- **Powerful Support for Time-Series Data**: Easily manipulate time axes and change sampling rates.
- **High Performance**: Built on NumPy and SciPy, allowing fast computations even with large datasets.
- **Rich Data Input/Output**: Supports reading and writing various data formats like WAV, FLAC, CSV, and MAT files.
- **Integration with Visualization Libraries**: Seamlessly integrate with libraries like Matplotlib and Seaborn for easy data visualization.

## Installation

```bash
pip install wandas
```

## Quick Start

```python
import wandas as wd

# Read a WAV file
signal = wd.read_wav('audio_sample.wav')

# Plot the signal
signal.plot()

# Apply a low-pass filter
filtered_signal = signal.low_pass_filter(cutoff=1000)

# Perform Fourier transform for spectral analysis
spectrum = filtered_signal.fft()

# Plot the spectrum
spectrum.plot()

# Write the filtered signal to a WAV file
filtered_signal.to_wav('filtered_audio.wav')
```

## Documentation

For detailed usage and API references, please visit the [official documentation](https://wandas.readthedocs.io/).

## Supported Data Formats

- **Audio Files**: WAV, FLAC, MP3 (read-only)
- **Data Files**: CSV, Excel, MAT files
- **Real-Time Data**: Streaming data from microphones and sensors (planned for future release)

## Community and Contribution

Wandas is an open-source project, and we welcome contributions from everyone.

### Bug Reports and Feature Requests

- **Bug Reports**: Please provide details in the [Issue Tracker](https://github.com/yourusername/wandas/issues).
- **Feature Requests**: If you have ideas for new features or improvements, feel free to open an issue.

### How to Contribute

1. Fork this repository.
2. Create a feature branch. (`git checkout -b feature/new_feature`)
3. Commit your changes. (`git commit -m 'Add new_feature'`)
4. Push to the branch. (`git push origin feature/new_feature`)
5. Open a Pull Request.

## License

This project is licensed under the [MIT License](LICENSE).

---

Experience efficient signal processing with Wandas!
