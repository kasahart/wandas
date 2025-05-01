# Wandas: **W**aveform **An**alysis **Da**ta **S**tructures

**Wandas** is an open-source library for efficient signal analysis in Python. Wandas provides comprehensive functionality for signal processing and seamless integration with Matplotlib.

## Features

- **Comprehensive Signal Processing Functions**: Easily perform basic signal processing operations including filtering, Fourier transforms, and STFT
- **Integration with Visualization Libraries**: Seamlessly integrate with Matplotlib for easy data visualization
- **Lazy Evaluation**: Efficiently process large data using dask
- **Various Analysis Tools**: Frequency analysis, octave band analysis, time-frequency analysis, and more

## Usage Examples

### Loading and Visualizing Audio Files

```python
import wandas as wd

cf = wd.read_wav("data/sample.wav")
cf.describe()
```

![Waveform and spectrogram display](../assets/images/read_wav_describe.png)

### Filtering

```python
signal = wd.generate_sin(freqs=[5000, 1000], duration=1)
# Apply low pass filter
signal.low_pass_filter(cutoff=1000).fft().plot()
```

![Low-pass filter results](../assets/images/low_pass_filter.png)

For detailed documentation and usage examples, see the [Tutorial](tutorial/index.md) and [Cookbook](how_to/index.md).

## Documentation Structure

- [Tutorial](tutorial/index.md) - 5-minute getting started guide
- [Cookbook](how_to/index.md) - Recipe collection for common tasks
- [API Reference](api/index.md) - Detailed API specifications
- [Theory & Architecture](explanation/index.md) - Design philosophy and algorithm explanations
- [Contributing Guide](contributing.md) - Rules and methods for contribution

## License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).
