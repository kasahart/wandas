# Wandas

[English](README.md) | [日本語](https://github.com/kasahart/wandas/blob/main/README.ja.md)

![Wandas logo](https://github.com/kasahart/wandas/blob/main/images/logo.png?raw=true)

[![PyPI](https://img.shields.io/pypi/v/wandas)](https://pypi.org/project/wandas/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/wandas)](https://pypi.org/project/wandas/)
[![CI](https://github.com/kasahart/wandas/actions/workflows/ci.yml/badge.svg)](https://github.com/kasahart/wandas/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/kasahart/wandas/graph/badge.svg?token=53NPNQQZZ8)](https://codecov.io/gh/kasahart/wandas)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/kasahart/wandas/blob/main/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/wandas)](https://pypi.org/project/wandas/)

**Signal analysis that feels like working with data frames.**

Wandas gives waveform and time-series data a pandas-like home: a `ChannelFrame` keeps the samples, sampling rate, channel labels, units, metadata, and operation history together while you inspect, clean, transform, and plot signals.

Instead of juggling `array`, `sampling_rate`, `channels`, and a notebook full of helper variables, you can write a single readable analysis chain.

```python
import numpy as np
import wandas as wd

sr = 48_000
t = np.arange(sr) / sr
samples = np.vstack([
    np.sin(2 * np.pi * 440 * t),
    0.5 * np.sin(2 * np.pi * 880 * t),
]).astype(np.float32)

signal = wd.from_numpy(
    samples,
    sampling_rate=sr,
    label="demo tone",
    ch_labels=["440 Hz", "880 Hz"],
)

clean = signal.remove_dc().normalize()
clean.describe()

spectrum = clean.welch(n_fft=4096)
spectrogram = clean.stft(n_fft=1024)
```

`describe()` gives you a quick visual summary of the waveform, spectrum, and spectrogram.

![Wandas describe output](https://github.com/kasahart/wandas/blob/main/images/read_wav_describe.png?raw=true)

## Why try Wandas?

- **Frame-first signal analysis**: use objects that know their sampling rate, duration, channels, labels, units, and metadata.
- **A smooth path from raw data to insight**: read, trim, filter, normalize, resample, summarize, transform, and plot with consistent methods.
- **Time, frequency, and time-frequency views**: move from `ChannelFrame` to `SpectralFrame`, `SpectrogramFrame`, or `NOctFrame` without rebuilding context by hand.
- **Practical acoustics included**: RMS trends, sound level, A-weighting, octave-band spectra, loudness, and roughness are available when you need them.
- **Works with real workflows**: read WAV/FLAC/OGG/AIFF/SND/CSV, URLs, bytes, file-like objects, NumPy arrays, folders of recordings, and Wandas WDF files with the `io` extra.
- **Built for interactive exploration**: `describe()`, Matplotlib-friendly plotting, and marimo learning apps make quick inspection easy.

## Install

For the best first experience, install the interactive display and learning-app extra:

```bash
pip install "wandas[marimo]"
```

For a small core install:

```bash
pip install wandas
```

Core install includes waveform frames, CSV/WAV reading, processing, plotting, and `describe()` figure/export workflows when you use options such as `is_close=False` or `image_save`. The default interactive `frame.describe()` display path uses the `marimo` extra.

Optional extras can be mixed as needed:

```bash
pip install "wandas[io]"              # WDF save/load support
pip install "wandas[effects]"         # librosa-backed audio effects
pip install "wandas[marimo]"          # marimo learning apps and interactive display support
pip install "wandas[psychoacoustic]"  # loudness, roughness, octave-band helpers
pip install "wandas[ml]"              # Torch/TensorFlow tensor helpers

pip install "wandas[marimo,io,effects,psychoacoustic]"
```

## Start with your own data

### Read and inspect a recording

```python
import wandas as wd

signal = wd.read("recording.wav", start=0, end=10)
signal.info()
signal.describe(fmin=20, fmax=8_000)
```

### Clean it before analysis

```python
clean = (
    signal
    .remove_dc()
    .band_pass_filter(80, min(8_000, 0.45 * signal.sampling_rate))
    .normalize()
)

clean.rms_plot(Aw=True)
```

### Look at the frequency content

```python
spectrum = clean.welch(n_fft=2048)
spectrum.plot()

spectrogram = clean.stft(n_fft=2048, hop_length=512)
spectrogram.plot()

# N-octave spectra require the psychoacoustic extra.
third_octave = clean.noct_spectrum(
    n=3,
    fmax=min(20_000, 0.4 * clean.sampling_rate),
)  # requires wandas[psychoacoustic]
third_octave.plot()
```

### Compare channels and acoustic metrics

```python
# SPL-style dB plots require calibrated pressure data.
calibration_gain = 1.0  # Pa per sample; replace with your microphone calibration
pressure = signal * calibration_gain
for channel in pressure.channels:
    channel.unit = "Pa"
    channel.ref = 20e-6

level = pressure.sound_level(freq_weighting="A", time_weighting="Fast", dB=True)
level.plot(ylabel="LA Fast [dB re 20 uPa]")

# Psychoacoustic metrics require the psychoacoustic extra.
loudness = pressure.loudness_zwtv(field_type="free")
roughness = pressure.roughness_dw(overlap=0.5)
```

## Small top-level API

```python
import numpy as np
import wandas as wd

signal = wd.read("audio.wav")          # WAV, CSV, supported audio, URL, bytes, file-like
saved = wd.load("analysis.wdf")        # Wandas native WDF; requires wandas[io]
data = np.zeros((2, 48_000), dtype=np.float32)
array_signal = wd.from_numpy(data, sampling_rate=48_000)
dataset = wd.from_folder("recordings/", recursive=True)
formats = wd.supported_formats()
```

`read_wav()`, `read_csv()`, and `from_ndarray()` remain available for existing code, but new examples use `read()` and `from_numpy()`.

## Core objects

- `ChannelFrame`: time-domain waveform or sensor data with channels.
- `SpectralFrame`: FFT, Welch, coherence, CSD, and transfer-function results.
- `SpectrogramFrame`: STFT and time-frequency data.
- `NOctFrame`: octave and fractional-octave spectra.
- `ChannelFrameDataset`: a folder-backed collection of channel frames for batch workflows.

## Good fits

Wandas is especially useful when you want to:

- prototype signal-processing pipelines in notebooks or marimo apps;
- keep channel metadata attached while trying filters and transforms;
- inspect acoustic recordings quickly before deeper analysis;
- compare many WAV/CSV files with the same API;
- build readable examples for signal-processing education.

## Learn more

- [Documentation](https://kasahart.github.io/wandas/) - Guides, API reference, and examples.
- [Learning Path](https://github.com/kasahart/wandas/tree/main/learning-path/) - marimo app-based walkthroughs.
- [Tutorial](https://kasahart.github.io/wandas/tutorial/) - A guided walkthrough of the core workflow.
- [Issue Tracker](https://github.com/kasahart/wandas/issues) - Report bugs or propose ideas.

## Project status

Wandas is actively evolving. The package currently targets Python 3.10+ and is published under the MIT License. If you use it in a production workflow, pin the version and check release notes when upgrading.

## Contributing

Contributions are welcome.

For setup, quality checks, documentation rules, and pull request workflow, see [docs/src/contributing.md](https://kasahart.github.io/wandas/contributing/).

## License

Released under the [MIT License](https://github.com/kasahart/wandas/blob/main/LICENSE).
