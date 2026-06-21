# Wandas

English | [日本語](README.ja.md)

![Wandas logo](https://github.com/kasahart/wandas/blob/main/images/logo.png?raw=true)

[![PyPI](https://img.shields.io/pypi/v/wandas)](https://pypi.org/project/wandas/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/wandas)](https://pypi.org/project/wandas/)
[![CI](https://github.com/kasahart/wandas/actions/workflows/ci.yml/badge.svg)](https://github.com/kasahart/wandas/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/kasahart/wandas/graph/badge.svg?token=53NPNQQZZ8)](https://codecov.io/gh/kasahart/wandas)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/kasahart/wandas/blob/main/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/wandas)](https://pypi.org/project/wandas/)

Data structures for waveform analysis.

Wandas brings pandas-like workflows to time-domain, spectral, and spectrogram analysis.

## Overview

Wandas is an open-source Python library for signal and waveform analysis with chainable, frame-based APIs.

It helps you move from raw data to inspection, filtering, spectral analysis, and plotting without losing context such as sampling rate, channel labels, and metadata.

## Why Wandas

- Work with waveform data using familiar, pandas-like objects instead of ad hoc NumPy arrays.
- Keep metadata, channel information, and operation history attached as analysis grows.
- Move smoothly between time-domain, spectral, and spectrogram views with a consistent API.
- Use built-in plotting and summary helpers to inspect signals quickly.
- Scale to larger data with Dask-backed lazy execution where available.

## Quick Start

Install from PyPI with the recommended marimo extra:

```bash
pip install "wandas[marimo]"
```

For a minimal core-only install:

```bash
pip install wandas
```

### Installation Options

The core-only install keeps waveform, CSV/WAV, processing, plotting, and `describe()` figure/export workflows available when you use non-display options such as `is_close=False` or `image_save`. The default interactive `frame.describe()` display path requires `wandas[marimo]`.

Install optional extras when you need additional file formats or heavier analysis features:

```bash
pip install "wandas[io]"              # WDF save/load support
pip install "wandas[effects]"         # librosa-backed audio effects
pip install "wandas[marimo]"          # marimo learning apps and interactive display support
pip install "wandas[psychoacoustic]"  # loudness, roughness, octave-band helpers
pip install "wandas[ml]"              # Torch/TensorFlow tensor helpers
```

Combine extras as needed:

```bash
pip install "wandas[marimo,io,effects,psychoacoustic]"
```

Then read a signal file and inspect it in one short path:

```python
import wandas as wd

# Read a signal file and inspect it.
signal = wd.read("audio.wav")
signal.describe()
```

`describe()` gives you a quick visual summary of the waveform, spectrum, and spectrogram.

![cf.describe](https://github.com/kasahart/wandas/blob/main/images/read_wav_describe.png?raw=true)

## Public API

For most workflows, start with the small top-level API:

```python
import numpy as np
import wandas as wd

signal = wd.read("audio.wav")      # WAV, CSV, supported audio, URL, bytes, file-like
saved = wd.load("analysis.wdf")    # Wandas native WDF
data = np.zeros((1, 48000), dtype=np.float32)
array_signal = wd.from_numpy(data, sampling_rate=48000)
dataset = wd.from_folder("recordings/")
```

`read_wav()`, `read_csv()`, and `from_ndarray()` remain available for existing code, but new examples use `read()` and `from_numpy()`.

## What You Can Do

- Read waveform and sensor data from registered reader formats: WAV, FLAC, OGG, AIFF/AIF, SND, and CSV. WDF is available through the separate save/load API.
- Filter, resample, normalize, and summarize signals with method chaining.
- Run FFT, STFT, Welch, coherence, transfer-function, and octave-style analyses.
- Compute psychoacoustic metrics such as loudness and roughness.
- Plot waveforms, spectra, and spectrograms directly with Matplotlib-friendly APIs.

## Learn More

- [Documentation](https://kasahart.github.io/wandas/) - Guides, API reference, and examples.
- [Learning Path](https://github.com/kasahart/wandas/tree/main/learning-path/) - marimo app-based walkthroughs.
- [Examples](https://github.com/kasahart/wandas/tree/main/examples/) - Small runnable scripts and sample data.

## Contributing

Contributions are welcome.

For setup, quality checks, documentation rules, and pull request workflow, see [docs/src/contributing.md](https://kasahart.github.io/wandas/contributing/).

If you want to report a bug or propose an idea, please use the [Issue Tracker](https://github.com/kasahart/wandas/issues).

## License

Released under the [MIT License](https://github.com/kasahart/wandas/blob/main/LICENSE).
