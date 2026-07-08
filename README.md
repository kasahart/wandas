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

Instead of juggling `array`, `sampling_rate`, `channels`, and a notebook full of helper variables, you can create one frame, keep the context attached, and check whether the result matches the signal you meant to analyze.

That frame-first style is especially useful when analysis needs to be shared with teammates or AI agents: the code, labels, units, metadata, operation history, and generated figures stay close enough together to make implementation and review easier.

## Known-signal check

Start with a signal whose answer is already known: one channel contains 750 Hz and 1500 Hz tones with a DC offset, and the other contains 1500 Hz and 3000 Hz tones with a different DC offset. First use `describe()` to see the whole frame, then use focused plots to check the cleanup and spectrum.

```python
import numpy as np
import wandas as wd

sr = 48_000
t = np.arange(sr) / sr
labels = ["750 Hz + 1500 Hz", "1500 Hz + 3000 Hz"]


def tone(components, *, offset=0.0):
    return offset + sum(amplitude * np.sin(2 * np.pi * freq * t) for freq, amplitude in components)


samples = np.vstack([
    tone([(750, 0.20), (1500, 0.05)], offset=0.25),
    tone([(1500, 0.10), (3000, 0.02)], offset=-0.10),
]).astype(np.float64)

signal = wd.from_numpy(
    samples,
    sampling_rate=sr,
    label="known signal",
    ch_labels=labels,
    ch_units="Pa",
)

signal.describe(fmax=4_000, image_save="readme_known_signal_describe.png")

clean = signal.remove_dc()
spectrum = clean.welch(n_fft=4096)

clean.plot(overlay=True, xlim=(0, 0.02), title="Known signal after remove_dc()", label=labels)
spectrum.plot(overlay=True, xlim=(0, 4_000), title="Welch spectrum of the known signal", label=labels)
```

The first figures are Wandas `describe()` output from the original frame: waveform, spectrogram, and spectrum are collected in one view for each channel.

![Wandas describe output for the generated 750 Hz and 1500 Hz channel](images/readme_known_signal_describe_0.png)

![Wandas describe output for the generated 1500 Hz and 3000 Hz channel](images/readme_known_signal_describe_1.png)

The focused waveform view shows that the DC offset disappears after `remove_dc()`.

![Wandas waveform plot after removing DC offset from the generated signal](images/readme_known_signal_waveform.png)

The focused spectrum view then shows the tone components we put in: 750 Hz and 1500 Hz for the first channel, 1500 Hz and 3000 Hz for the second channel.

![Wandas Welch spectrum plot for the generated signal](images/readme_known_signal_spectrum.png)

## Why try Wandas?

- **Frame-first signal analysis**: use objects that know their sampling rate, duration, channels, labels, units, and metadata.
- **Reviewable workflows**: keep code, data context, operation history, and figures connected so teammates and AI agents can inspect the same analysis.
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

## Use Your Own Data

Once the known-signal check makes sense, replace the generated array with a recording. The same frame-first workflow still applies.

```python
import wandas as wd

recording = wd.read("recording.wav", start=0, end=10)
recording.describe(fmin=20, fmax=8_000, image_save="recording_overview.png")
```

For SPL, loudness, roughness, or octave-band work, use calibrated pressure data. Those examples need the `wandas[psychoacoustic]` extra. WDF save/load support lives in `wandas[io]`.

## Small top-level API

- `wd.read("audio.wav")`: WAV, CSV, supported audio, URL, bytes, and file-like input.
- `wd.from_numpy(data, sampling_rate=48_000)`: create a frame from an array.
- `wd.from_folder("recordings/", recursive=True)`: build a folder-backed dataset.
- `wd.load("analysis.wdf")`: load Wandas native WDF files with `wandas[io]`.
- `wd.supported_formats()`: inspect registered reader formats.

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
