# Wandas

English | [日本語](https://github.com/kasahart/wandas/blob/main/README.ja.md)

![Wandas logo](https://github.com/kasahart/wandas/blob/main/images/logo.png?raw=true)

[![PyPI](https://img.shields.io/pypi/v/wandas)](https://pypi.org/project/wandas/)
[![CI](https://github.com/kasahart/wandas/actions/workflows/ci.yml/badge.svg)](https://github.com/kasahart/wandas/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/kasahart/wandas/graph/badge.svg?token=53NPNQQZZ8)](https://codecov.io/gh/kasahart/wandas)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/kasahart/wandas/blob/main/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/wandas)](https://pypi.org/project/wandas/)

**Signal analysis that feels like working with a data frame.**

Wandas is a Python library that treats audio, vibration and sensor waveforms as one frame (`ChannelFrame`) that carries the sampling rate, channel names, units and processing history. Reading, preprocessing, frequency analysis and plotting are written as a method chain.

```python
import wandas as wd

x = wd.read("https://raw.githubusercontent.com/kasahart/wandas/main/learning-path/sample_audio.wav", end=15)  # first 15 s
x.describe()                                               # waveform, spectrogram and spectrum at a glance
x.remove_dc().high_pass_filter(cutoff=100).welch().plot()  # preprocess, then a Welch spectrum
```

![Wandas describe() output](https://raw.githubusercontent.com/kasahart/wandas/main/images/readme_sample_audio_describe_0.png)

## Why Wandas

With NumPy and SciPy alone, you keep track of the sampling rate, channel names, units and "what was done" yourself.

```python
# NumPy / SciPy: you carry the array and the sampling rate yourself
import scipy.signal

a, fs = x.data, x.sampling_rate
sos = scipy.signal.butter(4, 100, "highpass", fs=fs, output="sos")
y_np = scipy.signal.sosfiltfilt(sos, a - a.mean(axis=-1, keepdims=True))

# Wandas: the same 4th-order zero-phase Butterworth
y = x.remove_dc().high_pass_filter(cutoff=100)
y.operation_history  # [{'operation': 'wandas.audio.remove_dc', ...}, ...]
```

- Methods never change the original frame; each returns a new one, and the processing history stays on the result.
- A processing chain can be saved and re-applied to another recording ([`RecipePlan`](https://kasahart.github.io/wandas/how-to/pipeline-recipes/)).

## What you can do

| Area | Main features |
|------|---------------|
| Input | WAV / FLAC / OGG / MP3 / CSV, URLs, NumPy arrays |
| Preprocessing | DC removal, high/low/band-pass, A-weighting, resampling, trimming |
| Levels | Per-channel calibration, RMS, dB SPL |
| Frequency analysis | FFT, Welch, STFT, 1/N-octave bands, coherence, transfer functions |
| Psychoacoustics | Loudness, roughness, sharpness (via MoSQITo) |
| Many files | Build a recording list from folders or CSV and process it in one go |

## Installation

```bash
pip install wandas
pip install "wandas[marimo]"          # audio playback in describe() on Jupyter / marimo
pip install "wandas[psychoacoustic]"  # psychoacoustic metrics
```

## Check with a known signal

Add a DC offset to 750 Hz and 1500 Hz tones, then remove the DC and apply a 1 kHz low-pass filter.

```python
import numpy as np
import wandas as wd

sr = 48_000
t = np.arange(sr) / sr
samples = 0.25 + 0.20 * np.sin(2 * np.pi * 750 * t) + 0.05 * np.sin(2 * np.pi * 1500 * t)

signal = wd.from_numpy(samples, sampling_rate=sr, ch_labels=["Original"], ch_units="Pa")
processed = signal.remove_dc().low_pass_filter(cutoff=1_000).rename_channels({0: "After DC removal + 1 kHz low-pass"})

comparison = signal.concat_frame(processed)
comparison.plot(overlay=True, xlim=(0, 0.02))
comparison.fft().plot(overlay=True, xlim=(0, 4_000)).set_ylim(30, 90)  # dB SPL (re 20 µPa)
```

![Overlaid waveforms of the original and processed signal](https://raw.githubusercontent.com/kasahart/wandas/main/images/readme_known_signal_waveform.png)

![Overlaid FFT spectra of the original and processed signal](https://raw.githubusercontent.com/kasahart/wandas/main/images/readme_known_signal_spectrum.png)

The 750 Hz tone stays and the 1500 Hz tone is attenuated.

## Work with many files

```python
import wandas as wd

# recordings/
# ├── label=ok/  a.wav, b.wav, ...
# └── label=ng/  c.wav, d.wav, ...
dataset = wd.from_folder("recordings/", recursive=True, path_metadata=True)
ok = dataset.select(label="ok")
spectra = ok.trim(0, 5).resample(16_000).stft(n_fft=512)  # first 5 s
spectra[0].data  # NumPy array (channel, frequency, time); mono has no channel axis
```

`from_folder()` and `select()` do not open any audio file, so selection stays fast even for many files. Each recording is read and computed only when you take a result out, such as `spectra[0]`.

## Learn more

- [Documentation](https://kasahart.github.io/wandas/)
- [English Learning Path](https://kasahart.github.io/wandas/en/learning-path/00_why_wandas.html)

## License

[MIT License](https://github.com/kasahart/wandas/blob/main/LICENSE). Bug reports and ideas are welcome in the [issue tracker](https://github.com/kasahart/wandas/issues).
