---
name: wandas-visualization
description: Use when plotting waveforms, frequency spectra, spectrograms, or octave band charts with wandas, overlaying multiple signals on the same axes, or configuring describe() with frequency range and colormap settings.
---

# wandas: Visualization

Every frame type has a `.plot()` method returning a Matplotlib `Axes`. Use `.describe()` on `ChannelFrame` for a combined summary view.

## Plot Methods by Frame Type

| Frame | Method | Notes |
|-------|--------|-------|
| `ChannelFrame` | `.plot(title, ax, overlay=False)` | Waveform |
| `ChannelFrame` | `.rms_plot()` | RMS trend over time |
| `ChannelFrame` | `.describe(...)` | Waveform + spectrum + spectrogram |
| `SpectralFrame` | `.plot(title, ax, overlay=False, Aw=False)` | Frequency spectrum |
| `SpectralFrame` | `.plot_matrix()` | Multi-channel matrix view |
| `SpectrogramFrame` | `.plot(cmap, title)` | Time-frequency heatmap |
| `SpectrogramFrame` | `.plot_Aw()` | A-weighted spectrogram |
| `NOctFrame` | `.plot()` | Octave band bar chart |

## describe() Parameters

```python
signal.describe(
    fmin=0,           # lower frequency limit for spectrum/spectrogram (Hz)
    fmax=None,        # upper frequency limit
    cmap="jet",       # colormap for spectrogram
    vmin=None,        # spectrogram color range min (dB)
    vmax=None,        # spectrogram color range max (dB)
    xlim=None,        # time axis limits (s, s)
    ylim=None,        # amplitude axis limits
    Aw=False,         # A-weighted spectrogram
    waveform={},      # extra kwargs passed to waveform plot
    spectral={},      # extra kwargs passed to spectrum plot
)
```

## Patterns

**Overlay two spectra on the same axes:**
```python
import wandas as wd
import matplotlib.pyplot as plt

original = wd.read_wav("noisy.wav")
filtered = original.band_pass_filter(low_cutoff=100, high_cutoff=4000)

ax = original.fft().plot(overlay=True, label="Original")
filtered.fft().plot(ax=ax, overlay=True, label="Filtered", title="Filter Comparison")
plt.legend()
plt.show()
```

**Customized describe() with frequency focus:**
```python
signal = wd.read_wav("machine.wav")
signal.describe(
    fmin=100,
    fmax=5000,
    cmap="inferno",
    vmin=-80,
    vmax=-20,
)
```

## Common Mistakes

- Call `plt.show()` explicitly outside Jupyter - `.plot()` returns `Axes` but does not display.
- Pass `ax=` when overlaying - each `.plot()` call creates a new figure by default.
- `overlay=True` plots all channels of one frame together; for comparing two separate frames, use `ax=`.