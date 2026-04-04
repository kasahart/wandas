---
name: wandas-spectral-analysis
description: Use when performing FFT, STFT, Welch PSD estimation, octave band analysis, coherence, cross-spectral density, or transfer function analysis with wandas.
---

# wandas: Spectral Analysis

Transform `ChannelFrame` into frequency-domain frames. Each returns a different frame type.

## Transform Methods (on ChannelFrame)

| Method | Returns | Notes |
|--------|---------|-------|
| `.fft(n_fft=None, window="hann")` | `SpectralFrame` | Single-sided FFT |
| `.welch(n_fft=2048, hop_length=512, window="hann")` | `SpectralFrame` | Averaged PSD |
| `.stft(n_fft=2048, hop_length=512, window="hann")` | `SpectrogramFrame` | Time-frequency |
| `.noct_spectrum(fmin=25, n=3)` | `NOctFrame` | 1/N octave bands |
| `.coherence(n_fft=2048)` | `SpectralFrame` | Coherence between all channel pairs |
| `.csd(n_fft=2048)` | `SpectralFrame` | Cross-spectral density between all channel pairs |
| `.transfer_function(n_fft=2048)` | `SpectralFrame` | Frequency response between all channel pairs |

## Inverse Transforms

| Method | Returns |
|--------|---------|
| `spectral_frame.ifft()` | `ChannelFrame` |
| `spectrogram_frame.istft()` | `ChannelFrame` |

## Frame Transitions

```
ChannelFrame --fft/welch--> SpectralFrame --ifft--> ChannelFrame
     |
     +--stft--> SpectrogramFrame --istft--> ChannelFrame
     |
     +--noct_spectrum--> NOctFrame
```

## SpectralFrame Properties

```python
spectrum.freqs          # NDArray: frequency axis in Hz
```

## Patterns

**Basic FFT:**
```python
import wandas as wd

signal = wd.read_wav("audio.wav")
spectrum = signal.fft()
spectrum.plot(title="Frequency Spectrum")
print(spectrum.freqs[:5])   # first 5 frequency bins
```

**STFT spectrogram for anomaly detection:**
```python
sensor = wd.read_csv("sensor_log.csv", time_column="Time")
spectrogram = (
    sensor
    .high_pass_filter(cutoff=50)
    .stft(n_fft=2048, hop_length=512)
)
spectrogram.plot(cmap="inferno", title="Anomaly Detection")
```

**Coherence between two signals:**
```python
ref = wd.read_wav("ref.wav")
meas = wd.read_wav("meas.wav")
# Combine into a multi-channel frame, then compute coherence between all channel pairs
combined = ref.add_channel(meas)
coh = combined.coherence(n_fft=1024)
coh.plot(title="Coherence")
```

## Common Mistakes

- Do not call `.fft()` on a `SpectralFrame` - it is already frequency-domain.
- `.welch()` returns a `SpectralFrame` (averaged), not a `SpectrogramFrame`.
- Always specify `n_fft` and `hop_length` for `.stft()` - they control time vs. frequency resolution trade-off.