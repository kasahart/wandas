---
name: wandas-signal-processing
description: Use when applying filters (lowpass, highpass, bandpass, A-weighting), normalizing signals, resampling, trimming, adding fades, computing RMS trends, or separating harmonic and percussive components with wandas.
---

# wandas: Signal Processing

Time-domain operations on `ChannelFrame`. Every method returns a new frame - chain freely.

## Filter Methods

| Method | Effect |
|--------|--------|
| `.low_pass_filter(cutoff, order=4)` | Remove frequencies above `cutoff` Hz |
| `.high_pass_filter(cutoff, order=4)` | Remove frequencies below `cutoff` Hz |
| `.band_pass_filter(low_cutoff, high_cutoff, order=4)` | Keep `low_cutoff`-`high_cutoff` Hz range |
| `.a_weighting()` | Apply A-weighting curve (time domain only) |

## Normalization & Shaping

| Method | Effect |
|--------|--------|
| `.normalize(norm=float("inf"), axis=-1)` | Peak normalization by default (`norm=inf`) |
| `.remove_dc()` | Remove DC offset |
| `.resampling(target_sr)` | Change sampling rate |
| `.trim(start=0, end=None)` | Trim by time in seconds |
| `.fix_length(length=None)` | Pad or truncate to fixed sample count |
| `.fade(fade_ms=50)` | Symmetric fade-in/out (Tukey window) |

## Trend & Level

| Method | Effect |
|--------|--------|
| `.rms_trend(frame_length=2048, hop_length=512)` | RMS over time windows |
| `.sound_level(freq_weighting="Z")` | Overall sound level |

## Effects

| Method | Effect |
|--------|--------|
| `.hpss_harmonic(kernel_size=31)` | Harmonic component (HPSS) |
| `.hpss_percussive(kernel_size=31)` | Percussive component (HPSS) |
| `.apply(func, output_shape_func=lambda s: s)` | Custom function |

## Patterns

**Noise reduction pipeline:**
```python
import wandas as wd

signal = wd.read_wav("noisy.wav")
cleaned = (
    signal
    .high_pass_filter(cutoff=50)       # remove power-line hum
    .low_pass_filter(cutoff=8000)      # remove high-freq noise
    .normalize()                       # peak normalize
    .fade(fade_ms=10)                  # smooth edges
)
cleaned.describe()
```

**Harmonic-percussive separation:**
```python
signal = wd.read_wav("music.wav")
harmonic = signal.hpss_harmonic()
percussive = signal.hpss_percussive()
harmonic.describe()
```

## Common Mistakes

- `band_pass_filter` uses `low_cutoff` / `high_cutoff` - not `low` / `high`.
- `.a_weighting()` operates on `ChannelFrame` only - do not call it after `.fft()`.
- `normalize()` default is peak normalization (`norm=float("inf")`), not RMS.
