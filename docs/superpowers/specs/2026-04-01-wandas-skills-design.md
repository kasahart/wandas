# Wandas Skills Design

**Date:** 2026-04-01
**Status:** Approved

## Context

wandas is a Python library for waveform and signal analysis. Users working in Jupyter notebooks or scripts need AI-assisted guidance on the pandas-like API: how to load data, apply filters, perform spectral transforms, and visualize results. This spec defines 4 domain-split skills covering the user-facing workflow, available in both Claude Code (superpowers plugin) and GitHub Copilot formats.

## Goal

Create concise, accurate reference skills that help users write correct wandas code quickly, with working code examples and common-mistake warnings. Replace the existing single-skill `wandas-signal-analysis-helper` with 4 focused skills.

## Scope

**In scope:**
- 4 superpowers skills in `.claude/skills/`
- 4 GitHub Copilot skills in `.github/skills/`
- Replacement of existing `.github/skills/wandas-signal-analysis-helper/`

**Out of scope:**
- Developer/extension skills (adding new Frame types, custom operations at the library level)
- Psychoacoustic analysis (advanced domain)
- WDF I/O format

---

## Skill Structure

### 1. `wandas-getting-started`

**Trigger:** wandas initial use, loading data, understanding Frame types.

**Content:**
- Top-level I/O functions: `wd.read_wav()`, `wd.read_csv()`, `wd.from_numpy()`, `wd.generate_sin()`, `wd.from_folder()`
- Frame type overview: ChannelFrame (time), SpectralFrame (freq), SpectrogramFrame (time-freq), NOctFrame (octave bands)
- Inspection: `signal.describe()`, `signal.info()`, `.sampling_rate`, `.n_channels`, `.duration`
- Key concept: Dask lazy execution — `.compute()` to materialize to NumPy
- Common mistake: assuming operations are eager; they are lazy

**Code examples:**
1. Load WAV and inspect
2. Create synthetic signal and compare with loaded CSV

---

### 2. `wandas-signal-processing`

**Trigger:** Applying filters, normalization, resampling, fade, RMS, harmonic-percussive separation.

**Content:**
- Filters: `.low_pass_filter(cutoff, order=4)`, `.high_pass_filter(cutoff, order=4)`, `.band_pass_filter(low_cutoff, high_cutoff, order=4)`, `.a_weighting()`
- Normalization: `.normalize(norm=inf, axis=-1)`, `.remove_dc()`
- Temporal: `.resampling(target_rate)`, `.trim(start, end)`, `.fix_length(length)`, `.fade(fade_ms=50)`
- Trends: `.rms_trend(frame_length, hop_length)`, `.sound_level(...)`
- Effects: `.hpss_harmonic(...)`, `.hpss_percussive(...)`
- Custom: `.apply(func, output_shape_func=...)`
- Pattern: Method chaining
- Common mistakes: wrong param names on `band_pass_filter`, applying `a_weighting()` after FFT

**Code examples:**
1. Noise reduction pipeline (highpass + lowpass + normalize)
2. HPSS: separate harmonic from percussive

---

### 3. `wandas-spectral-analysis`

**Trigger:** FFT, STFT, Welch, spectrogram, coherence, CSD, transfer function, octave bands.

**Content:**
- Transforms: `.fft(n_fft, window)`, `.welch(n_fft, hop_length, window)`, `.stft(n_fft, hop_length, window)`
- Octave: `.noct_spectrum(n=3)` → NOctFrame
- Two-signal: `.coherence(other, n_fft)`, `.csd(other, n_fft)`, `.transfer_function(other, n_fft)`
- Inverse: `spectral.ifft()` → ChannelFrame, `spectrogram.istft()` → ChannelFrame
- Frame transition diagram: ChannelFrame ↔ SpectralFrame ↔ SpectrogramFrame / NOctFrame
- SpectralFrame properties: `.freqs`, `.magnitudes`, `.phases`
- Common mistakes: calling `.fft()` on SpectralFrame, forgetting n_fft/hop_length for stft, expecting welch to return SpectrogramFrame

**Code examples:**
1. FFT of filtered signal, plot spectrum
2. STFT spectrogram for anomaly detection (sensor data CSV)

---

### 4. `wandas-visualization`

**Trigger:** Plotting waveforms, spectra, spectrograms, overlaying signals, describe() configuration.

**Content:**
- Per-frame plot methods:
  - ChannelFrame: `.plot(title, ax)`, `.rms_plot()`, `.describe(...)`
  - SpectralFrame: `.plot(title, ax, label)`, `.plot_matrix()`
  - SpectrogramFrame: `.plot(cmap, title)`, `.plot_Aw()`
  - NOctFrame: `.plot()`
- Overlay pattern: `ax = frame1.plot(...); frame2.plot(ax=ax, ...)`
- Describe config: `fmin`, `fmax`, `cmap`, `vmin`, `vmax`, `xlim`, `ylim`, `Aw`, `waveform={}`, `spectral={}`
- Common mistakes: forgetting `plt.show()` outside Jupyter, not passing `ax=` when overlaying

**Code examples:**
1. Overlay filter comparison with legend
2. Customized `describe()` with frequency range and colormap settings

---

## File Locations

```
# Superpowers (Claude Code)
.claude/skills/wandas-getting-started/SKILL.md
.claude/skills/wandas-signal-processing/SKILL.md
.claude/skills/wandas-spectral-analysis/SKILL.md
.claude/skills/wandas-visualization/SKILL.md

# GitHub Copilot
.github/skills/wandas-getting-started/examples/workflows.md
.github/skills/wandas-signal-processing/examples/workflows.md
.github/skills/wandas-spectral-analysis/examples/workflows.md
.github/skills/wandas-visualization/examples/workflows.md

# Delete (replaced by above)
.github/skills/wandas-signal-analysis-helper/
```

## API Accuracy Notes

- `band_pass_filter` takes `low_cutoff` and `high_cutoff` (NOT `low`/`high`) — existing Copilot skill has this bug
- `normalize` takes `norm` (not `method`) — default `float("inf")` = peak normalization
- `describe()` takes kwargs directly (not a TypedDict argument), keyword-only after `normalize` and `is_close`

## Verification

1. Confirm each SKILL.md description triggers correctly for natural language queries
2. Spot-check all method signatures against `wandas/frames/mixins/channel_processing_mixin.py` and `channel_transform_mixin.py`
3. Run code examples manually in a notebook or Python session to confirm no errors
