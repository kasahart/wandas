# Known-Signal Overlay Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare one original mono signal with a method-chained DC-removal and 1 kHz low-pass result using `add_channel()` and overlaid waveform/FFT plots.

**Architecture:** Keep the processed branch named for history inspection, recombine it with the original through `add_channel()`, and chain FFT directly into `plot()`. Tests execute the published block and validate numerical, lineage, and plot semantics.

**Tech Stack:** Markdown, Python 3.10+, NumPy, Wandas, Matplotlib, pytest, `uv`, Ruff, ty

---

### Task 1: Update executable comparison behavior

**Files:**
- Modify: `tests/docs/test_readme_examples.py`
- Modify: `README.md`
- Modify: `README.ja.md`

- [ ] Change the numerical test to require `processed`, labels `Original` / `After DC removal + 1 kHz low-pass`, operation history ending in `remove_dc` and `lowpass_filter`, zero processed mean, preserved 750 Hz, and attenuated 1500 Hz.
- [ ] Run the focused test and confirm failure against the current remove-DC-only README.
- [ ] Replace both README blocks with this processing flow:

```python
processed = (
    signal
    .remove_dc()
    .low_pass_filter(cutoff=1_000)
    .rename_channels({0: "After DC removal + 1 kHz low-pass"})
)
comparison = signal.add_channel(processed)
```

- [ ] Plot `comparison` with `overlay=True`, then plot `comparison.fft(n_fft=sr)` on a linear 0–4,000 Hz axis and set the vertical range from the plotted peak down 60 dB.
- [ ] Update English and Japanese prose to explain waveform DC removal and 1500 Hz attenuation.
- [ ] Run focused numerical and plot tests until they pass.

### Task 2: Regenerate figures and verify

**Files:**
- Modify: `images/readme_known_signal_waveform.png`
- Modify: `images/readme_known_signal_spectrum.png`
- Modify: `docs/superpowers/plans/2026-07-10-known-signal-overlay-comparison.md`

- [ ] Execute the published English known-signal block and save its two figures to the existing image paths.
- [ ] Visually confirm two overlay lines, waveform offset removal, a peak-relative 60 dB FFT range, and 1500 Hz attenuation.
- [ ] Run `uv run pytest tests/docs`, `uv run ruff check wandas tests`, `uv run --extra marimo --extra psychoacoustic ty check wandas tests`, and `uv run --extra effects pytest`.
- [ ] Remove generated caches, run `git diff --check`, and commit with `docs: compare known signal before and after filtering`.
