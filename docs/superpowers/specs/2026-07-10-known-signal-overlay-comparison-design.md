# Known-Signal Overlay Comparison Design

## Goal

Use one deterministic mono signal to compare an original frame with a method-chained DC-removal and low-pass result, then overlay both waveform and FFT views in the bilingual README.

## Signal and Processing Flow

- Generate one mono signal containing 750 Hz and 1500 Hz tones plus a 0.25 DC offset.
- Create the original frame with the channel label `Original`.
- Build the processed frame with one readable method chain:

```python
processed = (
    signal
    .remove_dc()
    .low_pass_filter(cutoff=1_000)
    .rename_channels({0: "After DC removal + 1 kHz low-pass"})
)
```

- Build `comparison = signal.add_channel(processed)` so the original and processed signals share one two-channel frame.
- Keep `processed` and `comparison` as named values so readers can inspect `previous` and `operation_history` without duplicating the processing chain.

## Visualization

- Plot `comparison` with `overlay=True` over 0–0.02 seconds.
- Chain FFT directly to plotting and retain the returned axes.
- Use a linear frequency axis with `xlim=(0, 4_000)`.
- Set the FFT vertical range from the plotted peak down to peak minus 60 dB.
- Show DC removal in the waveform comparison and attenuation of the 1500 Hz component in the FFT comparison.
- Keep the existing two image paths and regenerate both images from the published code.

## Documentation and Tests

- Keep the English and Japanese Python blocks identical.
- Merge `origin/develop` so the branch includes #285, which makes overlay plots use every channel label by default.
- Omit explicit `label=comparison.labels` arguments from both README plots and rely on frame metadata.
- Explain the branch from one original signal to one processed signal and their recombination through `add_channel()`.
- Explain that `plot()` is terminal, while preprocessing and FFT can remain method-chained.
- Verify original and processed means, comparison labels, operation history, and the relative 750 Hz / 1500 Hz FFT amplitudes.
- Verify the published block contains no explicit plot `label` override and the rendered lines still use both comparison labels.
- Verify two overlaid lines, linear FFT x-axis, peak-relative 60 dB vertical range, titles, labels, and axis limits without pixel-exact comparisons.
- Run README/docs tests, Ruff, ty, and the full test suite before completion.

## Non-goals

- Do not change runtime behavior or public APIs.
- Do not duplicate the processing chain solely to create both figures.
- Do not add another test signal or additional filters.
