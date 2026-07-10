# Known-Signal Overlay Comparison Design

## Goal

Change the bilingual README known-signal walkthrough from a processed-only, two-input-channel example to a direct overlay comparison between one original test signal and its `remove_dc()` result.

## Signal and Processing Flow

- Generate one mono signal containing 750 Hz and 1500 Hz tones plus a DC offset.
- Create the original frame with the channel label `Original`.
- Produce `clean = signal.remove_dc()` without mutating `signal`.
- Build a two-channel comparison frame with `signal.add_channel(clean, suffix_on_dup=" after remove_dc()")`.
- Preserve the resulting labels `Original` and `Original after remove_dc()`.
- Transform the comparison frame with FFT so the original DC component and its removal remain visible in the spectral comparison.

## Visualization

- Keep the existing two-image README structure: one waveform image and one spectrum image.
- Plot the two comparison channels with `overlay=True` in both figures.
- Use a 0–0.02 second waveform window and a 0–4,000 Hz spectrum window.
- Regenerate `images/readme_known_signal_waveform.png` and `images/readme_known_signal_spectrum.png` from the published code.

## Documentation and Tests

- Keep the English and Japanese Python blocks identical.
- Explain that `add_channel()` creates a comparison frame while preserving the original and processed frames.
- Retain the `previous` and `operation_history` explanation for `clean`.
- Update numerical tests to verify the original DC mean, the processed zero mean, the comparison labels, and the 750 Hz / 1500 Hz FFT components.
- Update plot tests to verify two figures, two overlaid lines per figure, comparison labels, titles, and axis limits without pixel-exact comparisons.
- Run README/docs tests, Ruff, ty, and the full test suite before completion.

## Non-goals

- Do not change the runtime behavior of `remove_dc()`, `add_channel()`, FFT, or plotting.
- Do not add a second generated test signal or introduce filter processing.
