# Wandas Frames Design Prompt

Use this prompt when working on `wandas/frames/` or any code that manipulates `ChannelFrame`, `SpectralFrame`, or `SpectrogramFrame`.

## Core principles
- Frames are **immutable**: never mutate data, metadata, or `operation_history` in place.
- Always create a **new frame instance** when applying an operation.
- Update data, metadata, axes information, and `operation_history` **atomically**.

## Metadata & operation_history updates
- Prefer dedicated helpers (e.g. `_with_updated_metadata`, `replace(...)`) on frame classes to:
  - swap out the underlying array (NumPy/Dask),
  - update sampling rate, axes, and channel labels,
  - append an `operation_history` entry with parameters,
  - carry over user/recording metadata.
- Avoid scattered `frame.metadata[...] = ...` or direct `operation_history` mutations in callers; encapsulate these inside `frames/`.

## Where to put logic
- Keep **orchestration** in frames:
  - user-facing methods (e.g. `low_pass_filter`, `fft`, `stft`, `normalize`).
  - input validation, axis alignment, metadata management, and history recording.
- Keep **numerical logic** in `wandas/processing/`:
  - filtering, FFT, psychoacoustic metrics, resampling, stats, effects.

## When adding/modifying frame methods
- Mirror existing patterns in `ChannelFrame`, `SpectralFrame`, and `SpectrogramFrame`:
  - method naming (verbs like `normalize`, `resample`, `low_pass_filter`).
  - method chaining-friendly signatures (return a new frame of the same conceptual type).
- Ensure that:
  - sampling rate and axes are consistent with the operation performed,
  - channel labels remain aligned with the underlying data,
  - `operation_history` captures what/why/how for debugging and reproducibility.

Use this as a checklist whenever you change or add frame methods.
