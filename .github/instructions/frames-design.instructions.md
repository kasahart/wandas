---
description: "Frame architecture: immutability, metadata propagation, and runtime lineage rules for ChannelFrame, SpectralFrame, and SpectrogramFrame"
applyTo: "wandas/frames/**"
---
# Wandas Frames Design Prompt

Use this prompt when working on `wandas/frames/` or any code that manipulates `ChannelFrame`, `SpectralFrame`, or `SpectrogramFrame`.

## Core principles
- Frames are **immutable**: never mutate data, metadata, or runtime provenance in place.
- Always create a **new frame instance** when applying an operation.
- Update data, metadata, axes information, and `lineage` **atomically**.

## Metadata & lineage updates
- Prefer dedicated helpers (e.g. `_with_updated_metadata`, `replace(...)`) on frame classes to:
  - swap out the underlying array (NumPy/Dask),
  - update sampling rate, axes, and channel labels,
  - append runtime lineage with operation parameters,
  - carry over user/recording metadata.
- `operation_history` is a read-only compatibility view derived from lineage.
- Do not duplicate operation parameters into `metadata[operation_name]`; metadata should carry user, recording, and domain state.
- Avoid scattered `frame.metadata[...] = ...` or direct provenance mutations in callers; encapsulate these inside `frames/`.

## Where to put logic
- Keep **orchestration** in frames:
  - user-facing methods (e.g. `low_pass_filter`, `fft`, `stft`, `normalize`).
  - input validation, axis alignment, metadata management, and lineage recording.
- Keep **numerical logic** in `wandas/processing/`:
  - filtering, FFT, psychoacoustic metrics, resampling, stats, effects.

## When adding/modifying frame methods
- Mirror existing patterns in `ChannelFrame`, `SpectralFrame`, and `SpectrogramFrame`:
  - method naming (verbs like `normalize`, `resample`, `low_pass_filter`).
  - method chaining-friendly signatures (return a new frame of the same conceptual type).
- Ensure that:
  - sampling rate and axes are consistent with the operation performed,
  - channel labels remain aligned with the underlying data,
  - `lineage` captures operation provenance for debugging; `operation_history` exposes its flat compatibility summary.

Use this as a checklist whenever you change or add frame methods.

## Extension workflow

For the decision to add a new Frame family, the required constructor and axis
contracts, public exports, Recipe support, documentation, and complete test matrix,
follow the [Frame and Operation extension guide](../../docs/src/contributing/frame-operation-extensions.md).
