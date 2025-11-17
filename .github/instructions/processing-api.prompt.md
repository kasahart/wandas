# Wandas Processing API Prompt

Use this prompt when working on `wandas/processing/` or adding new signal processing operations.

## Layered responsibilities
- `wandas/processing/` implements **pure numerical logic**:
  - filter design and application,
  - FFT/STFT, spectral transforms, time-frequency analysis,
  - psychoacoustic metrics, stats, temporal features, effects.
- `wandas/frames/` is responsible for:
  - input validation and axis handling,
  - metadata and `operation_history` updates,
  - returning the correct frame type.

## Function design
- Prefer small, composable functions that:
  - take NumPy/Dask arrays and plain parameters as inputs,
  - return arrays or simple structs (no frames),
  - are easy to test in isolation.
- Reuse existing helpers for resampling, filtering, FFT, etc., instead of duplicating logic.

## API consistency
- Follow naming and signature patterns from nearby modules (e.g. `filters.py`, `spectral.py`, `temporal.py`).
- Keep public APIs stable and well-typed; introduce new options as keyword-only parameters when necessary.
- Avoid speculative flags or configuration options (YAGNI) until a concrete use case or test requires them.

Use this prompt to keep the processing layer focused, reusable, and easy to reason about.
