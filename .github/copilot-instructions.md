# Copilot Instructions for the Wandas Repository

These instructions are for AI coding agents (planner, implementer, reviewer) working on the Wandas codebase.

## 1. Big Picture & Architecture
- **Purpose**: Wandas provides pandas‑like data structures and operations for waveform/signal analysis (see `README.md`).
- **Core packages** (under `wandas/`):
  - `frames/`: user‑facing data structures (`ChannelFrame`, `SpectralFrame`, `SpectrogramFrame`) handling axes, metadata, and `operation_history`.
  - `processing/`: pure numerical logic for filters, spectral analysis, psychoacoustics, temporal/stats/effects; frame methods should delegate here.
  - `io/`: I/O helpers for WAV/WDF/CSV (`wav_io.py`, `wdf_io.py`, `readers.py`) plus sample data/datasets.
  - `visualization/`: plotting helpers returning Matplotlib `Axes` that build on frame methods (e.g. `plotting.py`).
  - `core/`, `utils/`, `datasets/`: shared protocol types, base classes, small utilities, dataset helpers.
- **Design goals**: immutable frame semantics, traceable `operation_history`, preserved metadata (including sampling rate and channel info), and optional Dask‑based lazy execution.

## 2. Development Workflow & Commands
- **Environment**: use `uv` for all Python commands (see `pyproject.toml`).
- **Setup**:
  - `uv sync` to install dependencies.
- **Tests** (or VS Code tasks with the same names):
  - `uv run pytest -n auto` (task: `Run pytest`) for normal runs.
  - `uv run pytest` (task: `Run pytest (serial)`) when debugging or tests are not parallel‑safe.
  - Use `tests/` as the source of truth for frame semantics, metadata rules, I/O contracts, and lazy behavior.
- **Type checking / lint**:
  - `uv run mypy --config-file=pyproject.toml` (task: `Run mypy wandas tests`).
  - `uv run ruff check wandas tests --config=pyproject.toml` (task: `Run ruff check`).
  - `uv run ruff format wandas tests` to apply the canonical formatting.
- **Docs**:
  - `uv run mkdocs build -f docs/mkdocs.yml` (task: `Build MkDocs Documentation`).
  - `uv run mkdocs serve -f docs/mkdocs.yml` (task: `Serve MkDocs Documentation`).

## 3. Frames, Immutability, and Metadata
- **Never mutate frames in place**: all frame operations must return new frame objects; treat underlying arrays/graphs as immutable from the API perspective.
- **Always maintain together** when implementing frame operations:
  - `operation_history` entries (what was done and with which parameters, in call‑order).
  - Sampling rate, time/frequency axes, and channel labels.
  - User/recording metadata carried from inputs to outputs, including when changing domains (time → spectral → spectrogram).
- **Metadata update encapsulation**:
  - Prefer helpers on frames (e.g. `_with_updated_metadata`, `replace(...)`) to update data + metadata + history atomically.
  - Avoid scattered direct dict updates to `metadata` or `operation_history`; keep them inside frame classes or dedicated utilities in `frames/`.
- **Dask laziness**:
  - Preserve lazy execution where present: build Dask graphs and avoid eager `.compute()` unless required by the public API or tests.
  - When refactoring, check spectral/roughness/spectrogram code paths to ensure you do not force computation earlier than before.

## 4. Processing API & Project‑Specific Patterns
- **Separation of concerns**:
  - Frame methods should be thin facades: validate inputs, manage metadata/history, and dispatch into `processing/` functions.
  - Numerical algorithms (FFT, filters, psychoacoustic metrics, statistics, effects) should live in `processing/` modules like `filters.py`, `spectral.py`, `psychoacoustic.py`, `temporal.py`, `stats.py`, `effects.py`.
- **Adding new operations**:
  - First add a function in the appropriate `processing/` module.
  - Then add a frame method in `frames/` that delegates to that function and wraps the result in a new frame with updated metadata/history.
  - Mirror parameter naming and default values of nearby functions/methods; avoid inventing new patterns unless necessary.
- **I/O patterns**:
  - Use `io/wav_io.py`, `io/wdf_io.py`, and `io/readers.py` as references for how sampling rate, channels, and metadata are handled, especially for WDF/HDF5 round‑trips.
  - Keep read/write functions thin: they should construct frames with correct axes/metadata and let frames/processing handle subsequent computation.
- **Visualization patterns**:
  - Plotting helpers live in `visualization/` and should return Matplotlib `Axes` objects.
  - They should call existing frame methods (e.g. `.fft()`, `.stft()`, `.describe()`) instead of duplicating numerical logic.
- **Usage examples**:
  - `README.md`, `learning-path/`, and `examples/` are the canonical references for method chaining patterns (e.g. `signal.normalize().low_pass_filter(...).resample(...).fft().plot(...)`).

## 5. Testing & Design Principles
- **Tests as spec**:
  - When changing behavior, locate relevant tests in `tests/` and update or extend them first; tests define expectations for metadata, axes, lazy behavior, and numerical tolerances.
- **Minimal surface (YAGNI)**:
  - Implement the smallest API surface that satisfies existing tests and documented use cases; avoid speculative flags or configuration.
- **Extensibility & maintainability**:
  - Prefer small, composable helpers in `processing/` and short, chainable frame methods over large monolithic functions.
  - Reuse existing helpers for resampling, filtering, spectral transforms, etc., instead of duplicating logic; factor out shared pieces when duplication is unavoidable.
- **Error handling**:
  - When raising new errors, follow a WHAT/WHY/HOW pattern in messages (what went wrong, why it matters here, how the caller can fix it).
- **Edge cases to mirror**:
  - Follow existing tests for NaN handling, multi‑channel audio, sampling‑rate changes, large Dask‑backed datasets, and psychoacoustic/spectral metrics.

## 6. Roles: Planner / Implementer / Reviewer
- **Planner**:
  - Use read‑only tools to map which `frames/`, `processing/`, and `io/` modules are affected.
  - Produce a concrete plan tied to specific files, tests to touch, and any risks around metadata consistency, Dask graphs, or performance.
- **Implementer**:
  - Follow the planner handoff; if assumptions change, update the plan before editing.
  - Keep frames immutable, preserve metadata/history, and honor Dask laziness as described above.
  - Run `pytest` plus `mypy`/`ruff` for the affected areas when feasible and record the commands used.
- **Reviewer**:
  - Re‑run the recorded commands (or their nearest equivalents).
  - Verify that frame immutability and metadata rules are respected, that new APIs align with existing naming/parameter patterns, and that tests cover the main branches and edge cases discussed in the planner handoff.
