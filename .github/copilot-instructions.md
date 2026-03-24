# Copilot Instructions for the Wandas Repository

These instructions are for Wandas custom agents. For substantive implementation, validation, and review work, the default workflow is `wandas-planner` -> `wandas-implementer` -> `wandas-reviewer`; use `wandas-publisher` only after review for PR publication handoff. GitHub Release drafting and tag-driven release publication stay in GitHub Actions.

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
  - If a `.venv` virtual environment is missing, run the VS Code task "Create Virtual Environment" (see `.vscode/tasks.json`) which executes:
    - `uv venv --allow-existing .venv && uv sync --frozen --all-groups`
- **Tests** (or VS Code tasks with the same names):
  - `uv run pytest -n auto --cov=wandas --cov-report=term-missing` (task: `Run pytest`) for repository test runs.
  - Use `tests/` as the source of truth for frame semantics, metadata rules, I/O contracts, and lazy behavior.
- **Type checking / lint**:
  - `uv run mypy --config-file=pyproject.toml` (task: `Run mypy wandas tests`).
  - `uv run ruff format wandas tests` (task: `Run ruff format`).
  - `uv run ruff check wandas tests --config=pyproject.toml -v` (task: `Run ruff check`) for non-mutating lint validation.
  - `uv run ruff check --fix wandas tests --config=pyproject.toml -v` (task: `Run ruff check --fix`) when automatic lint fixes are desired.
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
  - Example: `raise ValueError("Sampling rate mismatch (WHAT). Filters require matching rates to prevent phase distortion (WHY). Resample inputs to the same rate before filtering (HOW).")`
  - **Terminal compatibility**: Use ASCII-safe formatting (e.g., "freq x time" not "freq×time", "expected: value" not "expected → value").
  - **Multiline structure**: For complex errors with multiple pieces of information, format as:
    ```python
    raise ValueError(
        f"Invalid data shape\n"
        f"  Got: {actual_shape}\n"
        f"  Expected: {expected_shape}\n"
        f"Ensure input has correct dimensions before calling this method."
    )
    ```
  - **Test pattern updates**: When changing error messages, update test `pytest.raises(..., match=r"pattern")` to match the first line only (e.g., `r"Invalid data shape"` not the full multiline text). This keeps tests resilient to detail changes while ensuring the core error type is caught.
- **Edge cases to mirror**:
  - Follow existing tests for NaN handling, multi‑channel audio, sampling‑rate changes, large Dask‑backed datasets, and psychoacoustic/spectral metrics.

## 6. Workflow Roles: Planner / Implementer / Reviewer / Publisher
 **Delegation default**:
  - When the custom agents are available, the top-level/default agent should usually start substantive, ambiguous, or multi-file implementation, validation, and review work with `wandas-planner`, then hand off to `wandas-implementer` and `wandas-reviewer` as needed instead of performing that work directly.
  - This planner-first default applies to the top-level agent choosing which role to invoke. Once a role-specific custom agent is already active, it should perform its own role directly and hand off forward; it should not re-delegate that same role to itself.
  - Use `wandas-publisher` only after reviewer approval for publish-stage handoff work such as branching, staging, committing, pushing, and pull-request updates.
  - Release drafting and publication remain in GitHub Actions via `.github/workflows/release-drafter.yml`, `.github/release-drafter.yml`, and the tag-driven path in `.github/workflows/cd.yml`.
  - Narrow exceptions: trivial read-only guidance (for example, answering a simple repository question without changing files), explicit planning-only requests, and low-risk single-file customization maintenance in one existing `.github/` customization file when the change is limited to wording, links, or YAML/frontmatter fixes and does not alter tools, handoffs, roles, or workflow semantics.
  - Direct `wandas-implementer` or `wandas-reviewer` use remains valid when the user explicitly asks for that role, a prior handoff already exists, or the task is a narrow continuation with clear scope and validation context.
  - For new substantive repository work, quality-check execution, and multi-file customization changes, prefer the full Planner / Implementer / Reviewer flow.
- **Planner**:
  - Default first stop for new substantive, ambiguous, or multi-file work, including `.github/` customization changes.
  - Treat user- or issue-proposed implementations as inputs to evaluate, not approved plans to adopt unchanged. Use the planner to compare them against repository patterns, risks, and simpler alternatives before implementation starts.
  - Use read‑only tools to map which code modules or repository customization/workflow files are affected.
  - Produce a concrete plan tied to specific files, tests to touch, and any risks around metadata consistency, Dask graphs, or performance.
- **Implementer**:
  - Usually follows a planner handoff, but can handle explicit, tightly scoped follow-up implementation when the scope and validation context are already clear.
  - Follow the planner handoff when one is provided; if assumptions change, update the plan before editing.
  - Keep frames immutable, preserve metadata/history, and honor Dask laziness as described above.
  - Run the relevant VS Code tasks when they exist, and record the task names plus any direct `uv run ...` commands that were needed.
  - Run `Build MkDocs Documentation` only when `docs/`, `src/`, `README.md`, or other MkDocs-backed user-facing markdown changed; `.github/` customization-only changes normally do not require it.
- **Reviewer**:
  - Primarily reviews completed implementation work with existing validation evidence; it is not the default entry point for brand-new work.
  - Keep review read-only and verify the recorded validation evidence, problems, and changed files directly from the workspace instead of owning task execution.
  - Confirm non-mutating validation evidence such as `Run ruff check`; do not use `Run ruff check --fix` during review.
  - Verify that frame immutability and metadata rules are respected, that new APIs align with existing naming/parameter patterns, and that tests cover the main branches and edge cases discussed in the planner handoff.

## 7. Test Documentation Reference

For comprehensive testing guidance, refer to:
- **`.github/instructions/test-grand-policy.instructions.md`** — Auto-applied test policy (4 pillars + pyramid) when editing `tests/**` files, including coverage-aware testing guidance for touched code paths
