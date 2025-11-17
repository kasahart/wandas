# Wandas Testing & Quality Prompt

Use this prompt when adding or modifying behavior anywhere in the Wandas codebase.

## Core tools
- Use `uv` for all commands:
  - `uv sync` – install/update dependencies.
  - `uv run pytest` – run the test suite.
  - `uv run mypy --config-file=pyproject.toml` – static type checks.
  - `uv run ruff check wandas tests --config=pyproject.toml` – linting.
  - `uv run ruff format wandas tests` – formatting.

## Workflow expectations
- Prefer **TDD** for non-trivial changes:
  - write or update tests in `tests/` first,
  - then implement the minimal change to satisfy them.
- When changing behavior, identify and update relevant tests:
  - frame semantics, metadata, and operation history,
  - I/O contracts (WAV/WDF/CSV round-trips),
  - Dask-backed large data behavior.
- Keep a short command log of what you ran (pytest, mypy, ruff, mkdocs) to aid reviewers.

## Edge cases & quality
- Look at existing tests for how the project handles:
  - NaNs and missing data,
  - multi-channel audio and label alignment,
  - sampling rate changes,
  - lazy Dask computations and `.compute()` boundaries.
- Error messages should follow the **WHAT/WHY/HOW** pattern (see `docs/development/error_message_guide.md` when available).

Use this prompt to stay aligned with Wandas' testing, typing, and quality expectations.
