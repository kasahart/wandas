---
description: "TDD workflow and quality check commands (pytest, mypy, ruff, mkdocs)"
---
# Wandas Testing & Quality Prompt

Use this prompt when adding or modifying behavior anywhere in the Wandas codebase.

## Core tools
- Use `uv` for all commands, and prefer VS Code tasks when they exist:
  - `Create Virtual Environment` – `uv venv --allow-existing .venv && uv sync --frozen --all-groups`
  - `Run pytest` – `uv run pytest -n auto --cov=wandas --cov-report=term-missing`
  - `Run mypy wandas tests` – `uv run mypy --config-file=pyproject.toml`
  - `Run ruff format` – `uv run ruff format wandas tests`
  - `Run ruff check` – `uv run ruff check wandas tests --config=pyproject.toml -v`
  - `Run ruff check --fix` – `uv run ruff check --fix wandas tests --config=pyproject.toml -v`
  - `Build MkDocs Documentation` – `uv run mkdocs build -f docs/mkdocs.yml`
  - `Serve MkDocs Documentation` – `uv run mkdocs serve -f docs/mkdocs.yml`

## Workflow expectations
- Prefer **TDD** for non-trivial changes:
  - write or update tests in `tests/` first,
  - then implement the minimal change to satisfy them.
- Use non-mutating validation tasks for review and verification; reserve `Run ruff check --fix` for implementation or publishing when automatic fixes are intentional.
- When writing or modifying tests, follow the **test grand policy**:
  - [test-grand-policy.instructions.md](test-grand-policy.instructions.md) — 4 pillars & signal processing test pyramid
- When changing behavior, identify and update relevant tests:
  - frame semantics, metadata, and operation history,
  - I/O contracts (WAV/WDF/CSV round-trips),
  - Dask-backed large data behavior.
- Treat `Run pytest` coverage output as part of the definition of done for behavior changes:
  - capture the `--cov=wandas --cov-report=term-missing` result before and after implementation when coverage risk is non-trivial,
  - do not hand off silent coverage regressions on touched code paths,
  - if coverage drops or `term-missing` shows new uncovered lines, add focused tests for the changed branches, edge cases, and error handling paths or call out the gap explicitly as a warning.
- Keep a short command log of what you ran (pytest, mypy, ruff, mkdocs) to aid reviewers.

## Edge cases & quality
- Look at existing tests for how the project handles:
  - NaNs and missing data,
  - multi-channel audio and label alignment,
  - sampling rate changes,
  - lazy Dask computations and `.compute()` boundaries.
- Error messages should follow the **WHAT/WHY/HOW** pattern.

Use this prompt to stay aligned with Wandas' testing, typing, and quality expectations.
