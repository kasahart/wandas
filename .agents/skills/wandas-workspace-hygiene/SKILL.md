---
name: wandas-workspace-hygiene
description: Use when checking a Wandas workspace before editing, committing, pushing, or reporting completion.
---

# Wandas Workspace Hygiene

## When to use

Use before substantive edits and again before reporting completion. Treat `AGENTS.md` as the source of truth and this skill as a short execution adapter.

## Required checks

- Inspect dirty checkout state with `git status --short` before editing.
- Use a dedicated worktree under `.worktrees/` when available and appropriate.
- Preserve unrelated user changes; ask when the relationship to the task is unclear.
- Check ignored/generated artifacts with `git status --short --ignored`.
- Remove unintended `.coverage`, `coverage.xml`, `.pytest_cache/`, `.ruff_cache/`, `docs/site/`, `__pycache__/`, and scratch files.

## Output to report

Report the workspace path, branch, relevant dirty checkout state, ignored/generated artifacts found, and any cleanup performed.

## What not to do

Do not stage, move, revert, or overwrite unrelated user changes. Do not report a clean workspace without checking ignored/generated artifacts. Do not let this skill replace `AGENTS.md`.
