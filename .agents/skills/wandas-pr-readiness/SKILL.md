---
name: wandas-pr-readiness
description: Use when preparing to report a Wandas pull request as ready, complete, or ready to merge.
---

# Wandas PR Readiness

## When to use

Use before saying a PR is complete, ready for review, or ready to merge. Treat `AGENTS.md` as the source of truth and this skill as a short execution adapter.

## Required checks

- PR title/body full-scope check: confirm the title and body describe the complete current scope.
- Validation evidence: collect relevant `uv run pytest`, `uv run ruff check`, `uv run ty check`, docs builds, skipped checks, and reasons.
- SHA alignment: verify local `HEAD`, `origin/<branch>`, and the PR head SHA point to the intended commit after pushing.
- CI/review state: confirm there are no unresolved actionable review threads and no pending requested reviews.
- Finite post-push monitoring: wait for checks and review state to settle within the timebox requested by `AGENTS.md`.

## Output to report

Report the PR scope, validation evidence, SHA alignment result, CI/check status, review-thread status, and any skipped or timeboxed checks.

## What not to do

Do not report readiness from local tests alone. Do not let this skill override `AGENTS.md` or bury meaningful deferred work in PR comments.
