---
name: wandas-pr-readiness
description: Use when preparing to report a Wandas pull request as ready, complete, or ready to merge.
---

# Wandas PR Readiness

## When to use

Use before saying a PR is complete, ready for review, or ready to merge. This
skill owns the detailed readiness procedure; `AGENTS.md` owns the always-on
repository invariants.

## Required checks

- If the change triggered
  [`wandas-change-coherence`](../wandas-change-coherence/SKILL.md), apply its
  current-head contract-stability gate before requesting external review. Do
  not impose that procedure solely for a small, clear change.
- PR title/body full-scope check: confirm the title and body describe the complete current scope.
- Validation evidence: collect relevant `uv run pytest`, `uv run ruff check`, `uv run ty check`, docs builds, skipped checks, and reasons.
- Generated ignored artifacts are not a readiness gate. Confirm they are not
  staged, record them for one post-merge cleanup, and keep validation caches in
  place during active commit, push, CI, and review cycles.
- SHA alignment: verify local `HEAD`, `origin/<branch>`, and the PR head SHA point to the intended commit after pushing.
- CI/review state: confirm there are no unresolved actionable review threads and no pending requested reviews.
- Finite post-push monitoring: check immediately after a push, then recheck for
  up to 10 minutes by default and 30 minutes maximum unless the user requests a
  different bound.

## Output to report

Report the PR scope, triggered change-coherence outcome when applicable,
validation evidence, SHA alignment result, CI/check status, review-thread
status, and any skipped or timeboxed checks.

After merge, use
[`wandas-workspace-hygiene`](../wandas-workspace-hygiene/SKILL.md) to perform
the single generated-artifact cleanup while retiring the task worktree.

## What not to do

Do not report readiness from local tests alone. Do not bury meaningful deferred
work in PR comments. Do not repeatedly delete ignored validation artifacts
before merge.
