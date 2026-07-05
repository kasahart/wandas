---
description: "PR lifecycle harness: completion gates for publishing, review follow-up, issue triage, and workspace cleanup"
---
# PR Lifecycle Harness

AGENTS.md contains the cross-agent canonical PR lifecycle checklist. Use this file as supplemental detail for agents and tools that need the full publishing, issue triage, monitoring, and cleanup procedure.

Use this guidance when creating, updating, or preparing to merge a pull request. The goal is to make the repository-local agent workflow more autonomous without hiding state from the user.

## PR Completion Gate

Before reporting that a PR is ready to merge, verify and report:

- The PR targets the intended base branch.
- The PR title and description describe the full current scope, not only the latest review fix.
- The PR body lists validation evidence and any skipped checks with reasons.
- local `HEAD`, `origin/<branch>`, and the PR head SHA all match.
- Required checks are passing or explicitly identified as blocked.
- There are no unresolved actionable review threads.
- There are no pending requested reviews.

If the PR body has drifted during review, update it before final reporting.

## Issue Triage Gate

Before merge, inspect all linked or mentioned issues:

- Use `Closes` only for issues whose acceptance criteria are fully satisfied by the PR.
- Use `Related` for broader parent, phase, or design issues that must remain open.
- Create a follow-up issue when the PR intentionally leaves meaningful work behind.
- Link the follow-up issue from the PR body.

After merge, check closing issue references. If a completed source issue remains open, close it with a concise comment that links the merge result to the issue acceptance criteria.

## Workspace Hygiene Gate

Before final reporting or publishing:

- Run `git status --short --branch`.
- Run `git status --short --ignored` when tests, docs builds, notebooks, coverage, or local app tools were executed.
- Remove generated local artifacts such as `.coverage`, `coverage.xml`, `.pytest_cache/`, `.ruff_cache/`, `docs/site/`, `__pycache__/`, and tool-specific scratch files when they are not intended repository changes.
- Do not remove user-created untracked files or files outside the task scope.

If cleanup requires deleting ambiguous files, stop and ask instead of guessing.

## Finite Monitoring Gate

After pushing PR updates, monitor for a bounded period:

- Check CI/checks and review state once immediately after push.
- Recheck after a short settling period when review automation is expected.
- Stop after the agreed maximum wait. Default: 10 minutes. Maximum: 30 minutes unless the user explicitly asks for longer.

Report whether new reviews arrived, which comments were handled, and what remains open.

## Review Feedback Gate

When review feedback repeats around the same behavior, treat it as a contract or design signal:

- Pause before adding more defensive branches.
- Identify the underlying invariant that should make the bug impossible.
- Prefer constructor validation, explicit state boundaries, or shared helpers over scattered point fixes.
- Record deferred non-blocking concerns in a follow-up issue instead of expanding the PR late.
