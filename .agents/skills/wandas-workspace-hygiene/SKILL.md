---
name: wandas-workspace-hygiene
description: Use when checking a Wandas workspace before editing, committing, pushing, or reporting completion.
---

# Wandas Workspace Hygiene

## When to use

Use before substantive edits, at publication boundaries, and after merge when
retiring a task worktree. This skill owns the detailed hygiene procedure;
`AGENTS.md` owns the always-on repository invariants.

## Required checks

- Inspect dirty checkout state with `git status --short` before editing.
- Use a dedicated worktree under `.worktrees/` when available and appropriate.
- Preserve unrelated user changes; ask when the relationship to the task is unclear.
- Before commit or push, use `git status --short` to confirm that only intended
  tracked and untracked files can be staged. Generated ignored artifacts are not
  a merge-readiness gate.
- During an active pull request, do not repeatedly delete `.coverage`,
  `coverage.xml`, `.pytest_cache/`, `.ruff_cache/`, `docs/site/`, `__pycache__/`,
  or tool scratch files between validation, commit, push, CI, and review cycles.
- After merge, check `git status --short --ignored` and clean task-generated
  ignored artifacts once while retiring the dedicated worktree. If no pull
  request is used, perform that one cleanup at final handoff instead.
- Remove only artifacts inside the task worktree. Do not clean the primary
  checkout or another worktree on the task's behalf.

## Output to report

Report the workspace path, branch, relevant dirty checkout state, generated
artifacts deliberately retained during an active pull request, and the single
post-merge or final-handoff cleanup when it occurs.

## What not to do

Do not stage, move, revert, or overwrite unrelated user changes. Do not claim
that active ignored artifacts block merge, and do not repeatedly delete them to
make intermediate status output look clean.
