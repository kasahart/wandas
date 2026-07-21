---
name: wandas-workspace-hygiene
description: Use when checking a Wandas workspace before editing, committing, pushing, or reporting completion.
---

# Wandas Workspace Hygiene

## When to use

Use before substantive edits, at publication boundaries, and after merge when
retiring a task worktree. `AGENTS.md` owns the repository invariants.

## Required checks

- Inspect dirty checkout state with `git status --short` before editing.
- Use a dedicated worktree under `.worktrees/` when available and appropriate.
- Preserve unrelated user changes; ask when the relationship to the task is unclear.
- Do not stage or repeatedly delete ignored validation artifacts during an
  active pull request. Remove them once after merge when retiring the task
  worktree, or at final handoff when no pull request is used.
- Clean only the task worktree, never the primary checkout or another worktree.

## Output to report

Report the workspace path, branch, relevant dirty state, retained ignored
artifacts, and post-merge cleanup when it occurs.

## What not to do

Do not stage, move, revert, or overwrite unrelated user changes. Ignored
validation artifacts are not a merge blocker.
