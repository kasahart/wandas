---
name: wandas-issue-triage
description: Use when deciding whether a Wandas pull request closes issues, relates to issues, or needs follow-up issues.
---

# Wandas Issue Triage

## When to use

Use while preparing or updating PR issue references. This skill owns the
detailed issue-triage procedure; `AGENTS.md` owns the always-on repository
invariants.

## Required checks

- `Closes`: use only when the PR fully satisfies the issue acceptance criteria.
- `Related`: use for parent, partial, broad, or still-open work.
- Follow-up issue: create and link one for meaningful deferred work instead of hiding it in review comments.
- After merge, check closing references and close any completed-but-open source issue with a concise comment.

## Output to report

Report each linked issue, whether it is `Closes` or `Related`, why that relationship is correct, and any follow-up issue created.

## What not to do

Do not use `Closes` for partial work. Do not leave known deferred work
undocumented.
