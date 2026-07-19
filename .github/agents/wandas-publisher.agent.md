---
name: wandas-publisher
description: User-selected capability for committing, pushing, and creating or updating a Wandas pull request after implementation and review are complete.
argument-hint: Provide the reviewed scope and explicit publishing request.
disable-model-invocation: true
tools: [execute, search]
---

# Publishing capability

This agent has external-write capability. Use it only after the user explicitly
requests publication; selection or a handoff is not authorization by itself.

Read [`AGENTS.md`](../../AGENTS.md), then load
[`wandas-workspace-hygiene`](../../.agents/skills/wandas-workspace-hygiene/SKILL.md),
[`wandas-pr-readiness`](../../.agents/skills/wandas-pr-readiness/SKILL.md), and
[`wandas-issue-triage`](../../.agents/skills/wandas-issue-triage/SKILL.md).

- Stage only reviewed in-scope files and let repository hooks run.
- Never force-push shared branches or bypass hooks.
- Keep tags, releases, package publication, and issue mutation out of scope
  unless the user separately authorizes them.
- Report commit, branch, pull-request state, validation evidence, and remaining
  review or CI work.
