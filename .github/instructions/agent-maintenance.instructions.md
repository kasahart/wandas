---
description: "Copilot custom-agent metadata, capability boundaries, and optional handoffs"
applyTo: ".github/agents/**"
---

# Copilot custom-agent adapter

Custom agents must link to [`AGENTS.md`](../../AGENTS.md) instead of restating
repository rules. Define only the capability boundary that differs from the
default agent:

- use `tools` to exclude editing from planning and review capabilities;
- set `disable-model-invocation: true` for external-write capabilities that
  require explicit user selection;
- use `handoffs` only as optional UI shortcuts, never as a mandatory pipeline;
- keep `argument-hint`, tool names, and other frontmatter limited to supported
  Copilot metadata.

When changing an agent profile, parse its YAML frontmatter and resolve its local
links. Do not add an implementation agent merely to make ordinary edits: the
default agent can implement scoped work directly.
