# GitHub Copilot adapter

[`AGENTS.md`](../AGENTS.md) is the canonical Wandas contract. Copilot surfaces
that support agent instructions load it directly; on other surfaces, read it
before repository work. Copilot also discovers the reusable procedures in
`.agents/skills`.

The files under `.github/instructions/` are Copilot path-loading adapters.
Custom agents under `.github/agents/` are optional capability and permission
boundaries. A scoped task may be handled directly by the default agent;
planning, independent review, and publishing agents are selected only when
their isolation or tools add value. Handoff buttons are UI conveniences, not a
required Planner -> Implementer -> Reviewer pipeline.
