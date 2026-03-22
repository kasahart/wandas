---
description: "Agent maintenance: rules for creating/updating agent, instruction, and prompt files"
applyTo: ".github/agents/**"
---
# Agent Maintenance & Evolution Prompt

Use this prompt when modifying `.github/agents/` files, when creating new custom agents, or as manual guidance for related `.github/instructions/` maintenance.

## Core Principle: Keep Agents Connected
- **Linkage Rule**: If you create a new instruction file (e.g., `instructions/new-topic.instructions.md`), you **must** update the relevant agent file (e.g., `wandas-planner.agent.md`) to reference it using a Markdown link.
  - *Why*: Agents do not automatically see new files. Explicit links are required for context retrieval.
- **Handoff Rule**: If you create a new agent, ensure it is reachable via `handoffs` from an existing agent, and that it can hand off to the next logical step.
- **Automation Rule**: If a workflow is intended to continue automatically after a handoff button is selected, set `handoffs.send: true` and avoid body text that restricts handoff to explicit user approval unless that approval gate is intentionally required.
- **applyTo-based instructions**: Files ending in **`.instructions.md`** support `applyTo` frontmatter
  for path-specific auto-injection. Use this for policies that should automatically apply when
  editing specific file patterns.
- **Custom prompts**: Files ending in **`.prompt.md`** support `name`, `description`, `model`, `tools`,
  `argument-hint` frontmatter. Use this for manually-invoked prompts.

## Current Active Instruction Files

### Active `.instructions.md` Files (.github/instructions/)

Only the `.instructions.md` files in this directory are live agent guidance. Archived review notes should live outside `.github/instructions/` so they are not presented as active instructions.

| File | applyTo | Purpose |
|------|---------|---------|
| `testing-workflow.instructions.md` | (none) | General TDD workflow |
| `frames-design.instructions.md` | `wandas/frames/**` | Frame architecture |
| `processing-api.instructions.md` | `wandas/processing/**` | Processing layer |
| `io-contracts.instructions.md` | `wandas/io/**` | I/O contracts |
| `agent-maintenance.instructions.md` | `.github/agents/**` | This file |
| `test-grand-policy.instructions.md` | `tests/**` | Test quality: 4 pillars + pyramid |
| `test-frames-policy.instructions.md` | `tests/frames/**` | Frame test patterns |
| `test-processing-policy.instructions.md` | `tests/processing/**` | Processing test patterns |
| `test-io-policy.instructions.md` | `tests/io/**` | I/O test patterns |
| `test-visualization-policy.instructions.md` | `tests/visualization/**` | Visualization test patterns |

## Staying Current
- **Check Documentation**: Custom Agent features evolve rapidly. Before making structural changes, check the official docs:
  - [VS Code Copilot Custom Agents](https://code.visualstudio.com/docs/copilot/customization/custom-agents)
- **Web Search**: If the documentation seems outdated or you need advanced patterns, use the `fetch` tool to search for "VS Code Copilot Custom Agents best practices" or similar queries.

## Retrospective Workflow
When the `wandas-publisher` agent triggers a retrospective:
1. **Identify Friction**: Where did the agent misunderstand the task? (e.g., "Planner didn't know about the new I/O format").
2. **Update Instructions**: Clarify the relevant `.instructions.md` file.
3. **Update Context**: If the agent missed a file entirely, add a link in its `.agent.md` file.
4. **Verify**: Ensure the new instructions don't contradict `copilot-instructions.md`.

## Implementation Mode for Agent Updates
- **Who**: Use the full `wandas-planner` -> `wandas-implementer` -> `wandas-reviewer` flow for substantive or multi-file customization updates, and explicitly state "I am updating agent configurations" in the implementation prompt.
- **Discovery wording**: For multi-agent workflows, keep the planner framed as the preferred first-stop discovery agent, and describe user- or issue-proposed implementations as inputs to evaluate rather than approved plans to adopt unchanged.
- **Narrow exception**: A low-risk fix to one existing `.github/` customization file may be edited directly when it is only a wording, link, or YAML/frontmatter correction and does not change tools, handoffs, roles, or workflow semantics.
- **Docs validation**: Reserve `Build MkDocs Documentation` for changes to `docs/`, `src/`, `README.md`, or other MkDocs-backed user-facing markdown. For `.github/` customization-only work, validate by reading the modified Markdown/YAML, checking problems, and verifying linked paths instead.
- **Verification**: Since you cannot "test" an agent change with `pytest`, you must:
  1. Read the modified `.agent.md` file to verify syntax.
  2. Check that all file paths in links are valid (use `ls` or `fileSearch`).
  3. Verify that YAML frontmatter is valid.
