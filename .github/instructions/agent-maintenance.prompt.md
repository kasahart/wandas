# Agent Maintenance & Evolution Prompt

Use this prompt when modifying `.github/agents/` or `.github/instructions/` files, or when creating new custom agents.

## Core Principle: Keep Agents Connected
- **Linkage Rule**: If you create a new instruction file (e.g., `instructions/new-topic.prompt.md`), you **must** update the relevant agent file (e.g., `wandas-planner.agent.md`) to reference it using a Markdown link.
  - *Why*: Agents do not automatically see new files. Explicit links are required for context retrieval.
- **Handoff Rule**: If you create a new agent, ensure it is reachable via `handoffs` from an existing agent, and that it can hand off to the next logical step.

## Staying Current
- **Check Documentation**: Custom Agent features evolve rapidly. Before making structural changes, check the official docs:
  - [VS Code Copilot Custom Agents](https://code.visualstudio.com/docs/copilot/customization/custom-agents)
- **Web Search**: If the documentation seems outdated or you need advanced patterns, use the `fetch` tool to search for "VS Code Copilot Custom Agents best practices" or similar queries.

## Retrospective Workflow
When the `wandas-publisher` agent triggers a retrospective:
1. **Identify Friction**: Where did the agent misunderstand the task? (e.g., "Planner didn't know about the new I/O format").
2. **Update Instructions**: Clarify the relevant `.prompt.md` file.
3. **Update Context**: If the agent missed a file entirely, add a link in its `.agent.md` file.
4. **Verify**: Ensure the new instructions don't contradict `copilot-instructions.md`.

## Implementation Mode for Agent Updates
- **Who**: Use the `wandas-implementer` agent, but explicitly state "I am updating agent configurations" in the prompt.
- **Verification**: Since you cannot "test" an agent change with `pytest`, you must:
  1. Read the modified `.agent.md` file to verify syntax.
  2. Check that all file paths in links are valid (use `ls` or `fileSearch`).
  3. Verify that YAML frontmatter is valid.
