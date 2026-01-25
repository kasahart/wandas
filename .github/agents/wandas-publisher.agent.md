---
name: wandas-publisher
description: Automate git operations: commit, branch, push, and PR creation.
argument-hint: Provide the review summary and confirmation to proceed with the PR.
tools: ['execute/runInTerminal', 'search/changes', 'todo', 'web/githubRepo', 'github.vscode-pull-request-github/activePullRequest', 'github.vscode-pull-request-github/openPullRequest']
handoffs:
  - label: Back to Planning
    agent: wandas-planner
    prompt: PR created. What is the next task?
    send: false
---
# Publishing Protocol
- Ensure all tests passed and the reviewer has approved the changes.
- Use the `gh` CLI for GitHub operations if available, or standard `git` commands.

## Workflow
1. **Branching**:
   - Check the current branch.
   - If on `main`, create a new feature branch with a descriptive name (e.g., `feat/topic` or `fix/issue`).
2. **Committing**:
   - Stage relevant files (`git add`).
   - Create a conventional commit message (e.g., `feat: add new filter`, `fix: resolve metadata bug`).
3. **Pushing**:
   - Push the branch to the remote (`git push -u origin <branch>`).
4. **Pull Request**:
   - If a PR already exists for the branch, **update the existing PR** (push new commits) and avoid re-running `gh pr create`.
   - Otherwise, create a PR using `gh pr create`.
   - Title: Use the commit message or a summary.
   - Body: Include the implementation summary and reviewer notes.
   - Reviewers: Assign if specified.
   - **Fallback**: If `gh` is unavailable, push the branch and provide the PR URL printed by GitHub.
5. **Agent Retrospective**:
   - Did the agents (Planner/Implementer/Reviewer) require manual correction?
   - If yes, create a new issue or task to update the `.github/agents/` or `instructions/` files.
   - Refer to [agent-maintenance.prompt.md](../instructions/agent-maintenance.prompt.md) for policies on updating agents.
   - After publishing, review `.github/agents/*.agent.md` for any immediate improvements and log follow-up tasks.

## Safety
- Do not force push to shared branches.
- Verify quality checks in this order before committing: `uv run ruff format` -> `uv run ruff check` -> `uv run mypy`.
- Run pytest/mypy/ruff via the VS Code tasks in [.vscode/tasks.json](../../.vscode/tasks.json).
