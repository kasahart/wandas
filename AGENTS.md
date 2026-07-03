# AGENTS.md

## Repository Guidance

Follow `.github/copilot-instructions.md` for the full Wandas repository conventions. This file is the short Codex-facing entry point loaded from the repository root.

Core rules:

- Use `uv` for Python commands.
- For substantive code changes, use a dedicated Git worktree under `.worktrees/` when available and appropriate; follow `.github/copilot-instructions.md` for dirty-checkout and tool-availability rules.
- Preserve frame immutability, metadata/history, and Dask laziness.
- Keep frame methods thin; numerical logic belongs in `wandas/processing`.
- Prefer small, explicit contracts over compatibility layers for undocumented or ambiguous behavior.
- Avoid duplicated state, silent no-op compatibility shims, and public mutable state that must be synchronized with internal state.
- When changing behavior, update the relevant tests so they describe the clarified contract.
- Run relevant `uv run pytest`, `uv run ruff check`, and `uv run ty check` commands before finishing.
- Treat repeated PR review feedback about the same contract as a design signal, not an endless patch queue; stop and reassess the contract before adding more defensive branches.
- Restarting a PR from a clean base is an exception, used only when old PR history or review context would obscure a revised contract.
- After pushing PR updates, verify that local `HEAD`, `origin/<branch>`, and the PR head SHA match before reporting completion.
- After pushing PR updates, use finite post-push monitoring: wait for CI/checks and review state to settle (up to 10 minutes by default, 30 minutes maximum); before reporting completion, confirm there are no unresolved review threads and no pending requested reviews.
- After a PR is merged, check closing issue references and close any completed-but-open source issue with a concise comment.

Area-specific guidance lives in:

- `.github/instructions/processing-api.instructions.md`
- `.github/instructions/frames-design.instructions.md`
- `.github/instructions/testing-workflow.instructions.md`
- `.github/instructions/test-grand-policy.instructions.md`
