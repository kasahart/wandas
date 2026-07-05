# AGENTS.md

## Repository Guidance

This file is the cross-agent source of truth for Wandas repository work. Other harness files may adapt these rules for a specific tool, but they should not replace this checklist.

Core rules:

- Use `uv` for Python commands.
- For substantive code changes, use a dedicated Git worktree under `.worktrees/` when available and appropriate; inspect `git status --short` first and do not move, stage, revert, or overwrite unrelated user changes. If the current checkout has changes related to the task, continue there or ask before creating a fresh worktree. If the relationship is unclear, ask before editing.
- Preserve frame immutability, metadata/history, and Dask laziness.
- Keep frame methods thin; numerical logic belongs in `wandas/processing`.
- Prefer small, explicit contracts over compatibility layers for undocumented or ambiguous behavior.
- Avoid duplicated state, silent no-op compatibility shims, and public mutable state that must be synchronized with internal state.
- When changing behavior, update the relevant tests so they describe the clarified contract.
- Run relevant `uv run pytest`, `uv run ruff check`, and `uv run ty check` commands before finishing.
- Restarting a PR from a clean base is an exception, used only when old PR history or review context would obscure a revised contract.
- After a PR is merged, check closing issue references and close any completed-but-open source issue with a concise comment.

## PR Lifecycle Checklist

Use this direct checklist before reporting PR completion or readiness to merge. `.github/instructions/pr-lifecycle-harness.instructions.md` contains supplemental detail.

- PR title/body full-scope check: ensure the title and body describe the full current PR scope, not only the latest review fix.
- Validation evidence: list relevant `uv run pytest`, `uv run ruff check`, `uv run ty check`, docs builds, skipped checks, and reasons.
- Issue triage with `Closes` vs `Related`: use `Closes` only when the PR fully satisfies the issue acceptance criteria; use `Related` for broader or still-open work.
- Follow-up issue creation: create and link a follow-up issue for meaningful deferred work instead of burying it in review comments.
- Generated/ignored artifact cleanup: check ignored files after tests or docs builds and remove unintended `.coverage`, `coverage.xml`, `.pytest_cache/`, `.ruff_cache/`, `docs/site/`, `__pycache__/`, and tool scratch files.
- SHA alignment: after pushing, verify local `HEAD`, `origin/<branch>`, and the PR head SHA all point to the intended commit.
- Finite post-push monitoring: wait for CI/checks and review state to settle, up to 10 minutes by default and 30 minutes maximum unless the user asks for longer; before reporting completion, confirm there are no unresolved review threads and no pending requested reviews.
- Repeated review feedback is a design signal: stop and reassess the contract before adding more defensive branches for the same behavior.

## Instruction Source Matrix

- Codex: loads `AGENTS.md`; treat this file as the repository-canonical cross-agent checklist.
- GitHub Copilot: loads `.github/copilot-instructions.md` plus applicable `.github/instructions/*.instructions.md` files.
- Copilot custom agents: load `.github/agents/*.agent.md` only when that agent is selected; those files are role adapters.
- Skills: runtime procedures from the active agent environment; they are not repository source of truth.

Area-specific guidance lives in:

- `.github/instructions/processing-api.instructions.md`
- `.github/instructions/io-contracts.instructions.md`
- `.github/instructions/frames-design.instructions.md`
- `.github/instructions/testing-workflow.instructions.md`
- `.github/instructions/test-grand-policy.instructions.md`
- `.github/instructions/test-frames-policy.instructions.md`
- `.github/instructions/test-processing-policy.instructions.md`
- `.github/instructions/test-io-policy.instructions.md`
- `.github/instructions/test-visualization-policy.instructions.md`
- `.github/instructions/pr-lifecycle-harness.instructions.md`
- `.github/instructions/agent-maintenance.instructions.md`
