# AGENTS.md

## Repository Guidance

Follow `.github/copilot-instructions.md` for the full Wandas repository conventions. This file is the short Codex-facing entry point loaded from the repository root.

Core rules:

- Use `uv` for Python commands.
- For substantive code changes, use a dedicated Git worktree under `.worktrees/` before editing unless the user explicitly asks to work in the current checkout.
- Preserve frame immutability, metadata/history, and Dask laziness.
- Keep frame methods thin; numerical logic belongs in `wandas/processing`.
- Prefer small, explicit contracts over compatibility layers for undocumented or ambiguous behavior.
- Avoid duplicated state, silent no-op compatibility shims, and public mutable state that must be synchronized with internal state.
- When changing behavior, update the relevant tests so they describe the clarified contract.
- Run relevant `uv run pytest`, `uv run ruff check`, and `uv run ty check` commands before finishing.

Area-specific guidance lives in:

- `.github/instructions/processing-api.instructions.md`
- `.github/instructions/frames-design.instructions.md`
- `.github/instructions/testing-workflow.instructions.md`
- `.github/instructions/test-grand-policy.instructions.md`
