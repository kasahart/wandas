# Wandas Repository Contract

This file is the canonical, cross-agent contract for repository work. Keep it
limited to Wandas invariants that are non-obvious and needed throughout a task.

## Working invariants

- Use `uv` for Python commands.
- Inspect `git status --short` before editing. For substantive changes, use a
  dedicated worktree under `.worktrees/` when appropriate, and never move,
  stage, revert, or overwrite unrelated user changes.
- Preserve Frame immutability, metadata and lineage, and Dask laziness.
  `operation_history` remains a derived compatibility view of lineage.
- Keep Frame methods thin: orchestration and metadata belong in
  `wandas/frames`; numerical algorithms belong in `wandas/processing`.
- Prefer small, explicit contracts over undocumented compatibility layers,
  duplicated state, silent no-ops, or public mutable state that must be kept in
  sync with internal state.
- When behavior changes, update tests to describe the clarified contract.
- Before finishing, run the relevant `uv run pytest`, `uv run ruff check`, and
  `uv run ty check` commands, plus documentation or notebook checks when those
  artifacts change. Report commands that were skipped and why.

## Procedure routing

Load detailed procedures only when the task calls for them:

- Frame, Operation, Recipe, or related test extensions:
  [`wandas-frame-operation-extension`](.agents/skills/wandas-frame-operation-extension/SKILL.md)
- Test planning, implementation, and review across all domains:
  [`wandas-test-authoring`](.agents/skills/wandas-test-authoring/SKILL.md)
- I/O readers, writers, and format contracts:
  [I/O contracts](docs/src/contributing/io-contracts.md)
- Code or dependency changes that can affect representative scalability
  benchmark metrics or measured materialization boundaries:
  [`wandas-scalability-benchmark`](.agents/skills/wandas-scalability-benchmark/SKILL.md)
- Learning materials and executable notebooks:
  [`wandas-learning-material-authoring`](.agents/skills/wandas-learning-material-authoring/SKILL.md)
- Ambiguous, high-risk, cross-cutting, or related-finding changes:
  [`wandas-change-coherence`](.agents/skills/wandas-change-coherence/SKILL.md)
- Pull-request readiness and finite monitoring:
  [`wandas-pr-readiness`](.agents/skills/wandas-pr-readiness/SKILL.md)
- Issue closing, relation, and follow-up decisions:
  [`wandas-issue-triage`](.agents/skills/wandas-issue-triage/SKILL.md)

The [repository agent harness guide](docs/src/contributing/agent-harness.md)
explains instruction ownership, tool adapters, and where new guidance belongs.
