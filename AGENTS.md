# Wandas Repository Contract

- Use `uv` for Python commands. Check `git status --short` before editing and
  preserve unrelated changes.
- For each new task that will modify files, work in a dedicated Git worktree.
  If the current session is not already in one, stop before editing and ask the
  user to restart Codex from a worktree.
- Preserve Frame immutability, metadata, lineage, and Dask laziness.
  `operation_history` is a derived compatibility view of lineage.
- Keep orchestration and metadata in `wandas/frames`; keep numerical algorithms
  in `wandas/processing`.
- Update tests when behavior changes.
- Run relevant `uv run pytest`, `uv run ruff check`, and `uv run ty check`
  commands before finishing. Run documentation or notebook checks when those
  artifacts change, and report justified skips.

Load the following specialized guidance when the task matches:

- Frame, Operation, Recipe, or signal-processing behavior changes:
  [extension guide](docs/src/contributing/frame-operation-extensions.md)
- I/O format behavior: [I/O contracts](docs/src/contributing/io-contracts.md)
- Public API, deprecation, WDF, or Recipe schema compatibility:
  [public API and schema stability](docs/src/explanation/public-api-stability.md)
- Materialization, Dask graph, RecipePlan, `AudioOperation.process`, benchmark
  semantics, or dependency scalability:
  [`wandas-scalability-benchmark`](.agents/skills/wandas-scalability-benchmark/SKILL.md)
- Executable learning materials and supporting README, tutorial, or API examples:
  [`wandas-learning-material-authoring`](.agents/skills/wandas-learning-material-authoring/SKILL.md)
