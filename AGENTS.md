# Wandas Repository Contract

- Use `uv` for Python commands. Check `git status --short` before editing and
  preserve unrelated changes.
- Preserve Frame immutability, metadata, lineage, and Dask laziness.
  `operation_history` is a derived compatibility view of lineage.
- Keep orchestration and metadata in `wandas/frames`; keep numerical algorithms
  in `wandas/processing`.
- Update tests when behavior changes.
- Run relevant `uv run pytest`, `uv run ruff check`, and `uv run ty check`
  commands before finishing. Run documentation or notebook checks when those
  artifacts change, and report justified skips.

Load specialized guidance only for:

- Frame, Operation, or Recipe extensions:
  [extension guide](docs/src/contributing/frame-operation-extensions.md)
- I/O format behavior: [I/O contracts](docs/src/contributing/io-contracts.md)
- Materialization, Dask graph, RecipePlan, `AudioOperation.process`, benchmark
  semantics, or dependency scalability:
  [`wandas-scalability-benchmark`](.agents/skills/wandas-scalability-benchmark/SKILL.md)
- Executable learning materials and supporting README, tutorial, or API examples:
  [`wandas-learning-material-authoring`](.agents/skills/wandas-learning-material-authoring/SKILL.md)
