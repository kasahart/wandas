# Repository Agent Guidance / リポジトリ Agent ガイダンス

Wandas keeps agent guidance intentionally small. Modern coding models should
infer ordinary engineering workflow from the task and repository rather than
follow a repository-specific planning or review pipeline.
Wandas の Agent guidance は意図的に小さく保ちます。通常の開発手順は repository 固有の
planning／review pipeline で規定せず、最新の coding model が task と repository から判断します。

## Ownership / 正本

- [`AGENTS.md`](https://github.com/kasahart/wandas/blob/main/AGENTS.md) contains only non-obvious invariants needed
  across repository work.
- Detailed developer documents own domain contracts.
- `.agents/skills` is reserved for specialized, repeatable work that benefits
  from exact commands or domain-specific checks.
- `CLAUDE.md` and `.github/copilot-instructions.md` only point to `AGENTS.md`.

Do not add generic planner, implementer, reviewer, publisher, workspace-hygiene,
or test-authoring agents. Do not add path-specific adapters when `AGENTS.md`, a
developer guide, or native repository discovery already provides the context.
Deterministic product contracts belong in product tests; agent prose should not
be frozen by a large policy test suite.

汎用 planner、implementer、reviewer、publisher、workspace hygiene、test authoring
agent は追加しません。`AGENTS.md`、developer guide、native discovery で十分な場合は
path-specific adapter も追加しません。決定論的な製品契約は製品 test で検査し、Agent 向け prose を
大規模な policy test suite で固定しません。

Before adding guidance, ask whether a capable model would predictably make a
Wandas-specific mistake without it. If not, prefer code, tests, API design, or
ordinary documentation.
