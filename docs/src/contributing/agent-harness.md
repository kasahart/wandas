# Repository Agent Guidance / リポジトリ Agent ガイダンス

Wandas keeps agent guidance intentionally small. Modern coding models should
infer ordinary engineering workflow from the task and repository rather than
follow a repository-specific planning or review pipeline.
Wandas の Agent guidance は意図的に小さく保ちます。通常の開発手順は repository 固有の
planning／review pipeline で規定せず、最新の coding model が task と repository から判断します。

## Ownership / 正本

- [`AGENTS.md`](https://github.com/kasahart/wandas/blob/main/AGENTS.md) contains only non-obvious invariants needed
  across repository work.
  [`AGENTS.md`](https://github.com/kasahart/wandas/blob/main/AGENTS.md)には、repository
  全体で必要な非自明のinvariantだけを置きます。
- Detailed developer documents own domain contracts.
  詳細なdeveloper documentがdomain contractを所有します。
- `.agents/skills` is reserved for specialized, repeatable work that benefits
  from exact commands or domain-specific checks.
  `.agents/skills`は、正確なcommandやdomain固有checkが有効な専門的・反復的作業に限定します。
- `CLAUDE.md` and `.github/copilot-instructions.md` only point to `AGENTS.md`.
  `CLAUDE.md`と`.github/copilot-instructions.md`は`AGENTS.md`だけを参照します。

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
guidanceを追加する前に、有能なmodelでもそれがなければWandas固有の誤りを予測可能な形で
起こすかを確認します。そうでなければ、code、test、API design、または通常のdocumentationを
優先します。
