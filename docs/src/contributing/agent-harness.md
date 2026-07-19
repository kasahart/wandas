# Repository Agent Harness / リポジトリ Agent ハーネス

The Wandas agent harness separates durable repository contracts, on-demand
procedures, and vendor adapters. The same repository meaning has one owner even
when several tools expose it differently.
Wandas の Agent harness は、永続的な repository contract、必要時に読む手順、vendor adapter を
分離します。複数の tool が異なる方法で公開する場合でも、同じ意味の正本は1か所だけです。

## Ownership map / 正本の対応

| Layer / 層 | Owner / 正本 | Content / 内容 |
| --- | --- | --- |
| Canonical contract / 共通契約 | [`AGENTS.md`](https://github.com/kasahart/wandas/blob/main/AGENTS.md) | Always-needed Wandas invariants and procedure routes / 常に必要な Wandas invariant と手順への route |
| Reusable procedures / 再利用手順 | [`.agents/skills`](https://github.com/kasahart/wandas/tree/main/.agents/skills) | Task-triggered workflows, references, and helpers / task 起点の workflow、reference、helper |
| Codex adapter | Native discovery; optional `agents/openai.yaml` beside a Skill / native discovery と Skill 内の任意 `agents/openai.yaml` | UI metadata and tool dependencies only / UI metadata と tool dependency のみ |
| Claude Code adapter | [`CLAUDE.md`](https://github.com/kasahart/wandas/blob/main/CLAUDE.md) | `@AGENTS.md` import and the `.agents/skills` routing difference / `@AGENTS.md` import と `.agents/skills` の読み込み差分 |
| GitHub Copilot adapter | [`.github/copilot-instructions.md`](https://github.com/kasahart/wandas/blob/main/.github/copilot-instructions.md), [`.github/instructions`](https://github.com/kasahart/wandas/tree/main/.github/instructions), and [`.github/agents`](https://github.com/kasahart/wandas/tree/main/.github/agents) | Surface loading, path routing, tool boundaries, and optional handoffs / surface の読み込み、path routing、tool 境界、任意 handoff |

`wandas-frame-operation-extension` is the preferred progressive-disclosure
pattern: the Skill stays short and routes to the complete
[Frame and Operation extension guide](frame-operation-extensions.md). New
multi-step procedures should follow that pattern instead of expanding
`AGENTS.md`.
`wandas-frame-operation-extension` は progressive disclosure の推奨 pattern です。
Skill は短いまま、完全な [Frame・Operation 拡張ガイド](frame-operation-extensions.md)へ
route します。新しい複数 step の手順は `AGENTS.md` を増やさず、この pattern に従います。

Detailed testing policy follows the same model. The
[`wandas-test-authoring` Skill](https://github.com/kasahart/wandas/blob/main/.agents/skills/wandas-test-authoring/SKILL.md)
owns the workflow and vendor-neutral references. The existing
`.github/instructions/test-*.instructions.md` files are thin Copilot path adapters
that route test directories to that Skill and the matching domain reference.
詳細な test policy も同じ model に従います。`wandas-test-authoring` Skill が workflow と
vendor-neutral reference の正本です。既存の `.github/instructions/test-*.instructions.md` は、
test directory を Skill と対応 domain reference へ route する薄い Copilot path adapter です。

## Where guidance belongs / 情報の配置

- Put a rule in `AGENTS.md` only when it is Wandas-specific, non-obvious, and
  needed for almost every repository task.
  Wandas 固有で非自明かつ、ほぼ全 repository task に必要な rule だけを `AGENTS.md` に置きます。
- Put a repeatable multi-step workflow in `.agents/skills`; put large reference
  material in a linked developer guide or Skill reference file.
  反復する複数 step の workflow は `.agents/skills` に置き、大きな reference は link した
  developer guide または Skill reference file に置きます。
- Put deterministic contracts in tests, CI, hooks, or shared validators when
  they can be checked mechanically.
  機械的に検査できる契約は、prose ではなく test、CI、hook、共通 validator に置きます。
- Put only loading behavior, permissions, tool names, handoffs, and UI metadata
  in vendor files. Link to the owner instead of copying repository rules.
  vendor file には loading、permission、tool 名、handoff、UI metadata のみを置き、
  repository rule は複製せず正本へ link します。

Small, clear tasks may be implemented directly. Planning, subagents, and
independent review are optional controls for ambiguity, risk, context isolation,
or useful independence; they are not mandatory stages.
小さく明確な task は直接実装できます。planning、subagent、independent review は、曖昧さ、
risk、context 分離、独立性が有効な場合の任意 control であり、必須 stage ではありません。

## Adapter decisions / Adapter の判断

Codex reads root and nested `AGENTS.md` files and discovers repository Skills in
`.agents/skills`, so Wandas needs no extra Codex instruction copy. GitHub
Copilot also supports `AGENTS.md` on agent-capable surfaces and recognizes
`.agents/skills`; `.github` files therefore contain only Copilot-specific
routing and capabilities.
Codex は root／nested `AGENTS.md` と `.agents/skills` を native に発見するため、Codex 用の
instruction copy は不要です。GitHub Copilot も agent 対応 surface で `AGENTS.md` と
`.agents/skills` を認識するため、`.github` file は Copilot 固有の routing と capability のみを
持ちます。

Claude Code currently reads `CLAUDE.md`, not `AGENTS.md`, so the root adapter
uses the officially documented `@AGENTS.md` import. Claude Code's
`.claude/skills` is a current feature, not legacy. Wandas does not mirror Skills
there: Copilot also scans that directory, duplicate names have product-specific
precedence, and directory symlinks are unreliable on Windows without extra
privileges. Claude follows the Skill links imported from `AGENTS.md` and reads
the canonical `.agents/skills/.../SKILL.md` on demand.
Claude Code は現在 `AGENTS.md` ではなく `CLAUDE.md` を読むため、root adapter は公式仕様の
`@AGENTS.md` import を使います。`.claude/skills` は legacy ではなく現行機能です。ただし Wandas
では Skill を複製しません。Copilot もその directory を scan し、重複名の優先順位が product
依存になり、directory symlink は Windows で追加権限なしには安定しないためです。Claude は
import した `AGENTS.md` の link から、正本の `.agents/skills/.../SKILL.md` を必要時に読みます。

Official specifications used for these decisions:
これらの判断に使用した公式仕様:

- [OpenAI: Custom instructions with AGENTS.md](https://developers.openai.com/codex/guides/agents-md)
- [OpenAI: Build skills](https://developers.openai.com/codex/skills)
- [Anthropic: How Claude remembers your project](https://code.claude.com/docs/en/memory)
- [Anthropic: Extend Claude with skills](https://code.claude.com/docs/en/slash-commands)
- [GitHub: Custom instruction support](https://docs.github.com/en/copilot/reference/custom-instructions-support)
- [GitHub: About agent skills](https://docs.github.com/en/copilot/concepts/agents/about-agent-skills)
- [GitHub: Custom agents configuration](https://docs.github.com/en/copilot/reference/custom-agents-configuration)

## Test-policy ownership / Test policy の正本

The test Skill always loads its grand policy and conditionally loads one or more
Frame, Processing, I/O, and Visualization references for the changed behavior.
Codex discovers the Skill natively from `.agents/skills`; Claude follows the
route imported through `CLAUDE.md` and `AGENTS.md`; Copilot uses its path adapters.
No `.claude/skills` mirror, wrapper, or symlink is needed.
test Skill は grand policy を常に読み、変更内容に応じて Frame、Processing、I/O、Visualization の
reference を1つ以上読みます。Codex は `.agents/skills` から native に Skill を発見し、Claude は
`CLAUDE.md` と `AGENTS.md` から import された route をたどり、Copilot は path adapter を使います。
`.claude/skills` の mirror、wrapper、symlink は不要です。
