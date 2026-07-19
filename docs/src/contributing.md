# Contributing to Wandas / Wandasプロジェクトへの貢献

Thank you for your interest in contributing to the Wandas project.
Wandasプロジェクトへの貢献に興味を持っていただきありがとうございます。

## Development Environment Setup / 開発環境のセットアップ

This project uses `uv` for package management.
このプロジェクトではパッケージ管理に `uv` を使用しています。

1. Clone the repository.
   リポジトリをクローンします。
2. Install dependencies:
   依存関係をインストールします:

  ```bash
  uv sync --frozen --all-groups
  ```

## Branch Naming Policy / ブランチ命名ポリシー

The repository allows the following branch naming patterns. For normal contribution work, please create your branch with one of the prefixed patterns listed below.
このリポジトリでは以下のブランチ命名パターンを許可します。通常のコントリビューション作業では、以下のプレフィックス付きパターンのいずれかでブランチを作成してください。

- `main`
- `feat/*` - new features, for example `feat/add-plot-export`
  `feat/*` - 新機能追加。例: `feat/add-plot-export`
- `fix/*` - bug fixes, for example `fix/axis-label-bug`
  `fix/*` - バグ修正。例: `fix/axis-label-bug`
- `refactor/*` - internal refactors without intended behavior changes, for example `refactor/cleanup-fft-api`
  `refactor/*` - 意図した挙動変更を伴わない内部整理。例: `refactor/cleanup-fft-api`
- `chore/*` - maintenance work such as CI, tooling, or docs updates, for example `chore/update-ci`
  `chore/*` - CI、ツール、ドキュメント更新などの保守作業。例: `chore/update-ci`
- `release/vx.x.x` - release preparation branches, for example `release/v0.3.1`
  `release/vx.x.x` - リリース準備用ブランチ。例: `release/v0.3.1`

`main` is the always-releasable base branch and should not be used as the working branch for regular changes.
`main` は常にリリース可能なベースブランチであり、通常の変更作業用ブランチとしては使用しないでください。

Examples: `feat/add-plot-export`, `fix/axis-label-bug`, `release/v0.3.1`
例: `feat/add-plot-export`、`fix/axis-label-bug`、`release/v0.3.1`

## Running Tests / テストの実行

Tests are located in the `tests/` directory.
`tests/` ディレクトリにテストがあります。

- Preferred VS Code task:
  推奨 VS Code タスク:

  - `Run pytest`

- Run all tests (parallel with coverage):
  全テストの実行 (カバレッジ付き並列):

  ```bash
  uv run pytest -n auto --cov=wandas --cov-report=term-missing
  ```

## Code Quality Checks / コード品質チェック

Please perform the following checks before submitting a pull request.
プルリクエストを送る前に、以下のチェックを行ってください。

- Type check (ty):
  型チェック (ty):

  Preferred VS Code task:
  推奨 VS Code タスク:

  - `Run ty (red-knot) check`

  ```bash
  uv run ty check wandas tests
  ```

- Format (ruff):
  フォーマット (ruff):

  Preferred VS Code task:
  推奨 VS Code タスク:

  - `Run ruff format`

  ```bash
  uv run ruff format wandas tests
  ```

- Lint (ruff):
  リント (ruff):

  Preferred VS Code task:
  推奨 VS Code タスク:

  - `Run ruff check`

  ```bash
  uv run ruff check wandas tests --config=pyproject.toml -v
  ```

- Lint with auto-fix (ruff, implementer/publisher only when intended):
  自動修正付きリント (ruff, implementer/publisher が意図的に使う場合のみ):

  Preferred VS Code task:
  推奨 VS Code タスク:

  - `Run ruff check --fix`

  ```bash
  uv run ruff check --fix wandas tests --config=pyproject.toml -v
  ```

## Building Documentation / ドキュメントのビルド

Documentation is built with MkDocs.
ドキュメントは MkDocs で構築されています。

- Build:
  ビルド:

  ```bash
  uv run mkdocs build -f docs/mkdocs.yml
  ```

- Serve locally:
  ローカルサーバー起動:

  ```bash
  uv run mkdocs serve -f docs/mkdocs.yml
  ```

## Extending Frames and Operations / Frame・Operation の拡張

When adding a new Frame family, numerical Operation, public Frame method, or its
tests, follow the [Frame and Operation extension guide](contributing/frame-operation-extensions.md).
The guide includes the design decision, implementation boundaries, Recipe support,
test matrix, public documentation, and Agent reference route.
新しいFrame family、数値Operation、公開Frameメソッド、またはそのtestを追加する場合は、
[Frame・Operation拡張ガイド](contributing/frame-operation-extensions.md)に従ってください。
設計判断、実装境界、Recipe対応、test matrix、公開文書、Agentの参照経路をまとめています。

## Repository Agent Harness / リポジトリ Agent ハーネス

Repository-local agent guidance is organized as one canonical contract,
on-demand reusable procedures, and thin tool adapters. See the
[repository agent harness guide](contributing/agent-harness.md) before adding or
duplicating agent instructions.
Repository-local な Agent guidance は、1つの canonical contract、必要時に読む再利用手順、
薄い tool adapter に分けています。Agent instruction を追加または複製する前に、
[repository agent harness guide](contributing/agent-harness.md)を参照してください。

## Documentation Guidelines / ドキュメントガイドライン

### Bilingual Content / バイリンガル表記

All documentation is maintained in a bilingual format (English / Japanese) within a single file.
すべてのドキュメントは、単一ファイル内でバイリンガル形式（英語/日本語）で管理されています。

**Important / 重要**:

- When updating documentation, **always update both languages simultaneously**.
  ドキュメントを更新する際は、**必ず両言語を同時に更新してください**。
- Follow the established format: English text followed by Japanese translation.
  確立された形式に従ってください：英語テキストに続いて日本語訳。
- For code examples, use bilingual comments where appropriate.
  コード例では、適切な場合にバイリンガルコメントを使用してください。

**Format example / 形式の例**:

```markdown
## Section Title / セクションタイトル

English description of the section.
セクションの日本語説明。

- **Feature name**: English description.
  **機能名**: 日本語説明。
```

### Documentation Structure / ドキュメント構成

- `docs/src/index.md` - Home page / ホームページ
- `docs/src/tutorial/` - Tutorials / チュートリアル
- `docs/src/api/` - API reference / APIリファレンス
- `docs/src/explanation/` - Theory and architecture / 理論とアーキテクチャ
- `docs/src/contributing.md` - This file / このファイル
- `docs/src/contributing/agent-harness.md` - Repository agent instruction ownership / Repository Agent instruction の正本構成
- `docs/src/contributing/io-contracts.md` - I/O design and round-trip contracts / I/O 設計と round-trip 契約
- `docs/src/contributing/frame-operation-extensions.md` - Frame and Operation extension workflow / Frame・Operation拡張手順

### Review Checklist / レビューチェックリスト

When reviewing documentation PRs, verify:
ドキュメントのPRをレビューする際は、以下を確認してください：

- [ ] Both English and Japanese versions are updated.
      英語版と日本語版の両方が更新されている。
- [ ] Code examples are valid and tested.
      コード例が有効でテスト済みである。
- [ ] Links are correct and not broken.
      リンクが正しく、切れていない。
- [ ] Formatting is consistent with existing documentation.
      既存のドキュメントとフォーマットが一致している。
