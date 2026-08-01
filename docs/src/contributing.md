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

- Lint with auto-fix (ruff, only when modifying the affected files):
  自動修正付きリント (ruff, 対象ファイルを変更する場合のみ):

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

## Release-to-Agent Notification / リリースからAgentへの通知

When `WANDAS_AGENT_TOKEN` is configured, every strict `vX.Y.Z` tag dispatches
`wandas-updated` to `kasahart/wandas-agent`, which updates its Wandas submodule
to that exact tag.
`WANDAS_AGENT_TOKEN` が設定されている場合、厳密な
`vX.Y.Z` タグを作成すると、`kasahart/wandas-agent` へ `wandas-updated` が
送信され、Wandas submoduleがそのタグへ更新されます。

The cross-repository dispatch requires the `WANDAS_AGENT_TOKEN` repository
secret. Use a fine-grained personal access token scoped only to
`kasahart/wandas-agent`, with **Contents: Read and write** permission. GitHub
documents this as the required permission for the
[repository dispatch endpoint](https://docs.github.com/en/rest/repos/repos#create-a-repository-dispatch-event).
Cross-repository dispatchにはrepository secret `WANDAS_AGENT_TOKEN` が必要です。
`kasahart/wandas-agent` のみに限定し、**Contents: Read and write** 権限を与えた
fine-grained personal access tokenを使用してください。GitHubが
[repository dispatch endpoint](https://docs.github.com/en/rest/repos/repos#create-a-repository-dispatch-event)
に必要な権限として定義しています。

Store or rotate the token without putting it on the command line:
tokenをコマンドラインへ含めずに保存または更新します:

```bash
gh secret set WANDAS_AGENT_TOKEN --repo kasahart/wandas
```

If the secret is missing, the optional notification is skipped with a
repository-owned warning so that a successful package release does not produce a
failed workflow. Reconcile the downstream submodule manually, or configure the
credential and replay an existing tag explicitly:
secretが存在しない場合、任意の通知はrepository側のwarningを出してskipされ、
package releaseが成功しているのにworkflowが失敗扱いになることを避けます。
下流submoduleを手動で同期するか、credentialを設定して既存タグを明示的に再送します:

```bash
gh workflow run notify-agent.yml \
  --repo kasahart/wandas \
  --ref main \
  -f tag=v0.6.0
```

The manual input must be an existing strict SemVer tag. Verify both the source
workflow and the resulting `Update wandas submodule` run in
`kasahart/wandas-agent` before considering notification complete.
手動入力には、実在する厳密なSemVerタグが必要です。通知完了と判断する前に、
送信元workflowと`kasahart/wandas-agent`側の`Update wandas submodule` runの
両方を確認してください。

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

Keep `README.md` and `README.ja.md`, the home page, the main tutorial, and
existing bilingual release notes synchronized when their shared contract changes.
Other technical pages may be English-only; do not duplicate API contracts there.
`README.md`と`README.ja.md`、home page、main tutorial、既存のバイリンガルrelease noteは、
共通契約を変更する場合に同期します。その他の技術ページは英語のみでもよく、API契約を
重複記載しません。

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

- [ ] Maintained bilingual files are synchronized when their shared contract changes.
      維持対象のバイリンガル文書は、共通契約の変更時に同期されている。
- [ ] Code examples are valid and tested.
      コード例が有効でテスト済みである。
- [ ] Links are correct and not broken.
      リンクが正しく、切れていない。
- [ ] Formatting is consistent with existing documentation.
      既存のドキュメントとフォーマットが一致している。
- [ ] Public API docstrings use Google style, and the lightweight public export
      check passes.
      公開API docstringはGoogle styleを使用し、軽量な公開exportチェックが成功している。
