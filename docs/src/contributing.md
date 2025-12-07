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
   uv sync
   ```

## Running Tests / テストの実行

Tests are located in the `tests/` directory.
`tests/` ディレクトリにテストがあります。

- Run all tests (parallel):
  全テストの実行 (並列):
  ```bash
  uv run pytest -n auto
  ```
- Run tests serially (for debugging):
  シリアル実行 (デバッグ時など):
  ```bash
  uv run pytest
  ```

## Code Quality Checks / コード品質チェック

Please perform the following checks before submitting a pull request.
プルリクエストを送る前に、以下のチェックを行ってください。

- Type check (mypy):
  型チェック (mypy):
  ```bash
  uv run mypy --config-file=pyproject.toml
  ```
- Lint (ruff):
  リント (ruff):
  ```bash
  uv run ruff check wandas tests --config=pyproject.toml
  ```
- Format (ruff):
  フォーマット (ruff):
  ```bash
  uv run ruff format wandas tests
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
