# Pull Request

## Description / 説明

<!-- A clear and concise description of what this PR does and why.
     このPRが何をするものか、なぜ必要かを明確・簡潔に説明してください。 -->

<!-- Closes #<issue-number> / 関連Issue: #<issue-number> -->

---

## Type of Change / 変更の種類

<!-- Check all that apply. / 該当するものをすべてチェックしてください。 -->

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue) / バグ修正（既存機能を壊さない変更）
- [ ] ✨ New feature (non-breaking change that adds functionality) / 新機能（既存機能を壊さない機能追加）
- [ ] ⚠️ Breaking change (fix or feature that causes existing functionality to change) / 破壊的変更
- [ ] 🧹 Refactor / code cleanup / リファクタリング・コード整理
- [ ] 📝 Documentation update / ドキュメント更新
- [ ] 🧪 Test improvement / テスト改善
- [ ] 🔧 Chore / build / CI / その他の保守作業

---

## Changes / 変更内容

<!-- List the key changes made in this PR. / 主な変更点をリストアップしてください。 -->

-
-

---

## Testing / テスト

<!-- Describe how you tested your changes. / 変更をどのようにテストしたかを説明してください。 -->
<!-- Check the items that actually apply, and explain skipped checks below. -->

- [ ] Existing tests pass (`Run pytest` task / `uv run pytest -n auto --cov=wandas --cov-report=term-missing`) / 既存のテストが通ること
- [ ] Formatting applied (`Run ruff format` task / `uv run ruff format wandas tests`) / フォーマットを適用したこと
- [ ] New tests added for the changes / 変更に対応するテストを追加した
- [ ] Type checks pass (`Run mypy wandas tests` task / `uv run mypy --config-file=pyproject.toml`) / 型チェックが通ること
- [ ] Lint checks pass (`Run ruff check` task / `uv run ruff check wandas tests --config=pyproject.toml -v`) / Lintチェックが通ること
- [ ] Documentation build checked when `docs/`, `src/`, `README.md`, or other MkDocs-backed user-facing markdown changed (`Build MkDocs Documentation` task / `uv run mkdocs build -f docs/mkdocs.yml`) / `docs/`、`src/`、`README.md`、または MkDocs 対象のユーザー向け Markdown を変更した場合にドキュメントビルドを確認した

---

## Notes for Reviewers / レビュアーへのメモ

<!-- Any specific areas you'd like reviewers to focus on, or decisions that need discussion.
     レビュアーに特に確認してほしい点や、議論が必要な決定事項があれば記入してください。 -->
