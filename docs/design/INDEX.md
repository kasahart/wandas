# 設計ドキュメント一覧

**最終更新**: 2025年11月05日

## 📖 このディレクトリについて

このディレクトリには、wandasプロジェクトの設計パターン、ベストプラクティス、実装記録が含まれています。

## 🗂️ ディレクトリ構成

```text
docs/design/
├── working/              # Git管理外（.gitignore）
│   ├── plans/           # PLAN_*.md（実装前の計画）
│   └── drafts/          # *_SUMMARY.md（実装後の下書き）
├── guides/              # 永続的なガイドドキュメント（Git管理）
│   ├── api-improvements.md
│   └── metadata-encapsulation.md
├── INDEX.md             # このファイル
└── DOCUMENT_LIFECYCLE.md # ドキュメント管理ルール
```

## 📋 ドキュメント管理の原則

詳細は **[DOCUMENT_LIFECYCLE.md](./DOCUMENT_LIFECYCLE.md)** を参照してください。

**要約**:

- **working/** はGit管理外（自由に作成・編集・削除）
- **guides/** は永続的（Git管理、5-15個を目標）
- 四半期ごとにレビューして整理
- 清書してから公開する習慣
- **mainブランチへのマージ前に必ずレビュー**

## 🗃️ トピック別インデックス

### プロジェクト管理

**[DOCUMENT_LIFECYCLE.md](./DOCUMENT_LIFECYCLE.md)**
ドキュメントのライフサイクル管理ルール。新しいドキュメントを作成する前に必読。

**[guides/copilot-instructions-optimization.md](./guides/copilot-instructions-optimization.md)**

- GitHub Copilot instructions の最適化 (36KB → 10KB)
- 日本語から英語への移行
- パス固有の指示（Python, テスト, ドキュメント）
- 参照ドキュメント戦略
- モジュラー構造への改善

### API設計

**[guides/api-improvements.md](./guides/api-improvements.md)**

- `describe()`メソッドの改善（kwargs明示化）
- `plot()`メソッドの改善
- TypedDict活用パターン
- IDE補完と型安全性の向上

### アーキテクチャ設計

**[guides/metadata-encapsulation.md](./guides/metadata-encapsulation.md)**

- メタデータ更新のカプセル化パターン
- Operation層の責任分離
- YAGNI原則の適用
- 拡張性と保守性の向上

## 📚 ドキュメントの読み方

### 新機能を実装したい場合

1. **ガイドラインを確認**: `.github/copilot-instructions.md`
2. **類似のPLANを参照**: `working/plans/PLAN_*.md`
3. **設計パターンを学ぶ**: `guides/`のドキュメント

### 過去の実装経緯を知りたい場合

1. **SUMMARYを確認**: `working/drafts/*_SUMMARY.md`
2. **ガイドを参照**: `guides/*.md`
3. **関連PRを確認**: GitHub PRリンク

### リファクタリングの経緯を理解したい場合

1. **REFACTORING_SUMMARYを読む**: `working/drafts/REFACTORING_SUMMARY_*.md`
2. **関連するガイドを確認**: `guides/*.md`
3. **コード変更履歴を確認**: Git履歴

## 🔗 関連リソース

- **プロジェクトガイドライン**: `.github/copilot-instructions.md`（設計原則SOLID/YAGNI/KISS/DRYはここに記載）
- **APIドキュメント**: `docs/src/api/`
- **チュートリアル**: `docs/src/tutorial/`
- **使用例**: `examples/`
- **テスト**: `tests/`

## 📝 新しいドキュメントを作成する場合

詳細は **[DOCUMENT_LIFECYCLE.md](./DOCUMENT_LIFECYCLE.md)** を参照してください。

**簡単なワークフロー**:

1. **PLAN作成**: `working/plans/PLAN_<機能名>.md`（Git管理外）
2. **実装**: PLANを参照しながらコーディング
3. **SUMMARY下書き**: `working/drafts/<機能名>_SUMMARY.md`（Git管理外）
4. **清書・公開**: 重要な設計決定は `guides/` に移動（Git管理）

## 🎯 目標ドキュメント数

- **guides/**: 5-15個が理想（現在: 3個）
- **working/**: 無制限（Git管理外）
- **15個超えたら整理を検討**

## ⚠️ 重要な注意事項

### working/配下のファイルについて

- **working/plans/** と **working/drafts/** はGit管理外です
- これらのファイルは個人の作業メモであり、他の開発者には見えません
- **このINDEXには`guides/`配下のドキュメントのみを記載してください**
- 進行中のプロジェクトについては、PRやIssueで情報共有してください

### ドキュメントのレビュー

- **guides/** に新しいドキュメントを追加する際は、mainブランチへのマージ前に必ずレビューを実施してください
- レビュー観点は [DOCUMENT_LIFECYCLE.md](./DOCUMENT_LIFECYCLE.md) を参照してください

---

**Note**: このINDEXは定期的に更新されます。新しいガイドが追加されたら、このファイルに追記してください。
