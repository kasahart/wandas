# 設計ガイド / Design Guides

このディレクトリには、wandasプロジェクトの永続的な設計ガイドが含まれています。

## 📚 ガイド一覧

### プロジェクト管理

#### GitHub Copilot カスタム命令のベストプラクティス

- **[日本語版](./copilot-custom-instructions-best-practices.md)** - 完全版ガイド（703行）
- **[English Version](./copilot-custom-instructions-best-practices.en.md)** - Complete guide (703 lines)
- **[クイックリファレンス](./copilot-custom-instructions-quick-reference.md)** - 要点まとめ（日本語）

`.github/copilot-instructions.md` を効果的に作成・維持するための包括的なガイドです。

**内容**:
- 基本構造とファイル配置
- コンテンツ構成のベストプラクティス
- 記述スタイルガイド
- 効果的な例の提供方法
- 保守とメンテナンス
- よくある落とし穴
- 参考資料

**対象読者**:
- リポジトリメンテナー
- プロジェクトリーダー
- コーディング規約策定者
- 開発効率を向上させたいチーム

### API設計

#### [API改善パターン](./api-improvements.md)

- `describe()`メソッドの改善（kwargs明示化）
- `plot()`メソッドの改善
- TypedDict活用パターン
- IDE補完と型安全性の向上

### アーキテクチャ設計

#### [メタデータカプセル化](./metadata-encapsulation.md)

- メタデータ更新のカプセル化パターン
- Operation層の責任分離
- YAGNI原則の適用
- 拡張性と保守性の向上

## 📖 使い方

### 新しい機能を実装する場合

1. **ガイドラインを確認**: まず [.github/copilot-instructions.md](../../../.github/copilot-instructions.md) を読む
2. **設計パターンを学ぶ**: このディレクトリのガイドを参照
3. **実装プランを作成**: `../working/plans/PLAN_<機能名>.md` を作成（Git管理外）

### 過去の設計決定を理解したい場合

1. **ガイドを読む**: このディレクトリのガイドを参照
2. **INDEX を確認**: [../INDEX.md](../INDEX.md) で全体を把握
3. **関連PRを確認**: ガイド内のリンクから GitHub PR を参照

## 🔄 ドキュメントのライフサイクル

詳細は [../DOCUMENT_LIFECYCLE.md](../DOCUMENT_LIFECYCLE.md) を参照してください。

### 新しいガイドを作成する場合

1. **下書き作成**: `../working/drafts/<トピック>_SUMMARY.md`（Git管理外）
2. **清書**: このディレクトリに `.md` ファイルを作成
3. **INDEX更新**: `../INDEX.md` に追加
4. **レビュー**: mainブランチへのマージ前に必ずレビュー

### ガイド数の目標

- **現在**: 5個（API改善、メタデータカプセル化、Copilotベストプラクティス×3）
- **目標**: 5-15個
- **15個超えたら**: 統合を検討

## ⚠️ 重要な注意事項

### Git管理について

- ✅ このディレクトリのファイルは **Git管理** されています
- ✅ mainブランチへのマージ前に **必ずレビュー** してください
- ❌ 下書きや作業中のファイルは `../working/` に配置してください

### 品質基準

- 明確な構造と読みやすい文章
- 具体的なコード例を含む
- 将来の参照価値が高い内容
- プロジェクトの設計原則に準拠

### 更新について

- 定期的な見直し（3-6ヶ月ごと）
- 最終更新日を明記
- 古くなった情報は削除または更新
- 重要な変更は変更履歴に記録

## 🔗 関連リソース

- **設計ドキュメント一覧**: [../INDEX.md](../INDEX.md)
- **ドキュメントライフサイクル**: [../DOCUMENT_LIFECYCLE.md](../DOCUMENT_LIFECYCLE.md)
- **プロジェクトガイドライン**: [../../../.github/copilot-instructions.md](../../../.github/copilot-instructions.md)
- **APIドキュメント**: [../../src/api/](../../src/api/)
- **チュートリアル**: [../../src/tutorial/](../../src/tutorial/)

## 💡 ガイド作成のヒント

新しいガイドを作成する際は：

1. **クイックリファレンスを確認**: [copilot-custom-instructions-quick-reference.md](./copilot-custom-instructions-quick-reference.md)
2. **既存ガイドを参考にする**: 構成と書き方を学ぶ
3. **具体例を豊富に含める**: コード例、チェックリストなど
4. **簡潔に保つ**: 500-1500行が理想
5. **将来の参照価値を考える**: 長期的に有用な情報か？

---

**Note**: このディレクトリは継続的に進化します。改善提案は Issue または PR で歓迎します。
