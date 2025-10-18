# 設計ドキュメントのライフサイクル管理

## 目的

設計ドキュメントの肥大化を防ぎ、必要な情報だけを適切に管理する。

## ディレクトリ構成

```text
docs/design/
├── working/              # 作業用（gitignore）
│   ├── plans/           # PLAN_*.md（実装前の計画）
│   └── drafts/          # *_SUMMARY.md（下書き）
├── guides/              # 永続的なガイドドキュメント
│   ├── api-improvements.md
│   └── metadata-encapsulation.md
├── INDEX.md             # ドキュメント一覧
└── DOCUMENT_LIFECYCLE.md # このファイル
```

## ドキュメントのライフサイクル

```text
企画 → working/plans/PLAN_*.md (作業用、gitignore)
  ↓
実装 → PLANを参照
  ↓
完了 → working/drafts/*_SUMMARY.md作成（下書き）
  ↓
清書 → guides/*.md（永続、Git管理）
  ↓
整理 → working/以下は自動的に無視される
```

## ドキュメントの種類

### working/plans/PLAN_*.md（計画書）- 作業用、gitignore

**目的**: 実装前の設計と意思決定の記録

**特徴**:

- Git管理外（`.gitignore`に登録）
- ローカルで自由に編集可能
- チーム共有不要（個人の作業メモ）

**完了後**: 削除またはそのまま放置（gitignoreされているため問題なし）

### working/drafts/*_SUMMARY.md（サマリー下書き）- 作業用、gitignore

**目的**: 実装完了後の記録（下書き）

**特徴**:

- Git管理外（`.gitignore`に登録）
- 実装の詳細、試行錯誤の記録
- 生の情報、整理されていないメモ

**完了後**: 清書して`guides/`に移動、または削除

### guides/*.md（ガイドドキュメント）- 永続、Git管理

**目的**: プロジェクトの設計パターンとベストプラクティス

**特徴**:

- Git管理（チーム共有）
- 整理された情報
- 将来の参照価値が高い
- 例: `api-improvements.md`, `metadata-encapsulation.md`

**保持ルール**:

- ✅ 原則すべて保持
- 📦 類似のガイドは統合推奨
- 🔄 四半期ごとに見直し

## ワークフロー

### 1. 新機能の企画

```bash
# 作業用ディレクトリにPLANを作成（gitignoreされている）
vim docs/design/working/plans/PLAN_my_feature.md
```

### 2. 実装

PLANを参照しながらコーディング。

### 3. 実装完了後

```bash
# SUMMARYの下書きを作成（gitignoreされている）
vim docs/design/working/drafts/MY_FEATURE_SUMMARY.md

# 重要な設計決定があれば清書してガイドに追加
vim docs/design/guides/my-feature.md
git add docs/design/guides/my-feature.md
git commit -m "docs: Add my-feature design guide"
```

### 4. レビューとマージ

**重要**: `guides/`配下のドキュメントをmainブランチにマージする前に、必ずレビューを実施してください。

**レビュー観点**:

- [ ] 内容が明確で理解しやすいか
- [ ] 設計原則に沿っているか
- [ ] 既存のガイドと重複していないか
- [ ] 将来的な参照価値があるか
- [ ] INDEX.mdに適切に記載されているか

### 5. 定期的な整理

```bash
# working/以下は自動的にgitignoreされているため、
# 不要になったファイルは手動で削除するか、そのまま放置
rm docs/design/working/plans/PLAN_old_feature.md
rm docs/design/working/drafts/OLD_FEATURE_SUMMARY.md

# guides/以下のみGit管理
ls docs/design/guides/
```

## ドキュメント数の目安

### 理想的な状態

```text
docs/design/
├── working/              # Git管理外
│   ├── plans/           # 自由に作成・削除
│   └── drafts/          # 自由に作成・削除
├── guides/              # Git管理（5-10個が目標）
│   ├── api-improvements.md
│   ├── metadata-encapsulation.md
│   └── ...
├── INDEX.md
└── DOCUMENT_LIFECYCLE.md
```

### 整理が必要な状態

- `guides/`が15個以上 → 統合を検討
- 類似トピックのガイドが複数 → 1つに統合

## レビューチェックリスト

### 四半期レビュー

```markdown
## ドキュメントレビュー - YYYY年QX

### guides/ (計X個)

- [ ] 統合可能な類似ガイドはないか
- [ ] 古くなった情報はないか
- [ ] すべてINDEX.mdに記載されているか

### working/ (任意)

- [ ] 完了したPLANで清書すべきものはないか
- [ ] 不要なファイルを削除（任意、gitignoreされているため放置も可）

### アクション

- 統合: [リスト]
- 更新: [リスト]
```

## ベストプラクティス

### 1. PLANは自由に作成

```text
Good: 必要なときに作成、完了したら放置（gitignore）
Bad:  Gitにコミット → レポジトリが汚れる
```

### 2. SUMMARYは清書してから公開

```text
Good: working/drafts/で下書き → 清書してguides/に移動
Bad:  下書きをそのままコミット → 読みにくい
```

### 3. guidesは価値重視

```text
Good: 重要な設計決定・大規模機能・パターン
Bad:  些細なバグ修正（PRで十分）
```

### 4. 統合は積極的に

```text
Good: 関連する3つの改善 → 1つのガイドに統合
Bad:  小さな改善ごとにガイド → 散乱
```

## メリット

### Git管理の簡素化

- ✅ 作業用ファイルはGit管理外（`.gitignore`）
- ✅ レポジトリは整理されたガイドのみ
- ✅ PRレビューが容易（ガイドの追加/更新のみ）

### 柔軟な作業

- ✅ PLANは自由に作成・編集・削除
- ✅ 下書きは試行錯誤可能
- ✅ チーム共有不要な情報はローカルに保管

### ドキュメント品質の向上

- ✅ 公開前に清書する習慣
- ✅ 整理された情報のみがGit管理される
- ✅ 将来の参照価値が高い

## まとめ

### 原則

1. **Git管理の分離**: 作業用（working/）はgitignore、完成品（guides/）はGit管理
2. **清書の習慣**: 下書きを清書してから公開
3. **価値重視**: 将来の参照価値が高いものだけguides/に保管
4. **定期レビュー**: 四半期ごとにguides/を見直し

### 目標ドキュメント数

- **guides/**: 5-10個が理想
- **15個超えたら整理**
- **working/**: 無制限（gitignore）

### ワークフロー

1. ✅ PLANは`working/plans/`に作成（gitignore）
2. ✅ 実装完了後、`working/drafts/`にSUMMARY下書き
3. ✅ 重要な設計は清書して`guides/`に移動
4. ✅ 四半期レビューで`guides/`を整理
