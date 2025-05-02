# コントリビューションガイド

Wandasプロジェクトへの貢献に興味を持っていただきありがとうございます。このガイドでは、プロジェクトにコントリビュートする方法について説明します。

## コントリビューションの種類

以下のような形でWandasプロジェクトに貢献することができます：

- バグ報告や機能リクエスト
- ドキュメントの改善
- バグ修正
- 新機能の実装
- テストの追加や改善
- パフォーマンスの最適化

## 開発環境のセットアップ

### 1. リポジトリのクローン

```bash
git clone https://github.com/yourusername/wandas.git
cd wandas
```

### 2. 仮想環境のセットアップと依存関係のインストール

```bash
python -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate
pip install -e ".[dev]"
```

## 開発ワークフロー

### 1. 新しいブランチの作成

新しい機能やバグ修正には、新しいブランチを作成してください：

```bash
git checkout -b feature/your-feature-name
# または
git checkout -b fix/issue-description
```

### 2. コードスタイル

コードは[PEP 8](https://www.python.org/dev/peps/pep-0008/)スタイルガイドに従ってください。コードの品質を確保するために、以下のツールを使用しています：

- **Ruff**: コードリンターとフォーマッター
- **mypy**: 静的型チェック

コードをコミットする前に、以下のコマンドを実行してコードスタイルをチェックしてください：

```bash
# リンティング
ruff check wandas tests

# 型チェック
mypy wandas tests
```

### 3. テスト

新しい機能やバグ修正には、必ずテストを追加してください。テストは`pytest`を使用して実行します：

```bash
pytest
```

カバレッジレポートを生成するには：

```bash
pytest --cov=wandas tests/
```

### 4. ドキュメント

コードの変更にはドキュメントの更新が必要です：

- 関数やクラスには[NumPy形式](https://numpydoc.readthedocs.io/en/latest/format.html)のdocstringを記述
- 新しい機能については、適切なドキュメントページ（チュートリアル、ハウツー、APIリファレンス）を更新または追加
- 必要に応じて、`examples/`ディレクトリにサンプルコードを追加

ドキュメントをビルドして確認するには：

```bash
cd docs
mkdocs serve
```

その後、ブラウザで http://localhost:8000 にアクセスして確認できます。

### 5. プルリクエストの作成

変更が完了したら、プルリクエスト（PR）を作成してください：

1. 変更をコミットし、リモートリポジトリにプッシュ
   ```bash
   git add .
   git commit -m "説明的なコミットメッセージ"
   git push origin your-branch-name
   ```

2. GitHubリポジトリページでプルリクエストを作成
3. PRの説明には、変更内容、解決した問題、テスト方法などを記載

## レビュープロセス

すべてのPRは以下のプロセスでレビューされます：

1. 自動CIテストが成功すること
2. 少なくとも1人のメンテナによるコードレビュー
3. 必要に応じて変更の依頼と対応
4. マージ承認

## コミュニケーション

質問や議論があれば、以下の方法でコミュニケーションを取ることができます：

- GitHub Issues: バグ報告や機能リクエスト
- GitHub Discussions: 一般的な質問や議論
- プロジェクトのメーリングリスト（ある場合）

## 行動規範

プロジェクトに参加するすべての人は、[行動規範](CODE_OF_CONDUCT.md)に従うことが期待されています。他のコントリビューターに対して敬意を持って接し、協力的で建設的なコミュニケーションを心がけてください。