# GitHub Copilot リポジトリカスタム命令のベストプラクティス

**最終更新**: 2025年11月2日

## 目次

1. [はじめに](#はじめに)
2. [基本構造](#基本構造)
3. [コンテンツ構成のベストプラクティス](#コンテンツ構成のベストプラクティス)
4. [記述スタイルガイド](#記述スタイルガイド)
5. [効果的な例の提供](#効果的な例の提供)
6. [保守とメンテナンス](#保守とメンテナンス)
7. [よくある落とし穴](#よくある落とし穴)
8. [参考資料](#参考資料)

## はじめに

GitHub Copilotのリポジトリカスタム命令（`.github/copilot-instructions.md`）は、プロジェクト固有のコーディング規約、設計原則、ベストプラクティスをCopilotに伝えるための強力な機能です。

### なぜカスタム命令が重要か

- **一貫性**: プロジェクト全体で統一されたコーディングスタイルを維持
- **品質向上**: プロジェクト固有のベストプラクティスを自動的に適用
- **オンボーディング**: 新しい開発者がプロジェクトの規約を学びやすくなる
- **効率化**: 繰り返しの説明が不要になり、開発スピードが向上

### 対象読者

このガイドは以下の方を対象としています：

- GitHubリポジトリのメンテナー
- プロジェクトリーダー
- コーディング規約を策定する開発者
- チームの開発効率を向上させたい方

## 基本構造

### ファイル配置

```text
.github/
└── copilot-instructions.md
```

### YAMLフロントマター（オプション）

ファイルの先頭にYAMLフロントマターを追加することで、特定のファイルタイプにのみ適用できます：

```yaml
---
applyTo: '.py, .ipynb'
---
```

**サポートされる設定**:

- `applyTo`: カンマ区切りのファイル拡張子リスト（例: `'.py, .js, .ts'`）
- 拡張子はドットから始める必要があります

**使用例**:

```yaml
---
# Pythonファイルとノートブックにのみ適用
applyTo: '.py, .ipynb'
---

---
# JavaScriptとTypeScriptにのみ適用
applyTo: '.js, .ts, .jsx, .tsx'
---
```

### ファイルサイズの推奨

- **理想**: 500-1500行
- **最大**: 2000行程度まで
- それ以上の場合は、内容を複数のセクションに分割し、インデックスを作成

**現在のwandasプロジェクト**: 約970行（適切な範囲内）

## コンテンツ構成のベストプラクティス

### 1. 階層構造を明確にする

```markdown
# プロジェクト名 開発ガイドライン

## プロジェクト概要
[プロジェクトの簡潔な説明]

## 設計原則
### ドメイン固有の原則
### 普遍的な設計原則

## コーディング規約
### 1. 型ヒント
### 2. エラーハンドリング
### 3. テスト

## コード変更時の手順

## ツールとワークフロー
```

**推奨ポイント**:

- ✅ 明確な見出し階層（H1 → H2 → H3）
- ✅ 番号付きリストで手順を明示
- ✅ セクション間に関連性を持たせる
- ❌ 深すぎるネスト（H5以上は避ける）
- ❌ 重複するセクション

### 2. プロジェクト概要を最初に配置

```markdown
## プロジェクト概要
Wandas (**W**aveform **An**alysis **Da**ta **S**tructures) は、
音響信号・波形解析に特化したPythonライブラリです。
pandasライクなAPIで信号処理、スペクトル解析、可視化を提供します。
```

**目的**:

- Copilotにプロジェクトの目的とドメインを理解させる
- コンテキストに応じた適切なコード生成を促進
- 新しい開発者がプロジェクトを理解しやすくなる

### 3. 設計原則を明示する

設計原則は2つのカテゴリに分類することを推奨：

#### ドメイン固有の原則

プロジェクト特有の設計方針：

```markdown
### ドメイン固有の原則

1. **Pandasライクなインターフェース**: ユーザーがpandasの操作感で信号処理できるようにする
2. **型安全性**: mypyの厳格モードに準拠し、実行時エラーを防ぐ
3. **チェインメソッド**: メソッドチェーンで複数の処理を直感的に記述できるようにする
4. **遅延評価**: Dask配列を活用し、大規模データでもメモリ効率的に処理する
```

#### 普遍的な設計原則

SOLID原則、YAGNI、KISS、DRYなど：

```markdown
### 普遍的な設計原則

#### SOLID原則

1. **Single Responsibility Principle（単一責任の原則）**
   - 各クラス・関数は1つの責任のみを持つ
   - 変更する理由は1つだけ

2. **Open-Closed Principle（開放閉鎖の原則）**
   - 拡張には開いている（新機能を追加できる）
   - 修正には閉じている（既存コードを変更不要）
```

**なぜ両方が必要か**:

- ドメイン固有の原則: プロジェクトの独自性を保つ
- 普遍的な原則: ソフトウェア工学の基本を遵守

### 4. コーディング規約を具体例付きで記載

各規約には以下の要素を含める：

```markdown
### 1. 型ヒントとType Safety
- **すべての関数・メソッドに型ヒントを必須で付与**してください
- mypyの厳格モード (`strict = true`) に準拠してください

\`\`\`python
# 良い例
def process_signal(data: NDArrayReal, sampling_rate: float) -> NDArrayComplex:
    ...

# 悪い例
def process_signal(data, sampling_rate):  # 型ヒントなし
    ...
\`\`\`
```

**重要な要素**:

- ✅ 簡潔な説明
- ✅ 「良い例」と「悪い例」の対比
- ✅ 具体的なコード例
- ✅ 太字で強調すべき点を明示
- ❌ 抽象的な説明だけ
- ❌ コード例なし

### 5. 実装手順を段階的に記載

```markdown
## コード変更時の手順

### 0. 既存の設計ドキュメントを確認
- **変更を始める前に、必ず `docs/design/INDEX.md` を確認**してください

### 1. 変更プランの作成
- **変更プランを記載したMarkdownファイルを作成**してください
- ファイル名: `docs/design/working/plans/PLAN_<機能名>.md`

### 2. 変更プランのレビュー
実装前に以下の観点で**必ず**レビューを実施してください：

#### 設計チェックリスト
- [ ] 設計原則に沿っているか
- [ ] 後方互換性は保たれているか
```

**ポイント**:

- ✅ 番号付きで手順を明示
- ✅ 各ステップに具体的なアクションを記載
- ✅ チェックリスト形式で確認項目を提供
- ✅ 太字で重要な注意点を強調

### 6. ツールとコマンドを記載

```markdown
## ツールとワークフロー

### コード品質チェック
\`\`\`bash
# Ruffでリント・フォーマット
uv run ruff check wandas tests --fix
uv run ruff format wandas tests

# mypyで型チェック
uv run mypy --config-file=pyproject.toml

# テスト実行
uv run pytest

# カバレッジ付きテスト実行
uv run pytest --cov=wandas --cov-report=html --cov-report=term
\`\`\`
```

**目的**:

- 開発者がすぐに使えるコマンドを提供
- ワークフローの標準化
- ツールの使い方を統一

## 記述スタイルガイド

### 言語の選択

**推奨**: プロジェクトの主要言語に合わせる

- 日本語プロジェクト: 日本語で記述
- 国際プロジェクト: 英語で記述
- バイリンガル: 両方を併記（ただし保守コストに注意）

**wandasプロジェクトの例**:

- ガイドライン本体: 日本語
- コード例のdocstring: 英語
- コメント: 必要に応じて日本語と英語を併用

### 強調の使い方

```markdown
# 良い例
- **すべての関数・メソッドに型ヒントを必須で付与**してください
- **変更を始める前に、必ず `docs/design/INDEX.md` を確認**してください

# 悪い例（過度な強調）
- **すべて****の関数****・****メソッド****に****型ヒント****を****必須****で****付与**してください
```

**推奨**:

- ✅ 重要なアクションや原則を太字で強調
- ✅ ファイル名やコマンドは`バッククォート`で囲む
- ❌ 過度な強調（読みづらくなる）
- ❌ 全体を太字にする

### 指示の明確さ

```markdown
# 良い例（明確な指示）
- **すべての関数・メソッドに型ヒントを必須で付与**してください
- **カバレッジ100%を目標**にしてください（最低90%以上）

# 悪い例（曖昧な指示）
- 型ヒントを使ってください
- カバレッジを高くしてください
```

**推奨**:

- ✅ 具体的な基準を示す
- ✅ 「必須」「推奨」「任意」を明確に区別
- ✅ 例外条件を記載
- ❌ 曖昧な表現

## 効果的な例の提供

### 1. コード例の構造

```markdown
### 良い例と悪い例の対比

\`\`\`python
# 良い例: メソッドチェーンが可能
signal = (
    wd.read_wav("audio.wav")
    .normalize()
    .low_pass_filter(cutoff=1000)
    .resample(target_rate=16000)
)

# 悪い例: メソッドチェーン不可
signal = wd.read_wav("audio.wav")
normalize(signal)  # 関数として実装
low_pass_filter(signal, cutoff=1000)
\`\`\`
```

**ポイント**:

- ✅ コメントで「良い例」「悪い例」を明示
- ✅ 両方を対比させて提示
- ✅ なぜ良い/悪いかを説明

### 2. 実装パターンの例

```markdown
### エラーハンドリング

\`\`\`python
from pathlib import Path

def read_wav(filepath: Union[str, Path]) -> "ChannelFrame":
    """
    Read WAV file and create ChannelFrame.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the WAV file to read.
    
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file format is invalid or corrupted.
    """
    filepath = Path(filepath)
    
    # 入力検証
    if not filepath.exists():
        raise FileNotFoundError(
            f"WAV file not found: {filepath}\\n"
            f"Please check the file path and try again."
        )
    
    # 処理
    ...
\`\`\`
```

**ポイント**:

- ✅ 完全な実装例を提供
- ✅ docstringを含める
- ✅ エラーハンドリングのパターンを示す
- ✅ コメントで意図を説明

### 3. テストパターンの例

```markdown
### テスト作成の具体例

\`\`\`python
# tests/processing/test_new_feature.py

def test_new_feature_normal_case():
    """正常系: 基本的な動作を確認"""
    ...

def test_new_feature_with_edge_values():
    """境界値: 最小値・最大値での動作を確認"""
    ...

def test_new_feature_raises_error_on_invalid_input():
    """異常系: 不正な入力でエラーが発生することを確認"""
    ...

def test_new_feature_preserves_metadata():
    """メタデータが保持されることを確認"""
    ...
\`\`\`
```

**ポイント**:

- ✅ 複数のテストパターンを示す
- ✅ テスト名から意図が明確にわかる
- ✅ 日本語コメントで説明を追加

### 4. チェックリストの活用

```markdown
### 最終確認

#### 品質チェックリスト
- [ ] すべてのテストが通ることを確認（`uv run pytest`）
- [ ] カバレッジレポートで100%達成を確認（`uv run pytest --cov`）
- [ ] 型チェックが通ることを確認（`uv run mypy --config-file=pyproject.toml`）
- [ ] リントが通ることを確認（`uv run ruff check wandas tests`）
```

**ポイント**:

- ✅ マークダウンのチェックリスト構文を使用
- ✅ 確認すべき項目を明確にリスト化
- ✅ コマンドも併記

## 保守とメンテナンス

### 1. 定期的な見直し

**推奨頻度**: 3-6ヶ月ごと

**確認項目**:

```markdown
## 定期レビューチェックリスト

### 内容の妥当性
- [ ] 古くなった情報はないか
- [ ] 新しいツールやベストプラクティスを反映しているか
- [ ] プロジェクトの現状に合っているか

### 構成の改善
- [ ] 読みにくいセクションはないか
- [ ] 重複する内容はないか
- [ ] 追加すべき情報はないか

### 例の更新
- [ ] コード例は最新のAPIに対応しているか
- [ ] コマンド例は正しく動作するか
```

### 2. バージョン管理

ファイルの先頭に最終更新日を記載：

```markdown
# Wandas プロジェクト開発ガイドライン

**最終更新**: 2025年11月2日

## 変更履歴

- 2025年11月2日: 数値検証の原則を追加
- 2025年10月18日: ドキュメントライフサイクルルールを追加
- 2025年10月1日: 初版作成
```

### 3. チーム内での共有

```markdown
## このガイドラインについて

このガイドラインは進化し続けます。改善提案がある場合は：
1. Issueで議論を開始
2. このガイドライン自体の変更プラン（`working/plans/PLAN_update_guidelines.md`）を作成
3. プルリクエストを送信
```

**ポイント**:

- ✅ フィードバックプロセスを明確にする
- ✅ 継続的な改善を促す
- ✅ チームメンバーの貢献を歓迎

### 4. ファイルサイズの管理

**警告サイン**:

- ファイルが2000行を超える
- 1つのセクションが500行を超える
- 重複する内容が多い

**対策**:

1. **モジュール化**: 関連トピックごとに分離
2. **外部リンク**: 詳細は別ドキュメントへのリンク
3. **要約の活用**: 冗長な説明を削減

```markdown
# 良い例（要約 + リンク）
### テスト戦略
- カバレッジ100%を目標
- 詳細は [テスト戦略ガイド](docs/design/guides/testing-strategy.md) を参照

# 悪い例（すべてを含める）
### テスト戦略
（500行の詳細な説明）
```

## よくある落とし穴

### 1. 過度に詳細すぎる

❌ **悪い例**: すべての関数の実装方法を記載

```markdown
### array_sum関数の実装
\`\`\`python
def array_sum(arr: np.ndarray) -> float:
    """配列の合計を計算"""
    return np.sum(arr)
\`\`\`

### array_mean関数の実装
\`\`\`python
def array_mean(arr: np.ndarray) -> float:
    """配列の平均を計算"""
    return np.mean(arr)
\`\`\`
（以下、すべての関数を列挙）
```

✅ **良い例**: パターンを示す

```markdown
### NumPy関数の型ヒント

すべてのNumPy関数は型ヒントを付与：

\`\`\`python
from wandas.utils.types import NDArrayReal

def array_operation(arr: NDArrayReal) -> float:
    """
    Perform operation on array.
    
    Parameters
    ----------
    arr : NDArrayReal
        Input array.
    
    Returns
    -------
    float
        Result value.
    """
    return np.some_operation(arr)
\`\`\`
```

### 2. 曖昧な指示

❌ **悪い例**:

```markdown
- コードは綺麗に書いてください
- パフォーマンスに気をつけてください
```

✅ **良い例**:

```markdown
- **すべての関数に型ヒントを必須で付与**してください
- **大規模データ処理ではDaskの遅延評価を活用**してください
```

### 3. 例がない

❌ **悪い例**:

```markdown
### メソッドチェーン
メソッドチェーンを使ってください。
```

✅ **良い例**:

```markdown
### メソッドチェーン
\`\`\`python
# 良い例: メソッドチェーンが可能
signal = (
    wd.read_wav("audio.wav")
    .normalize()
    .low_pass_filter(cutoff=1000)
)
\`\`\`
```

### 4. 更新されていない情報

❌ **悪い例**:

```markdown
### テスト実行
\`\`\`bash
python -m pytest  # 古いコマンド
\`\`\`
```

✅ **良い例**:

```markdown
### テスト実行
\`\`\`bash
# 最新のプロジェクト設定
uv run pytest
\`\`\`

**最終確認**: 2025年11月2日
```

### 5. 一貫性のない用語

❌ **悪い例**:

```markdown
- dataframe を使う
- DataFrame を作成
- data frame に変換
```

✅ **良い例**:

```markdown
- ChannelFrame を使う（プロジェクトの型名に統一）
```

## 参考資料

### 公式ドキュメント

- [GitHub Copilot Documentation](https://docs.github.com/en/copilot)
- [GitHub Copilot Custom Instructions](https://docs.github.com/en/copilot/customizing-copilot/adding-custom-instructions-for-github-copilot)

### 関連ガイド

- [Wandas 設計ドキュメント一覧](../INDEX.md)
- [ドキュメントライフサイクル管理](../DOCUMENT_LIFECYCLE.md)
- [API改善パターン](./api-improvements.md)
- [メタデータカプセル化](./metadata-encapsulation.md)

### 実例

- [Wandas copilot-instructions.md](../../../.github/copilot-instructions.md) - 970行の包括的な例

### ベストプラクティス記事

- [Effective Documentation Practices](https://www.writethedocs.org/guide/writing/style-guides/)
- [Google Developer Documentation Style Guide](https://developers.google.com/style)

## まとめ

### 重要なポイント

1. **明確な構造**: 階層的なセクション構成
2. **具体的な例**: 良い例と悪い例の対比
3. **実践的な手順**: ステップバイステップのガイド
4. **定期的な更新**: 3-6ヶ月ごとの見直し
5. **適切なサイズ**: 500-1500行が理想

### チェックリスト

新しいカスタム命令を作成する際のチェックリスト：

```markdown
- [ ] プロジェクト概要を含めているか
- [ ] ドメイン固有の設計原則を記載しているか
- [ ] 普遍的な設計原則（SOLID等）を含めているか
- [ ] 具体的なコード例を提供しているか
- [ ] 良い例と悪い例を対比しているか
- [ ] 実装手順を段階的に記載しているか
- [ ] ツールとコマンドを記載しているか
- [ ] 最終更新日を記載しているか
- [ ] YAMLフロントマターで適用範囲を限定しているか（必要な場合）
- [ ] 500-1500行の範囲内に収まっているか
```

### 次のステップ

1. 既存の `copilot-instructions.md` をレビュー
2. 不足している情報を追加
3. 定期的な更新スケジュールを設定
4. チームでフィードバックを収集
5. 継続的な改善を実施

---

**Note**: このガイドは、wandasプロジェクトの経験に基づいて作成されています。プロジェクトの特性に応じて適宜調整してください。
