# Learning Path の日英対応 PoC

## 現在の構成

Learning Path は `learning-path/*.py` の marimo アプリである。MkDocs は
`docs/src` を `docs/site` へビルドした後、デプロイワークフローが各アプリを
`marimo export html` で一度ずつ `docs/site/learning-path/` へ出力している。
したがって、現在の日本語ページのURLは次のとおりである。

```text
/learning-path/01_getting_started.html
```

`mkdocs-static-i18n` は依存関係に存在するが、MkDocs設定では有効化されていない。
また、Learning Path のHTMLはMkDocsのMarkdownページではない。

このPoCの対象は、追跡対象の9教材のうち次の2教材だけである。

- `learning-path/01_getting_started.py`
- `learning-path/06_reusable_pipeline_recipes.py`

他の教材は日本語版だけをマニフェストに登録し、内容を変更していない。

## 要件と採用案

実行コードは各教材に1つだけ残す。説明文、見出し、実行結果、図タイトル、表、
リンクラベルは `learning-path/translations/*.json` のカタログから取得し、教材は
`-- --locale ja` または `-- --locale en` で選択されたカタログを使う。

`learning-path/manifest.json` が教材ID、ソース、利用可能locale、前後の教材、PoC
対象を一元管理する。`scripts/export_learning_path.py` はこのマニフェストだけを
読み、ローカル、CI、デプロイで同じexport経路を使う。

教材のタイトルのような翻訳対象の値は、manifestへ重複して書かず、同じlesson IDの
catalogに置く。これにより、教材の順序・URL・利用可能localeはmanifest、日英の文章は
catalogという一つずつの責務になり、タイトルの修正箇所も増やさない。

採用した構成は方式C（1つのmarimoソース + locale別カタログ）に、薄いマニフェストと
export/検証スクリプトを組み合わせたものだ。実行コードを複製せず、翻訳者はPython
ソースを編集せずにカタログを更新できる。カタログにキーを追加・削除したときは、
ソース参照との不一致をCIが検出する。

## 方式の比較

| 方式 | コード重複 | 翻訳漏れ/コード不一致 | marimoセル依存 | 静的export | 主な故障モード |
| --- | --- | --- | --- | --- | --- |
| A. `.ja.py` と `.en.py` を複製 | 大きい | 検査は可能だが差分が二重になる | 2ファイルを別々に維持 | 相性はよい | API変更、セル追加、出力修正の片側漏れ |
| B. 1ページに日英を併記 | なし | キー管理が不要でも翻訳漏れを機械検出しにくい | 1つだが教材の表示量が倍になる | 相性はよい | 学習者が読む順序と表示言語が曖昧 |
| C. 1ソース + カタログ | なし | キー、locale、placeholder、export HTMLを検査できる | localeを読むセルを依存グラフに追加 | 1ソースをlocaleごとに実行できる | カタログキーの誤り、リンク基準位置の誤り |
| D. gettext/生成ソース/MkDocsプラグイン | 方式次第 | gettextは成熟、生成ソースは差分検査が必要 | 生成後のmarimoソースが別物になり得る | MkDocs用はmarimoHTMLに直接適用されない | 生成物の同期、依存増、ビルド境界の不一致 |

Aは短期的には単純だが、今回の目的である「APIや教材コードを1か所だけ直す」と
反する。Bは初心者にとって常に二言語が見えるため、教材の焦点とページ長が悪化する。
Dのgettextは通常のMarkdownやPythonメッセージには有効だが、marimoセルの実行結果、
Matplotlibの引数、静的export後のリンクを一つの契約として扱うには追加の生成規約が
必要である。`mkdocs-static-i18n` もMkDocs管理下のMarkdownを対象にするため、今回の
追加exportを置き換えない。

残りの比較軸は次のように整理した。API変更時の修正箇所はAだけが日英2ソースに分かれ、
B/C/Dは実行コードを共有できる。ただしDは生成前後の同期契約が別に必要になる。翻訳
レビューは、Aがコード差分に埋もれ、Bが同じページ内で長くなり、Cは同じキーのja/enを
隣接行でレビューできる。DはPOなどの翻訳ツールを使える一方、生成されたmarimoソース
までレビュー対象になる。ローカル実行はA/Bが単一export、Cがlocale別export、Dが
カタログ生成とexportの複数段階になる。PoCのCI時間はBが最小だが常に2言語を表示し、
A/Cは対象lessonを2回実行する。Cは標準ライブラリだけで段階導入でき、Dは形式に応じた
追加依存または生成ツールが必要になる。3言語以上ではCのcatalog locale追加とmanifest
登録で拡張できるが、Aはソース複製数が増える。既存URL維持と未翻訳fallbackをmanifest
から一緒に検査できる点もCの利点である。

## 翻訳カタログ形式

PoCでは、1つのJSON内で各キーの `ja` と `en` を行配列として隣接させる形式を採用した。

```json
{
  "messages": {
    "title": {
      "ja": ["01 環境構築とウォームアップ"],
      "en": ["01 Getting Started with Wandas"]
    }
  }
}
```

長いMarkdownをJSONの巨大なエスケープ文字列にせず、行配列にすることで、空行、
コードフェンス、表、リンクを通常のレビュー差分として扱える。読み込みは標準
ライブラリの `json` だけで、Python 3.10以上の通常実行に依存を追加しない。

比較した形式の判断は次のとおりである。

- Pythonモジュール内の辞書: 実装は容易だが、翻訳者がPythonを編集することになるため不採用。
- JSON: 標準ライブラリ、locale/keyの集合比較、placeholder検査に適するため採用。
- YAML: 長文は読みやすいが、教材実行時の追加パーサ依存と暗黙型の問題があるため不採用。
- TOML: 表現力は十分だが、Python 3.10では `tomllib` がなく、追加依存が必要なため不採用。
- gettext PO: 翻訳レビューと既存gettextツールは強いが、Python標準ライブラリだけではPOを
  直接読むAPIがなく、Markdown/リンク/教材単位の契約を別途実装するため今回は不採用。
- Markdown断片: 長文の記述性は最もよいが、localeとキーの対応を独自規約で解析する必要があり、
  日英を同じキーの下でレビューしにくいため不採用。

カタログ値内の `{name}` は単純なnamed placeholderだけ許可する。CIは日英のplaceholder
集合、空の値、未知locale、Markdownコードフェンス、内部リンク形式を検証する。

## locale の受け渡しとmarimoの仕様

使用中の依存関係では `pyproject.toml` が `marimo>=0.23.3`、`uv.lock` が `0.23.9`
を固定している。ローカルの `marimo export html --help` と公式ドキュメントを確認した
結果、exportサブコマンドはファイル名の後の `--` 以降を教材のコマンドライン引数へ渡す。

```bash
uv run --no-sync marimo export html \
  learning-path/01_getting_started.py \
  -o /tmp/01.html -f -- --locale en
```

教材側は標準ライブラリの `argparse.parse_known_args()` で `--locale` を読み、通常の
`marimo edit` や既存のオフラインテストが持つ他の引数は無視する。未知のlocaleは
`argparse` が拒否する。これはmarimo内部APIやソース文字列書換えに依存しない。

marimo自身のlocale設定は、公式ガイドによれば日付・時刻、数値、相対時刻の表示形式を
主に変更し、本文、UIテキスト、エラーメッセージを翻訳しない。教材本文の切り替えには
使わない。

静的exportの実測では `<html lang="en">` が生成された。使用中の0.23.9のexport CLIに
`lang`指定はなく、教材ごとのHTML全体の`lang`を設定する公開APIも確認できなかった。
`html_head_file` のようなrun-mode設定やexport後のHTML文字列置換には依存しない。この
PoCではブラウザUIの既定langを変更しないことを制限として受け入れる。将来marimoが
静的exportの公開設定としてlangを提供した場合だけ再評価する。

参照した公式資料:

- [Run notebooks as scripts](https://docs.marimo.io/guides/scripts/)
- [Static HTML](https://docs.marimo.io/guides/exporting/static_html/)
- [marimo i18n](https://docs.marimo.io/guides/configuration/internationalization/)

## URLと未翻訳教材

日本語は既存URLを維持する。

```text
/learning-path/01_getting_started.html
/learning-path/06_reusable_pipeline_recipes.html
```

英語は同じソースをlocaleだけ変えて次へ出力する。

```text
/en/learning-path/01_getting_started.html
/en/learning-path/06_reusable_pipeline_recipes.html
```

リンクはサイトルートではなく相対URLにして、GitHub Pagesの `/wandas/` 配下でも動く
ようにする。日英対応ページには `日本語 | English` の静的リンクを置く。
本文のlocale切り替えはexport時に完了させ、ページは静的HTMLとして配信する。カスタム
JavaScript、サーバー実行、Pyodide、marimo WASM、CookieやLocal Storageは必要としない。

未翻訳教材の方針は「英語ページから日本語版へフォールバック」である。例えば英語の
01ページの前後リンクは、まだ英語化されていない00/02へ `Japanese only` と表示して
日本語URLへ向ける。英語ページが存在しない教材への `/en/` リンクは作らない。日本語
ページでは従来どおり日本語の前後リンクを表示する。

06の英語本文からMkDocsのhow-to/APIへ移るリンクも、現時点で英語MkDocsページがない
ため、日本語の既存ページへ相対リンクする。MkDocs本体の英語化後に、同じリンク生成
ヘルパーで英語ページへ切り替える。

## CI契約

`tests/docs/test_learning_path_i18n.py` と `scripts/validate_learning_path_i18n.py`
は次を検査する。

- マニフェストのlesson/source/locale/前後リンクが妥当である。
- 01/06の全参照キーに `ja` と `en` があり、値が空でない。
- 未知locale、未知キー、参照されないキーを拒否する。
- 日英のplaceholder集合が一致する。
- Markdownコードフェンスと内部リンク形式が壊れていない。
- 同じmarimoソースを使う01/06の日本語・英語4ページを静的exportする。
- export HTMLに日本語/英語タイトル、言語切り替え、前後リンクがある。
- 未翻訳教材への英語リンクが存在せず、主要なWandasコードマーカーが両localeにある。
- HTMLにPython tracebackや主要なimport errorがない。

既存の `tests/docs/test_learning_path_executable_contracts.py` は全教材の
`marimo check --strict` とオフライン実行を引き続き担当する。PoCのexportは01/06の4回
だけであり、全教材を日英2回実行する契約はまだ追加しない。

## ローカル実行とデプロイ

カタログだけを検証する。

```bash
uv run --no-sync python scripts/validate_learning_path_i18n.py
```

PoCの日本語・英語を同じ経路でexportし、生成HTMLを検証する。

```bash
uv run --no-sync python scripts/export_learning_path.py \
  --poc --output /tmp/wandas-learning-path-site --jobs 2
uv run --no-sync python scripts/validate_learning_path_i18n.py \
  --site /tmp/wandas-learning-path-site
```

個別の確認には次を使える。

```bash
uv run --no-sync python scripts/export_learning_path.py \
  --lesson 01_getting_started --locale en --output /tmp/wandas-site
uv run --no-sync python scripts/export_learning_path.py \
  --lesson 06_reusable_pipeline_recipes --locale ja --output /tmp/wandas-site
```

デプロイワークフローはMkDocsをビルドした後、次の1コマンドで全教材を出力する。

```bash
uv run --no-sync python scripts/export_learning_path.py --all --output docs/site
```

生成HTMLはリポジトリへ追加しない。

## MkDocs本体との関係

今回 `mkdocs-static-i18n` は有効化しない。将来MkDocs本体を多言語化する場合は、例えば
`docs/src/en/` を英語ツリー、既存の `docs/src/` を日本語ツリーとして設定し、プラグイン
が生成する `/en/` の通常ページと、exportスクリプトが生成する `/en/learning-path/` を
同じサイトルートに置く。MkDocsのnavとLearning Pathマニフェストは別の責務として保ち、
`/en/learning-path/` をMarkdownページの変換対象にしない。

Material for MkDocsの言語切り替えは通常ページのlocale URLへリンクできる。Learning Path
のmarimo HTMLには独自の静的リンクがあるため、Materialのヘッダー切り替えをそのまま
marimoページへ注入することはせず、サイト全体の切り替え設計が固まった時点で同じURL
規約に合わせる。

## 全教材への移行手順

1. `manifest.json` に対象教材のlocaleと前後関係を登録する。
2. 教材内の学習者向け文章、print出力、図/表ラベルを `t("key")` に置き換える。
   変数名、Wandas API、単位、URL、数式はそのままにし、見えるコメントは英語または
   言語に依存しない短い説明へ整理する。
3. JSONカタログへ日本語を移し、同じキーの英訳を追加する。未翻訳ならlocaleを`ja`
   のみにして、英語リンクを生成しない。
4. `validate_learning_path_i18n.py`、`marimo check --strict`、該当lessonのexportを
   実行する。
5. 英語カタログのレビュー後、マニフェストのlocaleを`["ja", "en"]`へ変更する。
6. 全教材が揃ったら、デプロイの `--all` はそのまま使い、必要ならCIのPoC対象を全教材
   へ段階的に拡大する。

現在の9教材を日英化した場合、静的ページ数は日本語9ページ + 英語9ページの18ページ
になる。exportは図やデータ処理を実行するため、CIではまず代表教材を必須にし、全教材
の英語exportは翻訳済みマニフェストの割合とCI実行時間を見て並列数・ジョブ分割を決める。
デプロイは公開成果物を一貫させるため全localeをexportする。

## PoCで判明した制限と故障モード

- marimoのUI本文とHTML全体の`lang`は翻訳カタログの対象外である。
- カタログは行配列JSONなので、Markdown専用エディタのプレビューはない。レビューでは
  キー単位の差分と生成HTMLを確認する。
- `--locale` は教材が解釈する引数であり、marimo本文翻訳の機能ではない。exportスクリプト
  が必ず検証済みlocaleを渡す。
- `docs/site` のようなMkDocs出力と同じルートに英語ツリーを作るため、出力先を手書きする
  shell loopへ戻すと日本語/英語の片側漏れが起きる。デプロイはスクリプトだけを呼ぶ。
- カタログのリンク基準位置を間違えると、日英で同じMarkdownでもリンク先が変わる。
  そのため06のhow-to/APIリンクとナビゲーションはlocale別に検証する。
- 依存APIの変更は、共有Pythonソースと既存の全教材オフライン契約で検出する。翻訳内容
  の意味の正しさや表現品質は自動検査できないため、翻訳レビューは別途必要である。

## 採用案を撤回する条件

次のいずれかが現れた場合は、カタログ方式を見直す。

- 教材の大半が動的UIや入力状態による多言語切り替えを必要とし、静的2回exportで表現できない。
- カタログレビューが大規模化し、JSON行配列よりPOや翻訳管理ツールの方が明らかに保守しやすい。
- marimoの公開API変更で、同じソースのlocale別静的exportが安定して再現できなくなる。
- CIのexport時間・メモリが教材数に対して許容できず、代表exportだけでは品質を保証できない。
- MkDocs本体の多言語構成が `/en/learning-path/` を別の生成物で占有し、URL契約を保てない。

この場合でも、Aのソース複製へ戻る前に、共有Pythonロジック + gettext/Markdownカタログ、
またはMkDocsとmarimoのビルド境界を統合する生成方式を比較し直す。
