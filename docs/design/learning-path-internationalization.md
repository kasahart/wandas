# Learning Path の日英対応 PoC

## 現在の構成と対象

Learning Path は `learning-path/[0-9][0-9]_*.py` の marimo アプリである。MkDocs は
`docs/src` を `docs/site` へビルドし、デプロイ時に Learning Path を追加で静的HTMLへ
exportする。日本語の既存URLは次のとおりである。

```text
/learning-path/01_getting_started.html
```

`mkdocs-static-i18n` は通常のMkDocs用依存関係にあるが、Learning PathはMkDocsの
Markdownではないため、このPoCでは有効化しない。対象は次の2教材であり、他の教材の
内容は変更しない。

```text
learning-path/01_getting_started.py
learning-path/06_reusable_pipeline_recipes.py
```

使用中の `pyproject.toml` は `marimo>=0.23.3`、ロック済みの環境は marimo 0.23.9
である。0.23.9の `marimo export html --help` には locale オプションはないが、
`NAME [ARGS]...` と `--` 以降をノートブックへ渡す仕様がある。

## 採用案

方式C「1つのmarimoソース + locale別翻訳カタログ」を採用する。

- Pythonの実行コードは教材ごとに1つだけ持つ。
- 説明、見出し、表、動的な読者向け結果、リンクラベルはカタログから生成する。
- 日本語は既存の `/learning-path/`、英語は `/en/learning-path/` へ出力する。
- localeごとに同じソースを別プロセスで実行し、静的HTMLを生成する。
- 翻訳実装の初期化と動的な翻訳結果は `hide_code=True` のセルに置く。
- 学習者向けセルには翻訳キー、`t()`、locale/catalog/helperを置かない。
- 本文の切り替えにJavaScript、Pyodide、WASM、サーバー、Cookie、Local Storageを使わない。

タイトルは教材カタログに置く。manifestには教材の順序、source、locale、URLを決める
情報だけを置き、同じタイトルを二重管理しない。

## 方式比較

| 方式 | コード/文章の重複 | API変更 | 漏れ検出とレビュー | marimo/静的export | 段階導入・将来性 | 主な故障モード |
| --- | --- | --- | --- | --- | --- | --- |
| A. `.ja.py` と `.en.py` | コードも説明も大きく重複 | 日英2ファイル | 差分が二重。片側のセル追加を検出し続ける必要 | exportは単純 | 未翻訳は単純だが、3言語で複製が増える | API、セル依存、コード修正の片側漏れ |
| B. 1ページ併記 | コードは共有、表示文は同じページに2つ | 1箇所 | 翻訳キーの漏れを機械的に扱いにくく、ページが長い | exportは単純 | locale単位の公開と相性が悪い | 学習者の読む言語が曖昧、表示量が倍 |
| C. 1ソース + カタログ | 実行コードなし、翻訳文はlocaleごと | 1ソース | key/locale/placeholder/HTMLをCIで検査でき、隣接したja/enをレビューできる | 同じソースを2回export。セル依存はhidden setupだけ増える | jaのみ→日英へmanifestを変更して導入 | カタログキー、相対リンク、locale引数の誤り |
| D. gettext/生成ソース/MkDocsプラグイン | 方式次第 | 共有可能 | POツールは強いが、生成ソースとmarimo出力の同期契約が増える | MkDocsプラグインは追加export済みHTMLを直接処理しない | 大規模化には候補 | 生成境界、追加依存、生成物のレビュー漏れ |

Cは、Pythonコード重複、API変更時の修正箇所、静的HTML配信、未翻訳教材の混在、CIでの
不一致検出を同時に満たす。Aは短期的には分かりやすいが、今回の「API変更を1箇所だけ
直す」という条件に反する。Bは表示言語を選べず、Cより翻訳漏れを明確に検出しにくい。
Dはgettext自体を否定しないが、現在の9教材規模では、marimo実行・図・前後リンクまで
含む契約を保つための生成基盤が過剰である。

## manifest、順序、export plan

`learning-path/manifest.json` の lessons 配列がLearning Pathの唯一の順序である。
`previous` と `next` はmanifestに保存せず、配列位置から導出する。これにより、教材の
追加・削除・並べ替えで前後関係を三重管理しない。

`load_manifest()` は次を検査する。

- `git ls-files -z -- learning-path` から得た追跡済みの numbered source集合と、manifestのsource集合が完全一致する。
- ignored/untrackedのscratch `.py` は公開集合に入らない。
- Gitが使えない、または `git ls-files` に失敗した場合は、危険な `Path.glob()` fallbackを使わず明示的に失敗する。
- lesson ID、source、`Path(source).stem`、存在性が一致し、ID/sourceが重複しない。
- 既定localeは `ja` で、各lessonは `ja` を持つ。
- manifestに書かれた `previous`/`next` は拒否する。
- `build_export_plan()` が各localeの出力先を決定し、重複を拒否する。

`scripts/export_learning_path.py --all --dry-run` は、manifestの全日本語ページと、
manifestで `en` を宣言したページだけの英語ページを、決定的な順序で表示する。現在は
日本語9ページ、英語2ページの11項目である。CIはこの全planをdry-runで検査し、重い実行
はPoCの4ページに限定する。

## 翻訳カタログ

カタログは標準ライブラリの `json` だけで読める行配列JSONとする。

```text
learning-path/translations/common.json
learning-path/translations/01_getting_started.json
learning-path/translations/06_reusable_pipeline_recipes.json
```

`common.json` は次のhelper所有キーだけを持ち、教材カタログへ複製しない。

- `language.ja`, `language.en`
- `navigation.heading`, `navigation.previous`, `navigation.next`
- `navigation.japanese_only`

教材固有の title、説明、表、図の解説、動的結果はlesson catalogに置く。commonとlesson
で同名keyがあれば拒否する。commonのキー集合は固定し、未知key・欠落keyも拒否するため、
共通文言の未使用・勝手な追加を放置しない。

YAMLは読みやすいが実行時依存が増え、暗黙型の事故もある。Python辞書は翻訳者がPythonを
編集する。TOMLはPython 3.10の標準ライブラリに `tomllib` がなく、POは標準ライブラリ
だけでは直接読めず、Markdown断片はkey/localeの集合検査を別規約にする必要がある。
そのため今回は採用しない。gettextへの移行は、翻訳量とレビュー運用がJSONを上回った
場合に再評価する。

### placeholder

翻訳値の動的値は `[[name]]` だけを使う。

```text
config = {"sampling_rate": 48000}
\frac{a}{b}
サンプル数: [[samples]]
```

`name` は `[A-Za-z_][A-Za-z0-9_]*` に限定する。`str.format()`、format specifier、
属性アクセス、`{{`/`}}` escapingは使わないので、Markdown、Python辞書、JSON、LaTeXの
literalな `{` と `}` はそのまま書ける。日英のplaceholder集合を比較し、実行時には不足値
と余分な値の両方を lesson ID・key・locale付きで拒否する。

## localeの受け渡し

marimoの公開CLIを使い、exportスクリプトだけが翻訳カタログを持つlessonへlocaleを渡す。

```bash
uv run --no-sync marimo export html \
  learning-path/01_getting_started.py \
  -o docs/site/learning-path/01_getting_started.html \
  -f -- --locale ja

uv run --no-sync marimo export html \
  learning-path/01_getting_started.py \
  -o docs/site/en/learning-path/01_getting_started.html \
  -f -- --locale en
```

実際には `scripts/export_learning_path.py` がこのコマンドを構成する。翻訳カタログを
持たないlegacy教材は、`-- --locale` を付けず、PR前と同じmarimo export引数で出力する。
command constructionは `marimo_export_command()` というpure functionに分離し、テスト
している。教材側は `argparse.parse_known_args()` でlocaleを読み、marimoや他の引数を
private APIなしで共存させる。

export CLIに `--locale` を指定した場合、export planと実際の出力は指定したlocaleの
plan itemだけになる。指定localeを持たない教材へ別localeでフォールバックすることは
なく、選択結果がなければエラーにする。例えば次のコマンドは英語版1件だけを生成する。

```bash
uv run --no-sync python scripts/export_learning_path.py \
  --lesson 01_getting_started \
  --locale en \
  --output /tmp/wandas-site
```

`--locale` を省略した場合は従来どおり各lessonの全available localeを計画し、
`--all --locale en` は現在英語対応している01と06の英語版だけを計画する。

marimoのstatic HTML exportには、0.23.9のCLI上、ページ全体の `lang` をlocale別に設定
する公開引数/APIがない。実測では日本語・英語の両HTMLとも `<html lang="en">` になった。
export後のHTML文字列置換は壊れやすいため導入せず、これは残る制限として記録する。

参照した公式資料:

- [marimo static HTML export](https://docs.marimo.io/guides/exporting/static_html/)
- [marimo internationalization](https://docs.marimo.io/guides/configuration/internationalization/)
- [marimo scripts and CLI arguments](https://docs.marimo.io/guides/scripts/)

## URL、言語切り替え、navigation

出力先は次の固定規約である。

```text
/learning-path/<lesson>.html
/en/learning-path/<lesson>.html
```

リンクは相対URLで生成するため、GitHub Pagesの `/wandas/` サブパスでも動作する。
`language_switch_markdown()` は教材タイトル直下に配置し、対応localeの存在するリンク
だけを `日本語 | English` として表示する。`navigation_markdown()` は末尾にだけ配置し、
前後教材へ移動する。両方ともJavaScriptを使わず、日英ページは相互リンクする。

未翻訳教材へ移動する英語ページでは、存在しない `/en/` URLを作らない。対象が日本語
だけなら、日本語URLへのリンクに `(Japanese only)` を付ける。日本語ページには注記を
付けない。06の英語本文から日本語のMkDocs how-to/APIへ移るリンクも同じ注記を付ける。
リンク先と注記は `docs_reference_links()` に集約しており、将来MkDocsが英語化された
ときにこのhelperを変更する。

01の教材本文のまとめは `まとめ` / `Summary`、末尾の移動見出しは `教材間の移動` /
`Navigate between lessons` とし、「次のステップ」の同名見出しを重複させない。HTML
検証はタイトル直下のswitch、末尾navigation、見出しの重複、fallback注記、リンク先の
存在を文字列の全文snapshotや画素比較なしで確認する。

## learner-visible codeと図

locale/catalog/t/helper/navigationの初期化、静的Markdown、動的な数値・metadata表示、
RecipePlanのassertionはhidden cellに置く。教材を実行するために読者が必要とする
`json`、`numpy`、`wandas`、`wandas.pipeline` などのpublic importはvisible cellに置く。
visible cellは次のように、日英どちらでもそのまま読めるWandas/Pythonコードにする。

- Wandas API、Python変数名、単位は英語のままにする。
- 翻訳keyや `print(t(...))` をvisible codeへ残さない。
- 長いコメントやdocstringは説明用Markdownへ移し、コードから意味が明らかなコメントは削る。
- 06の `np.testing.assert_allclose()` とmetadata/history assertionはhidden verification cellに移し、成功は読みやすい結果として表示する。
- Matplotlibの図内タイトル、legend、軸ラベルへ新しい日本語を渡さない。図内はAPI名、単位、数値、短いASCIIラベルに限定し、説明は図の前後のlocalized `mo.md()` に置く。
- フォントファイル、font package、`japanize-matplotlib` は追加しない。

marimoのexport HTMLはhidden cellのsourceも実行用notebook metadataとして保持する。その
ためHTML全文からi18n文字列を禁止するのではなく、validatorがexport metadataの
`config.hide_code`を読み、visible cellにだけhelperがないことを確認する。source側の
AST検査も併用する。

## CI契約

次の契約を標準ライブラリ中心のvalidatorとテストで維持する。

- tracked numbered lesson集合とmanifest source集合が一致する。
- ID/source stem、source存在、locale、catalog、出力先、配列順navigationが妥当である。
- common/lesson key、ja/en coverage、空値、unknown locale/key、placeholder集合、literal brace、コードフェンス、内部リンクを検査する。
- 01/06のsourceに日英別 `.ja.py`/`.en.py` がなく、visible cellに翻訳実装がない。
- `marimo check --strict`、日本語export、英語export、offline executionでエラーやtracebackがない。
- `--all --dry-run` が全日本語lesson、英語を宣言したlesson、決定的順序だけを計画する。
- CIの実exportは01/06の4ページを `--jobs 2` で行い、生成HTMLのタイトル、switch、navigation、主要Wandasコード、fallback、Japanese-only注記、リンク存在を検証する。

CI jobはdocs変更時に次を実行する。

```bash
uv run --no-sync python scripts/validate_learning_path_i18n.py
uv run --no-sync python scripts/export_learning_path.py --all --dry-run
uv run --no-sync python scripts/export_learning_path.py \
  --poc --output "$RUNNER_TEMP/wandas-learning-path-poc" --jobs 2
uv run --no-sync python scripts/validate_learning_path_i18n.py \
  --site "$RUNNER_TEMP/wandas-learning-path-poc"
```

既存の `tests/docs/test_learning_path_executable_contracts.py` は全教材のoffline
`marimo check --strict`/実行契約を引き続き担当する。全教材を日英2回実行する重い契約は、
翻訳済み教材数と実測CI時間を見て段階的に拡大する。

## ローカルとデプロイ

PoCを確認する。

```bash
uv run --no-sync python scripts/export_learning_path.py \
  --poc --output /tmp/wandas-learning-path-poc --jobs 2
uv run --no-sync python scripts/validate_learning_path_i18n.py \
  --site /tmp/wandas-learning-path-poc
```

全サイトと同じ経路を一時ディレクトリで確認する場合は、MkDocs build後に次を実行する。

```bash
uv run --no-sync python scripts/export_learning_path.py \
  --all --output /tmp/wandas-full-site --jobs 1
uv run --no-sync python scripts/validate_learning_path_i18n.py \
  --site /tmp/wandas-full-site --all
```

デプロイのdefault並列度は `jobs=1` とする。Matplotlib/Dask/音声処理を並列実行して
メモリ使用量や非決定性を増やさず、従来の直列exportを維持する。PoCテストだけは明示的
に2並列を使って並列経路も検査する。対象数0でもThreadPoolExecutorを作らない。

GitHub Pagesへのdeployは次の順序を必須とする。

1. `marimo check` とカタログ検証。
2. MkDocs build。
3. `export_learning_path.py --all --jobs 1`。
4. `validate_learning_path_i18n.py --site docs/site --all`。
5. 検証成功後に `docs/site` を公開。

生成HTMLはGit管理しない。

## MkDocs本体との関係

同一PRでMkDocs全体の多言語化は行わない。将来は既存の日本語 `docs/src/` と英語
`docs/src/en/`（または採用する `mkdocs-static-i18n` の規約）を整理し、通常ページの
英語出力 `/en/` とLearning Pathの `/en/learning-path/` を同じサイトルートへ置く。
Learning Path HTMLをMkDocs Markdownのnavへ重ねず、manifest/export planを別責務として
維持する。Material for MkDocsの言語切り替えは通常ページのlocale URLと統合できるが、
marimoページ内の静的switchは残し、JavaScript注入に依存しない。`/en/learning-path/`を
通常ページの出力が占有しないことを統合時の契約にする。

SUPPORTED_LOCALESは現在 `("ja", "en")` に固定され、英語prefixもこの実装で定義して
いる。したがって「manifestへlocaleを追加するだけで3言語へ拡張できる」とは言わない。
第三言語を追加する場合は、locale検証、URL prefix、locale switch、カタログcoverage、
HTML validator、CI plan、fallbackをコードとテストへ追加し、出力リンクを実測してから
manifestを変更する。日英専用の単純さを、未検証の汎用frameworkより優先する。

## 全教材への移行手順と判断条件

全教材を展開するときは、教材ごとに次を繰り返す。

1. 既存sourceを1つのままにし、visible codeから説明用コメント、長いdocstring、文章printを整理する。
2. 翻訳対象を `t("key")` とhidden outputへ移し、図内の新しい日本語を作らない。
3. lesson catalogへja/enを追加し、common keyを複製しない。値とplaceholderをレビューする。
4. `manifest.json` の `catalog` と `en` を追加し、export plan、source検査、両localeのmarimo checkを通す。
5. HTMLのswitch、navigation、リンク、fallback、Wandas code markerを検証する。
6. 翻訳レビュー後にデプロイの全planへ含める。

manifestのlocaleをjaのみからja/enへ変える前提は、実行コードの意味がlocale間で同じ、
catalog key/placeholderが揃う、静的exportがネットワークなしで完了する、英語リンク先が
存在する、という4点である。翻訳品質・表現の自然さは機械検査だけでは保証できない。

次の場合は、全教材展開を止めて方式を再評価する。

- locale別静的exportでは表現できない動的UIや入力状態の翻訳が主流になる。
- marimoの公開CLIが変わり、同じsourceへ安全に引数を渡せなくなる。
- exportの時間・メモリが、representative CIとjobs=1 deployで許容できない。
- JSONのkey単位レビューがPOや別の翻訳管理方式より明らかに不利になる。
- MkDocs本体が `/en/learning-path/` を占有し、現在の相対URL契約を維持できない。
- HTMLのlang設定が必要条件となり、marimoに公式APIがないまま文字列置換しか選べない。

このPoCで残る制限は、HTML全体の `lang` がlocale別にならないこと、翻訳の意味品質を
自動判定できないこと、第三言語にコード変更が必要なこと、visible/hidden codeのHTML検査
が現在のmarimo export metadata（`notebook.cells[].config.hide_code`）に依存することである。
marimoを更新するときは `marimo export html --help`、`lang`、cell metadataを再確認する。
