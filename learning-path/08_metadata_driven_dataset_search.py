import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    # この教材で使用するライブラリを読み込む
    import pathlib

    import marimo as mo
    import pandas as pd

    import wandas as wd

    return mo, pathlib, pd, wd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # メタデータで必要な音声ファイルだけを選ぶ

    録音ファイルが増えると、「どのファイルを処理するか」を決めるだけでも時間がかかります。
    対象を選ぶために、すべての波形を先に読み込む必要はありません。フォルダ名やファイル名、
    CSVに記録した属性を使えば、必要なファイルだけを絞り込んでから信号処理を始められます。

    `metadata_resolver` は相対パスをファイル単位のメタデータへ変換し、
    `dataset.select()` はそのメタデータを完全一致で検索します。この方法は、測定対象、収録日、
    実験条件、確認状態など、プロジェクトごとに異なる属性へそのまま応用できます。

    3件の短い合成WAVとsidecar CSVからなるデモデータを同梱しているため、
    手元のWAVやCSVは必要ありません。

    このチュートリアルでは次の流れを実行します。

    1. グループと収録単位で分けた同梱サンプルを確認する
    2. パスから `group`、`batch`、`recording_id` を取り出す
    3. ファイルを選択する
    4. Dataset全体に処理チェーンを定義してからファイルを選択する
    5. sidecar CSVをlookupとして利用する
    """)
    return


@app.cell
def _(pathlib):
    # リポジトリに同梱されたデモデータの場所と件数を確認する
    root = pathlib.Path(__file__).parent / "data" / "metadata_search"
    _relative_paths = sorted(path.relative_to(root) for path in root.rglob("*.wav"))

    print(f"同梱データセット: {root}")
    print(f"WAV: {len(_relative_paths)}件")
    return (root,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. パス由来メタデータで選択する

    この例では `group/batch/filename.wav` という単純なフォルダ規則を使います。
    自分のフォルダ名・ファイル名の規則に合わせて、resolverの処理を置き換えられます。

    サンプルのフォルダ構造は次のとおりです。

    ```text
    demo_folder/
    ├── group_a/
    │   ├── batch_01/
    │   │   └── recording_001.wav
    │   └── batch_02/
    │       └── recording_002.wav
    └── group_b/
        └── batch_01/
            └── recording_003.wav
    ```

    resolverへ渡る `Path` はルートフォルダからの相対パスです。resolverは探索時に
    各ファイルにつき一度だけ呼ばれます。ここではファイルを開かず、パス文字列だけを解析します。
    """)
    return


@app.cell
def _(pathlib, root, wd):
    # 相対パスをメタデータへ変換してDatasetを作る
    def resolve_recording_metadata(path: pathlib.Path):
        group, batch, filename = path.parts
        return {
            "group": group,
            "batch": batch,
            "recording_id": filename.removesuffix(".wav"),
        }

    dataset = wd.from_folder(
        str(root),
        recursive=True,
        file_extensions=[".wav"],
        metadata_resolver=resolve_recording_metadata,
    )
    print("見つかったWAVファイル:", len(dataset), "件")
    return (dataset,)


@app.cell
def _(dataset):
    # groupとbatchの条件に一致するファイルだけを選択する
    selected = dataset.select(group="group_a", batch="batch_01")
    print("group_a / batch_01 に一致したファイル:", len(selected), "件")
    return (selected,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `select()` の複数条件はAND、値は完全一致です。未知のキーはタイプミスとして
    `KeyError` になり、一致しない条件は長さ0の有効なDatasetを返します。

    ### 遅延読み込み

    Wandasは、必要になった段階でファイルの内容を読み込みます。処理は次の3段階に分かれます。

    1. `from_folder()` と `select()`：パスとメタデータだけを使って候補を絞る
    2. `selected[0]`：選んだファイルの音声ヘッダーを読み、Frameを作る
    3. `frame.data`：選んだファイルの波形サンプルを実際に読み込む

    `loaded_count` は、これまでにFrameとしてロードしたファイル数です。`select()` の直後は0件で、
    `selected[0]` を実行すると1件になります。つまり、選択だけでは音声ヘッダーや波形を読みません。
    """)
    return


@app.cell
def _(selected):
    # selected[0]の前後でFrameの遅延読み込みを確認する
    print("select()直後にFrameとしてロード済み:", selected.get_metadata()["loaded_count"], "件")
    selected_frame = selected[0]
    assert selected_frame is not None
    print("selected[0]後にFrameとしてロード済み:", selected.get_metadata()["loaded_count"], "件")
    return (selected_frame,)


@app.cell
def _(selected_frame):
    # dataプロパティで波形サンプルを読み込む
    samples = selected_frame.data
    print("dataで読み込んだ波形の先頭5サンプル:", samples[:5])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Datasetに処理をまとめて定義してから選ぶ

    ファイルを選んだ後に、1件ずつ `normalize()` や `stft()` を呼ぶ必要はありません。
    Datasetに対して処理チェーンを定義すると、各ファイルへ同じ処理が自動的に適用されます。
    メタデータは処理後のDatasetにも保持されるため、`normalize().stft()` の後でも
    `select()` を使えます。要素へアクセスするとFrameと処理グラフが作られ、波形の読み込みと
    数値計算は `data` にアクセスするまで遅延されます。
    """)
    return


@app.cell
def _(dataset):
    # Dataset全体に処理を定義した後で、解析対象を選択する
    processed_dataset = dataset.normalize().stft(n_fft=128)
    processed_selected = processed_dataset.select(group="group_a", batch="batch_01")
    print("処理後に選択したファイル:", len(processed_selected), "件")
    return (processed_selected,)


@app.cell
def _(processed_selected):
    # 処理後のFrameにも選択に使ったメタデータが保持されることを確認する
    _selected_spectrogram = processed_selected[0]
    assert _selected_spectrogram is not None

    print("STFT後もgroupを保持:", _selected_spectrogram.metadata["group"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. CSVのメタデータでファイルを選ぶ

    ファイルの属性をCSVで管理している場合も、同じ方法で対象を選べます。
    CSVを `pandas.read_csv()` でDataFrameとして読み、内容を表で表示した後、
    lookup（パスをキーにした辞書）へ変換します。resolverは相対パスをキーにlookupを
    参照するだけです。信号CSVを誤って音声として
    読み込まないよう、`file_extensions=[".wav"]` を明示します。

    この例は `lookup[path.as_posix()]` を使うため、CSVにないWAVがあればDataset構築時に
    エラーになります。入力漏れに早く気づけるので、通常はこちらを推奨します。
    """)
    return


@app.cell
def _(mo, pd, root):
    # pandasで同梱CSVを読み込み、内容を表として表示する
    sidecar_path = root / "recordings.csv"
    recordings = pd.read_csv(sidecar_path)
    mo.vstack([mo.md("**pandasで読み込んだ recordings.csv**"), recordings])
    return (recordings,)


@app.cell
def _(recordings):
    # DataFrameを相対パスから属性を引けるlookupへ変換する
    lookup = recordings.set_index("path")[["condition", "priority"]].to_dict(orient="index")
    return (lookup,)


@app.cell
def _(lookup, root, wd):
    # CSV由来のメタデータを条件にファイルを選択する
    csv_dataset = wd.from_folder(
        str(root),
        recursive=True,
        file_extensions=[".wav"],
        metadata_resolver=lambda path: lookup[path.as_posix()],
    )
    reference_files = csv_dataset.select(condition="reference", priority=1)
    print("CSV条件に一致した件数:", len(reference_files))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## まとめ

    - resolverは相対パスから小さな `Mapping` を返す
    - `select()` は波形を読む前に完全一致で絞り込む
    - Dataset全体に `normalize()`、`stft()` などを定義してから選択できる
    - resolverメタデータはロードしたFrameと変換結果へ伝播する
    - 外部CSVはpandasで内容を確認し、lookupへ変換すれば同じresolver契約へ接続できる

    APIの詳細とエラー契約は
    [Frame Dataset utility reference](../api/utils.md#metadata-driven-file-selection--メタデータ駆動のファイル選択)
    を参照してください。
    """)
    return


if __name__ == "__main__":
    app.run()
