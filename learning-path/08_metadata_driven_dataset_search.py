import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import atexit
    import csv
    import pathlib
    import shutil
    import tempfile

    import marimo as mo
    import numpy as np
    from scipy.io import wavfile

    import wandas as wd

    return atexit, csv, mo, np, pathlib, shutil, tempfile, wavfile, wd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # メタデータで音声ファイルを選ぶ

    大量の録音から「fan の学習データ」「特定の回転数」といった条件で対象を絞るとき、
    波形を先に読む必要はありません。`metadata_resolver` が相対パスをファイル単位の
    メタデータへ変換し、`dataset.select()` が完全一致で選択します。

    ## この教材の開き方

    先に [02_working_with_data.py](02_working_with_data.py) でWAVの読み込みを試しておくと
    理解しやすくなります。リポジトリのルートで次を実行してください。

    ```bash
    uv sync --all-groups
    uv run marimo edit learning-path/08_metadata_driven_dataset_search.py
    ```

    ブラウザで読むだけなら `edit` を `run` に置き換えます。サンプルデータは自動生成されるため、
    手元のWAVやCSVは必要ありません。

    このチュートリアルでは次の流れを実行します。

    1. DCASE/ASDKit風のフォルダを用意する
    2. パスから `machine`、`split`、`section`、`domain` を解決する
    3. ヘッダーや波形を読まずにファイルを選択する
    4. 選択後に通常の Wandas 処理チェーンを適用する
    5. sidecar CSVをlookupとして利用する
    """)
    return


@app.cell
def _(atexit, np, pathlib, shutil, tempfile, wavfile):
    root = pathlib.Path(tempfile.mkdtemp(prefix="wandas_metadata_search_"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    sampling_rate = 8_000
    time = np.arange(512) / sampling_rate
    waveform = np.sin(2 * np.pi * 440 * time).reshape(1, -1)

    relative_paths = [
        pathlib.Path("fan/train/section_00_source.wav"),
        pathlib.Path("fan/test/section_01_target.wav"),
        pathlib.Path("pump/train/section_00_source.wav"),
    ]
    for _relative_path in relative_paths:
        output_path = root / _relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(output_path, sampling_rate, waveform.squeeze().astype(np.float32))

    print(f"デモ用フォルダ: {root}")
    print(f"作成したWAV: {len(relative_paths)}件")
    return relative_paths, root


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. パス由来メタデータで選択する

    この例では `machine/split/filename.wav` というフォルダ規則を使います。
    DCASE/ASDKitを知らなくても問題ありません。自分のフォルダ名・ファイル名の規則に
    `resolve_dcase()` の処理を置き換えられます。

    resolverへ渡る `Path` はルートフォルダからの相対パスです。resolverは探索時に
    各ファイルにつき一度だけ呼ばれます。ここではファイルを開かず、パス文字列だけを解析します。
    """)
    return


@app.cell
def _(pathlib, root, wd):
    def resolve_dcase(path: pathlib.Path):
        machine, split, filename = path.parts
        section, number, domain = filename.removesuffix(".wav").split("_")
        return {
            "machine": machine,
            "split": split,
            "section": f"{section}_{number}",
            "domain": domain,
        }

    dataset = wd.from_folder(
        str(root),
        recursive=True,
        file_extensions=[".wav"],
        metadata_resolver=resolve_dcase,
    )
    fan_train = dataset.select(machine="fan", split="train")

    print("探索直後のロード件数:", dataset.get_metadata()["loaded_count"])
    print("fan/train の件数:", len(fan_train))
    print("選択後のロード件数:", fan_train.get_metadata()["loaded_count"])
    return dataset, fan_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `select()` の複数条件はAND、値は完全一致です。未知のキーはタイプミスとして
    `KeyError` になり、一致しない条件は長さ0の有効なDatasetを返します。

    読み込みは次の3段階です。

    1. `from_folder()`：パスを探索し、resolverでメタデータを作る
    2. `dataset[i]`：サンプリングレートや長さなどの音声ヘッダーを読み、Frameを作る
    3. `frame.compute()`：波形サンプルを実際に読み込む

    したがって、`select()` は1の段階で実行でき、不要なファイルのヘッダーや波形を読みません。
    """)
    return


@app.cell
def _(fan_train):
    print("① select後のFrameロード件数:", fan_train.get_metadata()["loaded_count"])
    selected_frame = fan_train[0]
    assert selected_frame is not None
    print("② dataset[0]後のFrameロード件数:", fan_train.get_metadata()["loaded_count"])

    samples = selected_frame.compute()
    print("③ compute()後の先頭5サンプル:", samples[0, :5])

    processed = fan_train.normalize().stft(n_fft=128)
    spectrogram = processed[0]
    assert spectrogram is not None

    print("Frameへ伝播したメタデータ:", selected_frame.metadata)
    print("STFT後もmachineを保持:", spectrogram.metadata["machine"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. sidecar CSVをlookupとして使う

    v1ではWandasがCSVをjoinするのではなく、利用側で一度lookup（パスをキーにした辞書）へ変換します。
    resolverは相対パスをキーにlookupを参照するだけです。信号CSVを誤って音声として
    読み込まないよう、`file_extensions=[".wav"]` を明示します。

    この例は `lookup[path.as_posix()]` を使うため、CSVにないWAVがあればDataset構築時に
    エラーになります。入力漏れに早く気づけるので、通常はこちらを推奨します。
    """)
    return


@app.cell
def _(csv, relative_paths, root, wd):
    sidecar_path = root / "recordings.csv"
    with sidecar_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "load", "rpm"])
        writer.writeheader()
        for _index, _relative_path in enumerate(relative_paths):
            writer.writerow(
                {
                    "path": _relative_path.as_posix(),
                    "load": "low" if _index == 0 else "high",
                    "rpm": 1_000 + _index * 500,
                }
            )

    with sidecar_path.open(newline="") as stream:
        lookup = {row["path"]: {"load": row["load"], "rpm": int(row["rpm"])} for row in csv.DictReader(stream)}

    csv_dataset = wd.from_folder(
        str(root),
        recursive=True,
        file_extensions=[".wav"],
        metadata_resolver=lambda path: lookup[path.as_posix()],
    )
    low_load = csv_dataset.select(load="low", rpm=1_000)
    print("CSV条件に一致した件数:", len(low_load))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## まとめ

    - resolverは相対パスから小さな `Mapping` を返す
    - `select()` は波形を読む前に完全一致で絞り込む
    - 選択結果でも `sample()`、`apply()`、`normalize()`、`stft()` などを利用できる
    - resolverメタデータはロードしたFrameと変換結果へ伝播する
    - 外部CSVはlookupへ変換すれば同じresolver契約へ接続できる

    APIの詳細とエラー契約は
    [Frame Dataset utility reference](../api/utils.md#metadata-driven-file-selection--メタデータ駆動のファイル選択)
    を参照してください。
    """)
    return


if __name__ == "__main__":
    app.run()
