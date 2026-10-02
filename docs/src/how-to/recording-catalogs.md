# Use a File List or Recording Catalog

When you already know which recordings to process, pass their paths or a table
directly to a Dataset. You can select by metadata before opening audio files,
then reuse the existing lazy processing methods.

対象の音源が既に分かっている場合は、パス一覧や表をDatasetへ直接渡せます。
音声ファイルを開く前にmetadataで対象を選び、既存の遅延処理を使います。

## Start from selected paths / 選択済みパスから始める

```python
import wandas as wd

dataset = wd.from_files(["root_a/001.wav", "root_b/001.wav"])
frame = dataset[1]
if frame is not None:
    samples = frame.data
```

The collection keeps your order and repeated paths. It does not discover files,
sort, deduplicate, or match by basename. Relative paths are fixed against the
construction-time working directory; use `base_dir="/data"` for an explicit base.
A same-named recording in another folder remains a distinct item.

入力順と重複パスは保持されます。フォルダ探索・ソート・重複除去・basename照合は行いません。
相対パスは構築時の作業ディレクトリを基準に固定し、`base_dir`で基準を指定できます。

Use `from_folder()` when Wandas should discover a folder for you. Its existing
sorting, extension filtering, recursive search, and path metadata rules are unchanged.

## Select by attributes in a table / 表の属性で対象を選ぶ

```python
import pandas as pd
import wandas as wd

table = pd.DataFrame({
    "audio_file": ["fan/001.wav", "fan/001.wav", "pump/001.wav"],
    "condition": ["normal", "changed", "normal"],
    "trial": [1, 2, 3],
})
dataset = wd.from_table(table, path_column="audio_file", base_dir="/data")
selected = dataset.select(condition="changed").trim(0, 5)
frame = selected[0]
if frame is not None:
    trial = frame.metadata["trial"]  # 2, retained through the transform
    samples = frame.data
```

Name the source column explicitly. All other columns become metadata unless you
specify `metadata_columns=["condition", "trial"]`. One row is one observation:
two rows can use the same file with different attributes. DataFrame index values
are not implicit identifiers, and repeated sources do not gain a shared cache.

音源列は`path_column`で明示します。それ以外の列をmetadataにし、`metadata_columns`で
限定できます。同じ音源を使う異なる観測行も保持し、DataFrameのindexを暗黙のIDにしません。

DataFrame basic scalar types are retained and missing values become `None`.
Convert datetimes, complex/object values, or infinities explicitly, or exclude
those columns. Empty strings remain empty strings. Duplicate/non-string column
names and the reserved metadata column `_source_file` are rejected; rename or
exclude the reserved column. Source paths cannot be missing or blank.
The Dataset snapshots metadata; editing the input table or an obtained
`frame.metadata` dictionary does not change its observations.

## Read a CSV catalog / CSVカタログを読む

For a UTF-8 (optionally BOM-prefixed), comma-separated catalog containing paths
and attributes:

```python
dataset = wd.from_table("catalog.csv", path_column="audio_file")
```

CSV audio paths default to the CSV file's parent directory. `base_dir` changes
only the audio path base, not where the CSV itself is opened. DataFrames default
to construction-time cwd; if you load the CSV yourself with pandas, supply its
parent explicitly when that is the intended base.

CSV内の相対音声パスはCSVの親ディレクトリが既定の基準です。`base_dir`は音声パスの
基準だけを変え、CSV自身の場所は変えません。DataFrameは構築時の作業ディレクトリが
既定の基準です。

CSV metadata stays strings, preserving values such as `001` and empty cells.
For typed metadata, prepare a DataFrame with explicit conversions. This catalog
reader differs from `wd.read("signal.csv")`, which reads sampled signal values.
The [Dataset API reference](../api/utils.md) defines the complete input contract.

## Keep sources available until computation / 計算まで音源を保持する

Construction, metadata summaries, `select()`, and `sample()` do not open audio
sources. Reading the catalog itself is separate from reading its audio files.
Item access constructs and caches the Frame, including header inspection; sample
decoding is deferred until materialization such as `frame.data`.

A missing or broken recording stays in the collection. A Frame construction or
transform failure is logged and cached as `None`; access to other items continues.
A failure during later Dask computation raises at that boundary instead. Files
must remain available until computation finishes: returning a Frame does not
mean its file has already been read. Keep any temporary directory alive through
materialization. These APIs do not own, fingerprint, or persist your source files.

音源の欠落・破損で行を削除せず、Frame生成失敗は既存契約のログと`None`で扱います。
後のDask計算中の失敗は計算時に例外になります。Frameを取得しても音声読込が完了したとは
限らないため、テンポラリディレクトリを含め、計算終了まで元ファイルを保持してください。

This release adds local file-list and DataFrame/CSV inputs. URL, bytes, stream,
browser-handle collections, asynchronous loaders, and subset memory optimization
are outside this feature. Individual URL/bytes reads remain available via `read()`.
Collections retain metadata and attempted Frame caches in memory; selecting first
controls decoded audio volume but does not promise constant-memory catalogs.
