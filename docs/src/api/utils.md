# Utilities Module / ユーティリティモジュール

The `wandas.utils` module provides various utility functions used in the Wandas library.
`wandas.utils` モジュールは、Wandasライブラリで使用される様々なユーティリティ機能を提供します。

## Frame Dataset / フレームデータセット

Provides dataset utilities for managing multiple data frames.
複数のデータフレームを管理するためのデータセットユーティリティを提供します。

### Overview / 概要

The `FrameDataset` classes enable efficient batch processing of audio files in a folder. Key features include:
`FrameDataset` クラスは、フォルダ内の音声ファイルの効率的なバッチ処理を可能にします。主な機能：

- **Lazy Loading**: Load files only when accessed, reducing memory usage.
  **遅延読み込み**: アクセス時のみファイルを読み込み、メモリ使用量を削減。
- **Transformation Chaining**: Apply multiple processing operations efficiently.
  **変換のチェーン**: 複数の処理操作を効率的に適用。
- **Sampling**: Extract random subsets for testing or analysis.
  **サンプリング**: テストや分析のためにランダムなサブセットを抽出。
- **Summary Metadata**: Inspect dataset configuration, attempted-load counts, and
  whether a lazy transform is present. Dataset processing history is not exposed.
  **概要メタデータ**: Dataset 設定、読み込み試行済み件数、遅延 transform の有無を確認。
  Dataset の処理履歴は公開されません。

### Main Classes / 主なクラス

- **`ChannelFrameDataset`**: For time-domain data (WAV, FLAC, OGG, AIFF, SND, and CSV files).
  **`ChannelFrameDataset`**: 時間領域データ用（WAV、FLAC、OGG、AIFF、SND、CSVファイル）。
- **`SpectrogramFrameDataset`**: For time-frequency domain data (typically created from STFT).
  **`SpectrogramFrameDataset`**: 時間周波数領域データ用（通常はSTFTから作成）。

### Basic Usage / 基本的な使用方法

```python
from wandas.utils.frame_dataset import ChannelFrameDataset

# Create a dataset from a folder
# フォルダからデータセットを作成
dataset = ChannelFrameDataset.from_folder(
    folder_path="path/to/audio/files",
    sampling_rate=16000,  # Optional: resample all files to this rate / オプション: すべてのファイルをこのレートにリサンプリング
    file_extensions=[".wav", ".flac"],  # File types to include / 含めるファイルタイプ
    recursive=True,  # Search subdirectories / サブディレクトリを検索
    lazy_loading=True  # Load files on demand (recommended) / オンデマンドでファイルを読み込む（推奨）
)

# Access individual files
# 個別のファイルにアクセス
first_file = dataset[0]
if first_file is None:
    print("The first file could not be loaded.")
else:
    print(f"File: {first_file.label}")
    print(f"Duration: {first_file.duration}s")

# Get dataset information
# データセット情報を取得
metadata = dataset.get_metadata()
print(f"Total files: {metadata['file_count']}")
print(f"Loaded files: {metadata['loaded_count']}")
```

### Sampling / サンプリング

Extract random subsets of the dataset for testing or analysis:
テストや分析のためにデータセットのランダムなサブセットを抽出：

```python
# Sample by number of files
# ファイル数でサンプリング
sampled = dataset.sample(n=10, seed=42)

# Sample by ratio
# 比率でサンプリング
sampled = dataset.sample(ratio=0.1, seed=42)

# Default for a non-empty dataset:
# max(1, min(10, int(len(dataset) * 0.1))), capped at len(dataset)
# 空でない Dataset のデフォルト:
# max(1, min(10, int(len(dataset) * 0.1))) を len(dataset) 以下に制限
sampled = dataset.sample(seed=42)
```

For an empty dataset, `sample()` returns an empty dataset. An explicit `n` takes
precedence when both `n` and `ratio` are supplied. All sampling modes preserve lazy
Frame loading and cap the result at the number of discovered files.
空の Dataset では `sample()` も空の Dataset を返します。`n` と `ratio` を両方指定した場合は
`n` が優先されます。どの方法でも Frame の遅延読み込みは維持され、結果件数は探索済みファイル数を
超えません。

### Metadata-driven file selection / メタデータ駆動のファイル選択

#### Path-derived partitions / パス由来のパーティション

For common partitioned folders, set `path_metadata=True` instead of writing a resolver. Wandas inspects only each discovered relative path, so discovery remains lazy and does not open audio files.
一般的なパーティションフォルダでは resolver を書かずに `path_metadata=True` を指定できます。Wandas は探索した相対パスだけを調べるため、探索は遅延性を保ち、音声ファイルを開きません。

```python
dataset = wd.from_folder(
    "recordings/",
    recursive=True,
    path_metadata=True,
)
selected = dataset.select(partition_0="group_a", partition_1="batch_01")
```

The naming follows AWS Glue crawler conventions:
命名は AWS Glue crawler の規則に従います。

- Plain parent segments use their zero-based path positions: `group_a/batch_01/file.wav` becomes `{"partition_0": "group_a", "partition_1": "batch_01"}`. / 通常の親セグメントにはゼロ始まりのパス位置を使います。
- Hive-style segments use their keys: `group=group_a/batch=batch_01/file.wav` becomes `{"group": "group_a", "batch": "batch_01"}`. / Hive 形式のセグメントにはそのキーを使います。
- The root folder and filename are excluded. Files at uneven depths receive only the keys present in their own parent path; a missing key does not match a `select()` criterion. / ルートフォルダとファイル名は除外します。深さが異なる場合は各ファイルに存在するキーだけを持ち、不足キーは `select()` 条件に一致しません。
- Duplicate Hive keys, the reserved Hive key `_source_file`, and Hive keys in the generated `partition_<number>` namespace raise `ValueError` instead of overwriting or impersonating metadata. / Hive キーの重複、予約済み Hive キー `_source_file`、生成用 `partition_<number>` 名前空間の Hive キーは、メタデータの上書きや偽装をせず `ValueError` になります。

`path_metadata=True` and `metadata_resolver` are mutually exclusive because combining two metadata sources would make precedence ambiguous.
2つのメタデータ源の優先順位が曖昧になるため、`path_metadata=True` と `metadata_resolver` は同時に指定できません。

#### Custom metadata resolver / カスタムメタデータ resolver

`metadata_resolver` receives each path relative to `folder_path` once during file discovery. This makes it possible to select files without reading audio headers or waveform samples. Multiple `select()` criteria use exact-match AND semantics.
`metadata_resolver` はファイル探索時に、`folder_path` からの相対パスを各ファイルにつき一度受け取ります。音声ヘッダーや波形サンプルを読まずにファイルを選択でき、複数の `select()` 条件は完全一致の AND として扱われます。

Use a custom resolver when metadata comes from a project-specific filename rule rather than parent folders. For ordinary partition folders, prefer `path_metadata=True` above.
親フォルダではなくプロジェクト固有のファイル名規則からメタデータを得る場合に custom resolver を使います。通常のパーティションフォルダでは、上記の `path_metadata=True` を優先してください。

```python
from pathlib import Path
import wandas as wd

def resolve_recording_metadata(path: Path):
    filename = path.name
    recording_id, status = filename.removesuffix(".wav").split("__")
    return {
        "recording_id": recording_id,
        "status": status,
    }

dataset = wd.from_folder(
    "recordings/",
    recursive=True,
    file_extensions=[".wav"],
    metadata_resolver=resolve_recording_metadata,
)
selected = dataset.select(status="approved")
```

Sidecar CSV data remains application-owned in v1. Convert it to a lookup and reference that lookup from the same resolver contract. Specify WAV explicitly because reading a signal CSV currently materializes the whole CSV during header inspection.
v1 では sidecar CSV は利用側で管理します。lookup に変換し、同じ resolver 契約から参照してください。信号 CSV のヘッダー確認では現在 CSV 全体を実体化するため、WAV を明示的に指定します。

```python
import pandas as pd

recordings = pd.read_csv("recordings.csv")
lookup = recordings.set_index("path")[["condition", "priority"]].to_dict(orient="index")

dataset = wd.from_folder(
    "recordings/",
    recursive=True,
    file_extensions=[".wav"],
    metadata_resolver=lambda path: lookup[path.as_posix()],
)
selected = dataset.select(condition="reference", priority=1)
```

Indexing the lookup is recommended because a WAV missing from the CSV fails during dataset construction. Use `lookup.get(path.as_posix(), {})` only when files without CSV metadata are intentionally allowed.
CSVにないWAVをDataset構築時のエラーにできるため、lookupの添字アクセスを推奨します。CSVメタデータのないファイルも意図的に許可する場合だけ `lookup.get(path.as_posix(), {})` を使用します。

#### Loading stages / 読み込みの3段階

1. `from_folder()`: discover paths and resolve metadata; no audio headers or waveform samples are read. / パス探索とメタデータ解決。音声ヘッダーと波形は未読。
2. `dataset[i]`: inspect the audio header and create a Frame; waveform samples remain Dask-lazy. / 音声ヘッダーを確認してFrameを作成。波形はDask遅延のまま。
3. `frame.data`: materialize waveform samples. / 波形サンプルを実体化。

With `lazy_loading=False`, stage 2 runs for every file during construction, while stage 3 remains lazy.
`lazy_loading=False` では構築時に全ファイルの段階2を実行しますが、段階3は引き続き遅延します。

If stage 2 cannot load a file, or a lazy dataset transform raises or returns `None`,
integer access returns `None`. The result is cached as attempted, so later access does
not retry it automatically; exceptions are also logged. String lookup returns only
successfully loaded matches. Check an integer result explicitly before accessing Frame
properties.
段階2でファイルを読み込めない場合、または遅延 Dataset transform が例外を送出するか `None` を
返した場合、整数アクセスは `None` を返します。結果は試行済みとしてキャッシュされるため、
以後のアクセスでは自動再試行しません。例外はログにも記録されます。文字列検索は読み込みに
成功した一致だけを返します。Frame のプロパティを使う前に整数アクセスの結果を明示的に
確認してください。

#### Resolver and selection contracts / resolverと選択の契約

- The resolver receives a root-relative `Path` once per discovered file and must return a `Mapping[str, object]`. Wandas deep-copies the result. / resolverは探索した各ファイルにつき一度、ルート相対の`Path`を受け取り、`Mapping[str, object]`を返します。結果はディープコピーされます。
- Resolver exceptions, non-mapping results, non-string keys, and the reserved `_source_file` key fail during dataset construction with the affected path. / resolver例外、Mapping以外の戻り値、文字列以外のキー、予約キー`_source_file`は対象パスを含む構築時エラーになります。
- Multiple `select()` criteria use exact-match AND semantics. `select()` with no criteria returns an independent all-file view. / 複数条件は完全一致のANDです。条件なしの`select()`は全ファイルを保持する独立ビューです。
- An unknown key raises `KeyError`; no matches produce a valid empty Dataset. / 未知キーは`KeyError`、一致なしは有効な空Datasetです。

### Transformations / 変換

Apply processing operations to all files in the dataset:
データセット内のすべてのファイルに処理操作を適用：

```python
# Built-in transformations
# 組み込みの変換
resampled = dataset.resample(target_sr=8000)
trimmed = dataset.trim(start=0.5, end=2.0)

# Chain multiple transformations
# 複数の変換をチェーン
processed = (
    dataset
    .resample(target_sr=8000)
    .trim(start=0.5, end=2.0)
)

# Custom transformation
# カスタム変換
def custom_filter(frame):
    return frame.low_pass_filter(cutoff=1000)

filtered = dataset.apply(custom_filter)
```

`apply()`, `resample()`, `trim()`, and `normalize()` create lazy datasets of the
same runtime dataset subtype and do not mutate the source dataset. `stft()` is the
intentional domain transition: it returns `SpectrogramFrameDataset`. File metadata
resolved during discovery is deep-copied into derived datasets and attached to each
successful transformed Frame. Transform/load failures remain per-item `None` values;
they do not abort other items.
`apply()`、`resample()`、`trim()`、`normalize()` は元の Dataset を変更せず、同じ実行時
Dataset subtype の遅延 Dataset を作ります。`stft()` は意図的な domain 遷移であり、
`SpectrogramFrameDataset` を返します。探索時に解決したファイル metadata は派生 Dataset に
ディープコピーされ、変換に成功した各 Frame に付与されます。transform/load の失敗は項目ごとの
`None` として表され、他の項目の処理を中断しません。

### STFT - Spectrogram Generation / STFT - スペクトログラム生成

Convert time-domain data to spectrograms:
時間領域データをスペクトログラムに変換：

```python
# Create spectrogram dataset
# スペクトログラムデータセットを作成
spec_dataset = dataset.stft(
    n_fft=2048,
    hop_length=512,
    window="hann"
)

# Access a spectrogram
# スペクトログラムにアクセス
spec_frame = spec_dataset[0]
if spec_frame is not None:
    spec_frame.plot()
```

### Iteration / 反復処理

Process all files in the dataset:
データセット内のすべてのファイルを処理：

```python
for i in range(len(dataset)):
    frame = dataset[i]
    if frame is not None:
        # Process the frame
        # フレームを処理
        print(f"Processing {frame.label}...")
```

### Key Parameters / 主なパラメータ

**folder_path** (str): Path to the folder containing audio files.
音声ファイルを含むフォルダへのパス。

**sampling_rate** (Optional[int]): Target sampling rate. Files will be resampled if different from this rate.
ターゲットサンプリングレート。このレートと異なる場合、ファイルはリサンプリングされます。

**file_extensions** (Optional[list[str]]): List of file extensions to include. By default, all registered reader extensions from `wd.supported_formats()` are used: `[".aif", ".aiff", ".csv", ".flac", ".ogg", ".snd", ".wav"]`. Audio format availability follows the libsndfile library bundled or installed with SoundFile. MP3 is not a registered Wandas reader format.
含めるファイル拡張子のリスト。デフォルトでは、`wd.supported_formats()` が返す登録済み reader 拡張子 `[".aif", ".aiff", ".csv", ".flac", ".ogg", ".snd", ".wav"]` をすべて使用します。音声形式の利用可否は SoundFile に同梱またはインストールされた libsndfile に従います。MP3 は Wandas の登録済み reader 形式ではありません。

**lazy_loading** (bool): If True, files are loaded only when accessed. Default: True.
Trueの場合、ファイルはアクセス時にのみ読み込まれます。デフォルト: True。

**recursive** (bool): If True, search subdirectories recursively. Default: False.
Trueの場合、サブディレクトリを再帰的に検索します。デフォルト: False。

**path_metadata** (bool): If True, infer AWS Glue-style metadata from relative parent folders. Prefer this option for ordinary partitioned folders. Default: False.
Trueの場合、相対親フォルダからAWS Glue形式のメタデータを推論します。通常のパーティションフォルダではこの方法を優先します。デフォルト: False。

**metadata_resolver** (`Callable[[Path], Mapping[str, object]] | None`): Resolve file metadata from each root-relative path during discovery. Default: None.
探索時に各ルート相対パスからファイルメタデータを解決します。デフォルト: None。

### Unsupported and deprecated methods / 未対応・非推奨メソッド

`FrameDataset.save()` is not a supported persistence API and always raises
`NotImplementedError`. Save individual Frames with their supported Frame methods
instead.
`FrameDataset.save()` は対応済みの永続化 API ではなく、常に `NotImplementedError` を
送出します。代わりに、個々の Frame が対応する保存メソッドを使用してください。

`FrameDataset.get_by_label()` has been deprecated since 0.2.0 because duplicate
filenames make a first-match result ambiguous. Use `get_all_by_label()` (or string
indexing) to receive all successfully loaded matches. It is planned for removal no
earlier than 0.7.0.
重複ファイル名で最初の1件だけを返す動作が曖昧なため、`FrameDataset.get_by_label()` は
0.2.0 から非推奨です。読み込みに成功した一致をすべて得るには `get_all_by_label()`（または
文字列 indexing）を使ってください。削除は早くても 0.7.0 の予定です。

### Examples / 使用例

For detailed examples, see the `learning-path/` directory and the tutorial marimo apps listed in the Tutorial section.
詳細な例については、`learning-path/` ディレクトリとチュートリアル marimo アプリを参照してください。

### API Reference / APIリファレンス

::: wandas.utils.frame_dataset

## Sample Generation / サンプル生成

Provides functions for generating sample data for testing.
テスト用のサンプルデータを生成する機能を提供します。

::: wandas.utils.generate_sample

## Type Definitions / 型定義

Provides type definitions used in Wandas.
Wandasで使用される型定義を提供します。

::: wandas.utils.types

## General Utilities / 一般ユーティリティ

Provides other general utility functions.
その他の一般的なユーティリティ機能を提供します。

::: wandas.utils.util
