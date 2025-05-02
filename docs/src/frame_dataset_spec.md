# FrameDatasetの仕様

## 概要

`FrameDataset`は、フォルダ内のファイルを効率的に処理するための抽象基底クラスです。主な特徴として以下の機能を提供します：

- **遅延ロード**: 大規模データセットでも効率的に扱えるよう、必要になった時点でデータをロードします
- **変換チェーン**: データに対する変換を連鎖的に適用できます
- **サンプリング**: データセットから指定数または割合のサンプルを取得できます
- **ファイル出力**: 処理済みのデータを保存できます

## クラス階層

```
FrameDataset[F] (ABC)
├── _SampledFrameDataset[F] (内部クラス)
├── ChannelFrameDataset
└── SpectrogramFrameDataset
```

## クラスの詳細

### FrameDataset[F]

抽象基底クラスで、ジェネリック型パラメータ`F`は扱うフレームの型（`ChannelFrame`または`SpectrogramFrame`）を表します。

#### 主なメソッド

- `__init__`: データセットの初期化
- `_discover_files`: 対象となるファイルを検索
- `_load_all_files`: すべてのファイルをロード（非遅延モード）
- `_load_file`: 具体的なファイルロード処理（抽象メソッド、サブクラスで実装）
- `_ensure_loaded`: 指定インデックスのフレームがロードされているか確認し、必要であればロード
- `__len__`: データセット内のファイル数を返す
- `__getitem__`: 指定インデックスのフレームを取得
- `apply`: 関数をデータセット全体に適用して新しいデータセットを作成
- `save`: 処理済みフレームをファイルに保存
- `sample`: データセットからサンプルを取得
- `get_metadata`: データセットのメタデータを取得

### _SampledFrameDataset[F]

`FrameDataset`から派生した内部クラスで、サンプリングされたデータセットを表現します。元のデータセットへの参照と変換チェーンを保持します。

#### 特徴

- サンプリングされたインデックスマップを保持
- 変換チェーンを適用する洗練された内部実装
- 元のデータセットの構造を保持しつつ、サンプリングされたデータのみを処理

### ChannelFrameDataset

音声・信号ファイルを`ChannelFrame`オブジェクトとして扱うデータセットです。

#### 対応ファイル形式

- `.wav`
- `.mp3`
- `.flac`
- `.csv`

#### 特徴的なメソッド

- `_load_file`: 音声ファイルまたはCSVから`ChannelFrame`を作成
- `resample`: サンプリングレートを変更
- `trim`: 指定した時間範囲でトリミング
- `normalize`: 音声信号を正規化
- `stft`: 短時間フーリエ変換を適用し、`SpectrogramFrameDataset`を作成
- `from_folder`: ファイルからデータセットを作成するクラスメソッド

### SpectrogramFrameDataset

スペクトログラムデータを`SpectrogramFrame`オブジェクトとして扱うデータセットです。主に`ChannelFrameDataset.stft()`の結果として生成されることを想定しています。

#### 特徴的なメソッド

- `_load_file`: 現在は直接ファイルからのロードをサポートしていません
- `plot`: スペクトログラムをプロット

## 使用例

### 基本的な使用方法

```python
# 音声ファイルのデータセットを作成
dataset = ChannelFrameDataset.from_folder(
    "path/to/audio/files",
    sampling_rate=16000,
    lazy_loading=True
)

# 最初の10ファイルを処理
for i in range(min(10, len(dataset))):
    frame = dataset[i]
    # frameを使った処理
    print(f"File {i}: {frame.label}, SR: {frame.sampling_rate}, Duration: {frame.duration}s")
```

### 変換チェーンの例

```python
# 音声ファイルのデータセットを作成
dataset = ChannelFrameDataset.from_folder("path/to/audio/files")

# 変換チェーンを適用
processed_dataset = (
    dataset
    .resample(target_sr=16000)  # リサンプリング
    .trim(start=0.5, end=3.5)   # トリミング
    .normalize()                # 正規化
)

# STFTを適用してスペクトログラムデータセットを作成
spec_dataset = processed_dataset.stft(n_fft=2048, hop_length=512)

# 結果を保存
processed_dataset.save("output/processed_audio")
spec_dataset.save("output/spectrograms")
```

### サンプリングの例

```python
# 元のデータセットからサンプリング
sampled_dataset = dataset.sample(n=10, seed=42)

# サンプリングしたデータセットに変換を適用
processed_sampled = (
    sampled_dataset
    .resample(target_sr=16000)
    .normalize()
)
```

## 高度な機能

### 遅延ロード

大規模なデータセットを効率的に扱うため、`lazy_loading=True`（デフォルト）の場合、データは必要になった時点でロードされます。これにより、メモリ使用量を抑えつつ、大量のファイルを扱うことができます。

### 変換チェーン

`apply()`メソッドを使用して関数をデータセット全体に適用できます。変換は連鎖的に適用でき、各変換は前の変換の結果を入力として使用します。結果は新しいデータセットオブジェクトとして返されます。

### サンプリング

`sample()`メソッドを使用してデータセットから指定数または割合のサンプルを取得できます。サンプリングされたデータセットも元のデータセットと同様に操作でき、変換チェーンも適用できます。
