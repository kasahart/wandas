import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
async def _():
    import sys

    if sys.platform == "emscripten":
        import micropip

        # pydantic-core has no pure Python wheel on PyPI.
        # Pyodide ships pydantic in its own repo, so install it first
        # to satisfy wandas's dependency without hitting PyPI for pydantic-core.
        await micropip.install("pydantic")
        await micropip.install(
            [
                "wandas",
                "dask",
                "mosqito",
                "soundfile",
            ]
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 02 データの読み込みと操作
    ## 現実のデータをWandasで扱う

    このノートブックでは、実際のデータファイル（WAV, CSVなど）を読み込み、
    Wandasのデータ構造を理解し、基本的な操作を紹介します。

    **学習目標:**
    - さまざまなファイル形式からのデータ読み込み
    - ChannelFrameデータ構造の理解
    - チャンネルアクセスと操作
    - メタデータの管理

    **前提条件:**
    - 01_getting_started.ipynb を完了していること
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🎯 なぜデータ読み込みが重要か

    ### 信号処理ワークフローの第一歩

    **データ読み込み**は、信号処理ワークフローの最も重要な第一歩です。なぜなら：

    1. **現実世界のデータ**を扱う必要がある
    2. **さまざまなフォーマット**に対応する必要がある
    3. **データの品質**を確認する必要がある
    4. **適切な構造**でデータを保持する必要がある

    ### Wandasのデータ読み込みの特徴

    - **統一されたインターフェース**: さまざまなファイル形式を同じ方法で扱える
    - **自動メタデータ抽出**: サンプリングレート、チャンネル情報などを自動取得
    - **柔軟なオプション**: カスタム設定で特殊なフォーマットに対応
    - **エラーハンドリング**: 問題のあるデータを適切に処理
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 📂 サポートされるファイル形式

    ### 主要なフォーマット

    | フォーマット | 用途 | 特徴 |
    |------------|------|------|
    | **WAV** | 音声データ | 複数チャンネル、メタデータ豊富 |
    | **CSV** | 時系列データ | 表形式、柔軟な構造 |
    | **WDF** | Wandas専用 | 完全なメタデータ保存 |
    | **NumPy** | 配列データ | 高速、メモリ効率的 |

    ### ユースケース別の選択

    - **音声収録データ** → WAV
    - **センサーデータ** → CSV
    - **処理済みデータ** → WDF
    - **計算結果** → NumPy
    """)
    return


@app.cell
def _():
    # 必要なライブラリをインポート
    import urllib.request
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    import wandas as wd

    pathlib_path = Path

    # インタラクティブプロット設定
    # '%matplotlib widget' command supported automatically in marimo
    plt.rcParams["figure.figsize"] = (12, 6)

    print(f"Wandas: {wd.__version__}")
    print("✅ 準備完了")
    return np, pathlib_path, plt, urllib, wd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🎵 WAVファイルの読み込み

    ### なぜWAVが重要か

    WAVは**音声・振動データの標準フォーマット**です：
    - **無圧縮**: 高い音質を保証
    - **複数チャンネル**: ステレオやマイクアレイに対応
    - **メタデータ**: サンプリングレート、ビット深度などの情報
    - **広くサポート**: ほとんどの録音ソフトウェアで使用可能
    """)
    return


@app.cell
async def _(pathlib_path, urllib):
    # サンプルWAVファイルをダウンロード
    wav_url = "https://github.com/kasahart/wandas/raw/refs/heads/main/learning-path/sample_audio.wav"
    wav_path = pathlib_path("sample_audio.wav")

    # ダウンロード（既に存在しない場合）
    if not wav_path.exists():
        print("サンプルWAVファイルをダウンロード中...")
        import sys as _sys

        if _sys.platform == "emscripten":
            from pyodide.http import pyfetch as _pyfetch

            _response = await _pyfetch(wav_url)
            with open(str(wav_path), "wb") as _f:
                _f.write(await _response.bytes())
        else:
            urllib.request.urlretrieve(wav_url, wav_path)
        print(f"✅ ダウンロード完了: {wav_path}")
    else:
        print(f"✅ ファイル既に存在: {wav_path}")
    return (wav_path,)


@app.cell
def _(wav_path, wd):
    # WAVファイルを読み込み
    audio = wd.read_wav(str(wav_path))

    print("🎵 WAVファイル読み込み結果:")
    print(f"  ファイル: {wav_path.name}")
    print(f"  チャンネル数: {audio.n_channels}")
    print(f"  サンプリングレート: {audio.sampling_rate} Hz")
    print(f"  長さ: {audio.duration:.2f} 秒")
    print(f"  サンプル数: {audio.n_samples}")
    print(f"  データ型: {audio.data.dtype}")
    print(f"  チャンネル名: {audio.labels}")
    return (audio,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **🎧 聞いてみましょう！**

    WAVファイルの内容を実際に聴くことができます。
    """)
    return


@app.cell
def _(audio):
    # 読み込んだ音声データを可視化
    audio.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 📊 CSVファイルの読み込み

    ### なぜCSVが重要か

    CSVは**センサーデータや計測データの標準フォーマット**です：
    - **表形式**: 時間軸と複数のセンサー値を扱いやすい
    - **汎用性**: Excel, MATLAB, Pythonなど多くのツールで扱える
    - **柔軟性**: カスタムフォーマットに対応可能
    - **テキストベース**: バージョン管理システムで扱いやすい
    """)
    return


@app.cell
async def _(pathlib_path, urllib):
    # サンプルCSVファイルをダウンロード
    csv_path = pathlib_path("sensor_data.csv")
    csv_url = "https://raw.githubusercontent.com/kasahart/wandas/refs/heads/main/learning-path/sensor_data.csv"

    if not csv_path.exists():
        print("サンプルCSVファイルをダウンロード中...")
        import sys as _sys

        if _sys.platform == "emscripten":
            from pyodide.http import pyfetch as _pyfetch

            _response = await _pyfetch(csv_url)
            with open(str(csv_path), "wb") as _f:
                _f.write(await _response.bytes())
        else:
            urllib.request.urlretrieve(csv_url, csv_path)
        print(f"✅ ダウンロード完了: {csv_path}")
    else:
        print(f"✅ ファイル既に存在: {csv_path}")

    import pandas as pd

    df = pd.read_csv(csv_path)
    print(f"✅ サンプルCSVファイル準備完了: {csv_path}")
    print(f"   行数: {len(df)}")
    print(f"   列: {list(df.columns)}")
    return (csv_path,)


@app.cell
def _(csv_path, wd):
    # CSVファイルを読み込み
    sensor_data = wd.read_csv(
        csv_path,
        time_column="time",  # 時間軸の列名
        delimiter=",",  # 区切り文字
    )

    print("📊 CSVファイル読み込み結果:")
    print(f"  ファイル: {csv_path.name}")
    print(f"  チャンネル数: {sensor_data.n_channels}")
    print(f"  サンプリングレート: {sensor_data.sampling_rate:.1f} Hz")
    print(f"  長さ: {sensor_data.duration:.1f} 秒")
    print(f"  チャンネル名: {sensor_data.labels}")
    print(f"  データ形状: {sensor_data.shape}")
    return (sensor_data,)


@app.cell
def _(sensor_data):
    # センサーデータを可視化
    sensor_data.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🔢 NumPy配列からの作成

    ### なぜNumPy配列が重要か

    NumPy配列は**計算結果やシミュレーションデータ**の標準形式です：
    - **高速処理**: ベクトル化された演算
    - **メモリ効率**: 連続したメモリ配置
    - **相互運用**: 他の科学計算ライブラリとの互換性
    - **柔軟性**: 任意の次元数とデータ型
    """)
    return


@app.cell
def _(np, wd):
    # NumPy配列からChannelFrameを作成
    np.random.seed(123)
    _sampling_rate = 1000  # 1kHz
    _duration = 2.0
    _n_samples = int(_duration * _sampling_rate)
    _time = np.linspace(0, _duration, _n_samples)
    # ステレオ音声風のデータを作成
    left_channel = np.sin(2 * np.pi * 440 * _time) + 0.1 * np.random.randn(_n_samples)
    right_channel = np.sin(2 * np.pi * 440 * _time + np.pi / 4) + 0.1 * np.random.randn(_n_samples)
    stereo_data = np.vstack([left_channel, right_channel])
    stereo_audio = wd.from_numpy(data=stereo_data, sampling_rate=_sampling_rate, ch_labels=["Left", "Right"])
    # 2D配列にスタック
    print("🔢 NumPy配列からの作成結果:")
    print(f"  データ形状: {stereo_data.shape} (channels, samples)")
    # ChannelFrameを作成
    print(f"  サンプリングレート: {stereo_audio.sampling_rate} Hz")
    print(f"  チャンネル名: {stereo_audio.labels}")
    print(f"  データ型: {stereo_audio.data.dtype}")
    return (stereo_audio,)


@app.cell
def _(stereo_audio):
    # NumPyデータも可視化
    stereo_audio.plot(title="Stereo Audio from NumPy Array", overlay=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🏗️ ChannelFrameデータ構造の理解

    ### ChannelFrameの特徴

    ChannelFrameはWandasの**基本的なデータ構造**です：

    - **2Dデータ**: チャンネル × サンプルの行列
    - **リッチメタデータ**: サンプリングレート、チャンネル名、単位、処理履歴
    - **pandasライクAPI**: 直感的なアクセス方法
    - **型安全性**: ty対応の厳格な型付け
    - **遅延評価**: Daskとの統合で大規模データ対応
    """)
    return


@app.cell
def _(sensor_data):
    # ChannelFrameの構造を詳しく調べる
    print("🏗️ ChannelFrameの構造分析:")
    print(f"  データ型: {type(sensor_data)}")
    print(f"  データ形状: {sensor_data.shape}")
    print(f"  データ型（NumPy）: {sensor_data.data.dtype}")
    print(f"  メモリ使用量: {sensor_data.data.nbytes / 1024:.1f} KB")
    print()

    # メタデータ情報
    print("📋 メタデータ:")
    print(f"  サンプリングレート: {sensor_data.sampling_rate} Hz")
    print(f"  チャンネル数: {sensor_data.n_channels}")
    print(f"  サンプル数: {sensor_data.n_samples}")
    print(f"  長さ: {sensor_data.duration:.2f} 秒")
    print(f"  チャンネル名: {sensor_data.labels}")
    print(f"  処理履歴: {[op.name for op in sensor_data.operation_history]}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🔍 チャンネルアクセスと操作

    ### pandasライクなアクセス方法

    ChannelFrameは**pandasのDataFrameのようなアクセス**を提供します：
    - **インデックスアクセス**: `cf[0]`, `cf['channel_name']`
    - **スライシング**: `cf[0:2]`, `cf[['ch1', 'ch2']]`
    - **ブールインデックス**: `cf[cf.rms > threshold]`
    """)
    return


@app.cell
def _(sensor_data):
    # 基本的なチャンネルアクセス
    print("🔍 チャンネルアクセス方法:")
    first_channel = sensor_data[0]
    # インデックスでアクセス
    print(f"  cf[0]: {first_channel.shape} - {first_channel.labels}")
    _accel_x = sensor_data["accel_x"]
    print(f"  cf['accel_x']: {_accel_x.shape} - {_accel_x.labels}")
    # ラベルでアクセス
    accel_channels = sensor_data[["accel_x", "accel_y", "accel_z"]]
    print(f"  cf[['accel_x', 'accel_y', 'accel_z']]: {accel_channels.shape} - {accel_channels.labels}")
    first_two = sensor_data[0:2]
    # 複数チャンネル選択
    # スライシング
    print(f"  cf[0:2]: {first_two.shape} - {first_two.labels}")
    return


@app.cell
def _(np, sensor_data):
    # 実践的なチャンネル操作
    print("🎯 実践的なチャンネル操作:")

    # RMS値でチャンネルをフィルタリング
    rms_values = sensor_data.rms
    print(f"  RMS値: {dict(zip(sensor_data.labels, rms_values))}")

    # RMSが0.5以上のチャンネルのみ選択
    active_channels = sensor_data[rms_values > 0.5]
    print(f"  アクティブチャンネル: {active_channels.labels}")

    # 特定の時間範囲を抽出
    time_slice = sensor_data[:, 100:200]  # サンプル100-200
    print(f"  時間スライス: {time_slice.shape} samples")

    # チャンネル間の演算
    magnitude = np.sqrt(sensor_data["accel_x"] ** 2 + sensor_data["accel_y"] ** 2 + sensor_data["accel_z"] ** 2)
    print(f"  ベクトルの大きさ: {magnitude.shape}")
    return active_channels, time_slice


@app.cell
def _(active_channels, plt, sensor_data, time_slice):
    # チャンネル操作の可視化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 元のデータ
    sensor_data.plot(ax=axes[0, 0], title="All Channels", overlay=True)

    # 加速度チャンネルのみ
    sensor_data[["accel_x", "accel_y", "accel_z"]].plot(ax=axes[0, 1], title="Acceleration Only", overlay=False)

    # アクティブチャンネルのみ
    active_channels.plot(ax=axes[1, 0], title="Active Channels Only", overlay=True)

    # 時間スライス
    time_slice.plot(ax=axes[1, 1], title="Time Slice (100-200)", overlay=True)

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 🔍 クエリによるチャンネル選択

    より高度なチャンネル選択には`get_channel`メソッドの`query`パラメータを使用できます。

    #### なぜクエリが必要か

    チャンネルが多数ある場合、**条件に基づいてチャンネルを選択**したいことがあります：
    - 特定のラベルパターンに一致するチャンネル
    - 特定のインデックス範囲のチャンネル
    - メタデータに基づく選択
    """)
    return


@app.cell
def _(sensor_data):
    # get_channelメソッドのクエリ例
    import re

    print("🔍 get_channelメソッドのクエリ選択:")

    # 1. 文字列クエリ（完全一致）
    print("\n1. 📝 文字列クエリ（完全一致）")
    selected_exact = sensor_data.get_channel(query="accel_x")
    print(f"  query='accel_x': {selected_exact.labels}")

    # 2. 部分一致の例
    print("\n2. 🔍 部分一致の例")
    # ラベルに'ccel'を含むチャンネルを選択（正規表現で部分一致）
    selected_partial = sensor_data.get_channel(query=re.compile(r".*ccel.*"))
    print(f"  query=re.compile(r'.*ccel.*'): {selected_partial.labels}")

    # 3. 関数クエリ（述語）
    print("\n3. 🧮 関数クエリ（述語）")

    # RMS値が0.5以上のチャンネルを選択
    def high_energy_predicate(ch_metadata):
        # チャンネルのインデックスを取得
        ch_idx = sensor_data.labels.index(ch_metadata.label)
        return sensor_data.rms[ch_idx] > 0.5

    selected_predicate = sensor_data.get_channel(query=high_energy_predicate)
    print(f"  高エネルギー述語: {selected_predicate.labels}")

    # 4. 辞書クエリ（メタデータ属性）
    print("\n4. 📚 辞書クエリ（メタデータ属性）")
    # ラベルが特定の値に一致するチャンネル
    selected_dict = sensor_data.get_channel(query={"label": "temperature"})
    print(f"  query={{'label': 'temperature'}}: {selected_dict.labels}")

    # 5. 辞書クエリ（正規表現）
    print("\n5. 🔗 辞書クエリ（正規表現）")
    selected_dict_regex = sensor_data.get_channel(query={"label": re.compile(r"accel_.*")})
    print(f"  query={{'label': re.compile(r'accel_.*')}}: {selected_dict_regex.labels}")

    print("\n✅ get_channelクエリ選択完了！")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### get_channelクエリのポイント

    - **文字列**: `query='label_name'` - 完全一致
    - **正規表現**: `query=re.compile(r'pattern')` - パターンマッチ
    - **関数**: `query=lambda ch: condition` - カスタム条件
    - **辞書**: `query={'attr': value}` - メタデータ属性一致
    - **辞書+正規表現**: `query={'attr': re.compile(r'pattern')}` - 高度なマッチ

    `get_channel`メソッドはより柔軟で強力なチャンネル選択を提供します。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### チャンネル選択のポイント

    #### 基本的なアクセス方法
    - **インデックス**: `data[[0, 2]]` - 位置による選択
    - **ラベル**: `data[['ch1', 'ch2']]` - 名前による選択
    - **スライス**: `data[1:3]` - 範囲選択
    - **単一**: `data['ch1']` - 1つのチャンネル

    #### 高度な選択パターン
    - **条件**: リスト内包表記で動的選択
    - **統計情報**: RMS値などの指標を使った選択

    #### get_channelメソッドのクエリ
    - **文字列**: `query='label_name'` - 完全一致
    - **正規表現**: `query=re.compile(r'pattern')` - パターンマッチ
    - **関数**: `query=lambda ch: condition` - カスタム条件
    - **辞書**: `query={'attr': value}` - メタデータ属性一致

    これらの方法を組み合わせることで、柔軟なデータ分析が可能になります。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 💾 データの保存

    ### さまざまなフォーマットでの保存

    Wandasは**読み込み時と同じ柔軟性**でデータを保存できます：
    - **WAV**: 音声データとして保存
    - **WDF**: Wandas専用フォーマット（メタデータ完全保存）
    - **NumPy**: 高速処理用
    - **CSV**: 表計算ソフト用
    """)
    return


@app.cell
def _(pathlib_path, audio, np, sensor_data, stereo_audio):
    # 処理したデータを保存
    output_dir = pathlib_path("output")
    output_dir.mkdir(exist_ok=True)

    # WAV形式で保存
    wav_output = output_dir / "processed_audio.wav"
    audio.to_wav(wav_output)
    print(f"✅ WAV保存: {wav_output}")

    # WDF形式で保存（メタデータ完全保存）
    wdf_output = output_dir / "sensor_data.wdf"
    sensor_data.trim(start=0, end=1).save(wdf_output, overwrite=True)
    print(f"✅ WDF保存: {wdf_output}")

    # NumPy形式で保存
    np_output = output_dir / "stereo_audio.npy"
    np.save(np_output, stereo_audio.data)
    print(f"✅ NumPy保存: {np_output}")

    # CSV形式で保存
    csv_output = output_dir / "processed_sensors.csv"
    sensor_data.to_dataframe().to_csv(csv_output)
    print(f"✅ CSV保存: {csv_output}")
    return csv_output, np_output, output_dir, wdf_output


@app.cell
def _(csv_output, np, np_output, stereo_audio, wd, wdf_output):
    # 保存したデータを読み込んで確認
    print("🔄 保存データの読み込み確認:")

    # WDFファイルを読み込み
    loaded_wdf = wd.ChannelFrame.load(wdf_output)
    print(f"  WDF読み込み: {loaded_wdf.shape} - メタデータ保持: {len(loaded_wdf.operation_history)} operations")

    # NumPyファイルを読み込み
    loaded_np = wd.from_numpy(
        data=np.load(np_output), sampling_rate=stereo_audio.sampling_rate, ch_labels=stereo_audio.labels
    )
    print(f"  NumPy読み込み: {loaded_np.shape} - サンプリングレート: {loaded_np.sampling_rate} Hz")

    # CSVファイルを読み込み
    loaded_csv = wd.read_csv(csv_output, time_column="time")
    print(f"  CSV読み込み: {loaded_csv.shape} - チャンネル: {loaded_csv.labels}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🎯 実践的なワークフロー例

    ### センサーデータ分析の完全な例

    現実的な信号処理ワークフローを体験しましょう。
    """)
    return


@app.cell
def _(csv_path, output_dir, wd):
    # 完全なワークフロー: データ読み込み → 処理 → 保存
    print("🚀 完全な信号処理ワークフロー:")

    # 1. データ読み込み
    print("1. 📂 データ読み込み")
    data = wd.read_csv(csv_path, time_column="time")
    print(f"   読み込み完了: {data.shape}")

    # 2. 前処理
    print("2. 🔧 前処理")
    processed = (
        data.high_pass_filter(cutoff=0.5)  # 直流成分除去
        .low_pass_filter(cutoff=10)  # 高周波ノイズ除去
        .normalize()  # 正規化
    )
    print(f"   処理完了: {len(processed.operation_history)} operations")

    # 3. 特徴抽出
    print("3. 📊 特徴抽出")
    features = {
        "rms": processed.rms,
        "peak": processed.abs().data.max(),
        "crest_factor": processed.abs().data.max(-1) / processed.rms,
    }
    print(f"   RMS: {dict(zip(processed.labels, features['rms']))}")

    # 4. 可視化
    print("4. 📈 可視化")
    processed.describe()

    # 5. 保存
    print("5. 💾 保存")
    final_output = output_dir / "analyzed_sensor_data.wdf"
    processed.save(final_output, overwrite=True)
    print(f"   保存完了: {final_output}")

    print("\n✅ ワークフロー完了！")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🛠️ トラブルシューティング

    ### よくある問題と解決法

    #### 1. ファイル読み込みエラー
    - **ファイルが存在するか確認**: `Path(file).exists()`
    - **フォーマットが正しいか確認**: `file.suffix`
    - **エンコーディング問題**: CSVの場合 `encoding='utf-8'` を指定

    #### 2. サンプリングレートの問題
    - **明示的に指定**: `wd.read_wav(file, sampling_rate=44100)`
    - **自動検出**: `wd.read_wav(file)` でファイルから取得

    #### 3. メモリ不足
    - **チャンク読み込み**: 大きなファイルを分割して読み込み
    - **ダウンサンプリング**: `data.resample(target_sr=22050)`

    #### 4. チャンネル名の不一致
    - **明示的に指定**: `ch_labels=['ch1', 'ch2', 'ch3']`
    - **自動生成**: 指定しない場合 `Channel 0`, `Channel 1`... となる
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 📚 次のステップ

    データの読み込みと基本操作をマスターしました！

    **次のノートブック**: [03_signal_processing_basics.ipynb](03_signal_processing_basics.ipynb)

    ここでは、読み込んだデータを**フィルタリング**や**周波数分析**などの信号処理テクニックで加工する方法を紹介します。

    ### 🎯 これまでに学んだこと
    - ✅ さまざまなファイル形式（WAV, CSV, NumPy）からのデータ読み込み
    - ✅ ChannelFrameデータ構造の理解
    - ✅ pandasライクなチャンネルアクセス方法
    - ✅ メタデータの管理と処理履歴
    - ✅ 複数フォーマットでのデータ保存
    - ✅ 完全な信号処理ワークフローの実践

    ### 🚀 次の学習目標
    - FFTによる周波数領域変換
    - ローパス/ハイパス/バンドパスフィルタ
    - スペクトル分析
    - フィルタ設計の基礎

    ---

    **データ操作の基礎を身につけました。次の信号処理の世界へ！** 🎵
    """)
    return


if __name__ == "__main__":
    app.run()
