import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 01 環境構築とウォームアップ
    ## Wandasを動かしてみよう

    このmarimoアプリでは、Wandasをインストールし、基本的な環境設定を行って、最初の信号処理を体験します。

    **学習目標:**
    - Wandasのインストールと環境設定
    - marimo環境でのインタラクティブな操作
    - 最初の信号生成と可視化
    - 基本的な操作の習得

    **前提条件:**
    - Python 3.9以上
    - marimo環境
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🎯 なぜ環境構築が重要か

    ### 信号処理のワークフロー

    信号処理の作業では、以下のようなサイクルを繰り返します：

    1. **データ収集** → 2. **前処理** → 3. **分析** → 4. **可視化** → 5. **解釈**

    このサイクルを効率的に回すためには、適切な環境設定が不可欠です。Wandasは、このワークフローを**1つの統合された環境**で実現します。

    ### インタラクティブな探索の重要性

    信号処理では、**試行錯誤**が不可欠です：
    - フィルタのパラメータを調整しながら効果を確認
    - 異なる可視化方法を比較
    - 処理結果をリアルタイムで評価

    marimo + Wandasの組み合わせで、これを可能にします。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 📦 インストール

    ### 方法1: PyPIからインストール（推奨）

    最新の安定版をインストールします。
    """)
    return


@app.cell
def _():
    # Wandasの最新版をインストール
    # !pip install "wandas[marimo,io,psychoacoustic]"

    # 開発版の場合は（オプション）
    # !pip install git+https://github.com/kasahart/wandas.git

    print("Wandas learning path の推奨インストールコマンド:")
    print('!pip install "wandas[marimo,io,psychoacoustic]"')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 方法2: 開発環境の場合

    このリポジトリをクローンしている場合は：
    """)
    return


@app.cell
def _():
    # 開発インストール（このリポジトリの場合）
    # !pip install -e .

    print("開発インストールコマンド:")
    print("!pip install -e .")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 可視化ライブラリのインストール

    marimo 表示と学習パス内の WDF/音響解析セルのために必要です。
    """)
    return


@app.cell
def _():
    # marimo学習アプリ用のライブラリ
    # !pip install "wandas[marimo,io,psychoacoustic]"

    print("marimo環境のインストール:")
    print('!pip install "wandas[marimo,io,psychoacoustic]"')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🔧 基本的なインポートと確認

    ### 必要なライブラリのインポート
    """)
    return


@app.cell
def _():
    # 基本的なライブラリをインポート
    import matplotlib.pyplot as plt  # Matplotlib - 可視化ライブラリ
    import numpy as np  # NumPy - 数値計算の基礎ライブラリ

    import wandas as wd  # Wandasライブラリ本体 - 信号処理のメインライブラリ

    # バージョン情報を確認
    print(f"Wandas: {wd.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"Matplotlib: {plt.matplotlib.__version__}")

    print("\n✅ すべてのライブラリが正常にインポートされました！")
    return plt, wd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🎨 marimo環境の設定

    ### プロット表示の設定

    信号処理では、**グラフを見ながら探索的に確認**することが重要です：
    - 時間範囲や周波数範囲を絞って詳細を確認
    - 複数の処理結果を並べて比較
    - ラベルや凡例で注目点を確認

    この学習パスでは、marimo上でMatplotlibの静的な図を確認しながら進めます。
    """)
    return


@app.cell
def _(plt):
    # プロット表示の設定

    # プロットのサイズを設定 - 見やすさを調整
    plt.rcParams["figure.figsize"] = (10, 6)  # 図のサイズ (幅10インチ、高さ6インチ)
    plt.rcParams["figure.dpi"] = 100  # 解像度 (dots per inch)

    print("✅ プロット表示の設定が完了しました！")
    print("\nグラフ表示:")
    print("- 図は marimo のセル出力として表示されます")
    print("- 必要に応じて plot() の xlim/ylim などで表示範囲を調整します")
    print("- 比較したい結果は複数セルや複数軸に並べて確認します")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### marimoの確認

    marimoアプリとして実行できる環境か確認します。
    """)
    return


@app.cell
def _():
    # marimoが利用可能か確認 - アプリ実行環境の前提条件
    from importlib.metadata import version as _metadata_version

    marimo_version = _metadata_version("marimo")
    print(f"✅ marimo: {marimo_version} - marimoアプリとして実行できます")
    print("   起動例: marimo edit learning-path/01_getting_started.py")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🎵 最初の信号生成

    ### なぜ信号生成から始めるのか

    実際のデータ分析に入る前に、**既知の信号を生成**することで：
    - Wandasの基本操作を学ぶ
    - 期待される結果を確認できる
    - トラブルシューティングが容易

    これは信号処理の**「Hello, World!」**のようなものです。
    """)
    return


@app.cell
def _(wd):
    # シンプルな正弦波を生成 - Wandasの基本的な信号生成関数
    simple_tone = wd.generate_sin(
        freqs=[440],  # 440Hz (A4音) - 標準的なコンサートピッチ
        duration=1.0,  # 1秒間 - 信号の長さ
        sampling_rate=44100,  # CD品質 - 1秒間に44100サンプル
    )

    # 生成された信号の基本情報を表示
    print("生成された信号の情報:")
    print(f"  チャンネル数: {simple_tone.n_channels}")  # チャンネル数 (モノラル=1, ステレオ=2)
    print(f"  サンプリングレート: {simple_tone.sampling_rate} Hz")  # 1秒間のサンプル数
    print(f"  長さ: {simple_tone.duration:.1f} 秒")  # 信号の時間長
    print(f"  サンプル数: {simple_tone.n_samples}")  # 総サンプル数
    print(f"  チャンネル名: {simple_tone.labels}")  # 各チャンネルの名前
    return (simple_tone,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 信号の可視化

    Wandasの最も強力な機能の一つが、**一つのメソッドで包括的な可視化**ができることです。
    """)
    return


@app.cell
def _(simple_tone):
    # describe()メソッドで完全な分析を表示 - Wandasの強力な可視化機能
    # is_close=False: 図を自動で閉じず、marimo のセル出力として保持する設定
    simple_tone.describe(is_close=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **🎉 `describe()`メソッドで:**
    - **時間領域**: 波形の形状を確認
    - **周波数領域**: スペクトル（周波数成分）を確認
    - **時間周波数領域**: スペクトログラム（時間変化）を確認

    が一度に表示されました。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🔄 メソッドチェーンの体験

    ### Wandasの核心機能

    Wandasの最大の特徴は、**メソッドチェーン**による直感的な処理です。

    これはpandasのようなAPIで、処理を**自然言語のように**連鎖できます。
    """)
    return


@app.cell
def _(wd):
    # より複雑な信号を生成
    complex_signal = wd.generate_sin(
        freqs=[440, 880, 1320],  # 基本音 + 倍音 (440Hzの2倍と3倍)
        duration=2.0,  # 2秒間 - より長い信号
        sampling_rate=8000,  # 8kHz - 電話品質のサンプリングレート
    ).sum()

    # メソッドチェーンで処理 - pandasライクな直感的な処理
    processed = (
        complex_signal.fade(fade_ms=10).low_pass_filter(  # フェイドイン・アウトの時間 (10ミリ秒)
            cutoff=1000
        )  # 1kHzでローパスフィルタ
    )

    print("✅ メソッドチェーンによる処理が完了しました！")

    # 処理履歴を表示 - どのような処理が適用されたかを確認
    processed.print_operation_history()

    # 処理前後の比較 - 信号処理の効果を視覚的に確認
    combined_signal = complex_signal.add_channel(processed, suffix_on_dup="_processed")

    # TypedDictを使用した詳細設定 - 型安全な設定方法
    from wandas.visualization.types import DescribeParams

    config: DescribeParams = {
        "fmin": 100,  # 周波数軸の最小値 (100Hz)
        "fmax": 3000,  # 周波数軸の最大値 (3000Hz)
        "cmap": "jet",  # カラーマップ (jet: 虹色)
        "vmin": -80,  # スペクトログラムの最小値 (dB)
        "vmax": -20,  # スペクトログラムの最大値 (dB)
        "waveform": {"ylim": (-3, 3)},  # 波形のY軸範囲
        "spectral": {"xlim": (-60, 0)},  # スペクトルのX軸範囲 (dB)
    }

    # 設定を適用して詳細な分析を表示
    combined_signal.describe(**config)
    return (combined_signal,)


@app.cell
def _(combined_signal):
    combined_signal.fft().plot(overlay=True)  # 元の信号と処理後の信号を比較表示
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🎮 インタラクティブな実験

    ### パラメータを変更して実験

    信号処理の面白さは、**パラメータを変更しながら結果を確認**できることです。

    以下のセルで、さまざまなパラメータを試してみましょう。
    """)
    return


@app.cell
def _(wd):
    # 実験用の関数を定義 - パラメータを変更して信号処理を試すための関数
    def experiment_with_signal(frequency=440, duration=1.0, filter_cutoff=500):
        """
        周波数、時間、フィルタのカットオフを変更して実験。

        指定されたパラメータで正弦波信号を生成し、ローパスフィルタを適用して
        元の信号とフィルタ済み信号を比較できるようにします。

        Parameters
        ----------
        frequency : float, default=440
            基本周波数 [Hz]。この周波数とその2倍の周波数（倍音）で
            信号を生成します。A4音（440Hz）がデフォルトです。
        duration : float, default=1.0
            信号の長さ [秒]。生成される信号の時間長を指定します。
        filter_cutoff : float, default=500
            ローパスフィルタのカットオフ周波数 [Hz]。
            この周波数以上の成分が減衰されます。

        Returns
        -------
        ChannelFrame
            元の信号とフィルタ済み信号が結合されたChannelFrame。
            チャンネル名は元の信号が "signal"、フィルタ済みが "signal_filtered" となります。

        Examples
        --------
        >>> # デフォルトパラメータで実行
        >>> result = experiment_with_signal()
        >>> result.fft().plot(overlay=True)

        >>> # パラメータを変更して実験
        >>> result = experiment_with_signal(frequency=880, filter_cutoff=1500)
        >>> result.fft().plot(overlay=True)
        """

        # 信号生成 - 指定された周波数で基本音と倍音を作成
        signal = wd.generate_sin(
            freqs=[frequency, frequency * 2],  # 基本音 + 倍音
            duration=duration,  # 指定された長さ
            sampling_rate=4000,  # 4kHzサンプリング (実験用)
        ).sum()

        # フィルタ処理 - 指定されたカットオフ周波数でローパスフィルタ適用
        filtered = signal.low_pass_filter(cutoff=filter_cutoff)

        # 処理した信号を元の信号のchannel frameに追加 - 比較のため
        combined = signal.add_channel(filtered, suffix_on_dup="_filtered")
        return combined

    # デフォルトパラメータで実行 - 基本的な実験
    experiment_with_signal().fft().plot(overlay=True, title="Original and Filtered Signal Spectrum")
    return (experiment_with_signal,)


@app.cell
def _(experiment_with_signal):
    # パラメータを変更して実験
    # 例: より高い周波数で試す
    experiment_with_signal(frequency=880, filter_cutoff=1500).fft().plot(
        overlay=True, title="Original and Filtered Signal Spectrum"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🛠️ トラブルシューティング

    ### よくある問題と解決法

    #### 1. marimoアプリが起動しない
    ```python
    # 解決法
    !pip install "wandas[marimo,io,psychoacoustic]" --upgrade
    marimo edit learning-path/01_getting_started.py
    ```

    #### 2. バージョン互換性の問題
    - Python 3.9以上を使用してください
    - 最新版のWandasを使用してください

    #### 3. メモリ不足
    - 時間を短くする
    - サンプリングレートを下げる

    #### 4. プロットが表示されない
    - `marimo edit learning-path/01_getting_started.py` で起動しているか確認
    - ブラウザを再読み込みしてセルを再実行
    """)
    return


@app.cell
def _():
    # トラブルシューティング: marimo表示が動作しない場合の確認
    # !pip install "wandas[marimo,io,psychoacoustic]" --upgrade  # アップグレードが必要な場合
    print("marimo表示のトラブルシューティング:")
    print("1. marimo edit learning-path/01_getting_started.py で起動")
    print("2. wandas[marimo,io,psychoacoustic] がインストールされているか確認")
    print("3. ブラウザを再読み込みしてセルを再実行")
    from importlib.metadata import version as _metadata_version

    import matplotlib

    print(f"現在のMatplotlibバックエンド: {matplotlib.get_backend()}")
    print(f"✅ marimoバージョン: {_metadata_version('marimo')}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ☁️ S3からWAVを読み込む

    S3から取得したバイト列をそのまま`read_wav()`に渡せます。
    ファイル保存は不要です。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 📚 次のステップ

    環境構築と基本操作が完了しました！

    **次のmarimoアプリ**: [02_working_with_data.py](02_working_with_data.py)

    ここでは、実際のデータファイル（WAV, CSVなど）を読み込んで、Wandasのデータ構造について紹介します。

    ### 🎯 これまでに学んだこと
    - ✅ Wandasのインストールと環境設定
    - ✅ インタラクティブなmarimo環境の構築
    - ✅ 信号生成と基本的な可視化
    - ✅ メソッドチェーンによる直感的な処理
    - ✅ パラメータ変更による実験

    ### 🚀 次の学習目標
    - 実際のデータファイルの読み込み
    - ChannelFrameデータ構造の理解
    - チャンネル操作とインデックスアクセス
    - メタデータの管理

    ---

    **準備はできましたか？次のmarimoアプリへ進みましょう！** 🎵
    """)
    return


if __name__ == "__main__":
    app.run()
