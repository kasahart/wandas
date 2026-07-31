import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 00 Wandasとは何か
    ## 信号処理・音響解析のための新しいアプローチ

    このmarimoアプリでは、**なぜWandasが必要なのか**を説明し、信号処理の課題とWandasがどのように解決するかを紹介します。

    **学習目標:**
    - 信号処理の現状と課題を理解する
    - Wandasの特徴と利点を把握する
    - どんな問題を解決できるかを知る
    - 最初の動機付けを得る
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt

    import wandas as wd

    return plt, wd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🎯 信号処理の現状と課題

    ### 従来の信号処理の難しさ

    音響・振動データの解析では、以下のような課題があります：

    #### 1. **複雑なツールチェーン**

    **問題点:**
    - 複数のライブラリを組み合わせる必要がある
    - 各ステップでデータの形状を意識する必要がある
    - エラーが発生しやすい
    - コードが冗長になる
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 2. **データ管理の複雑さ**

    - **多次元データの扱い**: チャンネル数、サンプリングレート、時間軸の管理
    - **実行境界**: 遅延 graph 構築と、kernel・最終結果の実体化を区別
    - **メタデータ**: 単位、チャンネル名、処理履歴の管理
    - **型安全性**: NumPy配列のdtype管理
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 3. **再現性と保守性の欠如**

    - **処理履歴の追跡**: どのような処理を施したか分からない
    - **パラメータ管理**: フィルタ係数、サンプリングレートなどの管理
    - **ドキュメント化**: 分析プロセスの再現が難しい
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 従来の例
    """)
    return


@app.cell
def _(plt, wd):
    import numpy as np
    import scipy.signal

    _sampling_rate = 16000
    _duration = 1.0
    data = wd.generate_sin(freqs=[440], duration=1.0, sampling_rate=16000).data
    t = np.arange(int(_sampling_rate * _duration)) / _sampling_rate
    data = data + np.random.randn(len(data))
    (b, a) = scipy.signal.butter(4, 1000 / (_sampling_rate / 2))
    filtered = scipy.signal.filtfilt(b, a, data)
    window = scipy.signal.windows.hann(len(filtered))
    windowed = filtered * window
    fft_result = np.fft.fft(windowed, norm="forward")
    _freqs = np.fft.fftfreq(len(data), 1 / _sampling_rate)
    (_fig, (_ax1, _ax2)) = plt.subplots(2, 1, figsize=(10, 8))
    _fig.suptitle("Traditional Approach")
    t = np.arange(len(filtered)) / _sampling_rate
    _ax1.plot(t, filtered)
    _ax1.set(title="Time Domain Signal")
    _ax1.grid(True, alpha=0.3)
    magnitude_db = 20 * np.log10(np.abs(fft_result) + 1e-10)
    _ax2.plot(_freqs[: len(_freqs) // 2], magnitude_db[: len(_freqs) // 2])
    _ax2.set(title="Filtered Spectrum", ylim=(-60, 0))
    _ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("従来のアプローチでの処理が完了しました")
    return (np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🚀 Wandasの解決アプローチ

    ### Wandasの3つの柱

    #### 1. **PandasライクなAPI**
    ```python
    # Wandasのアプローチ：直感的なメソッドチェーン
    import wandas as wd

    # 読み込み → フィルタリング → FFT → 可視化
    result = (
        wd.read_wav('signal.wav')
        .low_pass_filter(cutoff=1000)
        .fft()
        .plot(title='Filtered Spectrum')
    )
    ```

    **利点:**
    - 直感的な操作
    - メソッドチェーンで処理を連鎖
    - pandasユーザーになじみやすい
    """)
    return


@app.cell
def _(np, plt, wd):
    signal = wd.generate_sin(freqs=[440], duration=1.0, sampling_rate=16000)
    signal = signal + np.random.randn(signal.n_samples)
    result = signal.low_pass_filter(cutoff=1000).fft()
    (_fig, (_ax1, _ax2)) = plt.subplots(2, 1, figsize=(10, 8))
    _fig.suptitle("Wandas Approach")
    result.previous.plot(ax=_ax1, title="Time Domain Signal")
    result.plot(ax=_ax2, title="Filtered Spectrum", ylim=(-50, 10))
    plt.tight_layout()
    print("Wandasのアプローチでの処理が完了しました")
    result.info()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 2. **統合されたデータ構造**

    **ChannelFrame**: 信号データを統一的に扱う
    - **多次元データ**: チャンネル × サンプルの2D配列
    - **メタデータ**: サンプリングレート、チャンネル名、単位、処理履歴
    - **型安全性**: ty対応の厳格な型付け
    - **遅延評価**: Dask graph の構築時には sample を計算しない

    遅延 graph の構築と実行時のメモリ使用は同じではありません。数値 kernel は連続した
    1 チャンネル全体を必要とすることがあり、operation によってはマルチチャンネル Frame
    全体を必要とします。`frame.data`、NumPy、tensor への変換では最終結果を実体化します。
    Wandas は単一の巨大な Frame を自由に分散するのではなく、サイズを制御した多数の
    収録ファイルを扱う workflow に適しています。正確な保証は
    [スケーラビリティ契約](https://kasahart.github.io/wandas/explanation/scalability-contract/)
    を参照してください。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 3. **包括的な可視化**

    **一つのメソッドで完全な分析**
    ```python
    # 波形、スペクトル、スペクトログラムを一度に表示
    signal.describe()
    ```

    **特徴:**
    - インタラクティブなプロット
    - 出版品質のグラフ
    - Matplotlib統合
    """)
    return


@app.cell
def _(wd):
    # 一つのメソッドで完全な分析
    # 波形、スペクトル、スペクトログラムを一度に表示
    signal_1 = wd.generate_sin(freqs=[440, 880], duration=2.0, sampling_rate=16000)
    signal_1.describe()
    print("describe()メソッドで包括的な分析が表示されました")
    print("時間領域、スペクトル領域、スペクトログラムが一度に確認できます")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 💡 Wandasが解決する具体的な問題

    ### ユースケース1: 環境音の分析

    **課題:** 複数地点で録音した環境音を比較したいが、チャンネル管理が複雑

    **Wandasでの解決:**
    ```python
    # 複数チャンネルの環境音分析
    recording = wd.read_wav('ambient_recording.wav')
    # 全チャンネルにバンドパスフィルタを適用
    filtered = recording.band_pass_filter(100, 8000)
    # RMS値で音圧レベルを比較
    rms_levels = filtered.rms
    print(f"Sound pressure levels: {rms_levels}")
    filtered.plot(title='Filtered Ambient Recording')
    ```
    """)
    return


@app.cell
def _(np, wd):
    # ユースケース1: 環境音の分析
    # 複数チャンネルの環境音分析
    fs = 51200  # サンプリング周波数
    _duration = 10  # 録音時間（秒）
    # サンプルとして複数チャンネルの環境音を生成（異なる場所のシミュレーション）
    np.random.seed(42)
    # 異なるノイズ特性を持つチャンネルを生成
    ch1_noise = np.random.randn(fs * _duration) * 0.1 + 0.05 * np.sin(
        np.linspace(0, 4 * np.pi, fs * _duration)
    )  # 低レベルノイズ + 低周波成分
    ch2_noise = np.random.randn(fs * _duration) * 0.15  # 中レベルノイズ
    ch3_noise = np.random.randn(fs * _duration) * 0.08 + 0.03 * np.sin(
        np.linspace(0, 8 * np.pi, fs * _duration)
    )  # 低レベルノイズ + 高周波成分
    recording = wd.from_numpy(
        data=np.array([ch1_noise, ch2_noise, ch3_noise]),
        sampling_rate=fs,
        ch_labels=["Location A", "Location B", "Location C"],
        ch_units="Pa",
    )
    print("環境音データのシミュレーション:")
    print(f"チャンネル数: {recording.n_channels}")
    print(f"チャンネル名: {recording.labels}")
    rms_levels = recording.a_weighting().rms
    print(f"各場所のA特性音圧レベル (RMS): {rms_levels}")
    recording.rms_plot(title="Filtered Ambient Recording", Aw=True)
    recording.noct_spectrum().plot(
        title="Octave Band Spectrum of Ambient Recording", Aw=True, overlay=True, ylim=(20, 80)
    )
    # RMS値で音圧レベルを比較
    # フィルタリング結果を可視化
    # オクターブ解析
    print("環境音分析が完了しました")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ユースケース2: 機械学習向けデータ前処理

    **課題:** フォルダに格納された多数の、サイズを制御したwavファイルを、
    MLモデルに入力するためにスペクトログラムを作成し、前処理を行う必要がある

    **Wandasを使った実装:**
    ```python
    # metadataで対象を選んでから、サイズを制御した録音単位で処理
    dataset = wd.from_folder('audio_dataset/', recursive=True, path_metadata=True)
    selected = dataset.select(split='train')
    # 前処理パイプライン（リサンプリング、トリミング、正規化）
    dataset = (selected
        .trim(start=0, end=5)      # 各録音の長さを制限
        .resample(target_sr=8000)  # サンプリングレート統一
        .normalize())              # 振幅正規化
    # 全ファイルにSTFTを適用してスペクトログラムを作成
    spectrograms = dataset.stft(n_fft=512, hop_length=256)
    # MLモデルでの処理と結果確認
    ml_results = spectrograms.apply(process_ml_function)
    # ISTFTで時間領域に戻して処理結果を検証
    reconstructed = ml_results.istft()
    ```

    **このユースケースで学ぶこと:**
    - metadataで先に選択し、サイズを制御した録音単位で処理する方法
    - ML向けのデータ前処理パイプライン
    - スペクトログラムベースの特徴抽出
    - 処理結果の検証
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### ステップ1: MLデータセットの準備

    機械学習モデルをトレーニングするためには、多数のデータが必要です。corpus 全体を
    1 つの Frame に連結せず、必要なファイルを metadata で選択し、サイズを制御した録音単位で
    処理します。ここでは、実際のwavファイルの代わりに、プログラムで様々な周波数特性を持つ
    小さな音声ファイルを生成してデータセットを作成します。
    """)
    return


@app.cell
def _(np, wd):
    import os
    import tempfile

    import wandas.utils.frame_dataset as frame_dataset_module

    channel_frame_dataset = frame_dataset_module.ChannelFrameDataset
    temp_dir = tempfile.mkdtemp()
    print(f"サンプルデータセットを作成: {temp_dir}")
    _sampling_rate = 16000
    _duration = 10.0
    n_files = 10
    for i in range(n_files):
        _split = "train" if i < 8 else "validation"
        _split_dir = os.path.join(temp_dir, f"split={_split}")
        os.makedirs(_split_dir, exist_ok=True)
        _freqs = [440 + i * 100, 880 + i * 50]
        audio = wd.generate_sin(freqs=_freqs, duration=_duration, sampling_rate=_sampling_rate)
        audio = audio + np.random.randn(audio.n_samples) * 0.1
        filename = os.path.join(_split_dir, f"audio_sample_{i + 1:03d}.wav")
        audio.to_wav(filename)
    print(f"{n_files}個のサンプル音声ファイルをsplit metadata付きで作成しました")
    return channel_frame_dataset, temp_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### ステップ2: データセットの読み込みと前処理

    作成したデータセットから `split="train"` を先に選び、選択した収録ファイルだけを
    MLモデルへの入力に適した形式へ前処理します。

    **前処理の内容:**
    - **select()**: path metadata だけを使い、sample を読む前に対象を選ぶ
    - **遅延読み込み**: データを必要になるまで各録音の sample を読み込まない
    - **trim()**: 音声の長さを統一（バッチ処理のため）
    - **resample()**: サンプリングレートを統一（MLモデルは固定レートを期待）

    **なぜ前処理が必要か:**
    - MLモデルは入力データの形式が統一されていることを前提としている
    - 異なるサンプリングレートや長さのデータを統一することで、バッチ処理が可能になる
    """)
    return


@app.cell
def _(channel_frame_dataset, temp_dir):
    # path metadata を解決してから、sample を読む前に train split を選択
    dataset = channel_frame_dataset.from_folder(
        folder_path=temp_dir,
        lazy_loading=True,  # 各録音のsampleは必要になるまで読み込まない
        recursive=True,
        path_metadata=True,
    )
    selected_dataset = dataset.select(split="train")

    print("データセット情報:")
    print(f"  探索したファイル数: {len(dataset)}")
    print(f"  trainとして選択したファイル数: {len(selected_dataset)}")
    _source_frame = selected_dataset[0]
    if _source_frame is None:
        raise RuntimeError("選択した最初の音声ファイルを読み込めませんでした")
    print(f"  サンプリングレート: {_source_frame.sampling_rate} Hz")
    print(f"  長さ: {_source_frame.duration} 秒")

    dataset = (
        selected_dataset.trim(start=0, end=5)  # 各録音の長さを先に制限
        .resample(target_sr=8000)  # 必要に応じてリサンプリング
        .normalize()  # 正規化
    )
    print("リサンプリング後のデータセット情報:")
    print(f"  データセットサイズ: {len(dataset)}")
    _processed_frame = dataset[0]
    if _processed_frame is None:
        raise RuntimeError("最初の音声ファイルを前処理できませんでした")
    print(f"  サンプリングレート: {_processed_frame.sampling_rate} Hz")
    print(f"  長さ: {_processed_frame.duration} 秒")

    # 全ファイルにSTFTを適用してスペクトログラムを作成
    spectrogram_dataset = dataset.stft(n_fft=512, hop_length=256)

    print("スペクトログラム作成完了:")
    print(f"  データセットサイズ: {len(spectrogram_dataset)}")
    _spectrogram_frame = spectrogram_dataset[0]
    if _spectrogram_frame is None:
        raise RuntimeError("最初の音声ファイルをスペクトログラムへ変換できませんでした")
    print(f"  周波数ビン数: {_spectrogram_frame.n_freq_bins}")
    print(f"  時間フレーム数: {_spectrogram_frame.n_frames}")

    # サンプルとして最初のスペクトログラムを表示
    _spectrogram_frame[0].plot(title="Spectrogram Sample for ML Input")
    return (spectrogram_dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### ステップ3: スペクトログラム変換とML処理

    前処理済みのデータをスペクトログラムに変換し、MLモデルでの処理をシミュレートします。

    **STFT (Short-Time Fourier Transform) の役割:**
    - 時間-周波数解析により、信号の時間変化を捉える
    - MLモデル（CNNなど）は画像のような2Dデータを入力として扱いやすい
    - n_fft=512: 周波数分解能と時間分解能のバランス
    - hop_length=256: フレームのオーバーラップ（50%）

    **ML処理のシミュレーション:**
    - 実際のMLモデル（TensorFlow, PyTorchなど）にスペクトログラムを入力
    - ここでは簡易的なノイズ除去を例として実装
    - apply()メソッドでデータセット全体に処理を適用
    """)
    return


@app.cell
def _(np, spectrogram_dataset):
    from wandas.frames.spectrogram import SpectrogramFrame

    def process_ml(frame: SpectrogramFrame) -> SpectrogramFrame:
        # ダミー関数、実際にはMLモデルへの入力処理を実装
        previous = frame[0]
        print(f"Processing ML input with shape: {previous.shape}")

        data = previous.data

        # ここで実際のML処理を行う
        data[np.abs(data) < 0.05] = 0  # 簡易なノイズ除去
        ml_out_data = data

        # ML処理結果をSpectrogramFrameとして返す
        ml_out = SpectrogramFrame.from_numpy(
            data=ml_out_data,  # チャンネル次元を追加
            sampling_rate=frame.sampling_rate,
            n_fft=frame.n_fft,
            hop_length=frame.hop_length,
            win_length=frame.win_length,
            window=frame.window,
            label=f"ML({frame.label})",
            metadata=frame.metadata,
            lineage=frame.lineage,
            channel_metadata=[frame.channels[0]],
            previous=previous,
        )
        return ml_out

    ml_results = spectrogram_dataset.apply(process_ml)
    _ml_result = ml_results[0]
    if _ml_result is None or _ml_result.previous is None:
        raise RuntimeError("最初のスペクトログラムをML処理できませんでした")
    _ml_result.previous.plot(vmin=-60, vmax=0, title="Original Spectrogram")
    _ml_result.plot(vmin=-60, vmax=0, title="ML Spectrogram")
    return (ml_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### ステップ4: 処理結果の検証

    ML処理の結果をISTFT (Inverse STFT) で時間領域に戻し、処理前後の比較を行います。

    **検証の重要性:**
    - ML処理が信号の重要な特徴を保持しているか確認
    - 処理結果が元の信号と整合性があるか検証
    - describe()メソッドで包括的な分析（波形・スペクトル・スペクトログラム）

    **ISTFTの役割:**
    - 周波数領域の処理結果を時間領域に戻す
    - 人間が聞き取りやすい形で結果を確認できる
    """)
    return


@app.cell
def _(ml_results):
    # ISTFTで時間信号に元して処理結果を確認
    ml_results[0].previous.istft().describe()
    ml_results[0].istft().describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### ステップ5: リソースのクリーンアップ

    一時的に作成したデータセットを削除し、リソースを解放します。
    """)
    return


@app.cell
def _(temp_dir):
    # クリーンアップ
    import shutil

    shutil.rmtree(temp_dir)
    print(f"一時ディレクトリを削除: {temp_dir}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ユースケース3: 品質管理と異常検知

    **課題:** 製造ラインの振動データを監視し、異常を検知したい

    **Wandasでの解決:**
    ```python
    # 振動データのバッチ処理と特徴抽出
    dataset = wd.ChannelFrameDataset.from_folder('vibration_data/')
    spectrograms = dataset.stft()
    # スペクトル特徴に基づく異常検知
    ```
    """)
    return


@app.cell
def _(np, wd):
    # 振動データのバッチ処理と特徴抽出

    # 振動データをシミュレーション（正常データと異常データ）
    np.random.seed(42)
    normal_vibration = wd.from_numpy(
        data=np.random.randn(1, 16000) * 0.1,  # 正常振動（1秒間）
        sampling_rate=16000,
        ch_labels=["Normal Vibration"],
    )

    abnormal_vibration = wd.from_numpy(
        data=np.random.randn(1, 16000) * 0.3 + np.sin(np.linspace(0, 4 * np.pi, 16000)),  # 異常振動（1秒間）
        sampling_rate=16000,
        ch_labels=["Abnormal Vibration"],
    )

    print("振動データの準備:")
    normal_vibration.info()
    abnormal_vibration.info()

    # 特徴抽出（RMS値を使用）
    def extract_features(vibration_data: wd.ChannelFrame) -> tuple[float, wd.ChannelFrame]:
        preprocessed = vibration_data.band_pass_filter(20, 1000)
        rms = preprocessed.rms
        return (rms[0], preprocessed)

    normal_features, normal_preprocessed = extract_features(normal_vibration)
    abnormal_features, abnormal_preprocessed = extract_features(abnormal_vibration)
    print("特徴抽出完了:")
    print(f"正常データ RMS: {normal_features:.3f}")
    print(f"異常データ RMS: {abnormal_features:.3f}")

    # 異常検知（簡易的な閾値ベース）
    threshold = (normal_features + abnormal_features) / 2
    print(f"閾値: {threshold:.3f}")

    if abnormal_features > threshold:
        print("⚠️ 異常が検知されました！")
    else:
        print("✅ 正常な状態です")

    # 正常データと異常データの比較
    ax = normal_preprocessed.rms_plot()
    abnormal_preprocessed.rms_plot(ax=ax, title="RMS Comparison")
    return abnormal_preprocessed, normal_preprocessed


@app.cell
def _(abnormal_preprocessed, normal_preprocessed):
    # データの詳細解析と視聴
    normal_preprocessed.describe()
    abnormal_preprocessed.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🎨 Wandasの特徴を体験してみよう

    ### 最初のWandasコード

    Wandasの便利さを一緒に確かめてみましょう。
    """)
    return


@app.cell
def _(plt, wd):
    # Wandasをインポート
    plt.rcParams["figure.figsize"] = (10, 6)
    print(f"Wandas version: {wd.__version__}")
    # インタラクティブプロット設定
    # '%matplotlib widget' command supported automatically in marimo
    print("Wandasが正常にインポートされました！")
    return


@app.cell
def _(wd):
    # サンプル信号を生成
    signal_2 = wd.generate_sin(freqs=[1000, 4000], duration=2.0, sampling_rate=16000)
    print("生成された信号:")
    signal_2.info()  # 2秒間
    return (signal_2,)


@app.cell
def _(signal_2):
    # Wandasのdescribe()メソッドで完全な分析を表示
    signal_2.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **🎉 たった1行のコードで:**
    - 時間領域の波形
    - 周波数領域のスペクトル
    - 時間周波数領域のスペクトログラム

    が一度に表示されました！
    """)
    return


@app.cell
def _(signal_2):
    # メソッドチェーンで信号処理
    processed = signal_2.low_pass_filter(cutoff=2000).normalize()
    print("フィルタ処理が完了しました！")
    print(f"処理履歴: {[op['operation'] for op in processed.operation_history]}")  # 660Hzローパスフィルタ  # 正規化
    return (processed,)


@app.cell
def _(plt, processed, signal_2):
    (_fig, (_ax1, _ax2)) = plt.subplots(1, 2, figsize=(12, 4))
    signal_2.fft().plot(ax=_ax1, title="Original Signal Spectrum")
    processed.fft().plot(ax=_ax2, title="Filtered Signal Spectrum")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 💡 Wandasを使うメリット

    ### 1. **生産性の向上**
    - 信号処理パイプラインを効率的に構築
    - 直感的なAPIで学習コストが低い
    - 統合されたツールで作業がスムーズ

    ### 2. **信頼性の向上**
    - 型安全性で実行時エラーを防ぐ
    - 処理履歴で再現性を確保
    - 包括的なテストで品質を保証

    ### 3. **拡張性の高さ**
    - Dask graphを遅延構築し、サイズを制御した多数の収録ファイルを処理
    - モジュール化でカスタム処理を追加
    - エコシステム統合（pandas, NumPy, Matplotlib）

    ### 4. **コミュニティとサポート**
    - オープンソースで透明性が高い
    - 活発な開発で継続的な改善
    - 包括的なドキュメントで学習しやすい
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🎯 こんな人におすすめ

    ### こんな方にぴったり
    - **音響・振動エンジニア**: 測定データの効率的な分析をしたい
    - **データサイエンティスト**: 信号処理をMLパイプラインに統合したい
    - **研究者**: 再現可能で信頼性の高い分析を行いたい
    - **学生**: 信号処理を直感的に学びたい

    ### すぐに始められる
    - **Python経験者**: pandasやNumPyを知っていればOK
    - **信号処理初心者**: 専門知識がなくても使える
    - **多数ファイルの利用者**: metadataで選択してから、収録単位で処理
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 📚 次のステップ

    Wandasの可能性を感じていただけましたか？

    **次のmarimoアプリ**: [01_getting_started.py](01_getting_started.py)

    ここでは実際にWandasをインストールし、環境を設定して、最初の信号処理を行ってみましょう。

    ---

    **さあ、一緒に信号処理の世界を探索しましょう！** 🚀

    質問があれば、[GitHub Issues](https://github.com/kasahart/wandas/issues) や
    [Discordコミュニティ](https://discord.gg/wandas) までお気軽にどうぞ。
    """)
    return


if __name__ == "__main__":
    app.run()
