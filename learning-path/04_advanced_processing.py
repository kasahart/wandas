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
    # 04 高度な信号処理
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 高度な信号処理手法の実践

    このノートブックでは、前回までに学んだ基本的な信号処理手法を発展させ、
    **実務で使えるパラメータ調整と比較分析**を行います。

    ### これまでに学んだ基礎
    - **FFT**: 全体の周波数成分を一度に解析（03で学習）
    - **基本的なフィルタリング**: ローパス、ハイパス、バンドパス（03で学習）
    - **Welch法の基本**: ノイズを低減したスペクトル推定（02・03で学習）

    ### 目次（このノートブックで扱う内容）
    1. **STFT（短時間フーリエ変換）のパラメータチューニング**
       - 時間-周波数分解能のトレードオフ
       - 窓サイズ変更による結果比較
    2. **Welch法によるパラメータ最適化**
       - 低SNR信号での分解能比較
       - セグメント数と推定安定性の関係
    3. **N-octave分析の実践**
       - 1/3-octaveバンドによる対数周波数分析
       - 音響評価に適した見方
    4. **A特性音圧レベル（時定数付き）の解析**
       - Fast（LAF）とSlow（LAS）の応答差
       - 騒音計設定に対応した実践的解釈
    5. **FrameDatasetへの一括処理適用**
       - 複数条件データへの同一処理適用
       - Welch法とN-octave分析の重ね合わせ比較

    ### 学習目標
    - STFTとWelch法のパラメータ選定を目的別に使い分けられるようになる
    - N-octave分析とA特性音圧レベルの読み方を習得する
    - FrameDatasetを使った複数条件の比較分析フローを実装できるようになる

    **注意**: Wandasは現在開発中のため、ウェーブレット変換など一部の高度な機能は
    今後実装予定です。このノートブックでは**現在利用可能な手法**に焦点を当てます。
    """)
    return


@app.cell
def _():
    # 必要なライブラリをインポート
    import matplotlib.pyplot as plt
    import numpy as np

    import wandas as wd

    # インタラクティブプロット設定
    # '%matplotlib inline' command supported automatically in marimo
    plt.rcParams["figure.figsize"] = (12, 6)

    print(f"Wandas: {wd.__version__}")
    print("✅ 準備完了")
    return np, plt, wd


@app.cell
def _(np, wd):
    np.random.seed(42)
    sampling_rate = 48000
    _duration = 4.0
    time = np.linspace(0, _duration, int(_duration * sampling_rate))
    time_varying_signal = np.zeros_like(time)
    _mask1 = (time >= 0) & (time < 1)
    time_varying_signal[_mask1] = 2.0 * np.sin(2 * np.pi * 10 * time[_mask1])
    _mask2 = (time >= 1) & (time < 2)
    time_varying_signal[_mask2] = 1.5 * np.sin(2 * np.pi * 50 * time[_mask2])
    _mask3 = (time >= 2) & (time < 3)
    time_varying_signal[_mask3] = 1.0 * np.sin(2 * np.pi * 150 * time[_mask3])
    _mask4 = (time >= 3) & (time < 4)
    time_varying_signal[_mask4] = 0.8 * np.sin(2 * np.pi * 80 * time[_mask4]) + 0.6 * np.sin(
        2 * np.pi * 120 * time[_mask4]
    )
    time_varying_signal = time_varying_signal + 0.1 * np.random.randn(len(time))
    time_varying_data = wd.from_numpy(
        data=time_varying_signal.reshape(1, -1), sampling_rate=sampling_rate, ch_labels=["Time-Varying Signal"]
    )
    print(f"信号作成: {time_varying_data.shape}, {time_varying_data.sampling_rate} Hz")
    print(f"継続時間: {time_varying_data.duration:.1f} 秒")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## STFT（短時間フーリエ変換）のパラメータチューニング

    ### なぜSTFTのパラメータ調整が重要か

    **STFT（Short-Time Fourier Transform）**は、信号を短い時間窓で区切り、
    各窓ごとにFFTを適用する手法です。これにより、
    **時間とともに変化する周波数成分**を捉えることができます。

    STFTには**時間分解能と周波数分解能のトレードオフ**が存在し、分析目的に応じた適切なパラメータ設定が必要です。

    ### 時間-周波数分解能のトレードオフ

    **窓サイズ（`n_fft`）が大きい場合**:
    - ✅ 周波数分解能が高い（細かい周波数成分を区別できる）
    - ❌ 時間分解能が低い（瞬間的な変化を捉えにくい）

    **窓サイズ（`n_fft`）が小さい場合**:
    - ✅ 時間分解能が高い（瞬間的な変化を捉えやすい）
    - ❌ 周波数分解能が低い（近接する周波数成分を区別しにくい）

    ### 実践的なパラメータ選択

    **48kHzサンプリングレートでの目安**:
    - **周波数分解能** = `sampling_rate / n_fft`
      - `n_fft=2048` → 約23.4 Hz（粗い）
      - `n_fft=4096` → 約11.7 Hz（中程度）
      - `n_fft=8192` → 約5.9 Hz（細かい）

    - **時間分解能** = `hop_length / sampling_rate`
      - `hop_length=512` → 約10.7 ms（細かい）
      - `hop_length=1024` → 約21.3 ms（中程度）
      - `hop_length=2048` → 約42.7 ms（粗い）

    このセクションでは、**パラメータ変更が結果に与える影響**を実践的に確認します。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### デモ信号の作成

    時間とともに周波数が変化する信号を作成し、STFTの効果を視覚的に確認します。

    **信号の構成**（4秒間）:
    - **0〜1秒**: 10Hz（低周波振動）
    - **1〜2秒**: 50Hz（電源周波数成分）
    - **2〜3秒**: 150Hz（機械の固有振動数）
    - **3〜4秒**: 80Hz + 120Hzの複合振動

    この信号により、STFTが**時間変化する周波数成分を捉える能力**を確認できます。
    """)
    return


@app.cell
def _(np, wd):
    np.random.seed(42)
    sampling_rate_1 = 48000
    _duration = 4.0
    time_1 = np.linspace(0, _duration, int(sampling_rate_1 * _duration))
    _signal = np.zeros_like(time_1)
    _mask1 = (time_1 >= 0) & (time_1 < 1)
    _signal[_mask1] = 1.0 * np.sin(2 * np.pi * 10 * time_1[_mask1])
    _mask2 = (time_1 >= 1) & (time_1 < 2)
    _signal[_mask2] = 1.0 * np.sin(2 * np.pi * 50 * time_1[_mask2])
    _mask3 = (time_1 >= 2) & (time_1 < 3)
    _signal[_mask3] = 1.0 * np.sin(2 * np.pi * 150 * time_1[_mask3])
    _mask4 = time_1 >= 3
    _signal[_mask4] = 0.7 * np.sin(2 * np.pi * 80 * time_1[_mask4]) + 0.7 * np.sin(2 * np.pi * 120 * time_1[_mask4])
    _signal = _signal + 0.05 * np.random.randn(len(time_1))
    time_varying_data_1 = wd.from_numpy(
        data=_signal.reshape(1, -1), sampling_rate=sampling_rate_1, ch_labels=["Time-Varying Signal"]
    )
    print("✅ 時間変化信号を作成:")
    print(f"  サンプリングレート: {time_varying_data_1.sampling_rate} Hz")
    print(f"  長さ: {time_varying_data_1.duration:.1f} 秒")
    print(f"  サンプル数: {time_varying_data_1.n_samples}")
    return sampling_rate_1, time_varying_data_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 時間領域での確認

    まず、作成した信号を時間領域で観察します。
    """)
    return


@app.cell
def _(plt, time_varying_data_1):
    (_fig, _ax) = plt.subplots(figsize=(12, 4))
    time_varying_data_1.plot(ax=_ax, title="Time-Varying Frequency Signal", xlabel="Time [s]", ylabel="Amplitude")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **波形の観察**:
    - 4つの異なる区間で振幅と周波数が明確に変化している
    - 周波数が高いほど波の間隔が狭くなる（2-3秒の区間）
    - 3-4秒の区間では複数の周波数が混在し、複雑な波形になる

    この波形だけでは「どの周波数成分がいつ現れたか」は定量的に判断できません。次にSTFTを適用して、時間-周波数平面での分析を行います。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### STFTの実行

    それでは、STFTを適用してスペクトログラムを作成します。

    **使用するパラメータ**（48kHzサンプリングレート）:
    - **`n_fft=2048`**: 周波数分解能 約23.4 Hz
    - **`hop_length=512`**: 時間分解能 約10.7 ms
    - **`window='hann'`**: Hann窓（スペクトル漏れと分解能のバランス）

    これらのパラメータは、**瞬時的な周波数変化を捉える**ために時間分解能を優先した設定です。
    """)
    return


@app.cell
def _(plt, time_varying_data_1):
    _spec = time_varying_data_1.stft(n_fft=2048, hop_length=512, window="hann")
    print("📊 STFT結果:")
    _spec.info()
    (_fig, _ax) = plt.subplots(figsize=(12, 6))
    _spec.plot(ax=_ax, title="Spectrogram (STFT)", ylim=(0, 200), vmin=-60, vmax=20)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### スペクトログラムの読み方と結果解釈

    **スペクトログラムの構成**:
    - **横軸**: 時間（秒） - 信号の時間進行
    - **縦軸**: 周波数（Hz） - 各時刻に含まれる周波数成分
    - **色**: パワー（dB） - 明るい色ほど強い周波数成分

    **観察される結果**:
    - **0-1秒**: 10Hz付近に明るい水平線 → 低周波振動が支配的
    - **1-2秒**: 50Hz付近に明るい水平線 → 電源周波数成分が現れる
    - **2-3秒**: 150Hz付近に明るい水平線 → より高い周波数成分
    - **3-4秒**: 80Hzと120Hz付近に2本の明るい線 → 複合振動を明確に分離

    **STFTの利点**:
    - 通常のFFTでは全時間の平均的なスペクトルしか得られないが、**STFTは時間変化を捉えられる**
    - 各時間窓での周波数成分を個別に分析できる
    - 過渡現象や非定常信号の分析に不可欠

    次に、パラメータを変更して分解能のトレードオフを体験します。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### パラメータ変更による分解能比較

    STFTの**窓サイズ（`n_fft`）**を変えることで、時間-周波数分解能のトレードオフを実際に確認します。

    **比較する3つの設定**:
    1. **小窓（`n_fft=1024`）**: 時間分解能優先 - 瞬時的変化に強い
    2. **中窓（`n_fft=2048`）**: バランス型 - 一般的な用途
    3. **大窓（`n_fft=4096`）**: 周波数分解能優先 - 近接周波数の分離に強い

    それぞれのスペクトログラムを並べて比較します。
    """)
    return


@app.cell
def _(plt, sampling_rate_1, time_varying_data_1):
    n_fft_sizes = [1024, 2048, 4096]
    spectrograms = {}
    for _n_fft in n_fft_sizes:
        _hop_length = _n_fft // 4
        _freq_res = sampling_rate_1 / _n_fft
        time_res = _hop_length / sampling_rate_1 * 1000
        _spec = time_varying_data_1.stft(n_fft=_n_fft, hop_length=_hop_length)
        spectrograms[_n_fft] = _spec
        print(f"n_fft={_n_fft}: 周波数分解能={_freq_res:.1f} Hz, 時間分解能={time_res:.1f} ms")
    (_fig, _axes) = plt.subplots(1, 3, figsize=(18, 5))
    for _ax, (_n_fft, _spec) in zip(_axes, spectrograms.items()):
        _freq_res = sampling_rate_1 / _n_fft
        time_res = _n_fft // 4 / sampling_rate_1 * 1000
        _spec.plot(
            ax=_ax,
            title=f"n_fft={_n_fft}\n(f_res={_freq_res:.1f}Hz, t_res={time_res:.1f}ms)",
            ylim=(0, 200),
            vmin=-60,
            vmax=20,
        )
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **パラメータ比較の結果**:

    **小窓（`n_fft=1024`）**:
    - ✅ 時間的な区切りがシャープ（各区間の境界がはっきり）
    - ❌ 周波数軸方向にぼやけている（80Hzと120Hzの分離が不明瞭）
    - **用途**: 急激な周波数変化や過渡現象の検出

    **中窓（`n_fft=2048`）**:
    - ✅ 時間と周波数のバランスが良い
    - ✅ 大抵のケースで十分な分解能
    - **用途**: 一般的な音響・振動分析（デフォルトとして推奨）

    **大窓（`n_fft=4096`）**:
    - ✅ 80Hzと120Hzを明確に分離できる
    - ❌ 時間方向にぼやけ（区間の境界が不明瞭）
    - **用途**: 近接する周波数成分の精密分離、定常信号の分析

    **実践的な選択指針**:
    - **過渡現象・衝撃音分析**: 小窓（高時間分解能）
    - **音声・音楽分析**: 中窓（バランス型）
    - **機械診断・周波数同定**: 大窓（高周波数分解能）

    次のセクションでは、Welch法によるスペクトル推定のパラメータ最適化を学びます。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Welch法によるパラメータ最適化

    ### なぜWelch法のパラメータ調整が重要か

    **Welch法**は、信号を重複する区間に分割し、各区間のパワースペクトルを平均化することで、ノイズの影響を低減し、より安定したスペクトル推定を実現します。

    **02のノートブック**では、Welch法の基本的な使い方とFFTとの違いを学びました。このセクションでは、**パラメータ調整による推定精度の最適化**を実践します。

    ### Welch法のパラメータとその役割

    **主要パラメータ**:
    - **`n_fft`**: 各セグメントのFFTサイズ - 周波数分解能を決定
    - **`hop_length`**: セグメント間のオーバーラップ - 推定の安定性に影響
    - **`window`**: 窓関数の種類 - スペクトル漏れの抑制

    **パラメータ間のトレードオフ**:
    - **`n_fft`が大きい**: 周波数分解能↑、セグメント数↓（平均化効果↓）
    - **`hop_length`が小さい**: セグメント数↑（平均化効果↑）、計算時間↑
    - **窓関数の選択**: スペクトル漏れ vs メインローブ幅

    ### 実践的な最適化戦略

    このセクションでは、**低SNR信号**に対して3つの異なる分解能設定を比較し、適切なパラメータ選択の指針を学びます。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 低SNR信号の作成

    まず、ノイズの多い信号を作成し、Welch法のパラメータ最適化の効果を確認します。

    **信号の構成**:
    - **信号成分**: 100Hz（振幅0.5）+ 150Hz（振幅0.3）
    - **ノイズ**: 強いガウシアンノイズ（標準偏差2.0）
    - **SNR**: マイナスのSNR（信号よりノイズが強い）

    この厳しい条件下で、Welch法のパラメータ調整によって信号成分を検出できるかを検証します。
    """)
    return


@app.cell
def _(np, sampling_rate_1, wd):
    np.random.seed(123)
    _duration = 10.0
    time_2 = np.linspace(0, _duration, int(sampling_rate_1 * _duration))
    _signal = 0.5 * np.sin(2 * np.pi * 100 * time_2) + 0.3 * np.sin(2 * np.pi * 150 * time_2)
    noise = 2.0 * np.random.randn(len(time_2))
    noisy_signal = _signal + noise
    noisy_data = wd.from_numpy(
        data=noisy_signal.reshape(1, -1), sampling_rate=sampling_rate_1, ch_labels=["Low SNR Signal"]
    )
    signal_power = _signal.var()
    noise_power = noise.var()
    snr_db = 10 * np.log10(signal_power / noise_power)
    print("✅ 低SNR信号を作成:")
    print(f"  理論SNR: {snr_db:.1f} dB")
    print("  信号周波数: 100Hz, 150Hz")
    print("  ノイズレベル: 強（σ=2.0）")
    print(f"  信号長: {noisy_data.duration:.1f} 秒")
    return noisy_data, time_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3つの分解能設定で比較

    異なるパラメータ設定でWelch法を実行し、**推定精度の違い**を確認します。

    **比較する3つの設定**:
    1. **低分解能（高速）**: `n_fft=2048`, `hop_length=1024`
       - 周波数分解能: 約23.4 Hz
       - セグメント数: 多い（平均化効果大）

    2. **中分解能（バランス）**: `n_fft=4096`, `hop_length=2048`
       - 周波数分解能: 約11.7 Hz
       - セグメント数: 中程度

    3. **高分解能（精密）**: `n_fft=8192`, `hop_length=4096`
       - 周波数分解能: 約5.9 Hz
       - セグメント数: 少ない（平均化効果小）

    それぞれのスペクトルを比較して、信号検出能力の差を確認します。
    """)
    return


@app.cell
def _(noisy_data, plt, sampling_rate_1):
    configs = [
        {"n_fft": 2048, "hop_length": 1024, "label": "低分解能（高速）"},
        {"n_fft": 4096, "hop_length": 2048, "label": "中分解能（バランス）"},
        {"n_fft": 8192, "hop_length": 4096, "label": "高分解能（精密）"},
    ]
    _welch_results = {}
    print("🔧 Welch法パラメータ比較:")
    for config in configs:
        _n_fft = config["n_fft"]
        _hop_length = config["hop_length"]
        label = config["label"]
        psd = noisy_data.welch(n_fft=_n_fft, hop_length=_hop_length)
        _welch_results[label] = psd
        _freq_res = sampling_rate_1 / _n_fft
        n_segments = int((noisy_data.n_samples - _n_fft) / _hop_length) + 1
        print(f"\n{label}:")
        print(f"  n_fft={_n_fft}, hop_length={_hop_length}")
        print(f"  周波数分解能: {_freq_res:.2f} Hz")
        print(f"  セグメント数: {n_segments} (平均化)")
    (_fig, _axes) = plt.subplots(1, 3, figsize=(18, 5))
    for _ax, (label, psd) in zip(_axes, _welch_results.items()):
        psd.plot(ax=_ax, title=label, xlim=(0, 300), ylim=(-30, 0))
        _ax.axvline(100, color="red", linestyle="--", alpha=0.7, linewidth=1, label="100Hz")
        _ax.axvline(150, color="orange", linestyle="--", alpha=0.7, linewidth=1, label="150Hz")
        _ax.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### パラメータ比較の結果解釈

    **低分解能（`n_fft=2048`）**:
    - ✅ 多数のセグメントによる平均化効果でノイズが滑らか
    - ✅ 100Hzと150Hzのピークが明確に検出できる
    - ❌ 周波数分解能が粗いため、ピークが広がっている
    - **用途**: ノイズが多く、大まかな周波数成分を知りたい場合

    **中分解能（`n_fft=4096`）**:
    - ✅ 周波数分解能と平均化のバランスが良い
    - ✅ ピークが適度にシャープで、ノイズレベルも許容範囲
    - ✅ 一般的な用途に最適
    - **用途**: 標準的な信号分析（推奨デフォルト設定）

    **高分解能（`n_fft=8192`）**:
    - ✅ 最も高い周波数分解能でピークがシャープ
    - ✅ 近接周波数の精密分離に優れている
    - ❌ セグメント数が少ないと、ノイズの変動が大きくなり、弱い信号が埋もれやすい
    - **用途**: 音が長く、精密な周波数分析が必要な場合

    **実践的な選択指針**:
    - **ノイズが多い場合**: 低〜中分解能（平均化を優先）
    - **信号が安定している場合**: 中〜高分解能（分解能を優先）
    - **データ長が短い場合**: 低分解能（セグメント数を確保）
    - **データ長が十分ある場合**: 中〜高分解能

    次のセクションでは、N-octave分析によるオーディオ/建築音響に特化した分析手法を学びます。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## N-octave分析

    ### なぜN-octave分析が必要か

    **N-octave分析**は、音響工学や建築音響で広く使われる周波数分析手法です。通常のFFTが**等間隔の周波数ビン**を使うのに対し、N-octave分析は**対数的に配置された周波数帯域**を使用します。

    ### 人間の聴覚特性との対応

    **人間の耳は対数的**に周波数を知覚します：
    - 100Hzと200Hzの差（100Hz）と、1000Hzと2000Hzの差（1000Hz）は**同じ音程差**（1オクターブ）として聞こえる
    - 音楽や音響では、等間隔ではなく**倍数関係**で周波数を扱う

    N-octave分析は、この人間の聴覚特性に合わせた周波数分割を提供します。

    ### 1/N-octaveバンドとは

    **1/N-octaveバンド**は、各帯域の中心周波数が以下の関係を持ちます：
    $$f_{\text{center},k+1} = f_{\text{center},k} \times 2^{1/N}$$

    一般的な設定:
    - **1-octave**: 粗い分析（建築音響の評価、規格に基づく一般的な騒音評価、簡易的な騒音源特定）
    - **1/3-octave**: 中程度の分解能（環境騒音の詳細分析、騒音対策の効果検証、機械の異音解析など精密な評価）
    - **1/12-octave**: 細かい分析（音楽の半音に対応）

    このセクションでは、**1/3-octave分析**を実践し、音響エネルギーの周波数分布を視覚化します。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 広帯域信号の作成

    N-octave分析の効果を確認するため、**広い周波数範囲**にわたる信号を作成します。

    **信号の構成**:
    - **63 Hz**: 低音（バスドラム相当）
    - **250 Hz**: 中低音（男性の声相当）
    - **1000 Hz**: 中音（基準周波数）
    - **4000 Hz**: 高音（シンバル相当）
    - **ノイズ**: 現実的な環境ノイズを模擬

    この信号により、N-octave分析が**広帯域信号をどのように帯域分割するか**を確認できます。
    """)
    return


@app.cell
def _(np, sampling_rate_1, time_2, wd):
    # N-オクターブ分析用の複合周波数信号を作成
    np.random.seed(456)
    composite_signal = (
        1.0 * np.sin(2 * np.pi * 63 * time_2)
        + 0.8 * np.sin(2 * np.pi * 250 * time_2)
        + 0.6 * np.sin(2 * np.pi * 1000 * time_2)
        + 0.5 * np.sin(2 * np.pi * 4000 * time_2)
        + 0.3 * np.sin(2 * np.pi * 8000 * time_2)
    )
    # 複数の周波数成分を持つ信号
    # 48kHzサンプリングレートでは、より広い周波数範囲が利用可能
    composite_signal = composite_signal + 0.05 * np.random.randn(len(time_2))
    composite_data = wd.from_numpy(
        data=composite_signal.reshape(1, -1), sampling_rate=sampling_rate_1, ch_labels=["Multi-Frequency Signal"]
    )  # 低域
    print(f"複合周波数信号作成: {composite_data.shape}")  # 中域
    print("含まれる周波数: 63Hz, 250Hz, 1kHz, 4kHz, 8kHz")  # 中高域
    # 軽いノイズを追加
    # ChannelFrame作成
    print(f"Nyquist周波数: {sampling_rate_1 / 2} Hz")  # 高域  # 超高域（48kHzだからこそ可能）
    return (composite_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1/3-octave分析の実行

    それでは、作成した信号に対して1/3-octave分析を実行します。

    **パラメータ**:
    - **`fraction=3`**: 1/3-octaveバンド（建築音響の標準）
    - 各帯域の帯域幅は中心周波数に比例する
    """)
    return


@app.cell
def _(composite_data, plt, sampling_rate_1):
    _noct_result = composite_data.noct_spectrum(fmin=25, fmax=20000, n=3)
    print("N-オクターブ分析結果:")
    print(f"  バンド数: {len(_noct_result.freqs)}")
    print(f"  周波数範囲: {_noct_result.freqs[0]:.1f} - {_noct_result.freqs[-1]:.1f} Hz")
    print(f"  サンプリングレート: {_noct_result.sampling_rate} Hz")
    print(f"  Nyquist周波数: {sampling_rate_1 / 2} Hz")
    (_fig, (_ax1, _ax2)) = plt.subplots(2, 1, figsize=(12, 10))
    composite_data.plot(ax=_ax1, title="Original Multi-Frequency Signal")
    _noct_result.plot(ax=_ax2, title="1/3 Octave Band Spectrum")
    _ax2.set_xscale("log")
    _ax2.grid(True, which="both", alpha=0.3)
    for freq in [63, 250, 1000, 4000, 8000]:
        _ax2.axvline(freq, color="red", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### N-octave分析結果の読み方

    **1/3-octaveスペクトルの特徴**:
    - **横軸**: 中心周波数（対数スケール）
    - **縦軸**: 各帯域のエネルギーレベル（dB）
    - **帯域幅**: 低周波数では狭く、高周波数では広い

    **観察される結果**:
    - **63 Hz付近**: 低音成分が明確にピークとして現れる
    - **250 Hz付近**: 中低音成分のピーク
    - **1000 Hz付近**: 中音成分のピーク（最も強い）
    - **4000 Hz付近**: 高音成分のピーク

    **FFTスペクトルとの違い**:
    - FFTは等間隔の周波数ビン → 高周波数帯域の分解能が相対的に高い
    - N-octaveは対数的な帯域 → 全周波数範囲で均等な相対分解能
    - N-octaveは人間の聴覚特性に合っている → 音響評価に適している

    **実用例**:
    - **建築音響**: 室内の音響特性評価
    - **騒音測定**: 環境騒音の周波数分布
    - **音楽分析**: 楽器の周波数特性
    - **機械診断**: 回転機械の振動評価
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A特性音圧レベル（時定数付き）の解析

    ### 騒音計の時定数とは

    **A特性音圧レベル（LA）**は、人間の聴覚特性に合わせた周波数重み付けを施した音圧レベルで、環境騒音の評価や騒音規制に広く使用されます。

    騒音計では、**時定数（Time Constant）**によって過去の音をどの程度反映するかを制御します：

    | 設定 | 時定数 | 記号 | 用途 |
    |------|--------|------|------|
    | **Fast（F）** | 125 ms | LAF | 瞬時的な騒音変動の把握 |
    | **Slow（S）** | 1000 ms | LAS | 安定した騒音レベルの評価 |

    **時定数の効果**：
    - **Fast**: 短い時定数により、騒音の瞬時的な変化に素早く応答
    - **Slow**: 長い時定数により、騒音レベルの変動を平滑化して安定評価

    Wandasの`sound_level()`メソッドは、これらの騒音計の特性を忠実に再現します。
    """)
    return


@app.cell
def _(np, wd):
    # A特性音圧レベル解析用の信号を作成
    # 突然レベルが変化する信号で時定数の効果を確認
    np.random.seed(789)
    sr_sl = 48000
    duration_sl = 6.0
    time_sl = np.linspace(0, duration_sl, int(sr_sl * duration_sl))

    # 段階的に音圧レベルが変化する信号（音響信号を模擬）
    # 0-2秒: 低レベル（60 dB相当）
    # 2-4秒: 高レベル（80 dB相当、突然増大）
    # 4-6秒: 再び低レベル（60 dB相当）
    p_ref = 2e-5  # 音圧の基準値 [Pa]

    # 各区間の実効音圧
    p_low = p_ref * 10 ** (60 / 20)  # 60 dB → 約 0.02 Pa
    p_high = p_ref * 10 ** (80 / 20)  # 80 dB → 約 0.20 Pa

    sound_signal = np.zeros(len(time_sl))
    mask_low1 = time_sl < 2.0
    mask_high = (time_sl >= 2.0) & (time_sl < 4.0)
    mask_low2 = time_sl >= 4.0

    # 1000 Hz正弦波（純音）に各レベルの振幅を設定
    sound_signal[mask_low1] = p_low * np.sqrt(2) * np.sin(2 * np.pi * 1000 * time_sl[mask_low1])
    sound_signal[mask_high] = p_high * np.sqrt(2) * np.sin(2 * np.pi * 1000 * time_sl[mask_high])
    sound_signal[mask_low2] = p_low * np.sqrt(2) * np.sin(2 * np.pi * 1000 * time_sl[mask_low2])

    # ChannelFrame作成（ch_units='Pa' により基準音圧 2e-5 Pa が自動設定される）
    sound_data = wd.from_numpy(
        data=sound_signal.reshape(1, -1), sampling_rate=sr_sl, ch_labels=["Sound Pressure"], ch_units=["Pa"]
    )

    print("✅ 音圧信号を作成:")
    print(f"  サンプリングレート: {sound_data.sampling_rate} Hz")
    print(f"  継続時間: {sound_data.duration:.1f} 秒")
    print(f"  基準音圧: {sound_data._channel_metadata[0].ref} Pa")
    print(f"  0-2秒: {20 * np.log10(p_low / p_ref):.0f} dB (低レベル)")
    print(f"  2-4秒: {20 * np.log10(p_high / p_ref):.0f} dB (高レベル、突然増大)")
    print(f"  4-6秒: {20 * np.log10(p_low / p_ref):.0f} dB (低レベル、突然減少)")
    return (sound_data,)


@app.cell
def _(plt, sound_data):
    # A特性音圧レベルをFast・Slow時定数で計算して比較
    # Fast（F）: 時定数 125ms、Slow（S）: 時定数 1000ms
    laf = sound_data.sound_level(freq_weighting="A", time_weighting="Fast", dB=True)
    # Fast時定数（125ms）でのA特性音圧レベル
    las = sound_data.sound_level(freq_weighting="A", time_weighting="Slow", dB=True)
    print("LAF（A特性・Fast）- 時定数: 125 ms")
    print(f"  出力形状: {laf.data.shape}")
    print("LAS（A特性・Slow）- 時定数: 1000 ms")
    print(f"  出力形状: {las.data.shape}")
    (_fig, _axes) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    # Slow時定数（1000ms）でのA特性音圧レベル
    laf.plot(ax=_axes[0], title="LAF（A特性・Fast時定数: 125 ms）", ylabel="音圧レベル [dB]", ylim=(20, 90))
    _axes[0].axvline(x=2.0, color="red", linestyle="--", alpha=0.7, label="レベル変化点")
    _axes[0].axvline(x=4.0, color="red", linestyle="--", alpha=0.7)
    _axes[0].axhline(y=60, color="gray", linestyle=":", alpha=0.5, label="60 dB (低レベル)")
    _axes[0].axhline(y=80, color="orange", linestyle=":", alpha=0.5, label="80 dB (高レベル)")
    _axes[0].legend(loc="upper right")
    las.plot(ax=_axes[1], title="LAS（A特性・Slow時定数: 1000 ms）", ylabel="音圧レベル [dB]", ylim=(20, 90))
    _axes[1].axvline(x=2.0, color="red", linestyle="--", alpha=0.7, label="レベル変化点")
    _axes[1].axvline(x=4.0, color="red", linestyle="--", alpha=0.7)
    _axes[1].axhline(y=60, color="gray", linestyle=":", alpha=0.5, label="60 dB (低レベル)")
    _axes[1].axhline(y=80, color="orange", linestyle=":", alpha=0.5, label="80 dB (高レベル)")
    # プロット比較
    _axes[1].legend(loc="upper right")
    plt.tight_layout()
    # LAF（Fast）
    plt.show()
    print("\n📊 定常状態での最大値（高レベル区間）:")
    print(f"  LAF最大値: {laf.data.max():.1f} dB")
    # LAS（Slow）
    print(f"  LAS最大値: {las.data.max():.1f} dB")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 解析結果の読み方

    **Fast（F）とSlow（S）の違い**:

    | 特性 | Fast（LAF） | Slow（LAS） |
    |------|------------|------------|
    | **時定数** | 125 ms | 1000 ms |
    | **応答速度** | 速い（瞬時変動を追跡） | 遅い（変動を平滑化） |
    | **レベル上昇時** | 素早く定常値へ到達 | ゆっくり定常値へ収束 |
    | **レベル下降時** | 素早く低下 | ゆっくり減衰 |

    **観察される結果**:
    - **LAF（Fast）**: 2秒と4秒の変化点でほぼ即座に応答し、素早く新しい定常値へ到達
    - **LAS（Slow）**: 変化点での立ち上がり・立ち下がりが緩やかで、定常値への収束に時間がかかる

    **実用的な使い分け**:
    - **LAF（Fast）**: 衝撃音・間欠騒音の最大値評価（工場騒音、交通騒音のピーク測定）
    - **LAS（Slow）**: 安定した騒音レベルの長時間評価（環境騒音、生活騒音の全体評価）

    **`sound_level()`メソッドの使い方**:
    ```python
    # A特性・Fast時定数（騒音計のLAF設定）
    laf = signal.sound_level(freq_weighting="A", time_weighting="Fast", dB=True)

    # A特性・Slow時定数（騒音計のLAS設定）
    las = signal.sound_level(freq_weighting="A", time_weighting="Slow", dB=True)

    # Z特性（フラット）・Fast時定数（線形RMS出力）
    z_rms = signal.sound_level(freq_weighting="Z", time_weighting="Fast", dB=False)
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## データセットへの一括処理適用

    FrameDatasetを使用すると、**全データに同じ処理を一度に適用**できます。

    ここでは、**Welch法とN-octave分析を一括適用**し、結果を重ねて比較することで、
    **異なる分析手法の特徴と相補性を視覚的に確認**します。

    **実践例**: 正常・軽度異常・重度異常の3つの条件の振動データを比較分析
    """)
    return


@app.cell
def _(np, sampling_rate_1, wd):
    np.random.seed(42)
    _duration = 5.0
    time_3 = np.linspace(0, _duration, int(sampling_rate_1 * _duration))
    normal_signal = (
        1.0 * np.sin(2 * np.pi * 100 * time_3)
        + 0.3 * np.sin(2 * np.pi * 200 * time_3)
        + 0.1 * np.random.randn(len(time_3))
    )
    mild_abnormal_signal = (
        1.0 * np.sin(2 * np.pi * 100 * time_3)
        + 0.3 * np.sin(2 * np.pi * 200 * time_3)
        + 0.5 * np.random.randn(len(time_3))
    )
    severe_abnormal_signal = (
        1.0 * np.sin(2 * np.pi * 100 * time_3)
        + 0.3 * np.sin(2 * np.pi * 200 * time_3)
        + 2.0 * np.random.randn(len(time_3))
    )
    impulse_positions = np.random.choice(len(time_3), size=5, replace=False)
    severe_abnormal_signal[impulse_positions] = severe_abnormal_signal[impulse_positions] + 5.0
    signals = [
        wd.from_numpy(data=normal_signal.reshape(1, -1), sampling_rate=sampling_rate_1, ch_labels=["Normal"]),
        wd.from_numpy(
            data=mild_abnormal_signal.reshape(1, -1), sampling_rate=sampling_rate_1, ch_labels=["Mild Abnormal"]
        ),
        wd.from_numpy(
            data=severe_abnormal_signal.reshape(1, -1), sampling_rate=sampling_rate_1, ch_labels=["Severe Abnormal"]
        ),
    ]
    print("✅ 3つの条件の振動信号を作成:")
    for _signal in signals:
        print(f"  {_signal.labels[0]}: {_signal.shape}, {_signal.sampling_rate} Hz")
    from wandas.utils.frame_dataset import ChannelFrameDataset

    temp_dir = "/tmp/wandas_comparison"
    import os

    os.makedirs(temp_dir, exist_ok=True)
    channel_frame_dataset = ChannelFrameDataset
    for i, _signal in enumerate(signals):
        filename = f"{temp_dir}/signal_{i}_{_signal.labels[0].lower().replace(' ', '_')}.wav"
        _signal.to_wav(filename)
    return channel_frame_dataset, temp_dir


@app.cell
def _(channel_frame_dataset, plt, temp_dir):
    # FrameDataset作成
    dataset = channel_frame_dataset.from_folder(temp_dir, lazy_loading=True)
    print(f"✅ FrameDataset作成: {len(dataset)} ファイル")
    _welch_results = dataset.apply(lambda x: x.welch(n_fft=2048, hop_length=1024))
    # Welch法とN-octave分析を一括適用
    noct_results = dataset.apply(lambda x: x.noct_spectrum(fmin=25, fmax=20000, n=3))
    (fig1, _ax1) = plt.subplots(figsize=(12, 6))
    for welch_result in _welch_results:
        # Welch法とN-octave分析の結果を重ねて比較
        # Welch法の結果を1つの図にまとめてプロット
        welch_result.plot(ax=_ax1, alpha=0.8, label=welch_result.label)
    _ax1.set_title("Welch Method Comparison Across Conditions", fontsize=14)
    _ax1.set_xlim(20, 1000)
    _ax1.set_ylim(30, 90)
    (fig2, _ax2) = plt.subplots(figsize=(12, 6))
    for _noct_result in noct_results:
        _noct_result.plot(ax=_ax2, alpha=0.8, label=_noct_result.label)
    _ax2.set_title("N-Octave Analysis Comparison Across Conditions", fontsize=14)
    # N-octave分析の結果を1つの図にまとめてプロット
    _ax2.set_xlim(20, 10000)
    _ax2.set_ylim(30, 90)
    _ax2.set_xscale("log")
    return


@app.cell
def _(temp_dir):
    # クリーンアップ
    import shutil

    shutil.rmtree(temp_dir)
    print(f"一時ディレクトリを削除しました: {temp_dir}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Welch法とN-octave分析の比較結果

    **Welch法とN-octave分析の特徴比較**:

    **Welch法の特徴**:
    - **等間隔周波数ビン**: 高周波数帯域でも一定の分解能
    - **詳細なピーク検出**: 100Hzと200Hzのピークを明確に分離
    - **ノイズの影響**: 平均化により安定した推定が可能
    - **用途**: 機械の固有周波数同定、精密な周波数分析

    **N-octave分析の特徴**:
    - **対数間隔周波数帯域**: 低周波数で広い帯域、高周波数で狭い帯域
    - **人間聴覚特性対応**: 音響・振動の評価に適したスケール
    - **広帯域特性評価**: 全体的なエネルギー分布を把握しやすい
    - **用途**: 騒音評価、建築音響、環境振動測定

    **実践的な使い分け**:
    - **精密な周波数同定が必要**: Welch法
    - **人間の感覚に合った評価**: N-octave分析
    - **両方の特徴を活かす**: Welch法でピーク検出、N-octaveで全体評価

    このように、FrameDatasetを使うことで**複数条件の系統的な比較と判定基準の設定**が可能になります。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## トラブルシューティング

    ### よくある問題と実践的な解決策

    #### 1. STFTパラメータ選択の迷い

    **症状**: どのFFTサイズとホップ長を選べばいいかわからない

    **解決策**:
    ```python
    # まず標準設定で試す（48kHz用）
    spec = signal.stft(n_fft=2048, hop_length=512)

    # 周波数分離を優先するなら
    spec_high_res = signal.stft(n_fft=4096, hop_length=1024)

    # 時間変化を細かく追いたいなら
    spec_high_time = signal.stft(n_fft=1024, hop_length=256)
    ```

    **判断基準**:
    - **近接周波数の分離が必要** -> FFTサイズを大きく（4096, 8192）
    - **急激な時間変化を追跡** -> ホップ長を小さく（256, 128）
    - **リアルタイム処理** -> 両方を小さく（1024/256）

    #### 2. Welch法の計算が遅い

    **原因**: FFTサイズと重複度が大きすぎる、またはセグメント数が多い

    **対策**:
    ```python
    # 高速化のポイント
    welch = signal.welch(
        n_fft=2048,        # 小さめのFFTサイズ
        hop_length=1024,   # 50%オーバーラップ（75%より軽い）
        win_length=2048    # n_fftと同じにする
    )
    ```

    **48kHzでの推奨設定**:
    - **バランス型**: n_fft=2048, hop_length=1024（本ノートブック採用）
    - **高速型**: n_fft=1024, hop_length=512
    - **高精度型**: n_fft=8192, hop_length=4096（長い信号のみ）

    #### 3. N-octave分析の周波数範囲エラー

    **症状**: `ValueError: fmax exceeds Nyquist frequency`

    **原因**: 最大周波数がNyquist周波数（sampling_rate/2）を超えている

    **解決策**:
    ```python
    nyquist = signal.sampling_rate / 2  # 48kHzなら24kHz

    # 安全な設定
    noct = signal.noct_spectrum(
        fmin=25,
        fmax=min(20000, nyquist * 0.9),  # Nyquistの90%まで
        n=3  # 1/3オクターブ
    )
    ```

    #### 4. A特性音圧レベルでFastとSlowの差が分かりにくい

    **症状**: LAFとLASが似た値になり、時定数の効果が見えない

    **原因**:
    - レベル変化の少ない定常信号を使っている
    - 変化区間が短く、Slow（1000ms）の追従差が出にくい

    **対策**:
    ```python
    # 段階的にレベルが変化する信号を使う
    laf = signal.sound_level(freq_weighting="A", time_weighting="Fast", dB=True)
    las = signal.sound_level(freq_weighting="A", time_weighting="Slow", dB=True)

    # 変化点を可視化して比較
    ax.axvline(2.0, linestyle="--", color="red")
    ax.axvline(4.0, linestyle="--", color="red")
    ```

    #### 5. FrameDataset比較でプロットラベルが揃わない

    **症状**: 重ね描きしたときに凡例名や系列対応が分かりにくい

    **原因**: 入力ファイル名やチャンネルラベルが揃っていない

    **対策**:
    ```python
    # 保存前にラベル命名規則を統一
    signal = wd.from_numpy(
        data=data,
        sampling_rate=48000,
        ch_labels=["Normal"]  # 条件名を明示
    )

    # あるいは読み込み後にラベルを確認してから比較
    for frame in dataset:
        print(frame.label)
    ```

    ### デバッグのヒント

    **スペクトルが期待と異なる場合**:
    1. 元信号を時間領域で確認（plot()）
    2. サンプリングレートが正しいか確認
    3. 窓関数の影響を考慮（Hann, Hamming, Blackmanで比較）
    4. ゼロパディングの効果を確認（n_fft > データ長）
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 次のステップ

    高度な信号処理手法を実践的に習得しました。

    **次のノートブック**: [05_custom_functions.ipynb](05_custom_functions.ipynb)

    ここでは、Wandasの処理チェーンに**独自関数を組み込む方法**を学び、
    現場課題に合わせた分析パイプラインを設計する実践に進みます。

    ### このノートブックで学んだこと

    **理論と実践の両面**:
    - STFTパラメータ調整: 時間-周波数分解能のトレードオフを体験
    - Welch法の最適化: パラメータによる性能の違いを比較
    - N-octaveバンド分析: 音響・振動解析で使う対数帯域分析
    - A特性音圧レベル: 時定数付きLAF/LASの計算と解釈
    - FrameDataset応用: 条件別データの効率的な比較分析

    **03との違い**: 基本的な処理方法から、パラメータ調整と比較評価の実践へ

    ### 次の学習目標

    05では以下を学びます:
    - 独自処理関数の作成と適用
    - 既存メソッドチェーンへの安全な組み込み
    - 再利用しやすい分析処理の設計

    ### 発展的なトピック（今後の実装予定）

    Wandasは開発中のため、以下の機能は将来実装予定です:
    - ウェーブレット変換: 時間-周波数-スケール解析
    - 高度なフィルタ設計: カスタムFIR/IIRフィルタ
    - 適応的信号処理: RLSアルゴリズムなど

    ---

    **高度な信号処理の実践スキルを習得しました。次は独自処理の実装に進みましょう。**
    """)
    return


if __name__ == "__main__":
    app.run()
